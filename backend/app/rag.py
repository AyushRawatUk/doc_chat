import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from app.config import INDEX_DIR, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from datasets import load_dataset
import json
from transformers import RobertaTokenizerFast
from itertools import islice, chain





# ==========================================
# 1. GEGLU Feed-Forward Network
# ==========================================
class GEGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.gate_proj(x)
        content, gate = projected.chunk(2, dim=-1)
        x = content * F.gelu(gate)
        x = self.dropout(x)
        return self.out_proj(x)


# ==========================================
# 2. Rotary Positional Embeddings (RoPE)
# ==========================================
def precompute_freqs_cis(dim, seq_len, theta=10000.0, device='cpu'):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs.to(device)).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat([-x2, x1], dim=-1)
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1).unsqueeze(0).unsqueeze(0)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return (x * freqs_cos) + (x_rotated * freqs_sin)


# ==========================================
# 3. Modern Attention
# ==========================================
class ModernAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout



    def forward(self, x, mask=None, freqs_cos=None, freqs_sin=None):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        float_mask = None
        if mask is not None:
            float_mask = torch.zeros_like(mask, dtype=x.dtype)
            float_mask = float_mask.masked_fill(~mask, -10000.0)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=float_mask,
            dropout_p= self.dropout if self.training else 0.0
        )
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, -1)
        return self.out_proj(attn_output)


# ==========================================
# 4. Transformer Block
# ==========================================
class ModernTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = ModernAttention(d_model, n_heads)
        self.ffn = GEGLUFFN(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, freqs_cos=None, freqs_sin=None):
        x = x + self.dropout(
            self.attn(
                self.norm1(x),
                mask=mask,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin
            )
        )

        x = x + self.dropout(
            self.ffn(self.norm2(x))
        )
        return x


# ==========================================
# 5. ModernBERT Model
# ==========================================
class ModernBERT(nn.Module):

    def __init__(self, vocab_size=30522, d_model=384, n_heads=6, num_layers=6, window_size=64, num_classes=2):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # ✅ store explicitly
        self.window_size = window_size
        self.vocab_size = vocab_size

        hidden_dim = int(d_model * (8 / 3))
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerLayer(d_model, n_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        self.lm_head.weight = self.token_emb.weight

        self._freqs_cos = None
        self._freqs_sin = None
        self._local_mask = None
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(0.1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_freqs(self, seq_len, device):
        if (self._freqs_cos is None or self._freqs_cos.shape[0] != seq_len or self._freqs_cos.device != device):
            self._freqs_cos, self._freqs_sin = precompute_freqs_cis(
            self.head_dim, seq_len, device=device
        )
        return self._freqs_cos, self._freqs_sin

    def get_local_mask(self, seq_len, device):
        if self._local_mask is None or self._local_mask.shape[-1] != seq_len:
            indices = torch.arange(seq_len, device=device)
            distance = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            mask = distance <= self.window_size
            self._local_mask = mask.unsqueeze(0).unsqueeze(0)
        return self._local_mask.to(device)

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        seq_len = input_ids.shape[1]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        x = self.token_emb(input_ids)
        freqs_cos, freqs_sin = self.get_freqs(seq_len, device)
        local_mask = self.get_local_mask(seq_len, device)

        for i, layer in enumerate(self.layers):

            combined_mask = attention_mask

            if i % 3 != 0:
                combined_mask = combined_mask & local_mask

            x = layer(x, mask=combined_mask, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        x = self.final_norm(x)
        mask = attention_mask.squeeze(1).squeeze(1)

        masked_x = x * mask.unsqueeze(-1)

        pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        pooled = self.embed_proj(pooled)
        pooled = F.normalize(pooled, p=2, dim=1)


        return pooled




class Embedding_model():
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


        SEQ_LEN = 128

        best_val_loss = float('inf')
        patience = 1
        patience_counter = 0
        training_history = []
        accumulation_steps = 4
        vocab_size = self.tokenizer.vocab_size
        epochs = 10


        model = ModernBERT(vocab_size=vocab_size, d_model=768, n_heads=12, num_layers=12, window_size=64)
        model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=torch.device('cpu')), strict=False)
        self.model = model

    def encode(self, text):
        tokenized_text = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            result = self.model(
                tokenized_text['input_ids'],
                tokenized_text["attention_mask"].bool()
            )
        return result.detach().numpy()


_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = Embedding_model()
    return _model
    



def split_text_into_chunks(
    pages_content: List[Dict[str, Any]], 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page text into overlapping chunks.
    Retains page references for accurate source citation.
    """
    chunks = []
    chunk_idx = 0
    
    for page_data in pages_content:
        text = page_data["text"]
        page_num = page_data["page"]
        
        # Safe boundary conditions
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_index": chunk_idx,
                "text": text.strip(),
                "page": page_num
            })
            chunk_idx += 1
            continue
            
        start = 0
        while start < len(text):
            # If remaining characters are less than or equal to the overlap size, 
            # they are already captured in the previous overlap step, so we stop
            if len(text) - start <= chunk_overlap:
                break
                
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to split at a logical boundary (space, newline) if possible to avoid cutting words
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                last_newline = chunk_text.rfind("\n")
                boundary = max(last_space, last_newline)
                
                # If a reasonable boundary is found, adjust the end of the chunk
                if boundary > chunk_size // 2:
                    end = start + boundary
                    chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text.strip(),
                    "page": page_num
                })
                chunk_idx += 1
                
            # Shift start by chunk size minus overlap to create the overlap region
            start = end - chunk_overlap
                
    return chunks


def build_and_save_index(document_id: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Generates embeddings for all text chunks of a document,
    builds a FAISS index, and saves it along with metadata JSON.
    """
    if not chunks:
        return
        
    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings (shape: [num_chunks, embedding_dim])
    embeddings = model.encode(texts)
    
    # Convert embeddings to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    # Create flat L2 index (or IP for cosine similarity if normalized)
    # Standard IndexFlatL2 is stable and robust for smaller documents
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save FAISS index
    index_path = os.path.join(INDEX_DIR, f"{document_id}.index")
    faiss.write_index(index, index_path)
    
    # Save chunks metadata
    meta_path = os.path.join(INDEX_DIR, f"{document_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Vector database created and saved for document {document_id} ({len(chunks)} chunks).")


def delete_document_index(document_id: str) -> None:
    """
    Deletes the FAISS index and metadata files for a given document.
    """
    index_path = os.path.join(INDEX_DIR, f"{document_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{document_id}.json")
    
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)
        
    print(f"Deleted vector index files for document {document_id}.")


def retrieve_relevant_chunks(
    query: str, 
    document_ids: List[str], 
    top_k: int = RETRIEVAL_TOP_K,
    doc_metadata_mapping: Dict[str, str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the most semantically relevant text chunks from the vector indices of specified documents.
    
    Returns a list of dicts:
    [
        {
            "doc_id": str,
            "filename": str,
            "chunk_index": int,
            "text": str,
            "page": int,
            "score": float
        },
        ...
    ]
    """
    if not document_ids:
        return []
        
    model = get_embedding_model()
    
    # Generate query embedding
    query_embedding = model.encode([query]).astype('float32')
    
    all_results = []
    
    for doc_id in document_ids:
        index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        meta_path = os.path.join(INDEX_DIR, f"{doc_id}.json")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print(f"Warning: Index files for document {doc_id} not found. Skipping.")
            continue
            
        try:
            # Load index and metadata
            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                
            # Search
            # D: distances (L2 distances, lower is better/closer)
            # I: indices of the matching vectors
            D, I = index.search(query_embedding, min(top_k, len(chunks)))
            
            # Map search results back to metadata
            filename = doc_metadata_mapping.get(doc_id, "Unknown File") if doc_metadata_mapping else "Document"
            
            for rank in range(len(I[0])):
                idx = int(I[0][rank])
                if idx == -1 or idx >= len(chunks):
                    continue
                    
                chunk = chunks[idx]
                distance = float(D[0][rank])
                
                # Convert L2 distance to a standard similarity score for readability
                # L2 distance is positive; closer to 0 means more similar
                # Simple conversion: 1 / (1 + distance)
                similarity_score = 1.0 / (1.0 + distance)
                
                all_results.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "page": chunk.get("page", 1),
                    "score": similarity_score
                })
        except Exception as e:
            print(f"Error searching vector index for document {doc_id}: {e}")
            continue
            
    # Sort all results combined across multiple documents by similarity score (descending)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top K chunks overall
    return all_results[:top_k]
