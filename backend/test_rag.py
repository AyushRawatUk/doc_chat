import os
import shutil
from app import parser, rag, config

def run_test():
    print("=== DOCUMIND AI: RAG PIPELINE SANITY TEST ===")
    
    # 1. Create a dummy test file
    test_file_path = os.path.join(config.UPLOAD_DIR, "test_doc.md")
    test_content = """# Artificial Intelligence and Machine Learning
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

## What is Deep Learning?
Deep learning is a subset of machine learning, which is in turn a subset of artificial intelligence. Deep learning is based on artificial neural networks, particularly deep neural networks, that learn from vast amounts of data.

## Applications of AI in Healthcare
AI in healthcare is transforming patient care. Intelligent algorithms can analyze medical imaging (like MRIs and X-rays) to detect anomalies such as tumors with high precision. Moreover, AI can help synthesize patient data to predict risk and suggest treatment paths.
"""
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
        
    print(f"1. Created dummy file: {test_file_path}")
    
    try:
        # 2. Test Parser
        print("\n2. Parsing document...")
        pages_content = parser.parse_document(test_file_path, ".md")
        print(f"Extracted pages: {len(pages_content)}")
        for idx, page in enumerate(pages_content):
            print(f"--- Page {page['page']} (First 150 chars) ---")
            print(page["text"][:150] + "...")
            
        # 3. Test Chunking
        print("\n3. Chunking document...")
        chunks = rag.split_text_into_chunks(pages_content, chunk_size=200, chunk_overlap=30)
        print(f"Created {len(chunks)} overlapping chunks.")
        for idx, chunk in enumerate(chunks):
            print(f"  Chunk {chunk['chunk_index']} (Page {chunk['page']}): {chunk['text'][:100]}...")
            
        # 4. Test Embeddings and Vector DB indexing
        print("\n4. Generating embeddings and building FAISS index...")
        document_id = "test_document_uuid_123"
        # Force loading the model
        rag.get_embedding_model()
        rag.build_and_save_index(document_id, chunks)
        
        index_file = os.path.join(config.INDEX_DIR, f"{document_id}.index")
        meta_file = os.path.join(config.INDEX_DIR, f"{document_id}.json")
        
        print(f"Index file exists: {os.path.exists(index_file)}")
        print(f"Metadata file exists: {os.path.exists(meta_file)}")
        
        # 5. Test Retrieval
        print("\n5. Testing retrieval semantic search...")
        query = "How is AI used in medical imaging and healthcare?"
        print(f"Query: '{query}'")
        
        mapping = {document_id: "test_doc.md"}
        results = rag.retrieve_relevant_chunks(query, [document_id], top_k=2, doc_metadata_mapping=mapping)
        
        print(f"Retrieved {len(results)} chunks:")
        for idx, res in enumerate(results):
            print(f"  [{idx+1}] Score: {res['score']:.4f} | Source: {res['filename']} (Page {res['page']})")
            print(f"      Text: \"{res['text']}\"")
            print("-" * 50)
            
        print("\n=== RAG SANITY TEST SUCCESSFUL ===")
        
    finally:
        # Cleanup test files
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        rag.delete_document_index("test_document_uuid_123")
        print("Cleaned up test assets.")

if __name__ == "__main__":
    run_test()
