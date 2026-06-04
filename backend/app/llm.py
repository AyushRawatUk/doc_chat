import json
import httpx
import asyncio
import os
import torch
from openai import OpenAI
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.config import OPENAI_API_KEY, OLLAMA_BASE_URL
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread









# Speed up HF downloads and avoid infinite hangs
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")   # 60s timeout per chunk
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # Mirror — faster in Asia

# Global cache for the HuggingFace local model
_hf_model = None
_hf_tokenizer = None
_hf_loaded_model_name = None # Track which model is currently loaded

def get_hf_model_and_tokenizer(model_name: str):
    """
    Loads and caches the Hugging Face CausalLM model and tokenizer.
    Reloads automatically if a different model_name is requested.
    """
    global _hf_model, _hf_tokenizer, _hf_loaded_model_name
    if _hf_model is None or _hf_tokenizer is None or _hf_loaded_model_name != model_name:
        print(f"Loading Hugging Face CausalLM model: {model_name} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32 if device == "cpu" else torch.float16
        ).to(device)
        _hf_loaded_model_name = model_name
        print(f"Hugging Face model '{model_name}' loaded on {device}!")
    return _hf_model, _hf_tokenizer


def format_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Formulates a structural conversation messages history incorporating RAG context
    and chat memory for the LLM.
    """
    # Build context string
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks):
        source_num = idx + 1
        filename = chunk["filename"]
        page_str = f" (Page {chunk['page']})" if chunk.get("page") else ""
        context_str += f"[{source_num}] Source: {filename}{page_str}\n"
        context_str += f"Content: {chunk['text']}\n"
        context_str += "-" * 40 + "\n\n"
        
    if context_str.strip():
        system_prompt = (
            "You are DocuMind AI, an intelligent, helpful document assistant. "
            "Your goal is to answer the user's question accurately using ONLY the provided document context below. "
            "If you do not know the answer or if the information is not present in the context, "
            "simply state that you cannot find this information in the uploaded documents. "
            "Do NOT invent or extrapolate facts beyond what is in the document.\n\n"
            "CITATION RULES:\n"
            "When referring to information from a source, you MUST cite it at the end of the sentence or block using "
            "the format [number] corresponding to the source index provided in the context (e.g. [1], [2]). "
            "Keep citations concise and accurate.\n\n"
            "DOCUMENT CONTEXT:\n"
            f"{context_str}"
        )
    else:
        system_prompt = (
            "You are DocuMind AI, an intelligent, helpful document assistant. "
            "No relevant document context was found for this specific query. "
            "Please answer the user's question to the best of your general knowledge. "
            "At the beginning of your response, clearly state that this is a general knowledge answer "
            "since no relevant context was found in the uploaded documents."
        )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (up to last 10 messages to avoid context explosion)
    # history is a list of dicts: [{"role": "user"/"assistant", "content": "..."}]
    for msg in history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
        
    # Add final query
    messages.append({"role": "user", "content": query})
    
    return messages


async def stream_openai_response(
    messages: List[Dict[str, str]], 
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Streams responses from OpenAI's API.
    """
    effective_key = api_key or OPENAI_API_KEY
    if not effective_key:
        yield "data: " + json.dumps({"error": "OpenAI API Key is missing. Please set it in Settings."}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
        
    try:
        client = OpenAI(api_key=effective_key)
        response = client.chat.completions.create(
            model=model_name or "gpt-3.5-turbo",
            messages=messages,
            stream=True,
            temperature=0.2
        )
        
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                # Format as SSE event
                yield f"data: {json.dumps({'token': content})}\n\n"
                
    except Exception as e:
        yield "data: " + json.dumps({"error": f"OpenAI error: {str(e)}"}) + "\n\n"
        
    yield "data: [DONE]\n\n"


async def stream_ollama_response(
    messages: List[Dict[str, str]], 
    model_name: str = "llama3"
) -> AsyncGenerator[str, None]:
    """
    Streams responses from a local Ollama instance.
    """
    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model_name or "llama3",
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.2
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", ollama_url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {json.dumps({'error': f'Ollama returned status {response.status_code}: {error_text.decode()}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                    
                async for line in response.iter_lines():
                    if not line:
                        continue
                        
                    data = json.loads(line)
                    # Ollama's response format: {"message": {"content": "token"}}
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                        
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': f'Failed to connect to local Ollama. Please make sure Ollama is running at {OLLAMA_BASE_URL}'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Ollama error: {str(e)}'})}\n\n"
        
    yield "data: [DONE]\n\n"


async def stream_extractive_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """
    Offline fallback: formats retrieved document chunks directly as the answer.
    No model download required — works 100% without internet.
    """
    if not retrieved_chunks:
        yield f"data: {json.dumps({'token': 'No relevant content found in the uploaded documents for your query.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Build a readable extractive answer from top chunks
    intro = f"Based on the uploaded documents, here is what I found regarding **\"{query}\"**:\n\n"
    yield f"data: {json.dumps({'token': intro})}\n\n"
    await asyncio.sleep(0.01)

    for idx, chunk in enumerate(retrieved_chunks[:3]):  # Show top 3 chunks
        filename = chunk.get("filename", "Document")
        page = chunk.get("page", "")
        text = chunk.get("text", "").strip()
        page_str = f", Page {page}" if page else ""

        section = f"**[{idx+1}] From: {filename}{page_str}**\n{text}\n\n"
        # Stream word by word for a nice effect
        words = section.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.005)

    yield "data: [DONE]\n\n"


async def stream_huggingface_response(
    messages: List[Dict[str, str]],
    model_name: str = "deepseek-ai/DeepSeek-V4-Pro"
) -> AsyncGenerator[str, None]:
    """
    Streams responses using HuggingFaceEndpoint and ChatHuggingFace wrapper.
    Queries the remote serverless model (like deepseek-ai/DeepSeek-V4-Pro) without local downloads.
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        # Check API key in env
        # Note: Users should set HUGGINGFACEHUB_API_TOKEN in their environment variables or .env file
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if api_token:
            api_token = api_token.strip().strip("'\"")
        
        # Initialize Endpoint
        llm = HuggingFaceEndpoint(
            repo_id=model_name or "deepseek-ai/DeepSeek-V4-Pro",
            task='text-generation',
            temperature=0.2,
            huggingfacehub_api_token=api_token
        )
        
        model = ChatHuggingFace(llm=llm)

        # Convert dictionary messages to LangChain Message objects (Solution 3 style)
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role in ("assistant", "ai"):
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Stream response using async astream
        has_tokens = False
        try:
            async for chunk in model.astream(lc_messages):
                if chunk.content:
                    has_tokens = True
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        except Exception as stream_err:
            print(f"Hugging Face streaming failed, attempting direct invoke fallback: {stream_err}")
            
        if not has_tokens:
            # Fallback to direct invoke (some models/endpoints do not support streaming)
            try:
                print("Remote model streaming empty. Attempting direct model.ainvoke fallback...")
                response = await model.ainvoke(lc_messages)
                if response and response.content:
                    has_tokens = True
                    yield f"data: {json.dumps({'token': response.content})}\n\n"
            except Exception as invoke_err:
                print(f"Direct invoke fallback failed: {invoke_err}")
                
        if not has_tokens:
            yield f"data: {json.dumps({'token': '⚠️ Remote model did not return any tokens. This can happen if the model is currently cold starting, overloaded, or the API token in your `.env` file contains syntax or formatting issues.'})}\n\n"
                
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Hugging Face (ChatHuggingFace) error: {str(e)}'})}\n\n"
        
    yield "data: [DONE]\n\n"

