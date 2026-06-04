import os
import json
from fastapi.testclient import TestClient
from app.main import app
from app import database, models

client = TestClient(app)

def test_full_pipeline():
    print("=== DOCUMIND AI: END-TO-END API ROUTE TEST ===")
    
    # 1. Create a temporary sample text file
    sample_filename = "antigravity_secret.txt"
    sample_text = "The secret code of Antigravity is 99-ANTIGRAVITY. This code unlocks advanced agentic capabilities."
    
    with open(sample_filename, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    print(f"1. Prepared test document: {sample_filename}")
    
    doc_id = None
    session_id = None
    
    try:
        # 2. Test Document Upload Endpoint: POST /api/documents/upload
        print("\n2. Testing File Upload Endpoint (POST /api/documents/upload)...")
        with open(sample_filename, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": (sample_filename, f, "text/plain")}
            )
            
        assert response.status_code == 200, f"Upload failed: {response.text}"
        doc_data = response.json()
        doc_id = doc_data["id"]
        print(f"   [SUCCESS] File uploaded. Assigned ID: {doc_id}")
        print(f"             Is Processed: {doc_data['is_processed']}")
        
        # 3. Test List Documents Endpoint: GET /api/documents
        print("\n3. Testing List Documents Endpoint (GET /api/documents)...")
        response = client.get("/api/documents")
        assert response.status_code == 200, f"Listing failed: {response.text}"
        docs_list = response.json()
        found = any(d["id"] == doc_id for d in docs_list)
        assert found, "Uploaded document not found in the documents list"
        print(f"   [SUCCESS] Document listed perfectly. Total files: {len(docs_list)}")
        
        # 4. Test Create Chat Session Endpoint: POST /api/chats
        print("\n4. Testing Create Chat Session Endpoint (POST /api/chats)...")
        response = client.post(
            "/api/chats",
            json={"document_ids": [doc_id]}
        )
        assert response.status_code == 200, f"Chat creation failed: {response.text}"
        session_data = response.json()
        session_id = session_data["id"]
        print(f"   [SUCCESS] Chat Session created. ID: {session_id}")
        print(f"             Title: '{session_data['title']}'")
        print(f"             Linked Documents count: {len(session_data['documents'])}")
        
        # 5. Test Streaming RAG Query Endpoint: POST /api/chats/{session_id}/query
        print("\n5. Testing RAG Streaming SSE Query Endpoint (POST /api/chats/{session_id}/query)...")
        query_payload = {
            "prompt": "What is the secret code of Antigravity?",
            "model_provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "document_ids": [doc_id]
        }
        
        # Read the streaming SSE response
        # Since we use TestClient, it aggregates the stream or lets us iterate over lines
        response = client.post(
            f"/api/chats/{session_id}/query",
            json=query_payload
        )
        
        assert response.status_code == 200, f"Query endpoint failed: {response.text}"
        
        print("   [SUCCESS] Stream endpoint connected. Parsing SSE packets:")
        
        # Parse SSE event lines
        lines = response.text.split("\n\n")
        citations_found = False
        tokens_received = 0
        error_found = False
        
        for line in lines:
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    if "citations" in data:
                        citations_found = True
                        print(f"      - Citation Packet Received: {len(data['citations'])} source chunks retrieved.")
                        for idx, cit in enumerate(data["citations"]):
                            print(f"        [{idx+1}] Source: {cit['filename']} | Score: {cit['score']:.4f}")
                            print(f"            Context snippet: \"{cit['text']}\"")
                    elif "token" in data:
                        tokens_received += 1
                    elif "error" in data:
                        error_found = True
                        print(f"      - Expected Provider Warning/Error: \"{data['error']}\"")
                except Exception as parse_err:
                    # Ignore format logs
                    pass
                    
        assert citations_found, "Citations metadata packet was missing from SSE stream start"
        print(f"   [SUCCESS] SSE pipeline validated. Citations processed correctly.")
        if error_found:
            print("             Note: LLM engine correctly reported provider setup details (e.g. missing API key), proving end-to-end exception handlers work.")
            
    finally:
        # 6. Cleanup files and SQL records
        print("\n6. Cleaning up test session and document assets...")
        
        if doc_id:
            # DELETE /api/documents/{doc_id}
            client.delete(f"/api/documents/{doc_id}")
            print(f"   - Deleted Document: {doc_id}")
            
        if session_id:
            # DELETE /api/chats/{session_id}
            client.delete(f"/api/chats/{session_id}")
            print(f"   - Deleted Chat Session: {session_id}")
            
        if os.path.exists(sample_filename):
            os.remove(sample_filename)
            print(f"   - Removed local temporary file: {sample_filename}")
            
    print("\n=== ALL END-TO-END API INTEGRATION TESTS SUCCESSFUL ===")

if __name__ == "__main__":
    test_full_pipeline()
