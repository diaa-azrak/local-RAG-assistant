import sqlite3
from vector_db import init_database, insert_document_chunk, search_top_k_chunks
from foundry_local_sdk import Configuration, FoundryLocalManager

def on_progress(percentage):
    """Callback function to display download progress in real time."""
    print(f"\r -> Downloading model: {percentage:.1f}%", end="", flush=True)

def main():
    print("=== Step 1: Starting Ingestion & Diagnostic Check ===", flush=True)

    # 1. Reset / Initialize SQLite Database
    db_filename = "rag_knowledge_base.db"
    conn = init_database(db_filename)
    print(f"[✓] SQLite database connected: '{db_filename}'", flush=True)

    # 2. Initialize Foundry Local SDK
    print("[✓] Initializing Foundry Local SDK...", flush=True)
    config = Configuration(app_name="LocalRAGAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 3. Load Embedding Model
    embedding_alias = "qwen3-embedding-0.6b"
    print(f"[✓] Fetching model metadata for '{embedding_alias}'...", flush=True)
    emb_model = manager.catalog.get_model(embedding_alias)

    # Download if not already cached
    print(f"[✓] Downloading embedding model...", flush=True)
    emb_model.download(on_progress)
    print("\n[✓] Download complete!", flush=True)

    print("[✓] Loading model into local memory...", flush=True)
    emb_model.load()
    
    # Corrected method name for Python SDK
    emb_client = emb_model.get_embedding_client()
    print("[✓] Embedding model loaded successfully!", flush=True)

    # 4. Insert Sample Data
    sample_docs = [
        ("Course FAQ", "Foundry Local runs LLM inference completely offline on CPU, GPU, or NPU with zero internet requirement."),
        ("Course FAQ", "SQLite is a lightweight, serverless database engine that stores all data in a single local file."),
        ("AI Fundamentals", "Retrieval-Augmented Generation (RAG) grounds LLM outputs in user data to prevent hallucinations.")
    ]

    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks")
    conn.commit()

    print("\n=== Step 2: Ingesting Passages into SQLite ===", flush=True)
    for source, text in sample_docs:
        res = emb_client.generate_embedding(text)
        vector = res.data[0].embedding
        insert_document_chunk(conn, source, text, vector)
        print(f" -> Inserted chunk from '{source}'", flush=True)

    # 5. Verify Row Count
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    count = cursor.fetchone()[0]
    print(f"\n[✓] Total Chunks Stored in DB: {count}", flush=True)

    # 6. Test Query Search
    print("\n=== Step 3: Running Semantic Vector Search ===", flush=True)
    test_query = "Does the system work without internet access?"
    print(f"Query: '{test_query}'", flush=True)

    q_res = emb_client.generate_embedding(test_query)
    q_vector = q_res.data[0].embedding

    top_matches = search_top_k_chunks(conn, q_vector, top_k=2)

    print("\n" + "=" * 60, flush=True)
    print("SEARCH RESULTS:", flush=True)
    print("=" * 60, flush=True)
    for score, source, text in top_matches:
        print(f"\n[Similarity Score: {score:.4f}] | Source: {source}", flush=True)
        print(f"Text: \"{text}\"", flush=True)
    print("=" * 60, flush=True)

    conn.close()
    print("\nDiagnostic complete! Phase 1 is fully verified.", flush=True)

if __name__ == "__main__":
    main()