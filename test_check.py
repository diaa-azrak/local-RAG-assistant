import sqlite3
from vector_db import search_top_k_chunks, init_database
from foundry_local_sdk import Configuration, FoundryLocalManager

def run_diagnostic_check():
    print("=== Running Pre-Phase 2 Diagnostic Check ===")
    
    # 1. Connect to SQLite
    conn = sqlite3.connect("rag_knowledge_base.db")
    cursor = conn.cursor()
    
    # Check stored row count
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_rows = cursor.fetchone()[0]
    print(f"[✓] Total chunks currently stored in SQLite: {total_rows}")
    
    if total_rows == 0:
        print("[!] Warning: Database is empty.")
        return

    # 2. Load Embedding Client
    config = Configuration(app_name="LocalRAGAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    emb_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    emb_model.load()
    emb_client = emb_model.create_embedding_client()

    # 3. Test Query A: Relevant Question
    q_relevant = "How does SQLite store data?"
    v_relevant = emb_client.generate_embeddings([q_relevant]).data[0].embedding
    matches_a = search_top_k_chunks(conn, v_relevant, top_k=1)
    
    print("\n--- Test A: Relevant Query ---")
    print(f"Query: '{q_relevant}'")
    print(f"Top Match Source: {matches_a[0][1]}")
    print(f"Similarity Score: {matches_a[0][0]:.4f}")

    # 4. Test Query B: Irrelevant Query
    q_irrelevant = "What is the capital of France?"
    v_irrelevant = emb_client.generate_embeddings([q_irrelevant]).data[0].embedding
    matches_b = search_top_k_chunks(conn, v_irrelevant, top_k=1)
    
    print("\n--- Test B: Irrelevant Query ---")
    print(f"Query: '{q_irrelevant}'")
    print(f"Top Match Source: {matches_b[0][1]}")
    print(f"Similarity Score: {matches_b[0][0]:.4f}")

    conn.close()
    
    print("\n" + "=" * 50)
    if matches_a[0][0] > matches_b[0][0]:
        print("[SUCCESS] Semantic search is accurately differentiating relevant context!")
    else:
        print("[ATTENTION] Vector scoring behavior unexpected. Needs review.")
    print("=" * 50)

if __name__ == "__main__":
    run_diagnostic_check()