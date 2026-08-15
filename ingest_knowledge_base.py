import os
import sqlite3
from vector_db import init_database, insert_document_chunk
from foundry_local_sdk import Configuration, FoundryLocalManager

def chunk_document_by_paragraphs(text_content):
    """Splits raw text content into individual non-empty paragraph chunks."""
    paragraphs = text_content.split("\n\n")
    cleaned_chunks = [p.strip().replace("\n", " ") for p in paragraphs if p.strip()]
    return cleaned_chunks

def main():
    print("=== Phase 2: Ingesting Knowledge Base Files ===", flush=True)

    kb_folder = "knowledge_base"
    db_filename = "rag_knowledge_base.db"

    if not os.path.exists(kb_folder):
        print(f"[!] Error: Folder '{kb_folder}' does not exist.", flush=True)
        return

    # 1. Connect to SQLite and wipe old data to ensure clean ingestion
    conn = init_database(db_filename)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks")
    conn.commit()
    print(f"[✓] SQLite database reset: '{db_filename}'", flush=True)

    # 2. Initialize Foundry Local embedding model
    print("[✓] Initializing Foundry Local SDK...", flush=True)
    config = Configuration(app_name="LocalRAGAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_alias = "qwen3-embedding-0.6b"
    print(f"[✓] Loading embedding model '{embedding_alias}'...", flush=True)
    emb_model = manager.catalog.get_model(embedding_alias)
    emb_model.download()
    emb_model.load()
    emb_client = emb_model.get_embedding_client()
    print("[✓] Embedding model ready!", flush=True)

    # 3. Process all text files in the knowledge_base directory
    txt_files = [f for f in os.listdir(kb_folder) if f.endswith(".txt")]
    print(f"\n[✓] Found {len(txt_files)} file(s) to process in '{kb_folder}/'", flush=True)

    total_chunks_ingested = 0

    for file_name in txt_files:
        file_path = os.path.join(kb_folder, file_name)
        print(f"\n--- Processing File: {file_name} ---", flush=True)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk document into paragraphs
        chunks = chunk_document_by_paragraphs(content)
        print(f" -> Split into {len(chunks)} paragraph chunk(s).", flush=True)

        # Generate embeddings & store in SQLite
        for idx, chunk_text in enumerate(chunks, 1):
            emb_res = emb_client.generate_embedding(chunk_text)
            vector = emb_res.data[0].embedding
            
            insert_document_chunk(conn, file_name, chunk_text, vector)
            print(f"   [Chunk {idx}] Saved embedding ({len(vector)} dims) to SQLite", flush=True)
            total_chunks_ingested += 1

    conn.close()

    print("\n" + "=" * 50, flush=True)
    print(f"INGESTION COMPLETE: {total_chunks_ingested} chunks stored in SQLite!", flush=True)
    print("=" * 50, flush=True)

if __name__ == "__main__":
    main()