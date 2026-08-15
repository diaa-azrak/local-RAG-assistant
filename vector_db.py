import sqlite3
import json
import math

# 1. Math Utility: Cosine Similarity
def cosine_similarity(v1, v2):
    """Calculates the similarity (-1.0 to 1.0) between two vector embeddings."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

# 2. Database Utility: SQLite Management
def init_database(db_path="rag_knowledge_base.db"):
    """Creates a local SQLite file and sets up the schema for document chunks."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT,
            chunk_text TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_document_chunk(conn, source_name, chunk_text, embedding_vector):
    """Inserts a text chunk and its vector embedding into SQLite."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO document_chunks (source_name, chunk_text, embedding) VALUES (?, ?, ?)",
        (source_name, chunk_text, json.dumps(embedding_vector))
    )
    conn.commit()

def search_top_k_chunks(conn, query_embedding, top_k=2):
    """Retrieves all chunks, computes cosine similarity against query, returns top matches."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, source_name, chunk_text, embedding FROM document_chunks")
    rows = cursor.fetchall()
    
    scored_results = []
    for row_id, source, chunk_text, emb_str in rows:
        stored_embedding = json.loads(emb_str)
        score = cosine_similarity(query_embedding, stored_embedding)
        scored_results.append((score, source, chunk_text))
        
    scored_results.sort(key=lambda item: item[0], reverse=True)
    return scored_results[:top_k]