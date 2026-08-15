import sqlite3
from vector_db import search_top_k_chunks
from foundry_local_sdk import Configuration, FoundryLocalManager

class LocalRetriever:
    def __init__(self, db_path="rag_knowledge_base.db", embedding_alias="qwen3-embedding-0.6b"):
        self.db_path = db_path
        
        # 1. Initialize SDK
        config = Configuration(app_name="LocalRAGAssistant")
        FoundryLocalManager.initialize(config)
        self.manager = FoundryLocalManager.instance
        
        # 2. Load Embedding Model
        self.emb_model = self.manager.catalog.get_model(embedding_alias)
        self.emb_model.load()
        self.emb_client = self.emb_model.get_embedding_client()

    def retrieve_chunks(self, query, top_k=2):
        """Generates query vector and returns top_k matching chunks from SQLite."""
        emb_res = self.emb_client.generate_embedding(query)
        query_vector = emb_res.data[0].embedding
        
        conn = sqlite3.connect(self.db_path)
        top_matches = search_top_k_chunks(conn, query_vector, top_k=top_k)
        conn.close()
        
        return top_matches

    def build_augmented_context(self, retrieved_chunks):
        """Formats retrieved chunks into a clean context block for LLM prompt injection."""
        context_blocks = []
        for score, source, text in retrieved_chunks:
            context_blocks.append(f"[Document Source: {source}]\n{text}")
        
        return "\n\n".join(context_blocks)

# Module Standalone Test
if __name__ == "__main__":
    print("=== Testing Standalone Retriever Module ===", flush=True)
    retriever = LocalRetriever()
    
    test_query = "How does SQLite store its database tables?"
    print(f"\n[Test Query]: '{test_query}'", flush=True)
    
    chunks = retriever.retrieve_chunks(test_query, top_k=2)
    
    print("\n--- Scored Vector Matches ---")
    for score, source, text in chunks:
        print(f" -> Score: {score:.4f} | Source: {source}")
        
    print("\n--- Formatted Augmented Context Block ---")
    context_text = retriever.build_augmented_context(chunks)
    print(context_text)