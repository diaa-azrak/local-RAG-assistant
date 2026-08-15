from retriever import LocalRetriever
from foundry_local_sdk import Configuration, FoundryLocalManager

def on_download_progress(percentage):
    """Displays real-time download progress for the LLM binary."""
    print(f"\r -> Downloading Chat LLM: {percentage:.1f}%", end="", flush=True)

class LocalRAGAssistant:
    def __init__(self, llm_alias="qwen2.5-0.5b"):
        print("=== Initializing Local RAG Assistant ===", flush=True)
        
        # 1. Initialize Retriever (loads database & embedding model)
        print("[1/3] Loading Retriever and Vector Database...", flush=True)
        self.retriever = LocalRetriever()

        # 2. Safely access the Foundry Local Manager instance
        print(f"[2/3] Accessing Foundry Local Manager...", flush=True)
        try:
            config = Configuration(app_name="LocalRAGAssistant")
            FoundryLocalManager.initialize(config)
        except Exception:
            # Singleton already initialized by LocalRetriever—reusing instance
            pass

        self.manager = FoundryLocalManager.instance

        # 3. Load Chat LLM
        print(f"[3/3] Fetching metadata for '{llm_alias}'...", flush=True)
        self.chat_model = self.manager.catalog.get_model(llm_alias)
        
        print(f"[✓] Downloading model weights (if not cached)...", flush=True)
        self.chat_model.download(on_download_progress)
        print("\n[✓] Chat model download complete!", flush=True)

        print("[✓] Loading model into device memory...", flush=True)
        self.chat_model.load()
        
        # Fixed Python SDK method name:
        self.chat_client = self.chat_model.get_chat_client()
        print("[✓] RAG Assistant is ready and completely offline!\n", flush=True)

    def ask(self, user_question, top_k=2):
        """Processes query through RAG pipeline: Retrieve -> Augment -> Generate."""
        # Step A: Retrieve relevant document chunks from SQLite
        retrieved_chunks = self.retriever.retrieve_chunks(user_question, top_k=top_k)
        context_block = self.retriever.build_augmented_context(retrieved_chunks)

        # Step B: Construct System Prompt with Hardened Guardrails
        system_prompt = (
            "CRITICAL INSTRUCTION:\n"
            "You are a strict factual QA assistant. Your job is to answer the user's query "
            "using ONLY the provided text in the CONTEXT section below.\n\n"
            "STRICT GUARDRAILS:\n"
            "1. NEVER use outside knowledge, prior training data, or general internet facts.\n"
            "2. Do NOT mention creators, origins, history, or dates unless explicitly written in the CONTEXT.\n"
            "3. If the exact answer is not stated in the CONTEXT, respond ONLY with:\n"
            "   \"I don't have enough information in my local knowledge base to answer this question.\"\n"
            "4. Keep your response concise, direct, and 100% faithful to the text provided.\n\n"
            f"=== CONTEXT START ===\n{context_block}\n=== CONTEXT END ==="
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        # Step C: Generate local inference response
        response = self.chat_client.complete_chat(messages)
        answer = response.choices[0].message.content

        return answer, retrieved_chunks


# Interactive Terminal Loop
def main():
    assistant = LocalRAGAssistant()

    print("=" * 60, flush=True)
    print("      OFFLINE LOCAL RAG ASSISTANT (Powered by Foundry Local)")
    print("=" * 60, flush=True)
    print("Type your question below (or type 'exit' to quit):\n", flush=True)

    while True:
        try:
            query = input("\nUser Question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting RAG Assistant. Goodbye!")
                break

            print("\n[Thinking...] Retrieving context & generating answer locally...", flush=True)
            answer, chunks = assistant.ask(query)

            print("\n" + "-" * 50)
            print("ASSISTANT RESPONSE:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

            print("\n[Retrieved Reference Sources]:")
            for score, source, text in chunks:
                print(f" • {source} (Similarity: {score:.4f})")

        except KeyboardInterrupt:
            print("\nExiting RAG Assistant. Goodbye!")
            break

if __name__ == "__main__":
    main()