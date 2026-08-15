import streamlit as st
from rag_agent import LocalRAGAssistant

# Page Config
st.set_page_config(
    page_title="Offline Local RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# 1. Cache the RAG Assistant instance so it only loads once into memory
@st.cache_resource(show_spinner=False)
def get_assistant():
    return LocalRAGAssistant()

# Header Section
st.title("🤖 Offline Local RAG Assistant")
st.caption("Powered by **Microsoft Foundry Local** & **SQLite Vector Store** — 100% On-Device & Private")

# Sidebar Details
with st.sidebar:
    st.header("⚙️ System Information")
    st.markdown("""
    - **LLM Runtime:** Microsoft Foundry Local
    - **Chat Model:** `qwen2.5-0.5b`
    - **Embedding Model:** `qwen3-embedding-0.6b`
    - **Database:** SQLite (`rag_knowledge_base.db`)
    - **Status:** 🔴 Completely Offline
    """)
    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 2. Initialize Assistant
with st.spinner("Initializing local models & loading vector database into memory..."):
    assistant = get_assistant()

# 3. Manage Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if present in history
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Reference Sources"):
                for score, source, text in message["sources"]:
                    st.write(f"**Source:** `{source}` | **Similarity:** `{score:.4f}`")
                    st.caption(f'"{text}"')

# 4. Handle User Input
if prompt := st.chat_input("Ask a question about your local documents..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching local SQLite vector DB & generating response..."):
            answer, chunks = assistant.ask(prompt)
            
            st.markdown(answer)
            
            # Display source citations in an expandable drawer
            if chunks:
                with st.expander("📚 View Retrieved Reference Sources"):
                    for score, source, text in chunks:
                        st.write(f"**Source:** `{source}` | **Similarity:** `{score:.4f}`")
                        st.caption(f'"{text}"')

    # Save assistant answer & context sources to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })