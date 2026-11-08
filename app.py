import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="Cloud Computing AI Assistant",
    page_icon="☁️",
    layout="wide"
)

INDEX_DIR = "faiss_index"

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        return None
    embeddings = load_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_env = os.getenv("GROQ_API_KEY", "")
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key_env = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    api_key_input = st.text_input(
        "Groq API Key:",
        value=api_key_env,
        type="password",
        help="Get a free key from https://console.groq.com/keys"
    )
    
    model_name = st.selectbox(
        "Select Model:",
        options=["openai/gpt-oss-20b","openai/gpt-oss-safeguard-20b"],
        index=0
    )
    
    retrieval_k = st.slider("Top Chunks (k):", min_value=2, max_value=8, value=4)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("☁️ Cloud Computing Textbook Assistant")
st.caption("Powered by Groq & LangChain • Lightning-fast responses from your 441-page PDF")

vectorstore = load_vectorstore()
if vectorstore is None:
    st.warning("⚠️ Vector index not found! Run `python ingest.py` in your terminal first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to answer any questions from your Cloud Computing book."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 View Citations & Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}**: {src['text']}")

if prompt := st.chat_input("Ask a question from the book..."):
    if not api_key_input:
        st.error("Please enter your Groq API Key in the sidebar or `.env` file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieved_docs = vectorstore.similarity_search(prompt, k=retrieval_k)
        
        context_parts = []
        sources = []
        for doc in retrieved_docs:
            page_num = doc.metadata.get("page", 0) + 1
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
            sources.append({
                "page": page_num,
                "text": doc.page_content[:250] + "..."
            })
        
        context_text = "\n\n".join(context_parts)

        system_instruction = (
            "You are a helpful and precise subject matter expert assistant for the textbook "
            "'Cloud Computing: A Practical Approach for Learning and Implementation'.\n"
            "Answer the user's question accurately using ONLY the provided textbook context.\n"
            "If the context does not contain enough information, state that the textbook does not cover it.\n"
            "Always cite the relevant page numbers in your answer.\n\n"
            f"--- TEXTBOOK CONTEXT ---\n{context_text}"
        )

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key_input
        )

        messages_payload = [SystemMessage(content=system_instruction)]
        for prev in st.session_state.messages[-4:]:
            if prev["role"] == "user":
                messages_payload.append(HumanMessage(content=prev["content"]))
            elif prev["role"] == "assistant":
                messages_payload.append(AIMessage(content=prev["content"]))

        def stream_generator():
            for chunk in llm.stream(messages_payload):
                yield chunk.content

        full_response = st.write_stream(stream_generator())

        with st.expander("📖 View Citations & Sources"):
            for src in sources:
                st.markdown(f"**Page {src['page']}**: {src['text']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })