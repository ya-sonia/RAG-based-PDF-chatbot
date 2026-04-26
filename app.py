import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama



st.set_page_config(
    page_title="Smart PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


st.markdown("""
<style>

/* App background (clean light gray) */
.stApp {
    background-color: #f5f7fb;
    color: #111;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 10px;
}

/* Chat container spacing */
.block-container {
    padding-top: 2rem;
}

/* User message */
.user-msg {
    background: #dbeafe;
    padding: 12px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: #111;
    width: fit-content;
    max-width: 80%;
    border: 1px solid #bfdbfe;
}

/* AI message */
.ai-msg {
    background: #ffffff;
    padding: 12px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: #111;
    width: fit-content;
    max-width: 80%;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title"> Smart PDF Chatbot (RAG + Ollama)</div>', unsafe_allow_html=True)
st.write("Upload a PDF and start a smart conversation with your document")


# ---------------- SESSION MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------------- LOAD PDF & BUILD INDEX ----------------
def process_pdf(pdf_file):

    with open("economics.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("economics.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db


# ---------------- RAG QUERY ----------------
def ask_rag(db, question):

    #  improve follow-up understanding
    history_context = "\n".join(
        [f"User: {q}\nAI: {a}" for q, a in st.session_state.chat_history[-3:]]
    )

    query = f"""
Previous conversation:
{history_context}

User question:
{question}
"""

    docs = db.max_marginal_relevance_search(query, k=3, fetch_k=8)

    context = "\n\n---\n\n".join([d.page_content[:1200] for d in docs])

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful AI tutor.

RULES:
- Use ONLY the given context
- If user says "explain it", "summarize it", understand previous topic
- Be simple and clear
- If answer not in context say: "Not found in document"
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Conversation:
{history_context}

Question:
{question}

Answer:
"""
            }
        ]
    )

    return response["message"]["content"]



with st.sidebar:
    st.header(" Upload PDF")

    pdf = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf:
        st.success("PDF uploaded successfully")

        if st.button(" Process PDF"):
            with st.spinner("Processing document..."):
                st.session_state.db = process_pdf(pdf)
                st.session_state.pdf_name = pdf.name
                st.success("PDF is ready for chat! ")


# ---------------- CHAT UI ----------------
if st.session_state.db:

    st.subheader(f"Chat with: {st.session_state.pdf_name}")

    user_input = st.text_input("Ask your question :")

    if st.button("Send") and user_input:

        answer = ask_rag(st.session_state.db, user_input)

        # store memory
        st.session_state.chat_history.append((user_input, answer))

    # ---------------- DISPLAY CHAT ----------------
    st.markdown("###  Conversation")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='user-msg'>🧑 {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-msg'>🤖 {a}</div>", unsafe_allow_html=True)

else:
    st.info("Upload a PDF to start chatting ")