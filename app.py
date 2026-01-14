import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
# Streamlit Page Config
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: black; }
h1 { font-size:44px !important; font-weight:800; color:#ffffff; }
h3 { font-size:28px !important; font-weight:700; color:#f1f1f1; }
label { font-size:22px !important; font-weight:700; color:#eaeaea; }
.stTextInput input { font-size:18px !important; }
.stButton > button { background-color:#22c55e; color:black; font-size:18px; font-weight:700; padding:10px 26px; border-radius:10px; border:none; }
.stButton > button:hover { background-color:#16a34a; }
.answer-box { background:#111827; border-left:6px solid #22c55e; border-radius:14px; padding:18px 22px; font-size:18px; color:#f9fafb; line-height:1.6; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic AI Chatbot")
st.markdown("<h3>Ask questions about the Agentic AI eBook 📖</h3>", unsafe_allow_html=True)


# Load Embeddings

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()


# Load Chroma Vector Store

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_ai",
        collection_name="agentic_ai",
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()


# Retriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Load Groq LLaMA 3.1 8B Instant

@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("Groq_API")
    )

llm = load_llm()

# Prompt Template

prompt = ChatPromptTemplate.from_template(
    """You are an AI research assistant.

Use the following context to answer the question.
If the answer is not in the context, say "I cannot answer based on the provided document."

Context:
{context}

Question:
{question}

Answer in clear, concise language:
"""
)


# Helper: format retrieved docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG Chain: Retrieve + Generate

def run_rag(question: str):
    docs_and_scores = retriever.invoke(question)
    
    docs = [d for d in docs_and_scores]
    # Approximate confidence based on default similarity
    scores = [d.metadata.get("score", 0.8) for d in docs_and_scores]
    
    context = format_docs(docs)
    
    response = llm.invoke(prompt.format(context=context, question=question))
    
    confidence = round(min(1.0, max(0.0, 1 - sum(scores)/len(scores))), 3) if scores else 0.0
    
    return {
        "answer": response.content,
        "confidence": confidence,
        "retrieved_context": context
    }


# Streamlit UI

question = st.text_input(
    "❓ Ask a question from the Agentic AI eBook",
    placeholder="e.g. What is Agentic AI?"
)

if st.button("🤖 Ask AI Chatbot"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):
            result = run_rag(question)
        
        st.markdown("### ✅ Answer")
        st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
        
        st.markdown(f"**Confidence:** {result['confidence']}")
        
        st.markdown("### 📄 Retrieved Context")
        st.markdown(f"<div class='answer-box'>{result['retrieved_context']}</div>", unsafe_allow_html=True)
        
        st.markdown("<hr><center>💡 Powered by Agentic AI RAG + Groq</center>", unsafe_allow_html=True)
