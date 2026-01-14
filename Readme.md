Agentic AI RAG Chatbot

flowchart TD
    A[Agentic AI PDF] --> B[PyPDFLoader]
    B --> C[Text Chunking]
    C --> D[HuggingFace Embeddings]
    D --> E[Chroma Vector DB]

    F[User Question] --> G[Retriever]
    E --> G
    G --> H[Relevant Chunks]
    H --> I[Groq LLaMA 3.1 8B]
    I --> J[Grounded Answer]

    J --> K[Streamlit UI]
    H --> K


A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the Agentic AI eBook using:

Chroma (vector database)

HuggingFace embeddings

Groq LLaMA-3.1-8B-Instant

Streamlit UI

Setup

pip install -r requirements.txt


export GROQ_API_KEY="your_key_here"
# Windows (PowerShell)
setx GROQ_API_KEY "your_key_here"

python ingest.py

streamlit run app.py