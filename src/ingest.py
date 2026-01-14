import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils import split_text
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Paths and settings
PDF_PATH = "./data/Ebook-Agentic-AI.pdf"
CHROMA_DIR = "./chroma_ai"
COLLECTION_NAME = "agentic_ai"

# Load PDF
print(f"[INFO] Loading PDF from {PDF_PATH} ...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[INFO] Loaded {len(docs)} pages from PDF")


# Split into chunks
print("[INFO] Splitting text into chunks...")
all_texts = []
for doc in docs:
    chunks = split_text(doc.page_content)
    all_texts.extend(chunks)
print(f"[INFO] Created {len(all_texts)} text chunks")


# Generate embeddings
print("[INFO] Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma vectorstore

print(f"[INFO] Creating Chroma vectorstore at {CHROMA_DIR} ...")
vectorstore = Chroma.from_texts(
    texts=all_texts,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
)

# Persist to disk
vectorstore.persist()
print(f"[SUCCESS] Chroma vectorstore created and saved at {CHROMA_DIR}")
