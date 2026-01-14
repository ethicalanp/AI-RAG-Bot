from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits a large text into smaller chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
