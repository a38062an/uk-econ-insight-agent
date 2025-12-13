from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

def get_semantic_chunks(text: str) -> List[Document]:
    """
    Splits text into chunks based on semantic similarity.
    This creates a more meaningful split than simple character counting.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embeddings)
    
    # Split the text
    docs = text_splitter.create_documents([text])
    
    return docs
