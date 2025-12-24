"""
Singleton instances for expensive models.
Centralized to avoid circular imports between modules.
"""
import spacy
from langchain_huggingface import HuggingFaceEmbeddings

# Spacy NLP model singleton
_nlp_instance = None

def get_spacy_model():
    """Singleton accessor for Spacy model. Loads once, returns cached instance."""
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = spacy.load("en_core_web_sm")
        print("Spacy model loaded into memory (singleton)")
    return _nlp_instance

# Embedding model singleton
_embedding_instance = None

def get_embedding_model():
    """Singleton accessor for embedding model. Loads once, returns cached instance."""
    global _embedding_instance
    if _embedding_instance is None:
        print("Loading embedding model into memory...")
        _embedding_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded")
    return _embedding_instance
