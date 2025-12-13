import feedparser
import newspaper
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.chunking_utils import get_semantic_chunks
import os
import shutil
from typing import List
from chromadb.config import Settings

# Configuration
DB_PATH = "./chroma_db"
RSS_FEED_URL = "https://feeds.bbci.co.uk/news/business/rss.xml" # BBC Business for prototype

def fetch_and_process_feed(feed_url: str = RSS_FEED_URL) -> List[Document]:
    """
    Fetches RSS feed, extracts articles, chunks content, and stores related metadata.
    """
    feed = feedparser.parse(feed_url)
    print(f"Found {len(feed.entries)} articles in feed.")
    
    all_chunks = []
    
    # Limit to top 5 articles for performance
    for entry in feed.entries[:5]: 
        print(f"Processing: {entry.title}")
        
        try:
            # Download and Parse
            article = newspaper.Article(entry.link)
            article.download()
            article.parse()
            
            # Skip if text is too short
            if len(article.text) < 500:
                continue
                
            # Semantic Chunking
            chunks = get_semantic_chunks(article.text)
            
            # Add Metadata (Important for retrieval)
            for chunk in chunks:
                chunk.metadata["source"] = entry.link
                chunk.metadata["title"] = entry.title
                
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Failed to process {entry.link}: {e}")
            
    return all_chunks

def ingest_data() -> None:
    """
    Orchestrates the ingestion process.
    """

        
    print("Fetching and chunking articles...")
    docs = fetch_and_process_feed()
    
    if not docs:
        print("No documents to ingest.")
        return

    print(f"Ingesting {len(docs)} chunks into ChromaDB...")
    
    # Initialize Embedding Function
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in Chroma

    Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=DB_PATH,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
    )
    
    print("Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()
