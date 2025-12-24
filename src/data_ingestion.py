import feedparser
import newspaper
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.chunking_utils import get_semantic_chunks
from src.models import get_spacy_model, get_embedding_model
import os
import shutil
from typing import List
from chromadb.config import Settings
import json
from datetime import datetime
import asyncio

# Configuration
from dotenv import load_dotenv
load_dotenv()

DB_PATH = "./chroma_db"
# Multiple trusted sources for UK Economy/Business
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",     # BBC Business
    "https://www.theguardian.com/uk/business/rss",        # Guardian Business
    "https://feeds.skynews.com/feeds/rss/business.xml"    # Sky News Business
]

# Helper Functions
def format_date(date_struct) -> str:
    """Makes the date readable (YYYY-MM-DD)."""
    if not date_struct:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime(*date_struct[:3]).strftime("%Y-%m-%d")
    except:
        return datetime.now().strftime("%Y-%m-%d")

def get_timestamp(date_struct) -> int:
    """Converts feedparser time struct to Unix timestamp (int)."""
    if not date_struct:
        return int(datetime.now().timestamp())
    try:
        # Notes on unpacking syntax:
        # The 'date_struct' (time.struct_time) is a 9-item tuple:
        # [0] Year (e.g., 2025)
        # [1] Month (1-12)
        # [2] Day (1-31)
        # [3] Hour (0-23)
        # [4] Minute (0-59)
        # [5] Second (0-61)
        # ... plus Day of Week, Day of Year, DST (which we ignore).
        #
        # *date_struct[:6] takes the first 6 items (Year -> Second)
        # and unpacks them as arguments into datetime().
        return int(datetime(*date_struct[:6]).timestamp())
    except:
        return int(datetime.now().timestamp())

def extract_entities_spacy(text: str) -> str:
    """
    Extract named entities using spaCy (faster and cheaper than LLM-based extraction).
    Uses cached model instance for efficiency.
    """
    nlp = get_spacy_model()
    doc = nlp(text[:10000]) # Limit text length for speed
    
    entities = {
        "organizations": set(),
        "people": set(),
        "locations": set()
    }
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["organizations"].add(ent.text)
        elif ent.label_ == "PERSON":
            entities["people"].add(ent.text)
        elif ent.label_ == "GPE":
            entities["locations"].add(ent.text)
            
    # Convert sets to list for JSON serialization
    final_entities = {k: list(v) for k, v in entities.items()}
    return json.dumps(final_entities)

def process_article(entry, feed_url: str) -> List[Document]:
    """Process a single article. Blocking I/O operations run in thread pool."""
    chunks = []
    try:
        print(f"Processing: {entry.title}")
        
        # Download and Parse
        article = newspaper.Article(entry.link)
        article.download()
        article.parse()
        article_text = article.text
        
        # Extract entities with spaCy (Fast & Free)
        entities_json = extract_entities_spacy(article.text)
        
        # Format Date & Timestamp
        date_struct = entry.get("published_parsed")
        formatted_date = format_date(date_struct)
        timestamp = get_timestamp(date_struct)
        
        # Skip if text is too short
        if len(article_text) < 500:
            return chunks
            
        # Semantic Chunking
        chunks = get_semantic_chunks(article_text)
        
        # Add Metadata (Important for retrieval)
        for chunk in chunks:
            chunk.metadata["source"] = entry.link
            chunk.metadata["title"] = entry.title
            chunk.metadata["date"] = formatted_date
            chunk.metadata["timestamp"] = timestamp
            chunk.metadata["entities"] = entities_json
            chunk.metadata["type"] = "news_chunk"
        
    except Exception as e:
        print(f"Failed to process article {entry.link}: {e}")
    
    return chunks

async def fetch_feed_async(feed_url: str) -> List[Document]:
    """Fetch and process RSS feed asynchronously."""
    print(f"Fetching feed: {feed_url}")
    all_chunks = []
    
    try:
        # Parse RSS feed
        feed = feedparser.parse(feed_url)
        print(f"Found {len(feed.entries)} articles in {feed_url}.")
        
        # Limit to top 5 articles per feed
        entries = feed.entries[:5]
        
        # Process each article
        # Note: newspaper3k is blocking, but since we're already async at the feed level,
        # we get good parallelization (3 feeds fetch simultaneously)
        for entry in entries:
            chunks = process_article(entry, feed_url)
            all_chunks.extend(chunks)
            
    except Exception as e:
        print(f"Failed to parse feed {feed_url}: {e}")
    
    return all_chunks

async def fetch_all_feeds_concurrent() -> List[Document]:
    """Fetch all RSS feeds concurrently using asyncio.
    
    Scales well - 100 feeds complete in time of slowest feed, not sum of all.
    """
    tasks = [fetch_feed_async(feed_url) for feed_url in RSS_FEEDS]
    results = await asyncio.gather(*tasks)
    
    # Flatten all chunks from all feeds
    all_chunks = []
    for chunks in results:
        all_chunks.extend(chunks)
    
    return all_chunks

def fetch_and_process_feed() -> List[Document]:
    """Main entry point for RSS feed ingestion. Uses concurrent async fetching."""
    return asyncio.run(fetch_all_feeds_concurrent())

def ingest_data(embedding_function=None) -> None:
    """
    Orchestrates the ingestion process.
    Accepts optional embedding_function to avoid re-initialization.
    """
    print("Fetching and chunking articles...")
    docs = fetch_and_process_feed()
    
    if not docs:
        print("No documents to ingest.")
        return

    print(f"Ingesting {len(docs)} chunks into ChromaDB...")
    
    # Use provided embedding function or get singleton
    if embedding_function is None:
        embedding_function = get_embedding_model()
    
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
