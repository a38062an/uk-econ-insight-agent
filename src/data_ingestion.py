import feedparser
import newspaper
import spacy
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.chunking_utils import get_semantic_chunks
import os
import shutil
from typing import List
from chromadb.config import Settings
import json
from datetime import datetime
from src.prompts import ENTITY_PROMPT
from src.orchestrator import get_llm
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

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

# Load Spacy Model for Entity Extraction (better over using LLM for speed and cost)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

# Optimized Extraction (No LLM Cost)
def extract_entities_spacy(text: str) -> str:
    """
    Use spaCy to grab entities (using over llm for speed and token cost)
    """
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

# Main Ingestion Functions
def fetch_and_process_feed() -> List[Document]:
    """
    Main loop:
    1. Check all feeds (BBC, Guardian, Sky)
    2. Download article text
    3. Run spaCy to find companies/people/locations
    4. Chunk it up so the LLM can read it
    """
    all_chunks = []
    
    for feed_url in RSS_FEEDS:
        print(f"Fetching feed: {feed_url}")
        try:
            feed = feedparser.parse(feed_url)
            print(f"Found {len(feed.entries)} articles in {feed_url}.")
            
            # Limit to top 3 articles per feed for prototype (3 feeds * 3 arts = 9 arts total)
            for entry in feed.entries[:5]: 
                print(f"Processing: {entry.title}")
                
                try:
                    # Download and Parse
                    article = newspaper.Article(entry.link)
                    article.download()
                    article.parse()
                    article_text = article.text
                    
                    # Extract entities with spaCy (Fast & Free)
                    entities_json = extract_entities_spacy(article.text)
                    
                    # Format Date & Timestamp
                    # Verified for BBC, Guardian, Sky News: All use 'published' / 'published_parsed'
                    date_struct = entry.get("published_parsed")
                        
                    formatted_date = format_date(date_struct)
                    timestamp = get_timestamp(date_struct)
                    
                    # Skip if text is too short
                    if len(article_text) < 500:
                        continue
                        
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
    
                    all_chunks.extend(chunks)
                    
                except Exception as e:
                    print(f"Failed to process article {entry.link}: {e}")
                    
        except Exception as e:
             print(f"Failed to parse feed {feed_url}: {e}")
             
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
