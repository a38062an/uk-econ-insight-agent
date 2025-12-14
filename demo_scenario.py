import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from chromadb.config import Settings
from src import orchestrator

# Load Env
load_dotenv()

# Setup helper to inject fake data
def inject_fake_data():
    print("--- 1. SETTING UP MOCK DATABASE ---")
    DB_PATH = "./chroma_db"
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, 
                         embedding_function=embedding_function, 
                         client_settings=Settings(anonymized_telemetry=False, 
                                                  is_persistent=True))

    now = datetime.now()
    past = now - timedelta(days=7)
    
    # 1. Simulate a REPORT from 7 days ago
    # "Inflation was low at 2%"
    old_report_text = """
    ## Market Report (7 Days Ago)
    ## Major Developments
    - Inflation is currently stable at 2.0%.
    - The Bank of England is considering cutting rates.
    - Tech stocks are flat.
    """
    old_report_doc = Document(
        page_content=old_report_text,
        metadata={
            "type": "report",
            "timestamp": int(past.timestamp()),
            "title": "Old Market Report",
            "date": past.strftime("%Y-%m-%d")
        }
    )
    
    # 2. Simulate NEWS from Today
    # "Inflation spiked to 5%"
    new_news_text = """
    BREAKING: Inflation jumps unexpectedly to 5.0% due to energy crisis.
    Bank of England actively discussing emergency rate hikes.
    Tech stocks plummet as investors panic.
    """
    new_news_doc = Document(
        page_content=new_news_text,
        metadata={
            "type": "news_chunk",
            "timestamp": int(now.timestamp()),
            "title": "Inflation Spike News",
            "date": now.strftime("%Y-%m-%d")
        }
    )

    print(f"Injecting 'Old Report' (Timestamp: {old_report_doc.metadata['timestamp']})")
    vectorstore.add_documents([old_report_doc])
    
    print(f"Injecting 'New News' (Timestamp: {new_news_doc.metadata['timestamp']})")
    vectorstore.add_documents([new_news_doc])
    
    print("Data Injection Complete.\n")

def run_demo():
    # Simulates the "Trend Analysis" requirement.
    # Since we can't wait a week for news to change, we inject fake 'Old' and 'New' data.
    print("--- 2. RUNNING TREND ANALYSIS DEMO ---")
    print("Triggering orchestrator.analyze_trend()...")
    print("Expected Outcome: It should spot that Inflation went from 2% -> 5%.")
    
    start_time = time.time()
    result = orchestrator.analyze_trend()
    end_time = time.time()
    
    print("\n--- 3. RESULT ---")
    print(result)
    print(f"\n(Analysis took {end_time - start_time:.2f} seconds)")
    print("--- DEMO COMPLETE ---")

if __name__ == "__main__":
    inject_fake_data()
    run_demo()
