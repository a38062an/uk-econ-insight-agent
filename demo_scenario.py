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
    vectorstore.add_documents([old_report_doc], ids=["mock_report_001"])
    
    print(f"Injecting 'New News' (Timestamp: {new_news_doc.metadata['timestamp']})")
    vectorstore.add_documents([new_news_doc], ids=["mock_news_001"])
    
    print("Data Injection Complete.\n")

def run_demo():
    print("=====================================================================")
    print("          UK ECONOMIC INSIGHT AGENT - WORKING DEMO EVIDENCE          ")
    print("=====================================================================")
    
    vectorstore = orchestrator.get_vectorstore()
    history_buffer = ""

    # SCENARIO 1: Periodic Report Generation
    print("\n\n--- [SCENARIO 1] GENERATING PERIODIC REPORT ---")
    print("(Simulates the hourly summary task)")
    start_time = time.time()
    report = orchestrator.generate_report()
    print(f"\n[Agent Output - Report Preview]:\n{report[:300]}...\n(Truncated for brevity)")
    print(f"(Time: {time.time() - start_time:.2f}s)")


    # SCENARIO 2: Conversational Q&A with Memory
    print("\n\n--- [SCENARIO 2] CONVERSATIONAL Q&A WITH MEMORY ---")
    
    # 2A. Specific Question
    q1 = "What is the inflation rate?"
    print(f"\n[Turn 1] User: '{q1}'")
    
    # Show Grounding
    docs = vectorstore.similarity_search(q1, k=3)
    print("   [GROUNDING CHECK] Retrieved Chunks:")
    for i, d in enumerate(docs):
        clean_content = d.page_content.replace('\n', ' ')[:80]
        print(f"   - Chunk {i+1}: {clean_content}...")
        
    a1 = orchestrator.answer_question(q1, history_buffer)
    history_buffer += f"User: {q1}\nAssistant: {a1}\n"
    print(f"   [Agent]: {a1}")


    # 2B. Follow-up (Memory Test)
    q2 = "Is that good?"
    print(f"\n[Turn 2] User: '{q2}' (Testing Memory of 'Inflation')")
    
    # Show Grounding (Likely empty/irrelevant for "Is that good?", relies on history)
    docs = vectorstore.similarity_search(q2, k=3)
    print("   [GROUNDING CHECK] Retrieval for 'Is that good?':")
    if not docs:
        print("   - No chunks (Correct. Relies on Chat History).")
    else:
        for i, d in enumerate(docs):
            print(f"   - Chunk {i+1}: {d.page_content[:40]}...")

    a2 = orchestrator.answer_question(q2, history_buffer)
    history_buffer += f"User: {q2}\nAssistant: {a2}\n"
    print(f"   [Agent]: {a2}")


    # SCENARIO 3: Trend Analysis
    print("\n\n--- [SCENARIO 3] QUERY-AWARE TREND ANALYSIS ---")
    q3 = "How is inflation trending compared to last week?"
    print(f"\n[Turn 3] User: '{q3}'")
    print("(Triggering analyze_trend logic...)")
    
    trend_result = orchestrator.analyze_trend("inflation")
    print(f"\n   [Agent Output - Trend Analysis]:\n{trend_result}")

    print("\n=====================================================================")
    print("                      END OF DEMO TRANSCRIPT                         ")
    print("=====================================================================")

if __name__ == "__main__":
    inject_fake_data()
    run_demo()
