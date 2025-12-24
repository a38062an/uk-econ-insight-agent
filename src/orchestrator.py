from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from src.prompts import FACT_PROMPT, SUMMARY_PROMPT, ROUTER_PROMPT, TREND_PROMPT
from src.models import get_embedding_model
from datetime import datetime, timedelta
from chromadb.config import Settings
import os
import json
from typing import Optional

# Suppress arnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
DB_PATH = "./chroma_db"
from dotenv import load_dotenv
load_dotenv()

def get_llm() -> Optional[ChatGroq]:
    """
    Initialize the Groq LLM with Llama-3.3-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return None
        
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

# Cached vectorstore instance (singleton pattern)
_vectorstore_instance = None

def get_vectorstore() -> Chroma:
    """
    Load the ChromaDB vector store using singleton pattern.
    Avoids re-initialization on every function call.
    """
    global _vectorstore_instance
    if _vectorstore_instance is None:
        print("Initializing vectorstore...")
        embedding_function = get_embedding_model()
        _vectorstore_instance = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=embedding_function, 
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        print("Vectorstore loaded")
    return _vectorstore_instance

def classify_intent(query: str) -> str:
    """Uses LLM to route the query to FACT_LOOKUP, TREND_ANALYSIS, or SUMMARY."""
    llm = get_llm()
    if not llm: return "FACT_LOOKUP"
    
    try:
        current_intent = (ROUTER_PROMPT | llm | StrOutputParser()).invoke({
            "question": query
        }).strip()
        # Fallback if model hallucinates extra text
        for valid in ["FACT_LOOKUP", "TREND_ANALYSIS", "SUMMARY", "GENERAL"]:
            if valid in current_intent:
                return valid
        return "FACT_LOOKUP"
    except:
        return "FACT_LOOKUP"

def analyze_trend(query: str = "") -> str:
    """Compares past reports with recent news. Query-Aware."""
    llm = get_llm()
    vectorstore = get_vectorstore()
    
    # Determine search term (Global vs Specific)
    search_term = query if query and len(query) > 10 else "market report economy"
    news_search_term = query if query and len(query) > 10 else "economy inflation"
    
    # 1. Get Past Report (Most recent one)
    # We fetch relevant reports and then SORT by timestamp to ensure we get the latest one.
    past_reports = vectorstore.similarity_search(search_term, k=10, filter={"type": "report"})
    
    if not past_reports:
        old_context = "No past reports found."
        last_report_time = 0 
    else:
        # Sort by timestamp descending (newest first)
        past_reports.sort(key=lambda x: x.metadata.get("timestamp", 0), reverse=True)
        latest_report = past_reports[0]
        
        old_context = latest_report.page_content
        last_report_time = latest_report.metadata.get("timestamp", 0)
        
    # 2. Get Recent News (Strictly AFTER the last report)
    # Filter: type is 'news_chunk' AND timestamp > last_report_time
    recent_news = vectorstore.similarity_search(
        news_search_term, 
        k=5, 
        filter={
            "$and": [
                {"type": {"$eq": "news_chunk"}},
                {"timestamp": {"$gt": last_report_time}}
            ]
        }
    )
    new_context = "\n".join([doc.page_content for doc in recent_news])
    
    chain = TREND_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "old_context": old_context, 
        "new_context": new_context, 
        "topic": search_term
    })


def lookup_facts(query: str, chat_history: str = "") -> str:
    """Performs retrieval for FACT_LOOKUP intent with history."""
    llm = get_llm()
    if not llm:
        return "Please enter your Groq API Key first."
        
    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return "I cannot find information in my database to answer that right now."
        
    context_text = "\n\n".join([d.page_content for d in docs])
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    chain = FACT_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "context": context_text, 
        "chat_history": chat_history, 
        "question": query,
        "date": today_str
    })


def generate_report() -> str:
    """
    Generates a market summary report.
    """
    llm = get_llm()
    if not llm:
        return "Please enter your Groq API Key first."
        
    vectorstore = get_vectorstore()
    
    # Broad search for recent ALL UK Econ news
    # We fetch a larger pool (k=20) and sort by timestamp to get the LATEST news.
    docs = vectorstore.similarity_search("UK Economy market updates", k=20, filter={"type": "news_chunk"})
    
    # Sort by timestamp descending (newest first)
    docs.sort(key=lambda x: x.metadata.get("timestamp", 0), reverse=True)
    
    # Take the top 10 most recent chunks
    recent_docs = docs[:10]
    
    if not recent_docs:
        return "No recent news found to generate a report."
        
    combined_text = "\n\n".join([f"Title: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}" for doc in recent_docs])
    
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "text": combined_text
    })

def save_report(report_content: str) -> str:
    """Saves the report to the reports/ directory."""
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    filename = f"reports/{datetime.now().strftime('%Y-%m-%d_%H-%M')}_Market_Report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    # Save report to memory to use in later prompts
    try:
        vectorstore = get_vectorstore()
        doc = Document(
            page_content=report_content,
            metadata={
                "source": "Agent Generated",
                "title": f"Market Report {datetime.now().strftime('%Y-%m-%d')}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": int(datetime.now().timestamp()),
                "type": "report"
            }
        )
        vectorstore.add_documents([doc])
        print("Report indexed to memory.")
    except Exception as e:
        print(f"Failed to index report: {e}")

    return filename

def answer_question(query: str, chat_history: str = "") -> str:
    """
    Retrieve relevant context and generate an answer using the LLM.
    """
    llm = get_llm()
    if not llm:
        return "Please ensure GROQ_API_KEY is set in your .env file."
        
    vectorstore = get_vectorstore()
    intent = classify_intent(query)
    
    print(f"Router Decision: {intent}")
    
    if intent == "TREND_ANALYSIS":
        return analyze_trend(query)
        
    elif intent == "SUMMARY":
        # Generate a fresh report-style summary
        return generate_report()

    elif intent == "GENERAL":
        return "I am the UK Economic Insight Agent. I can provide market reports, analyze trends, or answer specific questions about the economy. How can I help you today?"
        
    else: # FACT_LOOKUP
        return lookup_facts(query, chat_history)