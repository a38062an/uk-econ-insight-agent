import os
import logging
import glob
# Silence ChromaDB Logger
logging.getLogger('chromadb').setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import datetime, timedelta
from src import orchestrator
from src.models import get_spacy_model, get_embedding_model
import glob

class GlobalState:
    def __init__(self):
        self.last_ingestion_time = None

@st.cache_resource
def get_global_state() -> GlobalState:
    return GlobalState()

@st.cache_resource
def initialize_models():
    """Pre-load all models at app startup to eliminate cold-start latency.
    
    Cached by Streamlit and shared across all sessions.
    """
    print("=" * 60)
    print("INITIALIZING MODELS")
    print("=" * 60)
    
    # Pre-load Spacy model
    _ = get_spacy_model()
    
    # Pre-load Embedding model
    _ = get_embedding_model()
    
    # Pre-load Vectorstore (connects to ChromaDB)
    _ = orchestrator.get_vectorstore()
    
    print("=" * 60)
    print("ALL MODELS READY")
    print("=" * 60)
    return True

def run_periodic_ingestion() -> None:
    """Checks if it's time to run ingestion (every 1 hour). Uses Global State."""
    from src import data_ingestion
    state = get_global_state()
    now = datetime.now()
    
    if state.last_ingestion_time is None or (now - state.last_ingestion_time) > timedelta(hours=1):
        print(f"Starting hourly data ingestion... (Last run: {state.last_ingestion_time})")
        
        # UI Feedback inside spinner
        with st.spinner("Fetching latest news (Hourly Refresh)..."):
            data_ingestion.ingest_data()
            
        # Update GLOBAL timestamp
        state.last_ingestion_time = now
        st.session_state.last_run_display = now.strftime("%H:%M")
         
        try:
            print("Auto-generating report...")
            report = orchestrator.generate_report()
            orchestrator.save_report(report)
        except Exception as e:
            print(f"Auto-report failed: {e}")
        
    else:
        if state.last_ingestion_time:
            st.session_state.last_run_display = state.last_ingestion_time.strftime("%H:%M")

def manual_refresh() -> None:
    """Forces an immediate refresh."""
    from src import data_ingestion
    state = get_global_state()
    state.last_ingestion_time = None
    run_periodic_ingestion()

st.set_page_config(
    page_title="UK Economic Insight Agent",
    page_icon=None,
    layout="wide"
)

# Initialize all models at startup (cached, runs once)
initialize_models()

# Initialize Session State Variables that depend on Global State
if 'last_run_display' not in st.session_state:
    st.session_state.last_run_display = "Not Run Yet"

# Check schedule on every app rerun (Lightweight check against Global State)
run_periodic_ingestion()


st.title("UK Economic Insight Agent")

with st.sidebar:
    st.header("Controls")
    
    # API Key Input
    user_api_key = st.text_input(
        "Groq API Key", 
        type="password",
        help="Enter your Groq API key. Get one free at https://console.groq.com"
    )
    if user_api_key:
        os.environ["GROQ_API_KEY"] = user_api_key
        st.success("API key configured!")
    elif not os.getenv("GROQ_API_KEY"):
        st.warning("Please enter your Groq API key to use the chat feature")
    
    st.divider()
    
    # Status
    last_run = st.session_state.get("last_run_display", "Unknown")
    st.caption(f"Last Updated: {last_run}")
    
    # Calculate Next Run
    state = get_global_state()
    if state.last_ingestion_time:
        next_run = state.last_ingestion_time + timedelta(hours=1)
        st.caption(f"Next Run: {next_run.strftime('%H:%M')}")
    else:
        st.caption("Next Run: Pending...")
    
    # Manual Refresh
    if st.button("Force Feed Refresh Now"):
        manual_refresh()
        st.rerun()

    st.divider()
    
    st.info("System Status: Online")


# Main Interface Beginning
if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_report, tab_info = st.tabs(["Chat", "Reports", "Info"])

# Chat Tab
with tab_chat:
    st.subheader("Q&A Assistant")
    
    # Display Chat History
    for chat_entry in st.session_state.messages:
        with st.chat_message(chat_entry["role"]):
            st.markdown(chat_entry["content"])

    user_query = st.chat_input("Ask about the UK economy...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})        

        # Improve Memory: Retrieve last 3 user/assistant interactions for context
        history_context = ""
        recent_history = st.session_state.messages[-6:] # Last 3 turns (User+AI * 3)
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"

        with st.spinner("Thinking..."):
            try:
                model_response = orchestrator.answer_question(user_query, history_context)
            except Exception as e:
                if "authentication" in str(e).lower() or "api" in str(e).lower():
                    model_response = "API key not configured. Please add GROQ_API_KEY to Streamlit secrets (Settings > Secrets)."
                else:
                    model_response = f"An error occurred: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": model_response})
        
        st.rerun()

# Report Tab
with tab_report:
    st.subheader("Market Reports")
    if st.button("Generate New Report"):
        with st.spinner("Analyzing recent data..."):
            report = orchestrator.generate_report()
            saved_path = orchestrator.save_report(report)
            st.success(f"Report saved to {saved_path}")
            st.markdown(report)
    
    st.divider()
    
    # List History
    st.subheader("History")
    report_files = sorted(glob.glob("reports/*.md"), reverse=True)
    
    if not report_files:
        st.info("No reports found.")
        
    for f in report_files:
        with st.expander(os.path.basename(f)):
            with open(f, "r") as r:
                st.markdown(r.read())

# Info Tab
with tab_info:
    st.subheader("About")
    st.markdown("UK Economic Insight Agent v1.0")






