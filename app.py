import os
import logging
import glob
# Silence ChromaDB Logger
logging.getLogger('chromadb').setLevel(logging.ERROR)

import streamlit as st
from datetime import datetime, timedelta
from src import orchestrator, data_ingestion
import glob

class GlobalState:
    def __init__(self):
        self.last_ingestion_time = None

@st.cache_resource
def get_global_state() -> GlobalState:
    return GlobalState()

def run_periodic_ingestion() -> None:
    """Checks if it's time to run ingestion (every 1 hour). Uses Global State."""
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
         
        if "api_key" in st.session_state and st.session_state.api_key:
            try:
                print("Auto-generating report...")
                report = orchestrator.generate_report(st.session_state.api_key)
                orchestrator.save_report(report)
            except Exception as e:
                print(f"Auto-report failed: {e}")
        
    else:
        if state.last_ingestion_time:
            st.session_state.last_run_display = state.last_ingestion_time.strftime("%H:%M")

def manual_refresh() -> None:
    """Forces an immediate refresh."""
    state = get_global_state()
    state.last_ingestion_time = None # Reset global time to force run
    run_periodic_ingestion()

st.set_page_config(
    page_title="UK Economic Insight Agent",
    page_icon=None,
    layout="wide"
)

# Initialize Session State Variables that depend on Global State
if 'last_run_display' not in st.session_state:
    st.session_state.last_run_display = "Not Run Yet"

# Check schedule on every app rerun (Lightweight check against Global State)
run_periodic_ingestion()


st.title("UK Economic Insight Agent")

with st.sidebar:
    st.header("Controls")
    
    # API Key Input
    user_api_key = st.text_input("Groq API Key (Optional)", type="password")
    if user_api_key:
        st.session_state.api_key = user_api_key
    
    # Use the session state key if available
    api_key = st.session_state.get("api_key", "")
    
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

        if not api_key:
            model_response = "Please enter your Groq API Key in the sidebar."
        else:
            with st.spinner("Thinking..."):
                model_response = orchestrator.answer_question(user_query, api_key)
        
        st.session_state.messages.append({"role": "assistant", "content": model_response})
        
        st.rerun()

# Report Tab
with tab_report:
    st.subheader("Market Reports")
    if st.button("Generate New Report"):
        if not api_key:
            st.warning("Please provide an API Key.")
        else:
            with st.spinner("Analyzing recent data..."):
                report = orchestrator.generate_report(api_key)
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






