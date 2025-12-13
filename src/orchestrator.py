from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.prompts import QA_PROMPT, REPORT_PROMPT
from datetime import datetime
from chromadb.config import Settings
import os
from typing import Optional

# Configuration
DB_PATH = "./chroma_db"

def get_llm(api_key: str) -> Optional[ChatGroq]:
    """
    Initialize the Groq LLM with Llama-3.3-70b-versatile.
    """
    if not api_key:
        return None
        
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

def get_vectorstore() -> Chroma:
    """
    Load the ChromaDB vector store.
    """

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding_function, client_settings=Settings(anonymized_telemetry=False))

def answer_question(query: str, api_key: str) -> str:
    """
    Retrieve relevant context and generate an answer using the LLM.
    """
    llm = get_llm(api_key)
    if not llm:
        return "Please enter your Groq API Key in the Settings tab."
        
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # RAG Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query)

def generate_report(api_key: str) -> str:
    """
    Generates a market summary report.
    """
    llm = get_llm(api_key)
    if not llm:
        return "Please enter your Groq API Key first."
        
    vectorstore = get_vectorstore()
    
    # Broad search for report context
    docs = vectorstore.similarity_search("economy inflation interest rates", k=5)
    
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    
    chain = REPORT_PROMPT | llm | StrOutputParser()
    
    
    return chain.invoke({"text": combined_text})

def save_report(report_content: str) -> str:
    """Saves the report to the reports/ directory."""
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    filename = f"reports/{datetime.now().strftime('%Y-%m-%d_%H-%M')}_Market_Report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    return filename
