# UK Economic Insight Agent

A "Free Stack" agent that researches, monitors, and reports on the UK Economy using **Time-Aware RAG (Retrieval Augmented Generation)**.

## 1. The Architecture
This agent distinguishes itself by treating **Time** as a first-class citizen. It doesn't just retrieve "relevant" text; it retrieves "recent" text for reports and "new" text for trend analysis.

```mermaid
graph TD
    subgraph Data Pipeline
        A[BBC / Guardian / Sky] -->|Fetch| B[data_ingestion.py]
        B -->|Extract Entities (spaCy)| C[Metadata]
        B -->|Add Timestamp| D[ChromaDB]
    end

    subgraph Orchestration
        E[User Query] -->|app.py| F[orchestrator.py]
        F -->|Classify Intent| G{Router}
        G -->|'SUMMARY'| H[Generate Periodic Report]
        G -->|'TREND'| I[Analyze Trends]
        G -->|'FACT_LOOKUP'| J[Answer Question]
    end

    subgraph Memory
        D[(Vector Store)]
        H -->|Save Report| D
        I -->|Fetch Last Report| D
    end
```

## 2. The Components (Files)

### `src/orchestrator.py`
Central logic handler.
- **Router**: Classifies user queries into Reporting, Trend Analysis, or Fact Lookup.
- **Time-Aware Retrieval**:
    -   **Reporting**: Fetches latest 10 news items sorted by timestamp.
    -   **Trends**: Filters for news published *after* the last generated report.
    -   **Topic Awareness**: Filters trend analysis by specific user topics (e.g., "Inflation").
    -   *Note*: The Router is strict. Questions like "What did you just say?" are classified as **General** (non-economic) and receive a standard greeting. To test memory, ask **Economic Follow-ups** (see below).

### `src/data_ingestion.py`
Handles data fetching and processing.
1.  **Fetch**: Pulls from BBC, Guardian, and Sky News RSS feeds.
2.  **Chunk**: Uses Semantic Chunking (via embeddings) to split text by meaning.
3.  **Timestamp**: Adds Unix timestamps to metadata for time-based filtering.
4.  **Tag**: Uses spaCy to extract entities (Organizations, People) efficiently.

### `src/prompts.py`
LLM instructions for different tasks:
-   **`ROUTER_PROMPT`**: Determines user intent.
-   **`SUMMARY_PROMPT`**: Generates bulleted market summaries. Handles **conflicting reports** by explicitly noting discrepancies.
-   **`TREND_PROMPT`**: Compares two text contexts to identify shifts.
-   **`FACT_PROMPT`**: Answers specific questions based on retrieved chunks. 
    -   *Fallback*: If no economic data is found, it safely admits ignorance. If the question is general (e.g., "What is 2+2?"), it answers directly without retrieval.

### `app.py`
Streamlit web interface.
-   Manages secret keys securely via `.env`.
-   Runs the hourly ingestion scheduler.
-   Displays generated reports and interactive chat history.

## 3. How "Periodic Reporting" Works
Unlike standard RAG which searches for "Keywords", our Reporting Engine is **Chronological**.

1.  **Search**: We query the database for "UK Economy market updates" with a broad scope (`k=20`).
2.  **Sort**: We use Python to sort these results by their metadata `timestamp` (Newest First).
3.  **Cull**: We take exactly the **Top 10** most recent chunks.
4.  **Generate**: We feed these 10 chunks to the `SUMMARY_PROMPT` to write a "Last 24 Hours" style briefing.

## 4. How to Run It

### Step 1: Configuration
This project uses **Environment Variables** for security.
1.  Create a file named `.env` in the root directory.
2.  Add your Groq API Key:
    ```bash
    GROQ_API_KEY=gsk_your_key_here_...
    ```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```
1.  The app will auto-ingest data on startup.
2.  Go to the **Reports** tab to see generated summaries.
3.  Use the **Chat** tab to ask specific questions ("What is the interest rate?") or trend questions ("How does this compare to last week?").

### How to Test "Memory"
The agent remembers the context of the conversation. To verify this:
1.  **Ask a Fact**: "What is the current inflation rate?" (Agent answers: *It is 5%*).
2.  **Ask a Follow-up**: "Is **that** good?"
    -   The agent understands "**that**" refers to "Inflation at 5%" from the previous turn and answers accordingly.
    -   *Note*: Do not ask meta-questions like "What caused that?" immediately if the router interprets it as General. Stick to economic qualifiers ("Is that high?", "What caused the inflation?").
