from langchain_core.prompts import ChatPromptTemplate

# MAIN PROMPT TYPES: FACT (fallback/misclassification), TREND, SUMMARY / REPORT

# BASE PROMPT (foundation)
ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert intent classifier.
Classify the following user query into exactly one of these categories10:
1. FACT_LOOKUP: Specific questions about numbers, rates, events, OR qualitative assessments/opinions. (e.g., "What is the inflation rate?", "Is that good?", "Who is the CEO?")
2. TREND_ANALYSIS: Questions asking for comparison, changes over time, or evolution. (e.g., "How has the economy changed since last week?", "Is inflation getting worse?")
3. SUMMARY: Vague requests for an overview or briefing. (e.g., "What's happening?", "Give me a briefing", "Any news?")
4. GENERAL: Casual conversation, greetings, or questions unrelated to the economy. (e.g., "Hello", "How are you?", "Who are you?")

User Query: {question}

Return ONLY the category name. Do not explain.
""")

# TYPE OF SPECIFIC PROMPTS


FACT_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful UK Economic Advisor assistant.
Today's Date: {date}
Use the following pieces of context (retrieved from a vector database) to answer the user's question.

Context:
{context}

Chat History:
{chat_history}

User Question: {question}

Instructions:
- If the question is General Knowledge (e.g., math, definitions, greetings), answer it directly.
- If the question requires economic data NOT in the context, say "I cannot find information in my database to answer that right now."
- If the context contains conflicting information (e.g. Rate is 5% vs Rate is 2%), mention BOTH and the source of each.
- Do NOT make up numbers or facts.
- Keep your answer professional but conversational.
- Cite the source article titles if available in the context (implied).

Answer:
""")

TREND_PROMPT = ChatPromptTemplate.from_template("""
You are a senior analyst. Compare the following two data sets:

[OLD DATA (Past Reports/News)]:
{old_context}

[NEW DATA (Recent News)]:
{new_context}

Task:
You are analyzing trends for the topic: "{topic}".
1. Identify 3 major shifts or changes specifically related to "{topic}".
2. If nothing changed regarding "{topic}", clearly state that stability prevails.
3. Ignore minor noise; focus on macro trends.

Output format: Bullet points with "Old vs New" comparison.
""")


SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are an expert economic analyst specializing in the UK economy.
You have been provided with the following text content from recent news articles:

{text}

Your task is to produce a concise but comprehensive Markdown report on the **UK Economy**.
Ensure you follow this EXACT structure:

## Executive Summary
(Write a 150-word summary of the main events. If reports conflict, note the discrepancy.)

## Key Developments
(Bullet list of the top 3-5 facts)

## Market Sentiment
(Positive/Neutral/Negative with a 1-sentence reason)

## Active Entities
(Identify all companies, people, and government bodies mentioned)

## References
(A list of the titles/sources of the articles used in this report)

Format the output cleanly in Markdown.
""")

# HELPER PROMPTS
ENTITY_PROMPT = ChatPromptTemplate.from_template("""
Read the following article text:

{text}

Extract the names of all:
1. Companies/Organizations (e.g., Barclays, HSBC, Bank of England)
2. People/Public Figures (e.g., Rishi Sunak, Andrew Bailey)
3. Government Bodies (e.g., Treasury, ONS)

Return ONLY a valid JSON list of strings. Example: ["Barclays", "Rishi Sunak", "Treasury"]
If none are found, return [].
Do not include any other text or explanation.
""")
