Aviation Accident Self-RAG Assistant
====================================

This project implements a Self-RAG–style aviation accident investigation assistant that answers questions over real AAIB accident reports using an agentic RAG pipeline.
## Features

- Ingests multiple AAIB investigation PDFs from `./documents` automatically (VT-ANB, VT-SLH, VT-CHG, VT-PTE, VT-GDI).
- Splits text into chunks and builds a FAISS vector store using OpenAI `text-embedding-3-small`. 
- Uses a LangGraph `StateGraph` with explicit nodes for:
  - retrieval decision (Self-RAG style)  
  - document retrieval  
  - LLM-based relevance filtering  
  - context-based answer generation  
  - web-search fallback via Tavily. 
- Specialized aviation prompt that outputs: short summary, probable causes, contributing factors, and safety recommendations, grounded only in report context. 

## Architecture

- **Vector store**: FAISS over chunked report documents. 
- **LLM**: `ChatOpenAI` (e.g., `gpt-4o-mini`, temperature 0) for reasoning and control. 
- **Graph nodes**:
  - `decide_retrieval` → choose direct answer vs RAG.  
  - `retrieve` → query FAISS retriever.  
  - `is_relevant` → filter chunks using a Pydantic `RelevanceDecision`.  
  - `generate_from_context` → aviation-accident answer from context only.  
  - `rewrite_query` + `web_search` → Tavily-based web RAG when local docs are insufficient. 
- **Control flow**: LangGraph conditional edges and loop with max 3 relevance iterations to avoid infinite recursion.

## Tech Stack

- Python 3.10  
- LangChain / LangChain Community  
- LangGraph  
- OpenAI embeddings + ChatOpenAI  
- FAISS (CPU)  
- Tavily Search (optional web fallback)  
- PyPDF / LangChain PDF loader. 

## How It Works

1. Place AAIB PDF reports in `./documents`. 
2. Run the notebook to:
   - load and chunk PDFs  
   - build the FAISS index  
   - compile the LangGraph `StateGraph`. 
3. Call the graph with a question, e.g.:

```python
result = app.invoke({
    "question": "What was the probable cause of the VT-PTE accident?",
    "docs": [],
    "relevant_docs": [],
    "context": "",
    "answer": "",
})
print(result["answer"])
```


The agent decides whether to retrieve, selects relevant chunks from the reports, and returns a structured, report-grounded explanation.

## Example Questions

- “What was the probable cause of the VT-PTE accident?”  
- “List the safety recommendations issued in the VT-GDI report.”  
- “Compare the accidents involving VT-PTE and VT-GDI.”  
- “What human factors contributed to passenger injuries in the VT-SLH event?”
## Status

This is an ongoing research-engineering project exploring Self-RAG and agentic RAG patterns on safety-critical aviation data. Future work includes better evaluation, UI integration, and adding more accident reports.
