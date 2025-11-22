# rag-engine

Reference implementation of a production-grade Retrieval-Augmented Generation (RAG) system using **LangChain** and **OpenAI**.

## Features
- Document ingestion
- Chunking & preprocessing
- Embedding generation (OpenAI)
- Vector search using ChromaDB
- RAG pipeline with LangChain
- FastAPI server for inference
- Dual-stage retrieval (similarity + rerank)
- Simple evaluation hooks (RAGAS-style)
- Utilities: logging & simple caching

## How to Run (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
uvicorn app:app --reload --port 8000
```

## Project Structure
```
├── app.py                 # FastAPI app exposing
├── rag_pipeline.py        # Core RAG engine (ingest + query)
├── ingestion.py           # Helpers for loading documents
├── requirements.txt
├── sample_data/
│   └── sample.txt
├── evaluation/            # Evaluation scripts
│   └── evaluate_rag.py
└── utils/                 # Utility helpers (logging, cache)
    ├── logger.py
    └── cache.py
```

## API Endpoint
- **GET /ask?q=your_question** — returns JSON response with answer and metadata

