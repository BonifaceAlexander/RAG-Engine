from fastapi import FastAPI, HTTPException, Query
from typing import Annotated
from pydantic import BaseModel
import os
from rag_pipeline import RAGEngine
from ingestion import load_texts_from_folder
from utils.logger import get_logger
from utils.cache import SimpleCache

LOG = get_logger('app')
app = FastAPI(title='RAG Engine API')

engine: RAGEngine = None
cache = SimpleCache()

class AnswerResponse(BaseModel):
    query: str
    answer: str
    cached: bool = False

@app.on_event("startup")
async def startup_event():
    global engine
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        LOG.error('OPENAI_API_KEY not set; endpoints will error until set.')
    engine = RAGEngine(api_key=api_key, persist_directory='./chroma_db')
    # Load sample texts from sample_data
    texts = load_texts_from_folder('sample_data', use_openai=True)
    if texts:
        LOG.info(f'Ingesting {len(texts)} documents into vector store.')
        engine.ingest(texts)
    else:
        LOG.warning('No sample data found in sample_data/; ingestion skipped.')

@app.get('/ask', response_model=AnswerResponse)
async def ask(q: str = Query(..., min_length=1)):
    """Query the RAG engine. Uses a simple in-memory cache to reduce LLM calls for same questions."""
    cached = cache.get(q)
    if cached:
        return AnswerResponse(query=q, answer=cached, cached=True)

    try:
        answer = engine.query(q)
    except Exception as e:
        LOG.exception('Query failed')
        raise HTTPException(status_code=500, detail=str(e))

    # store in cache
    cache.set(q, answer)
    return AnswerResponse(query=q, answer=answer, cached=False)
