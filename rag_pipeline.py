# rag_pipeline.py
import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try to import LangChain text splitter and vectorstores (FAISS)
USE_FAISS = True
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
except Exception:
    USE_FAISS = False
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        # minimal fallback splitter
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=700, chunk_overlap=100):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text):
                chunks = []
                while len(text) > self.chunk_size:
                    chunks.append(text[: self.chunk_size])
                    text = text[self.chunk_size - self.chunk_overlap :]
                chunks.append(text)
                return chunks


# Try to import OpenAI embeddings helper from LangChain integration
OPENAI_EMBEDDINGS_AVAILABLE = True
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OPENAI_EMBEDDINGS_AVAILABLE = False

# Use modern OpenAI client for generation (works with openai>=1.x)
try:
    from openai import OpenAI as OpenAIClient

    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OpenAIClient = None  # type: ignore
    OPENAI_CLIENT_AVAILABLE = False


# ----------------------------
# Simple in-memory fallback vector store
# ----------------------------
class SimpleVectorStore:
    def __init__(self):
        self.docs: List[str] = []
        self.vects = None  # numpy array shape (n, d)

    def add_texts(self, texts: List[str], embedding_fn):
        embs = [embedding_fn(t) for t in texts]
        embs = [np.asarray(e, dtype=float) for e in embs]
        if embs:
            self.vects = np.vstack(embs)
        else:
            self.vects = np.zeros((0, 256))
        self.docs.extend(texts)

    def similarity_search(self, q_vec, k=5) -> List[Tuple[str, float]]:
        if self.vects is None or len(self.docs) == 0:
            return []
        sims = cosine_similarity(self.vects, q_vec.reshape(1, -1)).reshape(-1)
        idx = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in idx]


# ----------------------------
# Embedding function factory
# ----------------------------
def make_embedding_fn(api_key: Optional[str] = None):
    # Prefer LangChain OpenAIEmbeddings if available
    if OPENAI_EMBEDDINGS_AVAILABLE and (api_key or os.getenv("OPENAI_API_KEY")):
        embedder = OpenAIEmbeddings(
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        def embed(text: str):
            return np.array(embedder.embed_query(text), dtype=float)

        return embed

    # Deterministic fallback embedding when OpenAI embeddings unavailable
    def embed(text: str):
        b = text.encode("utf-8", errors="ignore")
        v = np.frombuffer(b, dtype=np.uint8).astype(float)
        if v.size < 256:
            v = np.pad(v, (0, 256 - v.size))
        else:
            v = v[:256]
        return v / (np.linalg.norm(v) + 1e-12)

    return embed


# ----------------------------
# Main RAG engine
# ----------------------------
class RAGEngine:
    def __init__(
        self, api_key: Optional[str] = None, persist_directory: str = "./faiss_index"
    ):
        """
        api_key: OpenAI API key string or None (will read from env).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.persist_directory = persist_directory
        self.embedding_fn = make_embedding_fn(self.api_key)
        # prefer FAISS if available
        self.use_faiss = USE_FAISS and OPENAI_EMBEDDINGS_AVAILABLE
        self.db = None  # FAISS instance or other vectorstore if used
        self.retriever = None
        self.fallback_store = SimpleVectorStore()

    # ----------------------------
    # Ingest documents (text list)
    # ----------------------------
    def ingest(self, texts: List[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs: List[str] = []
        for t in texts:
            docs.extend(splitter.split_text(t))

        # Try to use FAISS + OpenAI embeddings if configured
        if self.use_faiss:
            try:
                # Use LangChain's OpenAIEmbeddings wrapper
                embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                self.db = FAISS.from_texts(texts=docs, embedding=embeddings)
                # As a convenience, create a simple retriever if supported
                try:
                    self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
                except Exception:
                    self.retriever = None
                # Also store a simple fallback store (optional)
                self.fallback_store.add_texts(docs, self.embedding_fn)
                return
            except Exception:
                # If any failure, fallback to in-memory store
                self.use_faiss = False

        # Fallback: store in SimpleVectorStore using embedding_fn
        self.fallback_store.add_texts(docs, self.embedding_fn)

    # ----------------------------
    # Internal: assemble prompt from docs
    # ----------------------------
    def _assemble_prompt(self, docs: List[str], question: str) -> str:
        context = "\n\n---\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return prompt

    # ----------------------------
    # Query: returns structured 4-tuple
    # (answer, used_fallback, assembled_prompt_or_none, error_or_none)
    # ----------------------------
    def query(
        self, question: str, llm_model: str = "gpt-4o-mini"
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
        """
        Attempts to produce an LLM answer. Falls back to returning assembled prompt if no LLM available.
        Returns:
          - answer (string): final text OR assembled prompt when fallback
          - used_fallback (bool): True when no LLM generation was produced (fallback)
          - assembled_prompt_or_none (str|None): the assembled prompt when fallback used
          - error_or_none (str|None): textual exception if generation attempted but failed
        """
        # 1) Retrieve top documents (prefer retriever if available)
        docs: List[str] = []
        try:
            if getattr(self, "retriever", None) is not None:
                # Try retriever.get_relevant_documents if present
                try:
                    docs_objs = self.retriever.get_relevant_documents(question)
                    # convert to strings
                    for d in docs_objs:
                        txt = (
                            getattr(d, "page_content", None)
                            or getattr(d, "content", None)
                            or str(d)
                        )
                        docs.append(txt)
                except Exception:
                    # fallback: try using retriever directly with query
                    docs_objs = self.retriever.get_relevant_documents(question)
                    for d in docs_objs:
                        txt = (
                            getattr(d, "page_content", None)
                            or getattr(d, "content", None)
                            or str(d)
                        )
                        docs.append(txt)
        except Exception:
            # retriever not usable, proceed to fallback
            docs = []

        # fallback: use in-memory store
        if not docs:
            try:
                q_vec = self.embedding_fn(question)
                results = self.fallback_store.similarity_search(q_vec, k=5)
                docs = [t for t, s in results]
            except Exception:
                docs = []

        # assemble prompt (useful both for LLM input and fallback)
        assembled_prompt = self._assemble_prompt(docs, question)

        # 2) If OpenAI client available and API key present -> attempt generation
        if OPENAI_CLIENT_AVAILABLE and self.api_key:
            try:
                client = OpenAIClient(api_key=self.api_key)
                # create a response using the modern responses API
                completion = client.responses.create(
                    model=llm_model, input=assembled_prompt
                )
                # Extract text robustly
                answer_text = None
                try:
                    # new responses API shape: completion.output is a list of outputs
                    # content may be nested; attempt common access patterns
                    out = completion.output
                    if isinstance(out, list) and len(out) > 0:
                        first = out[0]
                        # typical shape: first.content -> list of dicts with 'text'
                        cnt = (
                            getattr(first, "content", None)
                            or first.get("content", None)
                            if isinstance(first, dict)
                            else None
                        )
                        if isinstance(cnt, list) and len(cnt) > 0:
                            # many SDK outputs put text under content[0].text
                            first_content = cnt[0]
                            # if it's an object-like with .get/.text
                            if (
                                isinstance(first_content, dict)
                                and "text" in first_content
                            ):
                                answer_text = first_content["text"]
                            elif hasattr(first_content, "text"):
                                answer_text = first_content.text
                        # fallback: try first['text'] or first.get('text')
                        if answer_text is None:
                            # try string representation
                            answer_text = (
                                getattr(first, "text", None) or first.get("text", None)
                                if isinstance(first, dict)
                                else None
                            )

                    # final fallback: try completion.output_text if present
                    if not answer_text:
                        answer_text = getattr(completion, "output_text", None)
                except Exception:
                    # As a last resort, stringify the whole completion
                    try:
                        answer_text = str(completion)
                    except Exception:
                        answer_text = None

                if answer_text:
                    return answer_text, False, None, None
                else:
                    # No extractable answer — treat as fallback with hint
                    return (
                        assembled_prompt,
                        True,
                        assembled_prompt,
                        "Could not parse LLM output",
                    )
            except Exception as e:
                # generation attempted but failed — return assembled prompt with error
                return assembled_prompt, True, assembled_prompt, repr(e)

        # 3) No OpenAI client or no api key -> explicit fallback
        return assembled_prompt, True, assembled_prompt, None

    # ----------------------------
    # get_top_docs helper used by UI
    # ----------------------------
    def get_top_docs(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-k retrieved documents with scores.
        Works for retriever (if present) or fallback_store.
        """
        # try retriever first
        try:
            if getattr(self, "retriever", None) is not None:
                try:
                    docs = self.retriever.get_relevant_documents(query)
                except Exception:
                    docs = self.retriever.get_relevant_documents(query)
                out = []
                for d in docs[:k]:
                    text = (
                        getattr(d, "page_content", None)
                        or getattr(d, "content", None)
                        or str(d)
                    )
                    score = getattr(d, "score", None)
                    out.append((text, float(score) if score is not None else 0.0))
                if out:
                    return out
        except Exception:
            pass

        # fallback to the in-memory store
        store = getattr(self, "fallback_store", None)
        if store is not None:
            try:
                q_vec = self.embedding_fn(query)
                results = store.similarity_search(q_vec, k=k)
                normalized = []
                for item in results:
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], str)
                    ):
                        normalized.append((item[0], float(item[1])))
                    else:
                        text = getattr(item, "page_content", None) or str(item)
                        normalized.append((text, 0.0))
                return normalized
            except Exception:
                return []

        return []
