# backend/memory.py
"""
Chroma-based memory for LearnTube (CareerNinja).
Uses a dedicated OpenAI client for embeddings, as Groq does not provide them.
Includes time.sleep() fix for rate limiting.
"""

import os
import time 
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Delay import of openai for when embeddings are actually needed
import openai
from openai import OpenAI 

# Try importing chromadb
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

# -----------------------------------------------------------------------
# Config (pick up from env)
# -----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # economical


# -----------------------------------------------------------------------
# Initialize OpenAI client globally for embeddings
# -----------------------------------------------------------------------
_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized for embeddings.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client for embeddings: {e}")

# -----------------------------------------------------------------------
# Initialize Chroma client (attempt multiple ways)
# -----------------------------------------------------------------------
_client = None

def _init_chroma_client():
    global _client
    if chromadb is None:
        logger.warning("chromadb module not installed. Memory features will be disabled.")
        return None

    # 1) Prefer trying the local Settings-based client (old-style)
    try:
        if Settings is not None:
            logger.info("Attempting to initialize Chroma client using Settings (duckdb+parquet)...")
            _client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_DIR
            ))
            logger.info("Chroma client created with Settings (duckdb+parquet).")
            return _client
    except ValueError as ve:
        logger.warning("Chroma Settings init failed (likely deprecated config). Falling back to default client. Details: %s", ve)
    except Exception as e:
        logger.warning("Chroma Settings init raised exception; will try fallback. Details: %s", e)

    # 2) Fallback to the new default client (works for newer Chroma versions)
    try:
        logger.info("Attempting to create Chroma client using chromadb.Client() (new API)...")
        _client = chromadb.Client()
        logger.info("Chroma client created using chromadb.Client().")
        return _client
    except Exception as e:
        logger.error("Failed to initialize chromadb.Client(): %s", e)
        return None

_client = _init_chroma_client()

if _client is None:
    logger.warning("Chroma client not available. Memory calls will be no-ops.")

# -----------------------------------------------------------------------
# Helpers / Collections
# -----------------------------------------------------------------------
def _collection_name(user_id: str) -> str:
    return f"learn_tube_{user_id}"

def ensure_collection(user_id: str):
    if _client is None:
        raise RuntimeError("Chroma client not initialized. Install/initialize Chroma to use memory.")
    name = _collection_name(user_id)
    try:
        return _client.get_collection(name)
    except Exception:
        # create collection if not exists (works for new client APIs)
        try:
            return _client.create_collection(name, metadata={"created_at": time.time()})
        except Exception as e:
            return _client.create_collection(name)

# -----------------------------------------------------------------------
# Embedding utility (OpenAI)
# -----------------------------------------------------------------------
def make_embedding(text: str) -> List[float]:
    """Return an embedding vector for the given text using OpenAI."""
    if _openai_client is None:
        raise EnvironmentError("OPENAI_API_KEY not set or client failed to initialize. Set it in your environment or .env.")
        
    text = (text or "").strip()
    if text == "":
        return [0.0]
        
    # --- FIXED: Increased Delay for Rate Limit ---
    time.sleep(1.0) 
    
    resp = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    
    # Access embedding via the new syntax
    emb = resp.data[0].embedding
    return emb

# -----------------------------------------------------------------------
# Public API (same interface as before)
# -----------------------------------------------------------------------
def save_interaction(user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None):
    """
    Save a text interaction into Chroma for a specific user.
    """
    if _client is None:
        logger.debug("Memory disabled: skipping save_interaction.")
        return {"status": "memory_disabled"}

    if metadata is None:
        metadata = {}

    collection = ensure_collection(user_id)

    if id is None:
        id = f"{int(time.time()*1000)}-{abs(hash(text)) % (10**8)}"

    try:
        emb = make_embedding(text)
        collection.add(ids=[id], documents=[text], metadatas=[metadata], embeddings=[emb])
        
        try:
            _client.persist()
        except Exception:
            pass
            
    except Exception as e:
        # FIXED: Removed the faulty nested try/except block. 
        # If embedding fails (e.g., 429 error), we log the error and skip saving the memory item.
        logger.warning("Failed to create/store embedding: %s. Skipping memory persistence.", e)
        # Check if the error is due to Chroma's serialization bug on NoneType
        if "failed to extract enum MetadataValue" in str(e):
             logger.error("CRITICAL: Chroma serialization bug encountered. Metadata may contain NoneType values.")
        return {"id": id, "status": "failed_embedding"}


    return {"id": id, "status": "saved"}

def get_recent_memory(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if _client is None:
        return []
    collection = ensure_collection(user_id)
    try:
        results = collection.get(limit=limit)
        docs = []
        for idx in range(len(results.get("ids", []))):
            docs.append({
                "id": results["ids"][idx],
                "document": results["documents"][idx],
                "metadata": results["metadatas"][idx]
            })
        return docs
    except Exception as e:
        logger.warning("get_recent_memory failed: %s", e)
        return []

def get_relevant_memory(user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if _client is None or not query:
        return []
    collection = ensure_collection(user_id)
    try:
        q_emb = make_embedding(query)
        results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=['metadatas', 'documents', 'ids', 'distances'])
        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if "distances" in results else [None]*len(ids)
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i]
            })
        return out
    except Exception as e:
        logger.warning("get_relevant_memory fallback: %s", e)
        try:
            results = collection.get(limit=top_k)
            out = []
            for i in range(len(results.get("ids", []))):
                out.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "distance": None
                })
            return out
        except Exception:
            return []

def build_memory_context(user_id: str, query: str, top_k: int = 5) -> str:
    relevant = get_relevant_memory(user_id, query, top_k=top_k)
    parts = []
    for i, r in enumerate(relevant, start=1):
        meta_str = json.dumps(r.get("metadata", {}), ensure_ascii=False)
        parts.append(f"[MEMORY {i}] meta={meta_str}\n{r.get('document')}\n")
    return "\n".join(parts)

# -----------------------------------------------------------------------
# Migration hint printed for user's benefit (not executed)
# -----------------------------------------------------------------------
MIGRATION_HINT = """
If you see a 'deprecated configuration' error from Chroma and you have data
in your existing CHROMA_DIR that you want to keep, run:

    pip install chroma-migrate
    chroma-migrate --source-dir ./chroma_db

After migration, reload your code. See Chroma docs:
https://docs.trychroma.com/deployment/migration

If you don't care for previous data, delete the folder and restart:

    rm -rf ./chroma_db
"""

if _client is None:
    logger.info(MIGRATION_HINT)