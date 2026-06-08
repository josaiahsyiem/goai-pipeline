"""
tools/error_memory.py
---------------------
GISclaw-inspired Error-Memory module.
Stores (error_pattern → working_fix) pairs in Qdrant.
When the retry loop encounters an error, retrieves similar
past errors and injects the proven fix into the prompt.
"""

import hashlib
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION = "error_memory"
VECTOR_DIM = 384  # nomic-embed-text / fallback hash embedding


def _get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=6333)


def _embed(text: str) -> list[float]:
    """
    Simple deterministic embedding via hash bucketing.
    Falls back gracefully if no embedding model is available.
    """
    try:
        from tools.rag import _embed_text
        return _embed_text(text)
    except Exception:
        # Fallback: hash-based pseudo-embedding
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b / 255.0) for b in h]
        # Pad or truncate to VECTOR_DIM
        while len(vec) < VECTOR_DIM:
            vec += vec
        return vec[:VECTOR_DIM]


def _ensure_collection():
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_DIM, distance=Distance.COSINE),
        )


def store_error_fix(error: str, fix_code: str, task_type: str = "general"):
    """
    Store a successful error→fix pair in Qdrant.
    Called after a retry succeeds.
    """
    try:
        _ensure_collection()
        client = _get_client()
        # Use first 300 chars of error as the key
        error_snippet = error[:300].strip()
        vector = _embed(error_snippet)
        point_id = abs(hash(error_snippet)) % (2**31)

        client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "error_snippet": error_snippet,
                    "fix_code":      fix_code[:1000],
                    "task_type":     task_type,
                    "success_count": 1,
                }
            )]
        )
        print(f"[ErrorMemory] Stored fix for error: {error_snippet[:60]}...")
    except Exception as e:
        print(f"[ErrorMemory] Store failed: {e}")


def retrieve_similar_fix(error: str, limit: int = 2) -> list[dict]:
    """
    Retrieve the most similar past error→fix pairs.
    Returns list of dicts with error_snippet and fix_code.
    """
    try:
        _ensure_collection()
        client = _get_client()
        error_snippet = error[:300].strip()
        vector = _embed(error_snippet)

        results = client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=limit,
            score_threshold=0.7,
        )
        return [
            {
                "error_snippet": r.payload.get("error_snippet", ""),
                "fix_code":      r.payload.get("fix_code", ""),
                "score":         r.score,
            }
            for r in results
        ]
    except Exception as e:
        print(f"[ErrorMemory] Retrieve failed: {e}")
        return []
