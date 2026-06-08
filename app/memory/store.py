"""
memory/store.py
---------------
Qdrant-backed vector memory for the GoAI pipeline.
Stores completed task results as embeddings so similar past queries
can be surfaced as context for new decompositions.
"""

import hashlib
import os
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION = "task_memory"
VECTOR_SIZE = 768


# ── Client ────────────────────────────────────────────────────────────────────

def get_qdrant() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=6333)


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """
    Returns a 768-dimensional embedding vector for the given text.
    Primary: Ollama nomic-embed-text.
    Fallback: deterministic hash-based vector (no external dependency).
    """
    import requests

    ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    try:
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["embedding"]
    except Exception:
        pass

    # Fallback: hash-based pseudo-embedding
    hash_val = hashlib.md5(text.encode()).hexdigest()
    vec = [int(hash_val[i:i + 2], 16) / 255.0 for i in range(0, 32, 2)]
    return (vec * (VECTOR_SIZE // len(vec) + 1))[:VECTOR_SIZE]


# ── Collection management ─────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Creates the Qdrant collection if it does not already exist."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[Memory] Created collection: {COLLECTION}")


# ── Public API ────────────────────────────────────────────────────────────────

def store_task(
    task_id: str,
    task: str,
    city: str,
    analysis_type: str,
    eval_score: float,
    ground_truth_correlation: float,
    top_results: list,
    working_code: str = "",
) -> bool:
    """
    Stores a completed task in vector memory.
    Returns True on success, False on failure.
    """
    try:
        ensure_collection()
        client = get_qdrant()
        embedding = get_embedding(f"{city} {task}")

        payload = {
            "task_id":                  task_id,
            "task":                     task,
            "city":                     city,
            "analysis_type":            analysis_type,
            "eval_score":               eval_score,
            "ground_truth_correlation": ground_truth_correlation,
            "top_results":              top_results[:3],
            "timestamp":                datetime.now(timezone.utc).isoformat(),
            "working_code": working_code,
        }

        # Qdrant point IDs must be unsigned 32-bit integers
        point_id = abs(hash(task_id)) % (2 ** 31)

        client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id, vector=embedding, payload=payload)],
        )
        print(f"[Memory] Stored task {task_id[:8]}")
        return True

    except Exception as e:
        print(f"[Memory] Store error: {e}")
        return False


def retrieve_similar(task: str, city: str, limit: int = 3) -> list:
    """
    Returns up to `limit` past tasks semantically similar to the query.
    Returns an empty list if Qdrant is unavailable.
    """
    try:
        ensure_collection()
        client = get_qdrant()
        embedding = get_embedding(f"{city} {task}")

        hits = client.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )

        results = [
            {
                "task":                     r.payload.get("task"),
                "city":                     r.payload.get("city"),
                "analysis_type":            r.payload.get("analysis_type"),
                "eval_score":               r.payload.get("eval_score"),
                "ground_truth_correlation": r.payload.get("ground_truth_correlation"),
                "top_results":              r.payload.get("top_results"),
                "similarity":               round(r.score, 4),
                "working_code": r.payload.get("working_code", ""),
            }
            for r in hits
        ]

        print(f"[Memory] Found {len(results)} similar past tasks")
        return results

    except Exception as e:
        print(f"[Memory] Retrieve error: {e}")
        return []
