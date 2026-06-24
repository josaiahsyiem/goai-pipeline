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
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small; Ollama fallback pads to match


# ── Client ────────────────────────────────────────────────────────────────────

def get_qdrant() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=6333)


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """
    Returns a 1536-dimensional embedding vector for the given text.
    Primary: OpenAI text-embedding-3-small (works everywhere, no local dependency).
    Fallback: Ollama nomic-embed-text (local, if available).
    Last resort: deterministic hash-based vector (non-semantic).
    """
    import requests

    # Primary: OpenAI embeddings
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai as _oai
            _client = _oai.OpenAI(api_key=openai_key)
            response = _client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as _e:
            print(f"[Memory] OpenAI embedding failed: {_e} — trying Ollama")

    # Secondary: Ollama (local dev only)
    ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    try:
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["embedding"]
    except Exception:
        pass

    # Last resort: hash-based pseudo-embedding — NOT semantic.
    print("[Memory] WARNING: All embedding backends unavailable — using hash fallback (non-semantic)")
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
    session_id: str = None,
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
            "session_id": session_id or "",
        }

        # Qdrant point IDs must be unsigned 32-bit integers.
        # Keyed on city+task (md5, stable across restarts) so re-running the
        # same query OVERWRITES its memory instead of accumulating duplicates.
        point_id = int(hashlib.md5(
            f"{city.lower()}|{task.lower()}".encode()).hexdigest()[:8], 16) % (2 ** 31)

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


def retrieve_similar(task: str, city: str, limit: int = 3, session_id: str = None, analysis_type: str = None) -> list:
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
            limit=limit * 3,
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
                "session_id": r.payload.get("session_id", ""),
            }
            for r in hits
        ]

        # Similarity floor — below this, "similar" is noise, not context
        results = [r for r in results if r.get("similarity", 0) >= 0.55]

        # Analysis-type gate: only reuse memories of the same analysis type.
        # Prevents cross-matches (e.g. hospital results reused for flood queries)
        # despite high embedding similarity between semantically similar queries.
        if analysis_type:
            typed = [r for r in results if r.get(
                "analysis_type") == analysis_type]
            if typed:
                results = typed
                print(
                    f"[Memory] Type-filtered to {len(results)} results matching '{analysis_type}'")

        # City gate: prefer same-city memories to avoid cross-city contamination
        # (e.g. a stored "Haryana" task being reused for a "Bengaluru" query).
        city_matched = [r for r in results
                        if str(r.get("city", "")).lower() == str(city).lower()]
        if city_matched:
            results = city_matched
            print(
                f"[Memory] City-filtered to {len(results)} results matching '{city}'")

        if session_id:
            same = [r for r in results if r.get("session_id") == session_id]
            rest = [r for r in results if r.get("session_id") != session_id]
            results = (same + rest)[:limit]
        else:
            results = results[:limit]

        print(f"[Memory] Found {len(results)} similar past tasks")
        return results

    except Exception as e:
        print(f"[Memory] Retrieve error: {e}")
        return []


def purge_memory(task_substring: str = None) -> int:
    """Deletes memories whose task contains task_substring (case-insensitive).
    With no argument, wipes the entire collection. Returns points deleted (-1 full wipe)."""
    try:
        client = get_qdrant()
        if task_substring is None:
            client.delete_collection(COLLECTION)
            ensure_collection()
            print(f"[Memory] Collection {COLLECTION} wiped")
            return -1
        sub = task_substring.lower()
        deleted = 0
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION, limit=100,
                offset=offset, with_payload=True)
            ids = [p.id for p in points
                   if sub in str(p.payload.get("task", "")).lower()]
            if ids:
                client.delete(collection_name=COLLECTION, points_selector=ids)
                deleted += len(ids)
            if offset is None:
                break
        print(
            f"[Memory] Purged {deleted} memories matching '{task_substring}'")
        return deleted
    except Exception as e:
        print(f"[Memory] Purge error: {e}")
        return 0
