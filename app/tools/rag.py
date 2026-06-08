"""
tools/rag.py
------------
RAG (Retrieval-Augmented Generation) over GIS tool handbooks.
Indexes handbook JSON files into Qdrant at startup.
Retrieves relevant documentation per query using hybrid search:
  - Dense vector search (Qdrant) — semantic similarity
  - BM25 sparse search — exact keyword matching (Paper 2)
Combined score gives better results for both conceptual and API-level queries.
"""

import json
import os
from pathlib import Path

from tools.llm_client import get_embedding

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
HANDBOOKS_DIR = "/app/tools/handbooks"
COLLECTION = "tool_docs"

# ── In-memory BM25 index (Paper 2 — hybrid retrieval) ────────────────────────
_bm25_index = None   # BM25Okapi instance
_bm25_chunks = []     # list of {"tool": ..., "text": ...}


def _get_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_HOST)


def _build_bm25_index(chunks: list):
    """Build in-memory BM25 index from handbook chunks."""
    global _bm25_index, _bm25_chunks
    try:
        from rank_bm25 import BM25Okapi
        _bm25_chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        _bm25_index = BM25Okapi(tokenized)
        print(f"[RAG] BM25 index built: {len(chunks)} chunks")
    except Exception as e:
        print(f"[RAG] BM25 index failed (will use dense only): {e}")
        _bm25_index = None


def index_handbooks():
    """
    Embed all handbook JSON files into Qdrant collection 'tool_docs'.
    Also builds in-memory BM25 index for hybrid retrieval.
    Called once at worker startup. Safe to call multiple times.
    """
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = _get_qdrant()

    # Create collection if not exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"[RAG] Created collection: {COLLECTION}")

    # Check if already indexed
    count = client.count(collection_name=COLLECTION).count
    if count > 0:
        print(f"[RAG] Handbooks already indexed: {count} chunks")
        # Still rebuild BM25 index from handbooks even if Qdrant already indexed
        _rebuild_bm25_from_handbooks()
        return

    points = []
    point_id = 1
    all_chunks = []

    for filepath in Path(HANDBOOKS_DIR).glob("*.json"):
        try:
            doc = json.loads(filepath.read_text())
        except Exception as e:
            print(f"[RAG] Failed to load {filepath}: {e}")
            continue

        name = doc.get("name", filepath.stem)

        chunk1_text = f"""Tool: {name}
Description: {doc.get('description', '')}
Notes: {doc.get('notes', '')}
Available features: {json.dumps(doc.get('available_features', {}), indent=2)}"""

        example = doc.get("example_code", "") or doc.get(
            "features_example", "")
        chunk2_text = f"""Tool: {name} - Example code:
{example}"""

        extras = []
        for key in doc:
            if "example" in key.lower() and key != "example_code":
                extras.append(f"{key}:\n{doc[key]}")

        for i, chunk_text in enumerate([chunk1_text, chunk2_text] + extras):
            if not chunk_text.strip():
                continue
            embedding = get_embedding(chunk_text)
            payload = {
                "tool":       name,
                "chunk_type": ["description", "example", "extra"][min(i, 2)],
                "text":       chunk_text,
                "source":     filepath.name,
            }
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))
            all_chunks.append({"tool": name, "text": chunk_text})
            point_id += 1

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[RAG] Indexed {len(points)} chunks from "
              f"{len(list(Path(HANDBOOKS_DIR).glob('*.json')))} handbooks")

    # Build BM25 index from the same chunks
    _build_bm25_index(all_chunks)


def _rebuild_bm25_from_handbooks():
    """Rebuild BM25 index from handbooks without re-indexing Qdrant."""
    all_chunks = []
    for filepath in Path(HANDBOOKS_DIR).glob("*.json"):
        try:
            doc = json.loads(filepath.read_text())
            name = doc.get("name", filepath.stem)

            chunk1_text = f"""Tool: {name}
Description: {doc.get('description', '')}
Notes: {doc.get('notes', '')}
Available features: {json.dumps(doc.get('available_features', {}), indent=2)}"""

            example = doc.get("example_code", "") or doc.get(
                "features_example", "")
            chunk2_text = f"""Tool: {name} - Example code:
{example}"""

            for chunk_text in [chunk1_text, chunk2_text]:
                if chunk_text.strip():
                    all_chunks.append({"tool": name, "text": chunk_text})
        except Exception:
            continue

    _build_bm25_index(all_chunks)


def retrieve_relevant_docs(query: str, top_k: int = 3) -> str:
    """
    Hybrid retrieval combining dense (Qdrant) + sparse (BM25) search.
    Paper 2: hybrid retrieval improves parameter grounding and API-level queries.

    - Dense finds semantically similar docs (e.g. "spatial join")
    - BM25 finds exact keyword matches (e.g. "predicate=", "gpd.sjoin", "EPSG")
    - Scores are combined with equal weight and top_k results returned
    """
    try:
        client = _get_qdrant()
        embedding = get_embedding(query)

        # ── Dense retrieval (Qdrant) ──────────────────────────────────────────
        dense_results = client.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            limit=top_k * 2,  # fetch more for re-ranking
        )

        dense_scores = {}
        for r in dense_results:
            text = r.payload.get("text", "")
            dense_scores[text] = r.score

        # ── Sparse retrieval (BM25) ───────────────────────────────────────────
        bm25_scores = {}
        if _bm25_index is not None and _bm25_chunks:
            try:
                tokenized_query = query.lower().split()
                scores = _bm25_index.get_scores(tokenized_query)
                # Normalise BM25 scores to [0, 1]
                max_score = max(scores) if max(scores) > 0 else 1.0
                for i, score in enumerate(scores):
                    if score > 0:
                        text = _bm25_chunks[i]["text"]
                        bm25_scores[text] = score / max_score
            except Exception as e:
                print(f"[RAG] BM25 scoring failed: {e}")

        # ── Combine scores (equal weight) ─────────────────────────────────────
        all_texts = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined = {}
        for text in all_texts:
            d = dense_scores.get(text, 0.0)
            b = bm25_scores.get(text, 0.0)
            combined[text] = 0.5 * d + 0.5 * b

        # Sort by combined score, return top_k unique tools
        sorted_texts = sorted(
            combined.items(), key=lambda x: x[1], reverse=True)

        chunks = []
        seen_tools = set()

        # Build tool lookup
        tool_for_text = {}
        for r in dense_results:
            tool_for_text[r.payload.get(
                "text", "")] = r.payload.get("tool", "")
        for chunk in _bm25_chunks:
            tool_for_text[chunk["text"]] = chunk["tool"]

        for text, score in sorted_texts:
            tool = tool_for_text.get(text, "unknown")
            if tool not in seen_tools:
                chunks.append(f"--- {tool} (score: {score:.3f}) ---\n{text}")
                seen_tools.add(tool)
            if len(seen_tools) >= top_k:
                break

        if not chunks:
            return ""

        return "\n\n".join(chunks)

    except Exception as e:
        print(f"[RAG] Retrieval failed: {e}")
        return ""


def _embed_text(text: str) -> list:
    """Public helper used by error_memory.py"""
    return get_embedding(text)
