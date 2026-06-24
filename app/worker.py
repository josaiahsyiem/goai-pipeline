"""
worker.py
---------
Celery worker for the GoAI pipeline.
Receives task IDs from the API, runs the full analysis pipeline,
and writes results back to PostgreSQL.
"""

import json
import os

import psycopg2
from celery import Celery
from dotenv import load_dotenv

from agents.orchestrator import run_pipeline
from tools.rag import index_handbooks

load_dotenv()

celery_app = Celery("goai", broker=os.getenv("REDIS_URL"))

# Index handbooks into Qdrant for RAG
try:
    index_handbooks()
except Exception as e:
    print(f"[RAG] Indexing failed: {e}")

# Feature #7: Pre-download WorldPop rasters for common countries in background
try:
    from agents.analysis_agent import predownload_worldpop
    predownload_worldpop()
except Exception as e:
    print(f"[Startup] WorldPop pre-download failed to start: {e}")


# Cleanup: remove processed result files older than 24h on worker startup,
# so generated GeoJSON/PNG outputs don't accumulate indefinitely.
def _cleanup_old_outputs(max_age_hours: int = 24):
    import time
    import glob
    processed_dir = "/data/processed"
    if not os.path.isdir(processed_dir):
        return
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for pattern in ("*.geojson", "*.png"):
        for fp in glob.glob(os.path.join(processed_dir, pattern)):
            try:
                # Never delete cached WorldPop rasters (they're .tif anyway) or
                # long-lived boundary caches that are expensive to rebuild.
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
                    removed += 1
            except Exception:
                continue
    if removed:
        print(
            f"[Startup] Cleaned up {removed} old result files (>{max_age_hours}h)")


try:
    _cleanup_old_outputs()
except Exception as e:
    print(f"[Startup] Output cleanup failed: {e}")


# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


# ── GeoJSON extraction ────────────────────────────────────────────────────────

def extract_geojson(result: dict) -> str | None:
    """Reads the RESULT_GEOJSON file written by the analysis sandbox.
    No re-execution of analysis code."""
    import re
    output = result.get("output", "")
    m = re.search(r"RESULT_GEOJSON:\s*(/\S+\.geojson)", output)
    if not m:
        print("[Worker] No RESULT_GEOJSON marker in output")
        return None
    path = m.group(1)
    if not os.path.exists(path):
        print(f"[Worker] Result file missing: {path}")
        return None
    try:
        with open(path) as f:
            parsed = json.load(f)
    except Exception as e:
        print(f"[Worker] GeoJSON read error: {e}")
        return None
    feats = parsed.get("features") or []
    if not feats:
        print("[Worker] GeoJSON: empty features")
        return None
    if len(feats) > 1000:
        parsed["features"] = feats[:1000]
        print(f"[Worker] GeoJSON capped 1000/{len(feats)}")
    print(f"[Worker] GeoJSON: {len(parsed['features'])} features extracted")
    return json.dumps(parsed)


# ── Task handler ──────────────────────────────────────────────────────────────

@celery_app.task
def process_task(
    task_id: str,
    task_text: str,
    city: str = "",
    upload_paths: list = None,
    domain_hint: str = None,
    session_id: str = None,
):
    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute("UPDATE tasks SET status=%s WHERE id=%s",
                    ("running", task_id))
        conn.commit()

        result = run_pipeline(
            task_text,
            city,
            task_id=task_id,
            upload_paths=upload_paths,
            domain_hint=domain_hint,
            session_id=session_id or task_id,
        )

        if result["success"]:
            geojson_str = extract_geojson(result)
            print(f"[Worker] GeoJSON extracted: {bool(geojson_str)}")

            cur.execute(
                "UPDATE tasks SET status=%s, result=%s, trace=%s, geojson=%s WHERE id=%s",
                (
                    "complete",
                    json.dumps({
                        "output":                   result["output"],
                        "plan":                     result["plan"],
                        "eval_score":               result.get("eval_score"),
                        "ground_truth_correlation": result.get("ground_truth_correlation"),
                        "total_time_s":             result["total_time_s"],
                        "methodology":              result.get("methodology", ""),
                    }),
                    json.dumps(result["trace"]),
                    geojson_str,
                    task_id,
                ),
            )
        else:
            cur.execute(
                "UPDATE tasks SET status=%s, result=%s, trace=%s WHERE id=%s",
                (
                    "failed",
                    json.dumps({"error": result["error"]}),
                    json.dumps(result["trace"]),
                    task_id,
                ),
            )

        conn.commit()

    except Exception as e:
        print(f"[Worker] Unhandled error for task {task_id}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        cur.execute(
            "UPDATE tasks SET status=%s, result=%s WHERE id=%s",
            ("failed", json.dumps({"error": str(e)}), task_id),
        )
        conn.commit()

    finally:
        cur.close()
        conn.close()
