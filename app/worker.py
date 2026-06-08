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
    import subprocess
    import sys

    code = result.get("code", "")
    if not code:
        print("[Worker] No code in result — skipping GeoJSON extraction")
        return None

    wrapped = """
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.validation import make_valid
from shapely.geometry import Point
import warnings
import json
import sys
import os
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterstats import zonal_stats
except ImportError:
    pass

default_buffer = 100

""" + code + """

out = result.head(50).copy()

for col in list(out.columns):
    if col == 'geometry':
        continue
    try:
        out[col] = out[col].astype(str)
    except Exception:
        out = out.drop(columns=[col])

out = out[out.geometry.notna() & out.geometry.is_valid].copy()

if len(out) == 0:
    sys.stdout.write("NO_FEATURES")
else:
    sys.stdout.write(out.to_json())
sys.stdout.flush()
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("[Worker] GeoJSON extraction timed out")
        return None

    if proc.returncode != 0:
        print(f"[Worker] GeoJSON sandbox error: {proc.stderr[:300]}")
        return None

    lines = [l for l in proc.stdout.strip().split('\n') if l.strip()]
    stdout = lines[-1] if lines else ""

    if not stdout or stdout == "NO_FEATURES":
        print("[Worker] GeoJSON: no features produced")
        return None

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as e:
        print(f"[Worker] GeoJSON parse error: {e}")
        print(f"[Worker] stdout sample: {stdout[:200]}")
        return None

    if "features" not in parsed or not parsed["features"]:
        print("[Worker] GeoJSON: missing or empty features")
        return None

    print(f"[Worker] GeoJSON: {len(parsed['features'])} features extracted")
    return stdout


# ── Task handler ──────────────────────────────────────────────────────────────

@celery_app.task
def process_task(
    task_id: str,
    task_text: str,
    city: str = "",
    upload_paths: list = None,
    domain_hint: str = None,
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
        cur.execute(
            "UPDATE tasks SET status=%s, result=%s WHERE id=%s",
            ("failed", json.dumps({"error": str(e)}), task_id),
        )
        conn.commit()

    finally:
        cur.close()
        conn.close()
