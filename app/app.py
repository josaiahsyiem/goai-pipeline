"""
app.py
------
FastAPI application for the GoAI pipeline.
Exposes endpoints for file uploads, query submission, result retrieval,
GeoJSON map data, Prometheus metrics, and health checks.
"""

import json
import os
import shutil
import uuid
from typing import Optional

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="GoAI — Geographic AI Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Prometheus metrics ────────────────────────────────────────────────────────

QUERIES_TOTAL = Counter("goai_queries_total",
                        "Total queries submitted")
ERRORS_TOTAL = Counter("goai_errors_total",           "Total failed queries")
EVAL_SCORE_AVG = Gauge("goai_eval_score_avg",         "Latest eval score")
QUERY_LATENCY = Gauge("goai_query_latency_seconds",
                      "Latest query latency in seconds")


# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id         TEXT PRIMARY KEY,
            status     TEXT NOT NULL,
            task_text  TEXT,
            city       TEXT DEFAULT '',
            result     TEXT,
            trace      TEXT,
            geojson    TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS geojson TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id          TEXT PRIMARY KEY,
            session_id  TEXT,
            filename    TEXT,
            file_path   TEXT,
            city        TEXT,
            description TEXT,
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS session_id TEXT")

    conn.commit()
    cur.close()
    conn.close()


init_db()


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    task:         str
    city:         str = ""
    upload_id:    Optional[str] = None
    session_id:   Optional[str] = None
    domain_hint:  Optional[str] = None


# ── Static frontend ───────────────────────────────────────────────────────────

@app.get("/")
def frontend():
    return FileResponse("static/index.html")


# ── Upload session endpoints ──────────────────────────────────────────────────

@app.post("/upload/session")
def create_upload_session():
    """Create a session ID to group multiple file uploads into one query."""
    return {"session_id": str(uuid.uuid4())}


@app.get("/upload/session/{session_id}")
def get_session_files(session_id: str):
    """Return all files uploaded under a session."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, file_path, city, description "
        "FROM uploads WHERE session_id = %s ORDER BY created_at",
        (session_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        "session_id": session_id,
        "files": [
            {
                "upload_id":   r[0],
                "filename":    r[1],
                "file_path":   r[2],
                "city":        r[3],
                "description": r[4],
            }
            for r in rows
        ],
    }


# ── File upload endpoint ──────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    file:        UploadFile = File(...),
    city:        str = Form(""),
    description: str = Form(""),
    session_id:  str = Form(""),
):
    filename = file.filename or "upload.geojson"
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in ("geojson", "json", "csv"):
        raise HTTPException(
            status_code=400,
            detail="Only GeoJSON (.geojson, .json) and CSV files are supported.",
        )

    upload_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{upload_id}.{ext}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_info: dict = {}
    try:
        if ext in ("geojson", "json"):
            import geopandas as gpd
            gdf = gpd.read_file(file_path)
            file_info = {
                "rows":       len(gdf),
                "columns":    list(gdf.columns),
                "crs":        str(gdf.crs),
                "geom_types": gdf.geom_type.value_counts().to_dict(),
            }
        else:
            import pandas as pd
            df = pd.read_csv(file_path)
            file_info = {"rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        file_info = {"error": str(e)}

    sid = session_id.strip() or upload_id
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO uploads (id, session_id, filename, file_path, city, description) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (upload_id, sid, filename, file_path, city, description),
    )
    conn.commit()
    cur.close()
    conn.close()

    return {
        "upload_id":  upload_id,
        "session_id": sid,
        "filename":   filename,
        "city":       city,
        "file_info":  file_info,
    }


@app.get("/upload/{upload_id}")
def get_upload(upload_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, city, description, file_path FROM uploads WHERE id = %s",
        (upload_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    return {
        "upload_id":   row[0],
        "filename":    row[1],
        "city":        row[2],
        "description": row[3],
        "file_path":   row[4],
    }


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.post("/query")
def create_query(req: QueryRequest):
    task_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()

    upload_paths: list = []
    if req.session_id:
        cur.execute(
            "SELECT file_path FROM uploads WHERE session_id = %s ORDER BY created_at",
            (req.session_id,),
        )
        upload_paths = [r[0] for r in cur.fetchall()]
    elif req.upload_id:
        cur.execute("SELECT file_path FROM uploads WHERE id = %s",
                    (req.upload_id,))
        row = cur.fetchone()
        if row:
            upload_paths = [row[0]]

    cur.execute(
        "INSERT INTO tasks (id, status, task_text, city) VALUES (%s, %s, %s, %s)",
        (task_id, "pending", req.task, req.city),
    )
    conn.commit()
    cur.close()
    conn.close()

    QUERIES_TOTAL.inc()

    from worker import process_task
    process_task.delay(task_id, req.task, req.city,
                       upload_paths or None, req.domain_hint or None)

    return {"task_id": task_id, "status": "pending"}


@app.get("/query/{task_id}")
def get_query(task_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, status, result, trace, city FROM tasks WHERE id = %s",
        (task_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    task_id_, status, result, trace, city = row

    if result and status == "complete":
        try:
            data = json.loads(result)
            if (score := data.get("eval_score")) is not None:
                EVAL_SCORE_AVG.set(score)
            if (latency := data.get("total_time_s")) is not None:
                QUERY_LATENCY.set(latency)
        except Exception:
            pass
    elif status == "failed":
        ERRORS_TOTAL.inc()

    return {
        "task_id": task_id_,
        "status":  status,
        "result":  result,
        "trace":   trace,
        "city":    city,
    }


@app.get("/query/{task_id}/geojson")
def get_query_geojson(task_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT status, geojson FROM tasks WHERE id = %s", (task_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    status, geojson = row

    if status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Task not complete (status: {status})",
        )
    if not geojson:
        raise HTTPException(
            status_code=404,
            detail="No GeoJSON available for this task",
        )

    return JSONResponse(
        content=json.loads(geojson),
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ── Observability ─────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
