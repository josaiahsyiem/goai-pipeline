# GoAI — System Design

## Overview

GoAI is a production-grade autonomous geographic information system (GIS) pipeline. It accepts any spatial question in natural language, autonomously retrieves the required geospatial data, runs a spatial analysis, evaluates the result, and renders a ranked choropleth map — all without human intervention.

The system is designed around two peer-reviewed papers from Penn State University's Geoinformation and Big Data Research Laboratory:

- **LLM-Geo** (Li & Ning, 2023) introduced the concept of autonomous GIS and demonstrated that LLMs can generate geoprocessing workflows from natural language. Their prototype, LLM-Geo, used GPT-4 to decompose spatial questions into directed acyclic graphs and execute them. doi:10.1080/17538947.2023.2278895
- **LLM-Find** (Ning et al., 2025) extended this work by building an autonomous data retrieval agent with a plug-and-play handbook system. The agent selects data sources and generates fetch code for OpenStreetMap, US Census, weather APIs, and satellite imagery. doi:10.1080/17538947.2025.2458688

GoAI extends both papers into a production system with any-city support, persistent memory, objective evaluation, and an interactive map interface — none of which either paper implemented.

---

## Architecture

```
User (browser)
     |
     | HTTP
     v
FastAPI (app.py)
     |
     | Celery task queue (Redis)
     v
Worker (worker.py)
     |
     v
Orchestrator (orchestrator.py)
     |
     |-- 1. Memory lookup    --> Qdrant (vector similarity search)
     |-- 2. Decompose        --> Groq LLM (structured JSON plan)
     |-- 3. Retrieve         --> Retrieval Agent (local files or OSMnx)
     |-- 4. Analyse          --> Analysis Agent (sandboxed subprocess)
     |-- 5. Evaluate         --> Scorer (LLM judge + Spearman correlation)
     |-- 6. Store            --> Qdrant + PostgreSQL
     |
     v
PostgreSQL (results, GeoJSON, trace)
     |
     v
FastAPI /query/{id}/geojson
     |
     v
Leaflet.js choropleth map
```

---

## Why this architecture

### Async task queue

Spatial analysis involving OSMnx can take 60–120 seconds for large cities. A synchronous API would time out. Using Celery with Redis as the broker allows the API to return a task ID immediately while the worker processes the query in the background. The frontend polls `/query/{task_id}` every 2 seconds to show live progress.

### Sandboxed code execution

Both LLM-Geo and LLM-Find execute LLM-generated Python code directly. GoAI runs all generated code in an isolated subprocess with a timeout. This means a buggy or infinite-loop code cannot crash the worker. It also enables self-debugging — if the subprocess returns a non-zero exit code, the error message is fed back to the LLM for correction. LLM-Geo identified this trial-and-error approach as necessary; GoAI implements it with up to 5 retry attempts per query.

### Hybrid deterministic and LLM approach

LLM-Geo Section 5.3 noted that LLMs struggle with complex spatial code. GoAI uses a hybrid approach:

- **Deterministic code** for known query types where the spatial operations are fixed: Mumbai flood analysis, road density, hospital coverage, green coverage. These paths never fail because we wrote and tested the code ourselves.
- **LLM-generated code** as a fallback for novel query types we have not seen before. Groq generates the code, the sandbox executes it, and errors trigger a retry with the error message included in the next prompt.

This matches what LLM-Geo described as "divide-and-conquer" — use LLM for reasoning (what to compute) and deterministic code for execution (how to compute it reliably).

### GIS guidance rules

LLM-Geo Table 2 identified that GPT-4 has "hazy memory" about spatial analysis prerequisites — CRS matching, data type alignment, duplicate removal after joins. GoAI maintains a `gis_guidance.json` file with 13 rules that are injected into every LLM prompt. Rules are added whenever a new bug is discovered during development. This is the "categorized guidance maintained by the GIS community" that LLM-Geo Section 6.3 called for.

### Plug-and-play handbook system

LLM-Find's core contribution was the handbook inventory — JSON files describing each data source with metadata, access methods, and example code. GoAI implements the same system with handbooks for Mumbai ward boundaries, lakes and rivers, drainage networks, and OpenStreetMap. The `data_source_index.json` gives the LLM a brief description of each source so it can select the right one for any query. New data sources can be added by creating a new handbook JSON — no code changes required.

---

## Error detection

LLM-Find Section 5 categorised two types of errors in LLM-generated code:

**Type 1 — unrunnable code:** The generated code crashes with a Python runtime error. GoAI catches this via the subprocess return code, extracts the error message, and feeds it back to the LLM for correction.

**Type 2 — runnable but wrong:** The code runs successfully but returns incorrect or irrelevant data. LLM-Find explicitly identified this as a gap in their system — they had no mechanism to detect it. GoAI addresses Type 2 errors in two ways:

1. **Feature count check** — if the result GeoDataFrame has fewer than 3 features, the analysis is flagged as a Type 2 failure. This catches cases where a spatial join silently fails and returns aggregated data at the wrong geographic level.
2. **Bounding box check** — the result centroid is compared against the Nominatim bounding box for the queried city. If the centroid falls outside the city boundary (expanded by 2 degrees), the result is rejected. This catches cases where the LLM fetches data for the wrong place entirely.

---

## Memory system

LLM-Geo Section 6.2 identified memory as a critical missing component: "Every autonomous system needs a memory component to store contextual and long-term information for future retrieval." GoAI implements this using Qdrant, a vector database, with nomic-embed-text embeddings (768 dimensions).

Every completed task is stored as a vector embedding of `"{city} {task}"` with metadata including the eval score, ground truth correlation, top results, and the working analysis code. When a new query arrives, the memory is searched for similar past tasks. If a match with cosine similarity above 0.95 is found, the working code from that past task is reused directly — skipping LLM code generation entirely. This is the self-growing capability described in LLM-Geo's autonomous goals.

---

## Evaluation

Neither LLM-Geo nor LLM-Find had objective accuracy measurement. GoAI evaluates every result on two dimensions:

**LLM judge (0–1):** Groq scores the result on four criteria — did it answer the question, are values realistic for the city, does the top result make geographic sense, are required columns present. This gives a quantitative quality score for every query type.

**Spearman rank correlation:** For Mumbai flood queries, the AI result is compared against a Phase 1 ground truth dataset produced by a manual QGIS analysis. The Spearman correlation between AI-ranked wards and ground truth ranks has consistently reached 1.0, confirming the pipeline produces results that match expert manual analysis exactly.

---

## What was not implemented

LLM-Geo identified several future research directions that GoAI has not yet addressed:

- **Online data discovery** — GoAI uses a fixed list of data sources. Autonomous crawling and handbook generation would allow the system to discover new data sources without human intervention.
- **Large Spatial Model** — training a foundation model on all available spatial data to give LLMs better geographic intuition.
- **Why questions** — GoAI answers "which" and "where" questions. Answering "why" would require hypothesis generation and experimental design capabilities.

These remain open research problems that the papers also left as future work.

---

## Production considerations

GoAI is currently deployed locally. Phase 9 will deploy to Render.com with the following changes:

- Replace nomic-embed-text (requires local Ollama) with Cohere free-tier embeddings
- Groq is already cloud-native — no changes needed
- PostgreSQL, Redis, and Qdrant deploy as managed services on Render

Prometheus metrics (`goai_queries_total`, `goai_eval_score_avg`, `goai_query_latency_seconds`) are already instrumented and will connect to a Grafana Cloud dashboard post-deployment.
