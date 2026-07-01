# NLGeo — Natural-Language GeoAI Pipeline

**Ask a geographic question in plain English. Get a ranked, interactive choropleth map for any city in the world.**

NLGeo turns a query like *"Which Mumbai wards have the highest flood-exposed population?"* or *"Pharmacies per ward in Berlin"* into a ranked ward-level map — no GIS expertise required. It fetches live OpenStreetMap data, runs deterministic or LLM-generated geospatial analysis in a sandbox, self-verifies the result, and renders an interactive Leaflet map.

🎥 **[Demo video](https://youtu.be/h7c9p2f2qqk)** 

---

## What it does

You type a question and a city. NLGeo:

1. Checks vector memory for a similar past query
2. Decomposes the question into a structured analysis plan
3. Fetches live boundaries and features (Overpass API, OSMnx) — or local files for Mumbai
4. Runs the analysis — deterministic where possible, LLM-generated code as fallback
5. Self-verifies with an LLM judge + spatial cross-validation
6. Returns a ranked choropleth map with per-ward metrics and a confidence score

Supports counts, density, per-capita (WorldPop rasters), proximity ("within 3 km of city centre"), multi-city comparison, file-upload spatial joins, and CSV/GeoJSON export.

---

## Verified results

Tested across 20+ cities. Every result carries an LLM evaluation score (0–1) and, where a ground truth or cross-validation exists, a Spearman correlation.

| City | Query | Score | Correlation | Time |
|---|---|---|---|---|
| Mumbai | Flood risk by ward | 0.90 | **1.00** | 6–10s |
| Berlin | Pharmacies per ward | 0.90 | 1.00 | 33–42s |
| Greater London | Hospitals per ward | 0.90 | 1.00 | 42s |
| Paris | Hospitals per arrondissement | 0.90 | 1.00 | 98s |
| Cairo | Schools per area | 0.90 | 1.00 | 131s |
| Seoul | Cafes per ward | 0.90 | 1.00 | ~90s |
| New Delhi | Hospitals per area | 0.90 | 1.00 | 28s |
| Paris | Restaurants per arrondissement | 0.90 | 1.00 | 156s |
| London | Parks and greenspace | 1.00 | — | 17s |
| Bengaluru | Hospital coverage (file upload) | 0.80 | — | 5.7s |
| Lagos | Most green space | 0.90 | — | 17s |
| Kolkata | Hospital density (upload) | 0.90 | — | 9.9s |

The **Mumbai flood benchmark** is the anchor: a deterministic implementation of a hand-built QGIS ground truth that reproduces the correct ward ranking at Spearman 1.0 on every run.

---

## Architecture

```
  user ──► FastAPI ──► Redis ──► Celery worker ──► 7-stage pipeline
           (enqueue)   (broker)  (async compute)    1 Memory
                                                     2 Decompose
           Postgres/PostGIS ◄──────────────────────►3 Retrieve  (Overpass/OSMnx)
           Qdrant (vector memory) ◄────────────────►4 Analyse   (deterministic→LLM)
           WorldPop rasters ◄──────────────────────►5 Evaluate  (judge + cross-val)
                                                     6 Store
           Leaflet frontend ◄──── GET /query/{id} ──7 Return
```

**Deterministic-first design.** Unlike prior LLM-GIS systems that route every query through the model, NLGeo tries verifiable deterministic paths first (50+ OSM feature types, zero LLM calls) and falls back to LLM-generated code only for novel queries. This makes common queries an order of magnitude faster (6–20s vs 90s+), cheaper on tokens, and resilient to LLM rate limits.

See [DESIGN.md](DESIGN.md) for the full design rationale.

---

## Tech stack

**Backend** FastAPI · Celery · Redis · PostgreSQL/PostGIS · Qdrant
**Geospatial** GeoPandas · Shapely · OSMnx · Overpass API · WorldPop · rasterio
**LLM** Groq (Llama-3.3-70B) · GPT-4o-mini fallback · RAG (dense + BM25)
**Frontend** Leaflet.js
**Ops** Docker · Azure · Prometheus · Grafana · Langfuse

---

## Run it locally

**Prerequisites:** Docker and Docker Compose.

```bash
# 1. Clone
git clone https://github.com/josaiahsyiem/goai-pipeline.git
cd goai-pipeline

# 2. Configure secrets
cp .env.example .env
#    edit .env — add your GROQ_API_KEY and OPENAI_API_KEY

# 3. Start the stack
docker compose up -d --build

# 4. Open the app
#    http://localhost:8000
```

Then type a question (e.g. `flood per ward`) and a city (e.g. `Mumbai`) and click **Analyse**.

### Dashboards

| Service | URL |
|---|---|
| Frontend | http://localhost:8000 |
| Grafana | http://localhost:3001 |
| Qdrant UI | http://localhost:6333/dashboard |

---

## Cloud deployment (Azure)

NLGeo is deployed on an **Azure Virtual Machine** (Standard B2s_v2, Central India region) running the full containerized stack via Docker Compose. The deployment covers:

- Multi-container orchestration (API, worker, PostgreSQL/PostGIS, Redis, Qdrant, Grafana) on a single VM
- Network security group rules exposing the API and dashboard ports
- Persistent volume mounts for geospatial data and vector memory
- SSH key–based access and a slimmed compose file (`docker-compose.azure.yml`) tuned for the VM's resource envelope

The same `docker compose up` workflow that runs locally runs on the VM — the deployment is configuration, not a code fork.

---

## Research foundation

NLGeo extends five peer-reviewed autonomous-GIS systems — LLM-Geo (Li & Ning, 2023), LLM-Find (Ning et al., 2025), GIS Copilot (Akinboyewa et al., 2025), GISclaw (Han et al., 2025), and GTChain (Zhang et al., 2025) — adding persistent quality-gated memory, a deterministic-first execution model, production REST API, any-city support, interactive maps, and full LLM observability that none of the originals had. The comparison table and rationale are in [DESIGN.md](DESIGN.md).

---

## Project status

Core pipeline complete and validated across 20+ cities. Active work: cloud deployment hardening, multi-city comparison polish, and a planned satellite/UHI extension. See [DESIGN.md](DESIGN.md) §9 for known limitations.

---

## License

Academic project — see repository for details.
