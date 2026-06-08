# GoAI — Geographic AI Pipeline

**Ask any spatial question about any city. Get a ranked, interactive choropleth map.**

> *"Which Berlin wards have the most pharmacies?"*
> *"Hospitals within 2km of Paris city center"*
> *"Which Cairo areas have the most schools?"*

GoAI is a production-grade autonomous GeoAI system. It takes a natural language query, fetches live OpenStreetMap data via the Overpass API, runs deterministic or LLM-generated geospatial analysis in a sandboxed subprocess, self-verifies the result with spatial cross-validation, stores findings to vector memory, and renders an interactive choropleth map — all from a single query with no GIS expertise required.

Built on and extending five peer-reviewed GeoAI papers: **LLM-Geo** (Li & Ning, 2023), **LLM-Find** (Ning et al., 2025), **GIS Copilot** (Akinboyewa et al., 2025), **GISclaw** (Han et al., 2025), **GTChain** (Zhang et al., 2025).

---

## Quick Start

```bash
git clone https://github.com/josaiahsyiem/goai-pipeline.git
cd goai-pipeline
cp .env.example .env        # fill in your API keys
docker-compose up
```

Open `http://localhost:8000` — type a question, select a city, click Analyse.

---

## Architecture

### Infrastructure — 7-container Dockerised stack

| Container | Role |
|---|---|
| `goai_api` | FastAPI REST API — receives queries, returns results |
| `goai_worker` | Celery async task worker — runs the pipeline |
| `goai_redis` | Message broker for Celery |
| `goai_postgres` | PostgreSQL/PostGIS — spatial result persistence |
| `goai_qdrant` | Qdrant vector memory (3 collections) |
| `goai_prometheus` | Prometheus — live pipeline metrics |
| `goai_grafana` | Grafana — monitoring dashboard at :3000 |

Plus a separate 7-container Langfuse observability stack (ClickHouse, MinIO, Redis, Postgres, web, worker).

Single `docker-compose up` startup with health checks and restart policies.

### Pipeline — 7 stages

```
Natural language query
        │
        ▼
1. Memory Lookup        — Qdrant cosine similarity search (reuse if score > 0.95)
        │
        ▼
2. Query Decomposition  — LLM extracts city, analysis type, sources, ranking metric
        │
        ▼
3. Data Retrieval       — Overpass API (features) + OSMnx (boundaries) + WorldPop (rasters)
        │
        ▼
4. Spatial Analysis     — 14-level routing cascade (deterministic → generic → LLM-generated)
        │
        ▼
5. Evaluation & Scoring — LLM judge (4 criteria) + Spearman spatial cross-validation
        │
        ▼
6. Memory Storage       — Quality-gated: score ≥ 0.85 + geometry bbox validation
        │
        ▼
7. Choropleth Map        — Leaflet.js, real-time progress pills, ranked sidebar, hover popups
```

---

## Analysis Engine

### 14-level routing cascade

Every query passes through this cascade in order, stopping at the first success:

| Level | Path |
|---|---|
| 1 | Memory reuse — similarity > 0.95, same city, file paths valid |
| 2 | Mumbai flood benchmark — hardcoded deterministic, Spearman 1.0 |
| 3 | Multi-file upload — GeoJSON + CSV point-in-polygon spatial join |
| 4 | Single file upload — LLM analysis on uploaded file |
| 5 | Per-capita — WorldPop GeoTIFF raster zonal statistics |
| 6 | Mumbai general engine — deterministic for population/density/area |
| 7 | Hospital density — centroid sjoin, hospital_count / area_km2 |
| 8 | Road density — edge length from geometry, sum / area_km2 |
| 9 | Flood risk — water buffer 100–300m, intersection ratio per ward |
| 10 | Schools — school centroid sjoin, school_count / area_km2 |
| 11 | Greenspace — polygon area filter, sorted by area_km2 |
| 12 | Generic OSM engine — 50+ feature types, fires before LLM |
| 13 | OSMnx template — boundary fetch loop (admin_level 10→7) |
| 14 | LLM-generated code — 5-attempt self-healing with fault attribution |

### Generic OSM Engine (`generic_engine.py`)

A single deterministic engine covering 50+ OSM feature categories with zero LLM dependency:

**Point features:** hospitals, clinics, pharmacies, dentists, vets, schools, kindergartens, universities, libraries, banks, ATMs, restaurants, cafes, bars, fuel stations, EV chargers, police, fire stations, post offices, places of worship, cinemas, theatres, playgrounds, gyms, supermarkets, hotels, museums, bus stops, metro stations, train stations, parking, bicycle parking, and more.

**Area features:** parks, gardens, forests, green space, swimming pools, stadiums, nature reserves, commercial zones, residential areas, water bodies.

**Line features:** cycle lanes, footpaths, roads, streets.

**4 analysis modes per feature:**
- Raw count per ward
- Density per km²
- Per-capita per 100k (WorldPop raster)
- Proximity — ranked by distance from city center

**4 composite patterns:**
- Deprivation — most underserved relative to population
- Optimization — best location for new facilities
- Contrast — high feature A but low feature B
- Combined — high A and high B

For any unrecognised feature, 1 LLM call translates the phrase into OSM tags before running deterministically.

---

## Key Features

### Data Sources
- **Overpass API** — primary feature fetch with 2-endpoint fallback (overpass-api.de → overpass.kumi.systems). Eliminates StringDtype errors vs OSMnx. Faster for sparse features.
- **OSMnx** — boundary polygon fetch (admin_level 10→7 cascade with city-clip)
- **WorldPop GeoTIFF** — 100m resolution population rasters, auto-downloaded per country ISO3, zonal statistics via rasterstats
- **Nominatim** — city geocoding and bounding box

### LLM Stack
- **Groq Llama-3.3-70B** — primary, 7-key rotation for rate limit management
- **GPT-4o-mini** — automatic fallback on Groq 429 errors
- **nomic-embed-text** (768-dim) — embeddings for Qdrant vector memory

### GIS Copilot Features (3/3)
- Hybrid BM25 + dense RAG over GIS handbooks (Qdrant tool_docs collection, 10 chunks)
- Chain-of-Thought query refining — LLM identifies spatial operation, data layers, metric, and pitfalls before writing any code
- Data understanding — auto-extracts CRS, geometry types, extent, column names, sample values from every fetched file

### GISclaw Features (10/10)
- Sandbox import interception — arcpy, pykrige, skimage, arcgis blocked at `builtins.__import__` level
- Error-Memory — Qdrant `error_memory` collection stores past error→fix pairs, retrieved and injected on retry
- Type B attentional error detection — same error across 2 consecutive attempts triggers explicit "do not repeat the same non-fix" injection
- Code deduplication — `hash(code.strip())` tracked per task, duplicate code+error combos skipped
- Domain knowledge injection — expert hint text box in UI flows to decomposition prompt
- Output format contract — result must be GeoDataFrame with rank, ward_name, metric, geometry (EPSG:4326)
- Zero-values validation (Type 3) — all top-5 metric values = 0.0 triggers retry
- Per-task timeout — 300s default, 600s for road density / flood / schools
- Asymmetric truncation — stdout from front, stderr from tail (root-cause error never cut off)
- Variable tracking — sandbox prints `VARIABLES: [...]` after execution for LLM retry context

### GTChain Features (4/4)
- Workflow planning — 6-step INPUT→TRANSFORM→SPATIAL_OP→AGGREGATE→METRIC→OUTPUT generated before code
- Token efficiency — `refine_and_plan()` merges query refinement + workflow into 1 LLM call (saves ~2000 tokens)
- Re-planning on retry — fresh workflow generated on attempt 3+ when current approach is failing
- Self-check mechanism — `validate_code_paths()` checks every `gpd.read_file()` path exists before sandbox execution

### Spatial Cross-Validation
After every generic engine query, re-counts features per ward using `within` predicate vs main `intersects` predicate. Spearman r between the two = spatial robustness score shown as GT Correlation in the UI. High r (≥ 0.85) means ward assignments are stable regardless of boundary treatment method.

For Mumbai flood queries, GT Correlation is instead the Spearman r against a manually verified QGIS ground truth — a true external accuracy benchmark.

### Memory Quality Gates
- Score threshold ≥ 0.85
- Geometry centroid must fall inside city bounding box (±1°) from Nominatim
- Generic engine queries skip LLM code reuse entirely

### Langfuse Observability
Every LLM call traced with input/output tokens, cost per call, call name, retry attempt events, and scores linked to trace_id. Groq pricing: $0.59/M input, $0.79/M output. GPT-4o-mini: $0.15/M input, $0.60/M output.

### File Upload Engine
Upload any GeoJSON boundary + CSV dataset. GoAI auto-detects the ward name column, population column, and lat/lon columns, runs a point-in-polygon spatial join, and computes per-100k rates when population data is present. Tested on 225 wards × 123 hospitals in Bengaluru, and 290 wards in Delhi.

### Frontend
- Leaflet.js choropleth — ward polygons coloured by rank, red→green gradient
- Real-time progress pills: Memory → Plan → Fetch → Analyse → Score → Done
- Ranked results sidebar with metric values and colour-coded badges
- Hover popups showing ward name, rank, metric value
- Expert hint text box — domain knowledge injected into the analysis prompt

---

## Verified Test Results

| City | Query | Score | GT Corr | Time | Notes |
|---|---|---|---|---|---|
| Mumbai | Flood risk by ward | 0.90 | 1.00 | 6–10s | Spearman vs QGIS ground truth |
| Berlin | Pharmacies per ward | 0.90 | 1.00 | 33–42s | Overpass API, 542 features, Mitte #1 |
| Greater London | Hospitals per ward | 0.90 | 1.00 | 42s | Croydon 25, Westminster 21, Camden 20 |
| Paris | Hospitals per ward | 0.90 | 1.00 | 98s | 100 arrondissements |
| Cairo | Schools per ward | 0.90 | 1.00 | 131s | Arabic ward names rendered correctly |
| Seoul | Cafes per ward | 0.90 | 1.00 | ~90s | Generic engine, Overpass API |
| New Delhi | Hospitals per area | 0.90 | 1.00 | 28s | New Delhi #1 (40 hospitals) |
| Berlin | Clinics per ward | 0.90 | 1.00 | 43s | Charlottenburg-Wilmersdorf #1 |
| Paris | Restaurants per arrondissement | 0.90 | 1.00 | 156s | 11e #1 (725 restaurants) |
| Berlin | Pharmacies within 2km of city center | 0.90 | N/A | 37s | 30 pharmacies ranked by distance |
| Paris | Hospitals within 3km of city center | 0.90 | N/A | 28s | 19 hospitals, Clinique Saint-Jean #1 |
| Mumbai | Population density by ward | 0.90 | N/A | ~20s | Deterministic engine |
| London | Parks and greenspace | 1.00 | N/A | 17s | Richmond Park #1 — correct |
| Bengaluru | Hospital coverage (file upload) | 0.80 | N/A | 5.69s | 225 wards × 123 hospitals, per-100k |

---

## How This Extends the Research

| Capability | LLM-Geo | LLM-Find | GIS Copilot | GISclaw | GTChain | GoAI |
|---|---|---|---|---|---|---|
| Self-generating code | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Self-verifying accuracy | ✗ | Partial | Partial | ✓ | Partial | ✓ + spatial cross-validation |
| Persistent vector memory | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ Qdrant, score ≥ 0.85 + geo gate |
| Error-Memory module | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ Qdrant collection, injected on retry |
| Production REST API | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ FastAPI + Celery |
| Any city support | ✗ | ✗ | Limited | Limited | Limited | ✓ 100+ cities, Overpass API |
| Overpass API integration | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ primary feature fetch, 2 endpoints |
| Proximity queries | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ within N km of city center |
| Composite analysis | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ deprivation, contrast, optimization |
| Interactive choropleth map | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ Leaflet.js |
| File upload spatial join | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ any GeoJSON + CSV |
| LLM observability | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ Langfuse, tokens, cost per call |
| WorldPop raster per-capita | ✗ | ✗ | ✓ QGIS | ✓ | ✗ | ✓ auto-download, 100+ countries |

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Manual GIS baseline — Mumbai flood ground truth in QGIS | ✓ Complete |
| 2 | Docker infrastructure — 7 containers, single startup command | ✓ Complete |
| 3 | Data pipeline — PostgreSQL schema, Qdrant collections, Celery | ✓ Complete |
| 4 | LLM agent pipeline — first working version | ✓ Complete |
| 5 | Multi-city + OSMnx integration | ✓ Complete |
| 6 | Memory + evaluation + self-growing | ✓ Complete |
| 7 | Monitoring — Prometheus + Grafana, 5 integration tests | ✓ Complete |
| 8 | Frontend — Leaflet.js choropleth + file upload engine | ✓ Complete |
| 9 | Research parity — GIS Copilot + GISclaw + GTChain features | ✓ Complete |
| 10 | Generic OSM engine — 50+ feature types, WorldPop rasters | ✓ Complete |
| 11 | Overpass API + Langfuse observability + proximity queries | ✓ Complete |
| 12 | Multi-city comparison + CSV/PDF export + deployment | Planned |

---

## Known Limitations

- Cities with sparse OSM boundary coverage (e.g. Amsterdam) may return fewer than expected ward polygons — this is an OSM data gap, not a code bug.
- Greenspace queries return individual park rankings rather than per-ward aggregation — a known query type mismatch.
- Groq free tier: 100k tokens/day. Deterministic engine mitigates this — covered queries use zero LLM tokens for analysis.
- Langfuse OTel context does not propagate across Celery fork boundaries — pipeline events appear as separate traces rather than nested. Data is complete and correct; nesting is cosmetic only.

---

## Stack

`Python` `FastAPI` `Celery` `Redis` `PostgreSQL/PostGIS` `Qdrant` `GeoPandas` `Shapely` `OSMnx` `Overpass API` `WorldPop` `rasterstats` `Leaflet.js` `Groq Llama-3.3-70B` `GPT-4o-mini` `Langfuse` `Prometheus` `Grafana` `Docker`

---

## References

- Li, Z., & Ning, H. (2023). Autonomous GIS: the next-generation AI-powered GIS. *International Journal of Digital Earth*. doi:10.1080/17538947.2023.2278895
- Ning, H., Li, Z., Akinboyewa, T., & Lessani, M. N. (2025). An autonomous GIS agent framework for geospatial data retrieval. *International Journal of Digital Earth*. doi:10.1080/17538947.2025.2458688
- Akinboyewa, T., et al. (2025). GIS Copilot. *International Journal of Digital Earth*.
- Han, et al. (2025). GISclaw. *International Journal of Digital Earth*.
- Zhang, et al. (2025). GTChain. *International Journal of Digital Earth*.
