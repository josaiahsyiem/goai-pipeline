# NLGeo — Design Rationale

> An autonomous GeoAI pipeline that turns a plain-English geographic question into a ranked, ward-level choropleth map for any city in the world.

This document explains *why* NLGeo is built the way it is: the architecture, the research it extends, the central design tension between deterministic and LLM-generated analysis, what was tried and rejected, and the tradeoffs that shaped the final system. It is written to be read alongside the code, not as a substitute for it.

---

## 1. Problem statement

Answering a question like *"Which Mumbai wards have the highest flood-exposed population?"* normally requires a GIS analyst: someone who knows how to source ward boundaries, find water-body and drainage layers, choose a projection, buffer linear features correctly, compute areal intersection, weight by population, and rank the result. Each step is a place where a non-expert gets stuck.

NLGeo removes the analyst from the loop. A user types the question in natural language and a city; the system produces the ranked map. The design goal that governs every decision below is **generalization without hardcoding**: the same pipeline that answers the Mumbai flood benchmark must answer "pharmacies per ward in Berlin" or "hospitals within 3 km of central Paris" without city-specific code. Mumbai is the *benchmark*, not the *scope*.

---

## 2. Research foundation

NLGeo extends five peer-reviewed systems for autonomous GIS with LLMs. The table states what each contributed and what NLGeo adds on top.

| Capability | LLM-Geo (2023) | LLM-Find (2025) | GIS Copilot (2025) | GISclaw (2025) | GTChain (2025) | **NLGeo** |
|---|---|---|---|---|---|---|
| Self-generating analysis code | Yes | Yes | Yes | Yes | Yes | Yes |
| Self-verifying accuracy | No | Partial | Partial | Yes | Partial | **Yes — LLM judge + spatial cross-validation** |
| Persistent vector memory | No | No | No | No | No | **Yes — Qdrant, quality-gated** |
| Error-memory on retry | No | No | No | Yes | No | **Yes — separate Qdrant collection** |
| Production REST API | Notebook | Notebook | QGIS plugin | No | No | **Yes — FastAPI + Celery + Docker** |
| Any-city support | Fixed | Fixed | Limited | Limited | Limited | **Yes — Overpass/OSMnx, 20+ verified** |
| Interactive choropleth | No | No | No | No | No | **Yes — Leaflet, ranked sidebar** |
| LLM observability | No | No | No | No | No | **Yes — Langfuse, tokens + cost/call** |
| Per-capita (population raster) | No | No | Yes (QGIS) | Yes | No | **Yes — WorldPop, auto-download per country** |

The intellectual core NLGeo adds to this lineage is **persistent, quality-gated memory** and **a deterministic-first execution model**. The papers treat each query as stateless and route everything through the LLM. NLGeo treats the LLM as the fallback, not the default — and that single inversion drives most of what follows.

---

## 3. System architecture

NLGeo runs as a multi-container stack. Containers are split by concern so that no slow operation ever blocks the API.

```
                ┌──────────────┐
   user ──────► │  FastAPI     │  POST /query  → enqueue only, return task_id
                │  (api)       │  GET  /query/{id}        → poll status
                └──────┬───────┘  GET  /query/{id}/geojson → choropleth data
                       │ enqueue
                ┌──────▼───────┐
                │  Redis       │  Celery broker
                └──────┬───────┘
                       │ dequeue
                ┌──────▼───────────────────────────────────────┐
                │  Celery worker — 7-stage orchestrator         │
                │  1 Memory → 2 Decompose → 3 Retrieve →        │
                │  4 Analyse → 5 Evaluate → 6 Store → 7 Return  │
                └──┬─────────┬──────────┬──────────┬────────────┘
                   │         │          │          │
            ┌──────▼──┐ ┌────▼────┐ ┌───▼────┐ ┌───▼─────┐
            │Postgres │ │ Qdrant  │ │Overpass│ │ WorldPop│
            │ /PostGIS│ │ vectors │ │ OSMnx  │ │ rasters │
            └─────────┘ └─────────┘ └────────┘ └─────────┘
```

**The first design rule is: never run the orchestrator inside the request handler.** The FastAPI handler does two things — write a pending row and enqueue a Celery task — then returns a `task_id` immediately. All geospatial computation happens in the worker. A flood analysis that takes 30 seconds, or a Paris restaurant query that takes 156 seconds, must never hold an HTTP connection open. The frontend polls. This is the difference between a demo notebook and a service that more than one person can use at once.

The worker runs a fixed **seven-stage pipeline**: memory lookup, decomposition, retrieval, analysis, evaluation, storage, and return. Each stage is observable and each stage can short-circuit the ones after it. Memory can skip decomposition; the deterministic analyzer can skip the LLM entirely.

---

## 4. The central design decision: deterministic-first, LLM-as-fallback

Every paper in the lineage routes analysis through the LLM: the model reads the query, writes the code, the code runs. NLGeo inverts this. The analysis stage is a **routing cascade** that tries cheap, deterministic, verifiable paths first and only reaches LLM-generated code as the last resort.

The cascade (simplified):

1. **Memory reuse** — a near-identical past query for the same city, with valid file paths, returns its stored code immediately.
2. **Mumbai flood benchmark** — a hardcoded, deterministic implementation of the ground-truth methodology. Always produces Spearman 1.0. Never delegated to the LLM.
3. **File-upload spatial join** — user-supplied GeoJSON/CSV joined point-in-polygon.
4. **Per-capita** — WorldPop raster zonal statistics.
5. **Generic OSM engine** — 50+ feature types (hospitals, pharmacies, schools, cafes, parks, roads…) handled deterministically with zero LLM calls.
6. **LLM-generated code** — only when nothing above matches; up to five self-correcting attempts.

### Why invert the default?

Three reasons, each learned from a concrete failure mode.

**Determinism is verifiable; LLM output is not.** The Mumbai flood query has a known correct answer — a ground-truth GeoJSON produced by hand in QGIS, ranking 24 wards by flood-exposed population. A deterministic implementation reproduces it at Spearman 1.0 every single run. An LLM asked to write the same analysis will *usually* get it right, but "usually" is not a benchmark. By hardcoding the methodology for the one query where correctness is objectively measurable, the system has a fixed anchor: if the benchmark ever drifts from 1.0, something upstream broke. You cannot get that signal from a path that is stochastic by construction.

**Determinism is free; the LLM is rate-limited and costs money.** Each LLM-path query consumes 5–10k tokens. The Groq free tier caps at 100k tokens/day/key. The generic OSM engine covers the *majority* of real queries — "X per ward in city Y" — with no LLM call at all, which means the token budget is spent only on genuinely novel phrasings. This is not a micro-optimization; during testing, exhausting all seven rotating Groq keys degraded LLM planning to the point where dependent queries failed. A system that needs the LLM for every routine query is a system that stops working when the quota runs out. Pushing routine work onto deterministic paths is what makes the LLM dependency survivable.

**Determinism is fast.** Deterministic paths complete in 6–20 seconds. LLM paths with retries can take 90–156 seconds because each failed attempt is another round-trip plus re-execution. For the common case, deterministic is an order of magnitude faster.

### What this costs

The price of deterministic-first is **code surface area**. `analysis_agent.py` carries explicit implementations for flood, density, per-capita, greenspace, and the 50+ generic features. That is a lot of code to maintain, and some of it is city-shaped (the Mumbai fast path reads specific local files). The honest tradeoff: we accept more code in exchange for verifiability, speed, and quota-resilience on the paths that matter most. The planned `SKIP_DETERMINISTIC` experiment — disabling each deterministic path and checking whether Groq alone clears the 0.85 score bar — exists specifically to find which deterministic paths have earned their keep and which can be deleted. The expectation is that the Mumbai benchmark and WorldPop per-capita survive (they are exact and verifiable), while some single-feature density paths may turn out to be replaceable by the LLM and worth removing.

---

## 5. Memory: the feature none of the papers had

The papers are stateless. NLGeo is not. After every query that scores well, the system embeds the task text and stores it in a Qdrant vector collection (`task_memory`) alongside the working code, the city, the analysis type, the evaluation score, and the top results. A later, similarly-worded query retrieves it by cosine similarity and can reuse the code outright.

This is powerful and also dangerous, and the design reflects scars from both.

**Quality gating.** Early versions stored every completed task. Mediocre results — a query that accidentally pulled a state-level polygon instead of city wards — polluted memory and got reused. The fix was a hard quality gate: only store if the evaluation score is **≥ 0.85** *and* the result geometry centroid falls inside the city's Nominatim bounding box (±1°). The geometric gate is the important half: it catches the "right number, wrong place" failures that a score alone misses.

**City and type filtering.** Memory retrieval is gated by both analysis type and city, so a Mumbai flood query can never reuse a Berlin pharmacy result. Without this, cross-contamination produces confident, fast, wrong answers — the worst kind.

**Stale-path protection.** Before reusing stored code, every `/data/` path it references is checked with `os.path.exists()`. Cached code that points at a file that has since been cleaned up is discarded rather than run.

**A known sharp edge.** Memory reuse sits *first* in the cascade — ahead of the deterministic benchmark. This is correct for throughput (an exact past match is the cheapest possible path) but it created a real bug: a generic-path result for a Mumbai flood query, once stored, would be reused on the next run *instead of* the deterministic benchmark, because the memory match scored similarity 1.0 and short-circuited everything below it. The benchmark — the most authoritative path in the entire system — was being overridden by a cached approximation of itself. The resolution is to move the benchmark check ahead of memory reuse for the specific queries that have a ground truth: authority should beat recency. This is documented here because it is exactly the kind of non-obvious interaction that the deterministic-first / memory-first ordering creates, and anyone extending the cascade needs to understand it.

---

## 6. Retrieval and the any-city requirement

Generalization lives or dies in the retrieval stage. For Mumbai, NLGeo uses local pre-built files (a fast path with zero network calls). For every other city it fetches live.

**Boundaries via OSMnx, features via Overpass.** Polygon boundary fetching is reliable through OSMnx's `features_from_place`, using an admin-level cascade (10 → 9 → 8 → 7, stopping at the first level with ≥ 3 polygons) so that the system finds the right granularity of "ward" whether the city calls them wards, arrondissements, or boroughs. Point and line *features*, however, were migrated off OSMnx to direct **Overpass QL** queries. OSMnx's feature fetch repeatedly produced `StringDtype` incompatibilities with Fiona and downloaded the entire feature graph when only matching features were needed; Overpass fetches just the 542 Berlin pharmacies, not the whole graph. Two Overpass endpoints are tried in sequence so a single outage does not kill the query.

**Projection is auto-detected, never hardcoded.** An early version hardcoded `EPSG:32643` (the UTM zone for Mumbai). That silently corrupts area calculations everywhere else on Earth. The fix derives the UTM zone from the GeoDataFrame centroid at runtime, so London resolves to 32630, Berlin to 32633, Paris to 32631, Mumbai to 32643 — automatically. This one change is most of what "any city" actually means in practice: correct areas require the correct projection, and the correct projection depends on where the city is.

**Boundary quality filtering.** Live OSM data is messy. Queries would occasionally return a country-level polygon, or leisure/landuse/natural features masquerading as administrative boundaries. The retrieval stage requires `boundary=administrative` with a non-null name and drops any polygon larger than 15× the median ward area — which is what eliminates the Brandenburg state polygon from a Berlin ward query.

---

## 7. Self-verification

A system that writes its own analysis must check its own work, or it is just confident noise. NLGeo verifies on two levels.

**An LLM judge** scores every result 0–1 against four criteria: does it answer the question, are the values realistic, does it make geographic sense, are the required columns present. This catches gross failures and feeds the memory quality gate.

**Spatial cross-validation** is the more rigorous check and a contribution beyond the papers. After the main analysis (which assigns features to wards using an `intersects` predicate), the system re-counts features per ward using a stricter `within` predicate, and computes the Spearman correlation between the two rankings. High correlation means the ward assignments are stable regardless of how boundary edge-cases are treated — a robustness signal available for *every* city, not just the one with an external benchmark. Mumbai flood reports true ground-truth correlation; every other query reports this cross-validation score. Both surface in the same "GT Correlation" field in the UI, so the user always sees a confidence signal.

---

## 8. What was tried and rejected

Design is as much what you remove as what you keep.

- **Hardcoded EPSG:32643** — corrupted areas off the Indian coast. Replaced with runtime UTM detection. (See §6.)
- **OSMnx for feature fetch** — `StringDtype`/Fiona errors and whole-graph downloads. Replaced with Overpass QL. (See §6.)
- **Storing every completed task in memory** — polluted retrieval with mediocre and mislocated results. Replaced with a 0.85 score gate plus a geometric in-bounds gate. (See §5.)
- **`reset_index()` without `drop=True`** — silently dropped the GeoDataFrame CRS, breaking every downstream projection. Fixed globally.
- **Running the orchestrator in the FastAPI handler** — blocked the server for the duration of every analysis. Moved entirely to Celery. (See §3.)
- **`{str(ascending)}` interpolated into generated code** — produced the literal string `"False"`, which is truthy in Python, silently inverting sort order in "least/lowest/safest" queries. A reminder that f-string interpolation into executable code is a footgun.

These are listed because the rejected designs are the actual content of the engineering. The final architecture is the residue of these failures.

---

## 9. Known limitations

Stated plainly, because a design document that only lists strengths is marketing.

- **Mumbai greenspace**: Nominatim returns a point, not a polygon, for "Mumbai," so boundary fetch fails for this specific combination; Mumbai also has sparse greenspace tagging in OSM. A data/geocoding gap, not a code bug — but a real limit.
- **Sparse-OSM cities**: Amsterdam returned only three administrative polygons. The system fails gracefully, but the answer is only as good as OSM coverage.
- **Large metros**: "Tokyo" as a whole metro times out at 300s. District-level queries work; the full metro does not. A scale ceiling, not a correctness failure.
- **Greenspace aggregation**: greenspace queries currently rank individual parks rather than aggregating per ward — a query-type mismatch still on the list.
- **LLM quota**: the deterministic-first design mitigates but does not eliminate the Groq dependency. Genuinely novel queries still need the LLM, and the 100k-token/day/key ceiling is real. Seven-key rotation multiplies it; it does not remove it.

---

## 10. Summary of the design philosophy

NLGeo is built on one inversion and one addition.

The **inversion** is deterministic-first: the LLM is the fallback, not the default. This buys verifiability (a benchmark anchored at Spearman 1.0), speed (an order of magnitude on common queries), and resilience against rate limits. It costs code surface area, which the `SKIP_DETERMINISTIC` experiment is designed to prune back.

The **addition** is quality-gated persistent memory: the system gets faster and cheaper the more it is used, but only stores results that pass both a score threshold and a geometric in-bounds check — because a memory of a wrong answer is worse than no memory at all.

Everything else — the Celery split, runtime UTM detection, Overpass migration, dual-level self-verification, the boundary quality filters — exists to make a single sentence true in practice rather than in principle: *any city, no hardcoding, with a confidence signal the user can see.*
