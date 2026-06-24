"""
agents/orchestrator.py — GoAI pipeline coordinator.

LANGFUSE 4.x CORRECT API:
  - start_as_current_observation() opened inside _pipeline()
  - All create_event() calls auto-nest via OTel context propagation
  - Scores posted via langfuse.score() with trace_id
"""

import json
import os
import time
import uuid

from agents.analysis_agent import run_analysis_for_task, generate_methodology_explanation
from agents.retrieval_agent import fetch_data_for_task
from eval.scorer import score_result
from memory.store import retrieve_similar, store_task
from tools.llm_client import smart_chat
from tools.prompts import GIS_EXPERT_SYSTEM_PROMPT

from langfuse_client import langfuse


# ── Langfuse helper ───────────────────────────────────────────────────────────

def _lf_event(name, input=None, output=None):
    try:
        kwargs = {"name": name}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        langfuse.create_event(**kwargs)
    except Exception:
        pass


# ── City normalization ────────────────────────────────────────────────────────

CITY_ALIASES = {
    "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai",
    "bangalore": "Bengaluru", "poona": "Pune", "new delhi": "Delhi",
    "ncr": "Delhi", "greater mumbai": "Mumbai", "greater london": "London",
    "london uk": "London", "london england": "London",
    "new york city": "New York", "nyc": "New York",
    "la": "Los Angeles", "sf": "San Francisco",
    "lunden": "London", "londn": "London",
    "mumbai india": "Mumbai", "delhi india": "Delhi",
    "paris france": "Paris", "berlin germany": "Berlin",
}


def normalize_city(city):
    if not city:
        return city
    n = CITY_ALIASES.get(city.lower().strip())
    if n:
        print(f"[Orchestrator] City normalized: '{city}' → '{n}'")
        return n
    return city.strip().title()


# ── Query rewriting ───────────────────────────────────────────────────────────

QUERY_REWRITES = {
    "areas with lots of doctors": "hospital density by ward",
    "where are the most hospitals": "hospital density by ward",
    "best areas for healthcare": "hospital density by ward",
    "areas with good schools": "school density by ward",
    "greenest areas": "green space coverage by ward",
    "most trees": "green space coverage by ward",
    "worst flooding": "flood risk by ward",
    "flood prone areas": "flood risk by ward",
    "most buses": "transit density by ward",
    "best transport": "public transport density by ward",
    "most congested": "road density by ward",
    "busiest roads": "road density by ward",
    "most people": "population density by ward",
    "most crowded": "population density by ward",
    "safest areas": "lowest flood risk by ward",
    "most parking": "parking density by ward",
    "best cycling": "cycling infrastructure density by ward",
    "most shops": "commercial density by ward",
}

GIS_KEYWORDS = [
    "density", "per capita", "per 100k", "per ward", "by ward",
    "by borough", "by district", "coverage", "proximity", "within",
    "flood risk", "flood", "greenspace", "population", "hospital", "school",
    "road", "transit", "cycling", "infrastructure", "analysis",
    "heat island", "uhi", "thermal", "surface temperature",
    "ndvi", "vegetation", "greenness", "heat map", "heat stress"
]


def rewrite_query(task, city):
    tl = task.lower().strip()
    for casual, professional in QUERY_REWRITES.items():
        if casual in tl:
            r = f"{professional} in {city}" if city else professional
            print(f"[Orchestrator] Query rewritten: '{task}' → '{r}'")
            return r
    if any(kw in tl for kw in GIS_KEYWORDS):
        return task
    try:
        prompt = (
            f'A user asked this geographic question in casual language:\n"{task}"\nCity: "{city}"\n\n'
            'Rewrite it as a precise GIS analysis query. Return ONLY the rewritten query, nothing else.'
        )
        r = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True,
                       call_name="query_rewrite").strip()
        if len(r) > 200 or "\n" in r or r.startswith("{"):
            return task
        if r != task:
            print(f"[Orchestrator] Query rewritten (LLM): '{task}' → '{r}'")
        return r
    except Exception:
        return task


# ── Task decomposition ────────────────────────────────────────────────────────

def decompose_task(task, city, memory_context=None, has_upload=False,
                   upload_files=None, domain_hint=None):
    memory_text = ""
    if memory_context:
        memory_text = "\n\nSimilar past tasks for context:\n"
        for m in memory_context:
            memory_text += (
                f"- Task: {m['task']} | City: {m['city']} | "
                f"Type: {m['analysis_type']} | Score: {m['eval_score']} | "
                f"Top results: {m['top_results']}\n"
            )
    upload_hint = ""
    if has_upload and upload_files:
        file_lines = [
            f"  - {os.path.basename(f.get('file_path', ''))} "
            f"({f.get('type', 'file')}, {f.get('rows', '?')} rows, columns: {f.get('columns', [])})"
            for f in upload_files
        ]
        upload_hint = (
            f"\nNote: User has uploaded {len(upload_files)} file(s):\n"
            + "\n".join(file_lines)
            + '\nUse "uploaded_file" as the required source.'
        )
    elif has_upload:
        upload_hint = '\nNote: User has uploaded a custom data file. Use "uploaded_file" as the required source.'

    domain_hint_text = ""
    if domain_hint:
        domain_hint_text = f"\nDomain knowledge from user: {domain_hint}\nUse this to set parameters, thresholds, or buffer distances."

    prompt = f"""You are given a geographic analysis question. Decompose it into a structured plan.

Question: "{task}"
City: "{city}"
{memory_text}{upload_hint}{domain_hint_text}
Return ONLY a JSON object with exactly these fields:
{{
    "city": "city name",
    "analysis_type": "flood_risk or greenspace or pollution or general",
    "required_sources": ["list of data source names needed"],
    "ranking_metric": "the main metric to rank by",
    "spatial_operations": ["list of operations in order"],
    "output_columns": ["list of output column names"],
    "parameters": {{}}
}}

For Mumbai flood risk: required_sources ["mumbai_wards","lakes_and_rivers","river_lines_streams_drains"], ranking_metric "flood_exposed_population".
If uploaded files: required_sources ["uploaded_file"].
Otherwise: required_sources ["openstreetmap"].
Return only the JSON object."""

    def _try_parse(raw: str):
        raw = raw.replace("```json", "").replace("```", "").strip()
        s = raw.find("{")
        if s == -1:
            return None
        e = raw.rfind("}") + 1
        candidate = raw[s:e] if e > s else raw[s:]
        # 1) direct parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 2) trailing-junk repair: shrink from the end
        for i in range(len(candidate) - 1, 0, -1):
            if candidate[i] == '}':
                try:
                    return json.loads(candidate[:i + 1])
                except json.JSONDecodeError:
                    continue
        # 3) truncation repair: close unterminated string, then append closers
        frag = candidate
        if frag.count('"') % 2 == 1:
            frag += '"'
        frag = frag.rstrip().rstrip(',')
        opens = frag.count('{') - frag.count('}')
        opens_sq = frag.count('[') - frag.count(']')
        frag += ']' * max(0, opens_sq) + '}' * max(0, opens)
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            return None

    parsed = None
    for attempt in range(3):
        text = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True,
                          call_name="decomposition")
        parsed = _try_parse(text or "")
        if parsed is not None and isinstance(parsed, dict) and parsed.get("city"):
            return parsed
        print(
            f"[Orchestrator] Decompose attempt {attempt+1} unparseable/truncated, retrying...")
    # Deterministic fallback — retrieval's keyword router doesn't need a perfect
    # plan; never hard-fail the pipeline on a cosmetic LLM hiccup.
    print("[Orchestrator] Decompose failed 3x — using deterministic fallback plan")
    return {
        "city": city,
        "analysis_type": "general",
        "required_sources": ["uploaded_file"] if has_upload else ["openstreetmap"],
        "ranking_metric": "auto",
        "spatial_operations": [],
        "output_columns": ["rank", "ward_name", "metric", "geometry"],
        "parameters": {},
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(task, city, task_id=None, upload_paths=None, domain_hint=None, session_id=None):
    trace = []
    start_total = time.time()
    try:
        result = _pipeline(task, city, task_id, upload_paths, domain_hint,
                           trace, start_total, session_id)
        return result
    finally:
        try:
            langfuse.flush()
        except Exception:
            pass


def _pipeline(task, city, task_id, upload_paths, domain_hint, trace, start_total, session_id=None):
    """Core pipeline — Langfuse span opened here so all events auto-nest via OTel."""

    _trace_id = uuid.uuid4().hex
    _lf_cm = None
    _pipeline_obs = None

    try:
        _lf_cm = langfuse.start_as_current_observation(
            name="goai_pipeline",
            as_type="span",
            trace_id=_trace_id,
            input={"task": task, "city": city,
                   "upload_count": len(upload_paths or [])}
        )
        _pipeline_obs = _lf_cm.__enter__()
    except Exception:
        pass

    def _close_span(result_dict):
        if _pipeline_obs is not None:
            try:
                _pipeline_obs.update(
                    output={
                        "success": result_dict.get("success"),
                        "score": result_dict.get("eval_score"),
                        "gt_correlation": result_dict.get("ground_truth_correlation"),
                        "output_preview": result_dict.get("output", "")[:300],
                        "working_code": result_dict.get("code", "")[:1000],
                    },
                    metadata={
                        "eval_score": result_dict.get("eval_score"),
                        "gt_correlation": result_dict.get("ground_truth_correlation"),
                        "analysis_path": result_dict.get("plan", {}).get("_analysis_path", "unknown") if result_dict.get("plan") else "unknown",
                        "total_time_s": result_dict.get("total_time_s"),
                    }
                )
            except Exception:
                pass
        if _lf_cm is not None:
            try:
                _lf_cm.__exit__(None, None, None)
            except Exception:
                pass

    def _fail(error_msg):
        r = {"success": False, "error": error_msg, "trace": trace}
        _close_span(r)
        return r

    # Step 1: Memory
    print("[Orchestrator] Checking memory for similar tasks...")
    # Quick keyword-based type hint for memory gate — avoids cross-matches
    # before the full LLM plan is available.
    _task_lower = task.lower()
    _type_hint = (
        "flood_risk" if any(w in _task_lower for w in ["flood", "waterlog", "inundation"]) else
        "greenspace" if any(w in _task_lower for w in ["green", "park", "vegetation", "ndvi"]) else
        "uhi" if any(w in _task_lower for w in ["heat", "uhi", "thermal", "temperature"]) else
        "hospital" if any(w in _task_lower for w in ["hospital", "clinic", "doctor", "health"]) else
        "general"
    )
    memory_context = retrieve_similar(
        task, city, limit=3, session_id=session_id, analysis_type=_type_hint)
    _lf_event("memory_lookup",
              input={"task": task, "city": city},
              output={
                  "similar_found": len(memory_context),
                  "similar_tasks": [{"task": m.get("task"), "score": m.get("eval_score"), "city": m.get("city")} for m in memory_context],
              })
    trace.append({
        "step": "memory_lookup", "status": "success",
        "similar_found": len(memory_context),
        "message": f"Found {len(memory_context)} similar past tasks" if memory_context else "No similar tasks found",
        "time_s": 0.0,
    })

    # Step 2: Normalize
    original_city, original_task = city, task
    city = normalize_city(city)
    task = rewrite_query(task, city)
    if city != original_city or task != original_task:
        trace.append({"step": "normalize", "status": "success",
                      "message": f"City: '{original_city}'→'{city}' | Query normalized", "time_s": 0.0})

    # Step 3: Decompose — pre-read upload schemas so the plan sees real columns
    _pre_upload_info = []
    if upload_paths:
        for _fp in upload_paths:
            if not os.path.exists(_fp):
                continue
            try:
                _ext = _fp.rsplit(".", 1)[-1].lower()
                if _ext in ("geojson", "json"):
                    import geopandas as _gpd_pre
                    _g = _gpd_pre.read_file(_fp)
                    _pre_upload_info.append({"file_path": _fp, "type": "geojson",
                                             "rows": len(_g), "columns": list(_g.columns), "crs": str(_g.crs)})
                else:
                    import pandas as _pd_pre
                    _d = _pd_pre.read_csv(_fp)
                    _pre_upload_info.append({"file_path": _fp, "type": "csv",
                                             "rows": len(_d), "columns": list(_d.columns), "crs": None})
            except Exception as _pe:
                print(f"[Orchestrator] Pre-read failed for {_fp}: {_pe}")
    print(f"[Orchestrator] Decomposing task: {task}")
    t0 = time.time()
    try:
        plan = decompose_task(task, city, memory_context,
                              has_upload=bool(upload_paths),
                              upload_files=_pre_upload_info,
                              domain_hint=domain_hint)
        _lf_event("decomposition",
                  input={"task": task, "city": city},
                  output={
                      "analysis_type": plan.get("analysis_type"),
                      "ranking_metric": plan.get("ranking_metric"),
                      "required_sources": plan.get("required_sources"),
                      "city": plan.get("city"),
                  })
    except Exception as e:
        _lf_event("pipeline_error", output={
                  "step": "decomposition", "error": str(e)})
        return _fail(f"Decomposition failed: {e}")
    plan["domain_hint"] = domain_hint or ""
    trace.append({
        "step": "decompose", "status": "success", "plan": plan,
        "message": f"Analysis type: {plan.get('analysis_type')} | Metric: {plan.get('ranking_metric')}",
        "time_s": round(time.time() - t0, 2),
    })
    print(f"[Orchestrator] Plan ready: {plan.get('analysis_type')}")

    # Step 4: Retrieve
    print("[Orchestrator] Starting retrieval...")
    t0 = time.time()
    try:
        if upload_paths:
            retrieved = {}
            all_file_info = _pre_upload_info
            for i, info in enumerate(all_file_info):
                fpath = info["file_path"]
                if info["type"] == "geojson":
                    code = f"import geopandas as gpd\nresult = gpd.read_file('{fpath}')"
                else:
                    code = f"import pandas as pd\nresult = pd.read_csv('{fpath}')"
                retrieved[f"uploaded_file_{i}"] = {
                    "code": code,
                    "output": f"ROWS: {info['rows']}\nCOLUMNS: {info['columns']}",
                    "attempts": 1, "file_path": fpath,
                }
                print(
                    f"[Orchestrator] Loaded {os.path.basename(fpath)}: {info['rows']} rows")
            _sat_kw = any(x in task.lower() for x in [
                "heat island", "urban heat", "uhi", "thermal", "surface temperature",
                "lst", "ndvi", "vegetation index", "vegetation health", "greenness",
                "heat map", "heat stress", "vegetation cover", "leaf area"])
            if _sat_kw:
                print(
                    "[Orchestrator] Upload + satellite query — fetching satellite layer too")
                try:
                    _sat_fetched = fetch_data_for_task(task, city)
                    for _k, _v in _sat_fetched.items():
                        if _k.startswith("satellite_") and "error" not in _v:
                            retrieved[_k] = _v
                except Exception as _se:
                    print(
                        f"[Orchestrator] Satellite fetch with upload failed: {_se}")
            plan["upload_paths"] = upload_paths
            plan["upload_files"] = all_file_info
            if all_file_info:
                plan["upload_path"] = all_file_info[0]["file_path"]
                plan["upload_columns"] = all_file_info[0]["columns"]
        else:
            retrieved = fetch_data_for_task(task, city)
    except Exception as e:
        _lf_event("pipeline_error", output={
                  "step": "retrieval", "error": str(e)})
        return _fail(f"Retrieval failed: {e}")

    successes = {k: v for k, v in retrieved.items() if "error" not in v}
    failures = {k: v for k, v in retrieved.items() if "error" in v}
    _lf_event("retrieval",
              input={"sources_requested": list(retrieved.keys())},
              output={
                  "successes": list(successes.keys()),
                  "failures": list(failures.keys()),
                  "file_paths": {k: v.get("file_path") for k, v in successes.items() if v.get("file_path")},
                  "row_counts": {k: v.get("output", "")[:80] for k, v in successes.items()},
              })
    trace.append({
        "step": "retrieve", "status": "success" if successes else "failed",
        "sources_fetched": list(successes.keys()), "sources_failed": list(failures.keys()),
        "message": f"Fetched: {', '.join(successes.keys())}", "time_s": round(time.time() - t0, 2),
    })
    if not successes:
        return _fail("All data sources failed")
    print(f"[Orchestrator] Retrieved: {list(successes.keys())}")

    # Step 5: Analyse
    print("[Orchestrator] Starting analysis...")
    t0 = time.time()
    try:
        analysis_result = run_analysis_for_task(task, successes, plan)
    except Exception as e:
        _lf_event("pipeline_error", output={
                  "step": "analysis", "error": str(e)})
        return _fail(f"Analysis failed: {e}")

    analysis_path = plan.get("_analysis_path", "LLM-generated")
    _lf_event("analysis_path",
              input={"task": task},
              output={
                  "path": analysis_path,
                  "analysis_type": plan.get("analysis_type"),
                  "metric": plan.get("ranking_metric"),
                  "attempts": analysis_result.get("attempts", 1),
                  "success": analysis_result.get("success"),
                  "output_preview": analysis_result.get("output", "")[:300],
              })
    trace.append({
        "step": "analyse", "status": "success" if analysis_result["success"] else "failed",
        "attempts": analysis_result.get("attempts", 1),
        "message": f"Path: {analysis_path} | Attempts: {analysis_result.get('attempts', 1)}",
        "time_s": round(time.time() - t0, 2),
    })
    if not analysis_result["success"]:
        return _fail(analysis_result["error"])

    # Step 6: Evaluate
    print("[Orchestrator] Evaluating result...")
    t0 = time.time()
    eval_scores = score_result(task, city, analysis_result["output"], plan,
                               cross_correlation=analysis_result.get("cross_correlation"))

    # Post scores to Langfuse scores tab
    try:
        langfuse.score(
            trace_id=_trace_id,
            name="eval_score",
            value=float(eval_scores.get("score", 0)),
            comment=eval_scores.get("reasoning", "")[:200],
        )
        gt = eval_scores.get("ground_truth_correlation")
        if gt is not None:
            langfuse.score(
                trace_id=_trace_id,
                name="gt_correlation",
                value=float(gt),
                comment="Spatial robustness: intersects vs within predicate",
            )
    except Exception:
        pass

    trace.append({
        "step": "evaluate", "status": "success",
        "eval_score": eval_scores.get("score"),
        "ground_truth_correlation": eval_scores.get("ground_truth_correlation"),
        "reasoning": eval_scores.get("reasoning"),
        "flags": eval_scores.get("flags", []),
        "message": f"Score: {eval_scores.get('score')} | {eval_scores.get('reasoning', '')[:80]}",
        "time_s": round(time.time() - t0, 2),
    })

    # Step 7: Store — dynamic threshold based on cross-validation correlation.
    # High spatial robustness (r >= 0.7) means the result is reproducible across
    # spatial join predicates, so we can trust it even at lower LLM-judge scores.
    gt_corr = eval_scores.get("ground_truth_correlation")
    min_score = 0.75 if (gt_corr is not None and gt_corr >= 0.7) else 0.85
    if gt_corr is not None and gt_corr >= 0.7:
        print(
            f"[Orchestrator] High spatial correlation ({gt_corr}) — store threshold relaxed to {min_score}")
    if eval_scores.get("score", 0) < min_score:
        print(
            f"[Orchestrator] Score {eval_scores.get('score')} below threshold {min_score} — skipping memory store")
    else:
        print("[Orchestrator] Storing in memory...")
    top_results = []
    import re as _re_top
    for line in analysis_result["output"].split("\n"):
        _m = _re_top.match(r'\s*#\d+\s+(.+?):\s', line)
        if _m:
            top_results.append(_m.group(1).strip())

    # Geometry validation before storing — parses RESULT_CENTROID printed by the
    # analysis sandbox; no re-execution. Only runs when score qualifies for store.
    _geo_valid = True
    if eval_scores.get("score", 0) >= min_score:
        try:
            import re as _re
            _cm = _re.search(r'RESULT_CENTROID:\s*(-?[\d.]+),(-?[\d.]+)',
                             analysis_result.get("output", ""))
            if _cm:
                import requests as _req
                _nom = _req.get('https://nominatim.openstreetmap.org/search',
                                params={'q': city,
                                        'format': 'json', 'limit': 1},
                                headers={'User-Agent': 'GoAI/1.0'}, timeout=8).json()
                if _nom:
                    _bb = _nom[0]['boundingbox']
                    _s, _n, _w, _e = float(
                        _bb[0])-1, float(_bb[1])+1, float(_bb[2])-1, float(_bb[3])+1
                    _cx, _cy = float(_cm.group(1)), float(_cm.group(2))
                    if not (_w <= _cx <= _e and _s <= _cy <= _n):
                        print(
                            f"[Orchestrator] Geometry outside {city} bbox — skipping store")
                        _geo_valid = False
        except Exception:
            pass

    if eval_scores.get("score", 0) >= min_score and _geo_valid:
        store_task(
            task_id=task_id or f"task_{int(time.time())}",
            task=task, city=city,
            analysis_type=plan.get("analysis_type", "general"),
            eval_score=eval_scores.get("score", 0.0),
            ground_truth_correlation=eval_scores.get(
                "ground_truth_correlation"),
            top_results=top_results,
            working_code=analysis_result.get("code", ""),
            session_id=session_id,
        )
    # Step 8: Return
    total_time = round(time.time() - start_total, 2)
    trace.append({"step": "complete", "status": "success",
                 "total_time_s": total_time})
    print(
        f"[Orchestrator] Complete in {total_time}s | Score: {eval_scores.get('score')} | Correlation: {eval_scores.get('ground_truth_correlation')}")

    _lf_event("pipeline_complete",
              input={"task": task, "city": city},
              output={
                  "success": True,
                  "score": eval_scores.get("score"),
                  "gt_correlation": eval_scores.get("ground_truth_correlation"),
                  "reasoning": eval_scores.get("reasoning"),
                  "top_results": top_results[:5],
                  "total_time_s": total_time,
                  "analysis_path": analysis_path,
                  "geo_valid": _geo_valid,
                  "output_preview": analysis_result["output"][:300],
                  "working_code": analysis_result.get("code", "")[:1000],
              })

    result = {
        "success": True,
        "output": analysis_result["output"],
        "plan": plan,
        "eval_score": eval_scores.get("score"),
        "ground_truth_correlation": eval_scores.get("ground_truth_correlation"),
        "trace": trace,
        "total_time_s": total_time,
        "code": analysis_result.get("code", ""),
        "methodology": generate_methodology_explanation(task, plan, analysis_result),
    }
    _close_span(result)
    return result
