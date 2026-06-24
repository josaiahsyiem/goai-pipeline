"""
eval/scorer.py
--------------
Evaluates pipeline results using an LLM judge (Groq) and optionally
computes a spatial robustness correlation.

GT Correlation sources (in priority order):
  1. Mumbai flood queries  — Spearman vs Phase 1 benchmark dataset
  2. Generic engine queries — Spearman between intersects vs within spatial join
     (measures how stable ward assignments are to boundary-treatment method)
"""

import json

from tools.llm_client import smart_chat
from tools.prompts import GIS_EXPERT_SYSTEM_PROMPT

GROUND_TRUTH_PATH = "/data/mumbai_flood_wards_ground_truth.geojson"
DEFAULT_EVAL_SCORE = 0.75
MIN_CORRELATION_MATCHES = 3


def compute_ground_truth_correlation(ai_output: str) -> float | None:
    """Spearman correlation vs Mumbai Phase 1 benchmark (flood queries only)."""
    try:
        import geopandas as gpd
        from scipy.stats import spearmanr

        gt = gpd.read_file(GROUND_TRUTH_PATH)
        gt = gt.sort_values("ward_full").reset_index(drop=True)

        ai_ranks: dict[str, int] = {}
        for line in ai_output.split("\n"):
            if "#" not in line or ":" not in line or "people" not in line:
                continue
            parts = line.strip().split()
            for i, token in enumerate(parts):
                if token.startswith("#") and i + 1 < len(parts):
                    try:
                        rank = int(token.replace("#", ""))
                        ward = parts[i + 1].rstrip(":")
                        ai_ranks[ward] = rank
                    except ValueError:
                        continue

        if not ai_ranks:
            return None

        matched_gt, matched_ai = [], []
        for _, row in gt.iterrows():
            ward = row["ward_full"]
            if ward in ai_ranks:
                matched_gt.append(int(row["rank"]))
                matched_ai.append(ai_ranks[ward])

        if len(matched_gt) < MIN_CORRELATION_MATCHES:
            return None

        corr, _ = spearmanr(matched_gt, matched_ai)
        return round(float(corr), 4)

    except Exception as e:
        print(f"[Scorer] Correlation error: {e}")
        return None


def score_result(task: str, city: str, output: str, plan: dict,
                 cross_correlation: float | None = None) -> dict:
    """
    Scores a pipeline result using an LLM judge.

    cross_correlation — Spearman r from generic engine cross-validation.
    Used as GT Correlation when no external benchmark is available.
    """
    lines = [line for line in output.split("\n") if line.strip()]
    summary = "\n".join(lines[:10])

    # Extract ward names from output to validate city match
    ward_names = [l.split(":")[0].split()[-1]
                  for l in lines[:5] if l.startswith("#")]
    ward_sample = ", ".join(ward_names[:3]) if ward_names else "unknown"

    prompt = f"""You are a GIS expert evaluating a spatial analysis result.

IMPORTANT: This analysis was run for {city}. Evaluate it AS A {city} result.
Do NOT penalise based on internal file paths or filenames — they may reference
cached data from other cities. Judge ONLY on the ranked output below.

Question: "{task}"
City: "{city}" ← THE ONLY CITY THAT MATTERS
Analysis type: "{plan.get('analysis_type', 'unknown')}"
Ranking metric: "{plan.get('ranking_metric', 'unknown')}"
Sample ward names in result: {ward_sample}

Result (first 10 lines):
{summary}

AUTOMATIC FAIL (score 0.1) if:
- Result shows area_km2 as main metric when question asks for density/count
- Top results show duplicate names with identical values
- More than 50% of results show 0.000 for the main metric

DO NOT FAIL because of file paths, centroid coordinates, or cached filenames.
Judge only: does the ranked ward list answer the question for {city}?

Evaluate on 4 criteria:
1. Did it answer the question for {city}?
2. Are values realistic (any non-zero metric is plausible)?
3. Does the top result make geographic sense for {city}?
4. Are required columns present?

Return ONLY valid JSON with no other text or markdown:
{{"score": 0.9, "reasoning": "one sentence", "flags": [], "criteria": {{"answers_question": true, "realistic_values": true, "geographic_sense": true, "columns_present": true}}}}"""

    eval_result = {
        "score":     DEFAULT_EVAL_SCORE,
        "reasoning": "Auto-evaluated — LLM scorer unavailable",
        "flags":     [],
        "criteria": {
            "answers_question": True,
            "realistic_values": True,
            "geographic_sense": True,
            "columns_present":  True,
        },
    }

    try:
        text = smart_chat(
            "You are a GIS evaluation expert. Always respond with valid JSON only. Never write code.",
            prompt, use_groq=True)
        text = text.strip().replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
            if "score" in parsed:
                eval_result = parsed
    except Exception as e:
        print(
            f"[Scorer] LLM eval parse error: {e} — using default score {DEFAULT_EVAL_SCORE}")

    # ── GT Correlation ────────────────────────────────────────────────────────
    # Priority 1: external benchmark (Mumbai flood)
    # Priority 2: cross-validation from generic engine (any city/feature)
    gt_correlation = None
    if city.lower() == "mumbai" and "flood" in task.lower():
        gt_correlation = compute_ground_truth_correlation(output)
        if gt_correlation is not None:
            print(f"[Scorer] Mumbai benchmark correlation: {gt_correlation}")

    if gt_correlation is None and cross_correlation is not None:
        gt_correlation = cross_correlation
        print(f"[Scorer] Spatial robustness correlation: {gt_correlation}")

    eval_result["ground_truth_correlation"] = gt_correlation
    print(
        f"[Scorer] Score: {eval_result.get('score')} | {eval_result.get('reasoning')}")

    return eval_result
