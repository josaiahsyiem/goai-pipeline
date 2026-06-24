"""
compare_benchmark.py
--------------------
Compares the latest GoAI pipeline result against the Phase 1
ground truth for the Mumbai flood benchmark query.

Run from inside the container or with /data mounted:
    python compare_benchmark.py

The ground truth was produced by a manual Phase 1 QGIS analysis
and is stored at /data/mumbai_flood_wards_ground_truth.geojson.
"""

import sys
import geopandas as gpd

sys.path.insert(0, "/app")

GROUND_TRUTH_PATH = "/data/mumbai_flood_wards_ground_truth.geojson"

# Most recent AI pipeline result — update this after each benchmark run
AI_TOP_RESULTS = [
    ("Goregaon",      46248),
    ("Andheri East",  31623),
    ("Borivali West", 31512),
]


def load_ground_truth(path: str) -> gpd.GeoDataFrame:
    gt = gpd.read_file(path)
    return gt.sort_values("rank").reset_index(drop=True)


def print_ground_truth(gt: gpd.GeoDataFrame, n: int = 5) -> None:
    print(f"Ground truth — top {n} wards (Phase 1 manual analysis):")
    for _, row in gt.head(n).iterrows():
        print(f"  #{int(row['rank'])} {row['ward_full']}: "
              f"{int(row['flood_exposed_population']):,} people")


def print_ai_results(results: list) -> None:
    print("\nAI pipeline result — top wards from last run:")
    for i, (ward, pop) in enumerate(results, start=1):
        print(f"  #{i} {ward}: {pop:,} people")


def compare(gt: gpd.GeoDataFrame, ai_results: list, top_n: int = 5) -> None:
    gt_top = gt.head(top_n)["ward_full"].tolist()
    ai_wards = [ward for ward, _ in ai_results]

    matches = [w for w in ai_wards if w in gt_top]

    print(f"\nGround truth top {top_n}: {gt_top}")
    print(f"AI top {len(ai_wards)}: {ai_wards}")
    print(f"Matching wards: {matches} ({len(matches)}/{len(ai_wards)})")

    if len(matches) >= 2:
        print("\nResults broadly match the benchmark.")
    else:
        print("\nResults differ from the benchmark — consider prompt tuning.")


def main():
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print_ground_truth(gt)
    print_ai_results(AI_TOP_RESULTS)
    compare(gt, AI_TOP_RESULTS)


if __name__ == "__main__":
    main()
