"""
test_integration.py
-------------------
Integration tests for the GoAI pipeline.
Run against a live stack: python app/tests/test_integration.py
Requires the API to be running at BASE_URL.
"""

import json
import time
import requests

BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120


# ── Helpers ───────────────────────────────────────────────────────────────────

def post_query(task: str, city: str = "Mumbai") -> str:
    response = requests.post(
        f"{BASE_URL}/query",
        json={"task": task, "city": city},
    )
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    data = response.json()
    assert "task_id" in data, "Response missing task_id"
    return data["task_id"]


def wait_for_result(task_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/query/{task_id}")
        data = response.json()
        if data["status"] in ("complete", "failed"):
            return data
        time.sleep(3)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    print("\n=== Test 1: Health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("PASSED")


def test_metrics_endpoint():
    print("\n=== Test 2: Metrics endpoint ===")
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "goai_queries_total" in response.text
    assert "goai_eval_score_avg" in response.text
    print("PASSED")


def test_mumbai_flood_query():
    print("\n=== Test 3: Mumbai flood benchmark ===")
    task_id = post_query(
        "Which Mumbai wards have highest flood exposed population?",
        "Mumbai",
    )
    print(f"Task ID: {task_id}")

    result = wait_for_result(task_id)
    assert result["status"] == "complete", f"Task failed: {result}"

    result_data = json.loads(result["result"])

    assert "output" in result_data, "Missing output field"
    assert "flood_exposed_population" in result_data["output"], "Output missing flood metric"
    assert "eval_score" in result_data, "Missing eval_score"
    assert "ground_truth_correlation" in result_data, "Missing ground_truth_correlation"

    corr = result_data["ground_truth_correlation"]
    assert corr is not None and corr >= 0.8, f"Ground truth correlation too low: {corr}"

    print(f"Eval score:              {result_data['eval_score']}")
    print(f"Ground truth correlation: {corr}")
    print("PASSED")


def test_memory_context():
    print("\n=== Test 4: Memory context ===")
    task_id = post_query(
        "Which Mumbai wards have highest flood exposed population?",
        "Mumbai",
    )
    result = wait_for_result(task_id)
    assert result["status"] == "complete", f"Task failed: {result}"

    trace = json.loads(result["trace"])
    memory_step = next(
        (s for s in trace if s["step"] == "memory_lookup"), None)
    assert memory_step is not None, "Memory lookup step missing from trace"
    assert memory_step["similar_found"] >= 1, "No similar tasks found in memory"

    print(f"Similar tasks found: {memory_step['similar_found']}")
    print("PASSED")


def test_nonsensical_query():
    print("\n=== Test 5: Nonsensical query (graceful handling) ===")
    task_id = post_query("purple elephant banana spaceship", "Mumbai")
    print(f"Task ID: {task_id}")

    result = wait_for_result(task_id)
    assert result["status"] in ("complete", "failed"), \
        f"Unexpected status: {result['status']}"

    print(f"Status returned: {result['status']}")
    print("PASSED — system handled gracefully")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("GoAI Integration Tests")
    print("=" * 50)

    tests = [
        test_health_endpoint,
        test_metrics_endpoint,
        test_mumbai_flood_query,
        test_memory_context,
        test_nonsensical_query,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        raise SystemExit(1)
