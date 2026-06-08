"""
tools/handbook_registry.py
--------------------------
Loads handbook JSON files and GIS guidance rules from disk.
Handbooks describe available data sources and how to fetch them.
Guidance rules are injected into every LLM prompt that generates code.
"""

import json
import os

TOOLS_DIR = os.path.dirname(__file__)
HANDBOOKS_DIR = os.path.join(TOOLS_DIR, "handbooks")


def load_all_handbooks() -> dict:
    """Returns a dict of all handbooks keyed by their 'name' field."""
    handbooks = {}
    for filename in os.listdir(HANDBOOKS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(HANDBOOKS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                handbooks[data["name"]] = data
    return handbooks


def get_handbook(name: str) -> dict | None:
    """Returns the handbook for the given source name, or None if not found."""
    return load_all_handbooks().get(name)


def load_data_source_index() -> dict:
    """Returns the data source index mapping source names to descriptions."""
    path = os.path.join(TOOLS_DIR, "data_source_index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_guidance() -> str:
    """
    Returns the GIS programming rules as a formatted string
    for injection into LLM prompts.
    """
    path = os.path.join(TOOLS_DIR, "gis_guidance.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = data["rules"]
    formatted = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(rules))
    return f"GIS programming rules — follow these in all generated code:\n{formatted}"
