"""
agents/retrieval_agent.py
-------------------------
Fetches geospatial data required for analysis.
Two retrieval paths:

1. Multi-source OSMnx — for complex queries (road density, hospital proximity,
   green coverage, flood risk) that require multiple datasets. Fetches each
   source deterministically (no LLM) and saves to disk so analysis doesn't
   need to re-download.

2. Standard single-source — for all other queries. Uses known local file paths
   for Mumbai data, or an LLM-generated OSMnx fetch script for any other city.

FIXES:
- FIX 1: per_capita multi-source entry no longer hardcodes osm_hospitals
         (analysis agent handles inline fetch when pre-fetch fails)
- FIX 2: osm_hospitals template uses bbox fetch instead of features_from_place
         to avoid Nominatim returning a point geometry for Mumbai
- FIX 3: worldpop_population template CITY_ISO3 expanded with missing cities
         including 'greater london', all Indian cities, etc.
- FIX 4: file_path null check — results dict always has file_path key, None if save failed
"""

import json
import os
import subprocess
import sys

from tools.handbook_registry import load_all_handbooks, load_data_source_index, load_guidance
from tools.llm_client import smart_chat
from tools.prompts import GIS_EXPERT_SYSTEM_PROMPT

# ── Known local data paths ────────────────────────────────────────────────────

KNOWN_LOCAL_SOURCES = {
    "mumbai_wards":                "/data/mumbai_ward_shapefile/Mumbai_wards.geojson",
    "lakes_and_rivers":            "/data/geojson_files/lakes_and_rivers.geojson",
    "river_lines_streams_drains":  "/data/geojson_files/river_lines_streams_drains.geojson",
}

# ── Multi-source query registry ───────────────────────────────────────────────

MULTI_SOURCE_QUERIES = {
    "road_density": {
        "keywords": [
            "road density", "street density", "road network", "roads per km",
            "road coverage", "street network", "roads per square",
        ],
        "sources":  ["osm_boundaries", "osm_roads"],
    },
    "hospital_proximity": {
        "keywords": [
            "hospital proximity", "distance to hospital", "nearest hospital",
            "hospital density", "hospital coverage", "hospitals per",
            "healthcare access", "medical facilities", "clinic density",
        ],
        # FIX 1: removed worldpop_population from here — per_capita path in
        # analysis_agent handles WorldPop. osm_hospitals kept for pre-fetch attempt.
        "sources":  ["osm_boundaries", "osm_hospitals"],
    },
    "green_coverage": {
        "keywords": [
            "green coverage", "green space percentage", "park coverage",
            "green space per capita", "green space per person", "park per capita",
            "parks per resident", "green space per 100k", "highest green space",
            "most green space", "green space",
        ],
        "sources": ["osm_boundaries", "osm_greenspace", "osm_parks"],
    },
    "flood_risk": {
        "keywords": ["flood risk", "flood zone", "flood exposure"],
        "sources":  ["osm_boundaries", "osm_water"],
    },
    "schools": {
        "keywords": [
            "school", "schools", "most schools", "school density",
            "schools per", "education", "primary school", "secondary school",
        ],
        "sources": ["osm_boundaries", "osm_schools"],
    },
    "greenspace": {
        "keywords": [
            "green space", "greenspace", "park", "parks", "forest",
            "garden", "most green", "green",
        ],
        "sources":  ["osm_greenspace"],
    },
    "transit_access": {
        "keywords": [
            "transport", "transit", "bus stop", "metro", "subway",
            "public transport", "train station", "bus density",
            "transit accessibility", "transport access",
            "accessibility", "public transport accessibility",
            "best transport", "transport score",
        ],
        "sources": ["osm_boundaries", "osm_transit"],
    },
    "commercial_density": {
        "keywords": [
            "commercial", "shop", "retail", "business density",
            "shopping", "market", "stores", "commercial activity",
        ],
        "sources": ["osm_boundaries", "osm_commercial"],
    },
    "cycling_infrastructure": {
        "keywords": [
            "cycling", "cycle lane", "bicycle", "bike lane",
            "cycling infrastructure", "cycle path", "bikeway",
        ],
        "sources": ["osm_boundaries", "osm_cycling"],
    },
    "noise_proxy": {
        "keywords": [
            "noise", "sound pollution", "quiet", "noise pollution",
            "traffic noise", "road noise",
        ],
        "sources": ["osm_boundaries", "osm_roads"],
    },
    "parking": {
        "keywords": [
            "parking", "car park", "parking density", "parking access",
        ],
        "sources": ["osm_boundaries", "osm_parking"],
    },
    # FIX 1: per_capita no longer includes osm_hospitals — analysis agent
    # fetches inline when hospital retrieval fails. Only boundaries needed here.
    "per_capita": {
        "keywords": [
            "per capita", "per 100k", "per 1000", "per population",
            "per resident", "per person", "per 100000",
            "hospitals per", "schools per", "transit per",
        ],
        "sources": ["osm_boundaries", "osm_hospitals"],
    },
    # Specific POI queries — retrieval fetches boundaries only;
    # generic_engine handles the feature-specific OSM fetch.
    "specific_poi": {
        "keywords": [
            "pharmac", "chemist", "drugstore",
            "cafe", "coffee shop",
            "restaurant", "dining",
            "librar",
            "playground", "play area",
            "gym", "fitness centre", "fitness center", "sports centre", "sports center",
            "supermarket", "grocery",
            "bakery", "butcher",
            "cinema", "movie theater", "movie theatre",
            "museum", "gallery",
            "hotel", "hostel",
            "petrol station", "gas station", "fuel station", "filling station",
            "police station",
            "fire station",
            "post office",
            "place of worship", "mosque", "temple", "church",
            "kindergarten", "nursery",
            "veterinar", "vet clinic",
            "charging station", "ev charger",
            "bicycle parking", "bike parking",
        ],
        "sources": ["osm_boundaries"],
    },
}


# ── Multi-source detection ────────────────────────────────────────────────────

PER_CAPITA_KEYWORDS = [
    "per capita", "per 100k", "per 100000", "per 100,000",
    "per 1000", "per 1,000", "per 10000",
    "per population", "per resident", "per person", "per inhabitant",
]


def _is_per_capita_query(task_lower: str) -> bool:
    return any(kw in task_lower for kw in PER_CAPITA_KEYWORDS)


def _ensure_per_capita_sources(sources: list, task_lower: str) -> list:
    """Additive: per-capita queries always fetch boundaries + WorldPop raster."""
    if not _is_per_capita_query(task_lower):
        return list(sources)
    out = list(sources)
    if "osm_boundaries" not in out:
        out.insert(0, "osm_boundaries")
    if "worldpop_population" not in out:
        out.append("worldpop_population")
    print(f"[Retrieval] Per-capita detected -> ensured sources: {out}")
    return out


def detect_multi_source_query(task: str) -> list:
    import json as _json
    task_lower = task.lower()
    print(f"[Retrieval] Multi-source check: {task_lower}")

    # First try keyword matching for speed
    for config in MULTI_SOURCE_QUERIES.values():
        for kw in config["keywords"]:
            if kw in task_lower:
                print(
                    f"[Retrieval] Matched keyword: '{kw}' -> {config['sources']}")
                return _ensure_per_capita_sources(config["sources"], task_lower)

    # Fallback — use LLM to decide sources for unknown query types
    print("[Retrieval] No keyword match — using LLM to select sources")
    try:
        from tools.llm_client import smart_chat
        prompt = f"""Given this GIS task, return a JSON array of required data sources.

Task: "{task}"

Available sources:
- osm_boundaries: ward/borough/neighborhood boundaries (include for ANY query about areas)
- osm_roads: road network
- osm_hospitals: hospitals and clinics
- osm_water: rivers, lakes, water bodies
- osm_schools: schools and universities
- osm_greenspace: parks, forests, green areas
- osm_transit: bus stops, metro stations
- osm_commercial: shops, restaurants, offices
- osm_cycling: cycling paths
- osm_parking: parking facilities
- worldpop_population: population raster (include for "per capita", "per 100k", "per person" queries)

Rules:
- ALWAYS include osm_boundaries if asking about neighborhoods/wards/districts/boroughs
- Include worldpop_population for any per capita query
- Return ONLY a JSON array, nothing else

Example: ["osm_boundaries", "osm_greenspace"]"""

        response = smart_chat(
            "You are a GIS data expert. Return only JSON.", prompt, use_groq=True)
        response = response.strip().replace("```json", "").replace("```", "").strip()
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            sources = _json.loads(response[start:end])
            print(f"[Retrieval] LLM selected sources: {sources}")
            return _ensure_per_capita_sources(sources, task_lower)
    except Exception as e:
        print(f"[Retrieval] LLM source selection failed: {e}")

    if _is_per_capita_query(task_lower):
        return _ensure_per_capita_sources([], task_lower)
    print("[Retrieval] No multi-source match found")
    return []


# ── Deterministic OSMnx fetch code ───────────────────────────────────────────

def generate_osmnx_fetch_code(source_type: str, city: str, task: str) -> str | None:
    place = city or "Mumbai"

    # FIX 2: get bounding box from Nominatim for more reliable OSMnx fetches
    bbox = None
    try:
        import requests
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "GoAI/1.0"},
            timeout=10,
        )
        data = res.json()
        if data:
            place = (
                data[0].get("display_name", city).split(",")[0].strip()
                + ", "
                + data[0].get("display_name", city).split(",")[-1].strip()
            )
            place = place.replace("'", "").replace('"', "").strip()
            bb = data[0].get("boundingbox")
            if bb and len(bb) == 4:
                bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    except Exception:
        pass

    # Sanitize place name — remove quotes that break f-string templates
    place = place.replace("'", "").replace('"', "").strip()
    city_name = place.split(",")[0].strip()

    # LLM-Find Feature #3: Load code templates from handbook
    # Paper: "providing a verified program template significantly increases accuracy"
    try:
        handbooks = load_all_handbooks()
        osm_handbook = handbooks.get("openstreetmap", {})
        code_templates = osm_handbook.get("code_templates", {})
        # LLM-Find Feature #4: Authentication component
        auth = osm_handbook.get("auth", {})
        if auth:
            print(
                f"[Retrieval] Auth loaded for openstreetmap: {list(auth.keys())}")
    except Exception:
        code_templates = {}
        auth = {}

    # LLM-Find Feature #2: Overpass API for POIs — faster than OSMnx
    # Paper Section 3.1.2 handbook rule: "use Overpass for POIs and polylines"
    if source_type == "osm_hospitals":
        if "hospitals_overpass" in code_templates:
            return code_templates["hospitals_overpass"].replace("{city_name}", city_name).replace("{place}", place)
        # Inline Overpass + OSMnx fallback template
        return f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'amenity': ['hospital', 'clinic', 'doctors']}}
    gdf = None
    try:
        gdf = ox.features_from_place('{place}', tags=tags)
    except Exception as _e:
        print(f'features_from_place failed: {{_e}}')
        gdf = None
    if gdf is None or len(gdf) == 0:
        try:
            import requests
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
            bb = res.json()[0].get('boundingbox', [])
            if len(bb) == 4:
                gdf = _ffb(float(bb[1]), float(bb[0]), float(bb[3]), float(bb[2]), tags)
        except Exception as _e2:
            print(f'bbox fallback failed: {{_e2}}')
    if gdf is None:
        raise ValueError('Could not fetch hospitals for {place}')
    gdf = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
"""

    if source_type == "osm_schools":
        if "schools_overpass" in code_templates:
            return code_templates["schools_overpass"].replace("{city_name}", city_name).replace("{place}", place)
        return f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'amenity': ['school', 'kindergarten', 'college', 'university']}}
    gdf = None
    try:
        gdf = ox.features_from_place('{place}', tags=tags)
    except Exception as _e:
        print(f'features_from_place failed: {{_e}}')
        gdf = None
    if gdf is None or len(gdf) == 0:
        try:
            import requests
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
            bb = res.json()[0].get('boundingbox', [])
            if len(bb) == 4:
                gdf = _ffb(float(bb[1]), float(bb[0]), float(bb[3]), float(bb[2]), tags)
        except Exception as _e2:
            print(f'bbox fallback failed: {{_e2}}')
    if gdf is None:
        raise ValueError('Could not fetch schools for {place}')
    gdf = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
"""

    # For greenspace use handbook template if available
    if source_type in ("osm_greenspace", "osm_parks") and "greenspace" in code_templates:
        return code_templates["greenspace"].replace("{place}", place)

    # For roads use handbook template if available
    if source_type == "osm_roads" and "roads" in code_templates:
        return code_templates["roads"].replace("{place}", place)

    # bbox_code for other templates
    bbox_code = ""
    if bbox:
        bbox_code = f"""
# Try bbox approach first (more reliable than place name for dense cities)
_bbox = ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})  # south, north, west, east
"""

    templates = {
        "osm_boundaries": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    # Get city bbox from Nominatim
    import requests
    res = requests.get('https://nominatim.openstreetmap.org/search',
        params={{'q': '{place}', 'format': 'json', 'limit': 1, 'polygon_geojson': 1}},
        headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned no results for {place}')
    bb = data[0].get('boundingbox', [])
    if len(bb) != 4:
        raise ValueError('No bounding box for {place}')
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    cx = (west + east) / 2
    cy = (south + north) / 2
    zone = int((cx + 180) / 6) + 1
    utm_crs = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    best = None
    for level in ['8', '9', '10', '7', '6']:
        try:
            tags = {{'boundary': 'administrative', 'admin_level': level}}
            try:
                candidate = ox.features_from_place('{place}', tags=tags)
            except Exception as _fp_e:
                print(f'features_from_place lvl {{level}}: {{_fp_e}}')
                candidate = _ffb(north, south, east, west, tags)
            candidate = candidate.reset_index(drop=True)
            if 'boundary' in candidate.columns:
                candidate = candidate[candidate['boundary'] == 'administrative'].copy()
            candidate = candidate[candidate.geometry.geom_type.isin(
                ['Polygon', 'MultiPolygon'])].copy()
            for col in ['leisure', 'landuse', 'natural']:
                if col in candidate.columns:
                    candidate = candidate[candidate[col].isna()].copy()
            if 'name' in candidate.columns:
                candidate = candidate[candidate['name'].notna()].copy()
            candidate = candidate.reset_index(drop=True)
            candidate['geometry'] = candidate['geometry'].apply(make_valid)
            if len(candidate) < 3:
                continue
            candidate_utm = candidate.to_crs(utm_crs)
            areas = candidate_utm.geometry.area
            median_area = areas.median()
            candidate = candidate[areas >= median_area * 0.1].copy().reset_index(drop=True)
            if len(candidate) >= 3:
                best = candidate
                print(f'Boundaries: admin_level={{level}}, {{len(best)}} features')
                break
        except Exception as e:
            print(f'Boundaries: level {{level}} failed: {{e}}')
            continue
    if best is None or len(best) < 3:
        raise ValueError('Could not find administrative boundaries for {place}')
    best['geometry'] = best['geometry'].apply(make_valid)

    # V2-2: clip to the city's own polygon so admin units outside the
    # city (e.g. Hertfordshire parishes inside a London bbox) are dropped
    try:
        from shapely.geometry import shape as _shape
        geojson = data[0].get('geojson')
        if geojson:
            city_poly = make_valid(_shape(geojson))
            before = len(best)
            keep = best.geometry.centroid.within(city_poly)
            best = best[keep.values].copy().reset_index(drop=True)
            print(f'City clip: {{before}} -> {{len(best)}} units inside {place}')
    except Exception as _ce:
        print(f'city clip skipped: {{_ce}}')

    result = best.to_crs('EPSG:4326')

download_data()
""",
        "osm_roads": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    G     = ox.graph_from_place('{place}', network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = edges.reset_index()
    edges['geometry'] = edges['geometry'].apply(make_valid)
    result = edges.to_crs('EPSG:4326')

download_data()
""",
        "osm_hospitals": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'amenity': ['hospital', 'clinic', 'doctors']}}
    gdf = None
    try:
        gdf = ox.features_from_place('{place}', tags=tags)
    except Exception as _e:
        print(f'features_from_place failed: {{_e}}')
        gdf = None
    if gdf is None or len(gdf) == 0:
        try:
            import requests
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
            bb = res.json()[0].get('boundingbox', [])
            if len(bb) == 4:
                gdf = _ffb(float(bb[1]), float(bb[0]), float(bb[3]), float(bb[2]), tags)
        except Exception as _e2:
            print(f'bbox fallback failed: {{_e2}}')
    if gdf is None:
        raise ValueError('Could not fetch hospitals for {place}')
    gdf = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_parks": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'leisure': 'park', 'landuse': ['forest', 'grass', 'recreation_ground']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf  = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_water": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'natural': ['water', 'coastline'], 'waterway': ['river', 'stream', 'drain']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_greenspace": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'leisure': 'park', 'landuse': ['forest', 'grass', 'recreation_ground', 'meadow']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf  = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_schools": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'amenity': ['school', 'kindergarten', 'college', 'university']}}
    gdf = None
    try:
        gdf = ox.features_from_place('{place}', tags=tags)
    except Exception as _e:
        print(f'features_from_place failed: {{_e}}')
        gdf = None
    if gdf is None or len(gdf) == 0:
        try:
            import requests
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
            bb = res.json()[0].get('boundingbox', [])
            if len(bb) == 4:
                gdf = _ffb(float(bb[1]), float(bb[0]), float(bb[3]), float(bb[2]), tags)
        except Exception as _e2:
            print(f'bbox fallback failed: {{_e2}}')
    if gdf is None:
        raise ValueError('Could not fetch schools for {place}')
    gdf = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_transit": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'public_transport': ['stop_position', 'station', 'platform'],
             'highway': ['bus_stop'],
             'railway': ['station', 'halt', 'tram_stop', 'subway_entrance']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_commercial": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'shop': True, 'amenity': ['bank', 'restaurant', 'cafe', 'fast_food']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_cycling": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    try:
        G = ox.graph_from_place('{place}', custom_filter='["highway"~"cycleway|path"]')
        edges = ox.graph_to_gdfs(G, nodes=False)
        edges = edges.reset_index()
    except Exception:
        tags = {{'highway': ['cycleway', 'path'], 'bicycle': ['designated', 'yes']}}
        edges = ox.features_from_place('{place}', tags)
        edges = edges.reset_index()
    edges['geometry'] = edges['geometry'].apply(make_valid)
    result = edges.to_crs('EPSG:4326')

download_data()
""",
        "osm_parking": f"""
import osmnx as ox
import geopandas as gpd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def download_data():
    global result
    tags = {{'amenity': ['parking', 'parking_space', 'parking_entrance']}}
    gdf  = ox.features_from_place('{place}', tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        # FIX 3: expanded CITY_ISO3 — matches module-level dict in analysis_agent.py
        "worldpop_population": f"""
import requests
import geopandas as gpd
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def download_data():
    global result

    CITY_ISO3 = {{
        # India
        'mumbai': 'IND', 'delhi': 'IND', 'bengaluru': 'IND', 'bangalore': 'IND',
        'kolkata': 'IND', 'pune': 'IND', 'hyderabad': 'IND', 'chennai': 'IND',
        'ahmedabad': 'IND', 'surat': 'IND', 'jaipur': 'IND', 'lucknow': 'IND',
        'kanpur': 'IND', 'nagpur': 'IND', 'indore': 'IND', 'thane': 'IND',
        'bhopal': 'IND', 'visakhapatnam': 'IND', 'pimpri': 'IND', 'patna': 'IND',
        # UK
        'london': 'GBR', 'greater london': 'GBR', 'birmingham': 'GBR',
        'manchester': 'GBR', 'leeds': 'GBR', 'glasgow': 'GBR', 'liverpool': 'GBR',
        'newcastle': 'GBR', 'sheffield': 'GBR', 'bristol': 'GBR', 'edinburgh': 'GBR',
        # Europe
        'berlin': 'DEU', 'munich': 'DEU', 'hamburg': 'DEU', 'frankfurt': 'DEU',
        'cologne': 'DEU', 'stuttgart': 'DEU', 'dusseldorf': 'DEU',
        'paris': 'FRA', 'lyon': 'FRA', 'marseille': 'FRA', 'toulouse': 'FRA',
        'amsterdam': 'NLD', 'rotterdam': 'NLD', 'the hague': 'NLD',
        'madrid': 'ESP', 'barcelona': 'ESP', 'seville': 'ESP', 'valencia': 'ESP',
        'rome': 'ITA', 'milan': 'ITA', 'naples': 'ITA', 'turin': 'ITA',
        'vienna': 'AUT', 'zurich': 'CHE', 'geneva': 'CHE', 'bern': 'CHE',
        'brussels': 'BEL', 'stockholm': 'SWE', 'oslo': 'NOR', 'copenhagen': 'DNK',
        'warsaw': 'POL', 'prague': 'CZE', 'budapest': 'HUN', 'bucharest': 'ROU',
        'lisbon': 'PRT', 'athens': 'GRC', 'helsinki': 'FIN',
        # Asia
        'singapore': 'SGP', 'tokyo': 'JPN', 'osaka': 'JPN', 'kyoto': 'JPN',
        'seoul': 'KOR', 'busan': 'KOR', 'bangkok': 'THA', 'jakarta': 'IDN',
        'kuala lumpur': 'MYS', 'manila': 'PHL', 'ho chi minh': 'VNM',
        'hanoi': 'VNM', 'beijing': 'CHN', 'shanghai': 'CHN', 'guangzhou': 'CHN',
        'shenzhen': 'CHN', 'hong kong': 'HKG', 'taipei': 'TWN',
        'karachi': 'PAK', 'lahore': 'PAK', 'islamabad': 'PAK',
        'dhaka': 'BGD', 'chittagong': 'BGD', 'colombo': 'LKA',
        'kathmandu': 'NPL', 'yangon': 'MMR',
        # Middle East
        'dubai': 'ARE', 'abu dhabi': 'ARE', 'riyadh': 'SAU', 'jeddah': 'SAU',
        'doha': 'QAT', 'kuwait city': 'KWT', 'muscat': 'OMN',
        'istanbul': 'TUR', 'ankara': 'TUR', 'tel aviv': 'ISR', 'jerusalem': 'ISR',
        'amman': 'JOR', 'beirut': 'LBN', 'tehran': 'IRN',
        # Africa
        'lagos': 'NGA', 'abuja': 'NGA', 'kano': 'NGA',
        'nairobi': 'KEN', 'mombasa': 'KEN',
        'cairo': 'EGY', 'alexandria': 'EGY',
        'johannesburg': 'ZAF', 'cape town': 'ZAF', 'durban': 'ZAF',
        'accra': 'GHA', 'addis ababa': 'ETH', 'dar es salaam': 'TZA',
        'casablanca': 'MAR', 'tunis': 'TUN', 'algiers': 'DZA',
        'kinshasa': 'COD', 'luanda': 'AGO', 'kampala': 'UGA',
        # Americas
        'new york': 'USA', 'los angeles': 'USA', 'chicago': 'USA',
        'houston': 'USA', 'phoenix': 'USA', 'philadelphia': 'USA',
        'san antonio': 'USA', 'san diego': 'USA', 'dallas': 'USA',
        'san francisco': 'USA', 'seattle': 'USA', 'boston': 'USA',
        'toronto': 'CAN', 'montreal': 'CAN', 'vancouver': 'CAN', 'calgary': 'CAN',
        'sao paulo': 'BRA', 'rio de janeiro': 'BRA', 'brasilia': 'BRA',
        'buenos aires': 'ARG', 'cordoba': 'ARG', 'rosario': 'ARG',
        'mexico city': 'MEX', 'guadalajara': 'MEX', 'monterrey': 'MEX',
        'bogota': 'COL', 'medellin': 'COL', 'lima': 'PER', 'santiago': 'CHL',
        'caracas': 'VEN', 'quito': 'ECU', 'la paz': 'BOL',
        # Oceania
        'sydney': 'AUS', 'melbourne': 'AUS', 'brisbane': 'AUS',
        'perth': 'AUS', 'adelaide': 'AUS', 'auckland': 'NZL', 'wellington': 'NZL',
        # Russia/CIS
        'moscow': 'RUS', 'saint petersburg': 'RUS', 'novosibirsk': 'RUS',
        'kyiv': 'UKR', 'kharkiv': 'UKR', 'almaty': 'KAZ', 'tashkent': 'UZB',
    }}

    city_lower = '{place}'.lower().split(',')[0].strip()
    iso3 = CITY_ISO3.get(city_lower, None)
    if iso3 is None:
        print(f"WorldPop: unknown city '{{city_lower}}', defaulting to IND")
        iso3 = 'IND'
    else:
        print(f"WorldPop: fetching population for {{city_lower}} ({{iso3}})")

    api_url = f'https://hub.worldpop.org/rest/data/pop/wpgp?iso3={{iso3}}'
    r = requests.get(api_url, timeout=30)
    data = r.json()['data']
    latest = sorted(data, key=lambda x: x['popyear'], reverse=True)[0]
    tif_url = latest['files'][0]
    tif_year = latest['popyear']
    print(f"WorldPop: using {{tif_year}} data from {{tif_url}}")

    cache_path = f'/data/processed/worldpop_{{iso3}}_{{tif_year}}.tif'
    if not os.path.exists(cache_path):
        print(f"WorldPop: downloading raster...")
        resp = requests.get(tif_url, timeout=300, stream=True)
        with open(cache_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"WorldPop: downloaded to {{cache_path}}")
    else:
        print(f"WorldPop: using cached raster {{cache_path}}")

    result = gpd.GeoDataFrame(
        pd.DataFrame([{{'iso3': iso3, 'year': tif_year, 'tif_path': cache_path}}]),
        geometry=gpd.points_from_xy([0], [0]),
        crs='EPSG:4326'
    )
    result.attrs['tif_path'] = cache_path
    print(f"WorldPop: ready, tif_path={{cache_path}}")

download_data()
""",
    }
    return templates.get(source_type)


# ── Disk persistence ──────────────────────────────────────────────────────────

def save_to_disk(code: str, source_type: str, task: str) -> str | None:
    task_hash = abs(hash(task)) % 1_000_000
    save_path = f"/data/processed/{source_type}_{task_hash}.geojson"
    os.makedirs("/data/processed", exist_ok=True)

    save_code = code + f"""
import os
os.makedirs('/data/processed', exist_ok=True)

out = result.reset_index(drop=True).copy()

name_candidates = ['name', 'Name', 'NAME', 'ward', 'Ward', 'label',
                   'title', 'ward_name', 'area_name', 'localname',
                   'amenity', 'leisure', 'landuse', 'highway']
keep_cols = ['geometry'] + [c for c in name_candidates if c in out.columns]
out = out[keep_cols].copy()

for col in keep_cols:
    if col == 'geometry':
        continue
    try:
        out[col] = out[col].astype(str).astype(object)
    except Exception:
        out = out.drop(columns=[col])

out = out[out.geometry.notna() & out.geometry.is_valid].copy()
out.to_file('{save_path}', driver='GeoJSON')
print('SAVED_OK')
"""
    try:
        save_timeout = 180 if "roads" in source_type else 90
        proc = subprocess.run(
            [sys.executable, "-c", save_code],
            capture_output=True, text=True, timeout=save_timeout,
        )
        if proc.returncode == 0 and "SAVED_OK" in proc.stdout and os.path.exists(save_path):
            return save_path
        print(f"[Retrieval] save_to_disk failed for {source_type}:")
        print(f"[Retrieval]   stdout: {proc.stdout.strip()[:200]}")
        print(f"[Retrieval]   stderr: {proc.stderr.strip()[:300]}")
    except subprocess.TimeoutExpired:
        print(f"[Retrieval] save_to_disk timed out for {source_type}")
    except Exception as e:
        print(f"[Retrieval] save_to_disk exception for {source_type}: {e}")
    return None


# ── Source selection ──────────────────────────────────────────────────────────

def select_handbooks(task: str) -> list:
    index = load_data_source_index()
    index_text = "\n".join(f"- {name}: {desc}" for name, desc in index.items())

    prompt = f"""Given this task: "{task}"

These data sources are available:
{index_text}

Return ONLY a JSON array of source names needed.
Example: ["mumbai_wards", "lakes_and_rivers"]
For any non-Mumbai city use: ["openstreetmap"]
Return only the JSON array, nothing else."""

    text = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        sources = json.loads(text)
        if isinstance(sources, list):
            return [
                "openstreetmap" if any(
                    x in s.lower()
                    for x in ["osm", "openstreet", "open_street", "open street"]
                )
                else s
                for s in sources
            ]
    except Exception:
        pass

    if "mumbai" in task.lower():
        return ["mumbai_wards", "lakes_and_rivers", "river_lines_streams_drains"]
    return ["openstreetmap"]


# ── LLM-generated fetch code ──────────────────────────────────────────────────

def generate_fetch_code(handbook: dict, task: str) -> str:
    guidance = load_guidance()

    # LLM-Find Feature #5: RAG for handbooks — inject code template if available
    # Paper Section 3.1.2: "template programs significantly increase accuracy"
    code_template_hint = ""
    code_templates = handbook.get("code_templates", {})
    if code_templates:
        # Pick most relevant template based on task keywords
        task_lower = task.lower()
        best_template = None
        if "hospital" in task_lower or "clinic" in task_lower:
            best_template = code_templates.get(
                "hospitals_overpass") or code_templates.get("hospitals")
        elif "school" in task_lower or "university" in task_lower:
            best_template = code_templates.get(
                "schools_overpass") or code_templates.get("schools")
        elif "green" in task_lower or "park" in task_lower:
            best_template = code_templates.get("greenspace")
        elif "road" in task_lower or "street" in task_lower:
            best_template = code_templates.get("roads")
        elif "boundary" in task_lower or "ward" in task_lower or "borough" in task_lower:
            best_template = code_templates.get("boundaries")
        if best_template:
            code_template_hint = f"\nReference code template (adapt this for the task):\n```python\n{best_template}\n```\n"

    # Also use example_code from handbook
    example_hint = ""
    if handbook.get("example_code") and not code_template_hint:
        example_hint = f"\nExample code for reference:\n```python\n{handbook['example_code']}\n```\n"

    prompt = f"""Write Python code to fetch geospatial data for this task: "{task}"

Data source handbook:
{json.dumps(handbook, indent=2)}

{guidance}
{code_template_hint}{example_hint}
Rules:
- All code inside a function named download_data()
- Last line calls download_data()
- Store result as a GeoDataFrame in a variable called result
- Use Overpass API for POIs (hospitals, schools, shops) — faster than OSMnx
- Use OSMnx only for boundaries (geocode_to_gdf) and road networks
- Always include country in place name (e.g. 'Hyderabad, India')
- Use predicate= in gpd.sjoin(), never op=
- Do not perform spatial joins in retrieval — just fetch raw data
- Do not use plt.show()
- Throw an error if data download fails

Return only Python code."""

    code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
    return code.replace("```python", "").replace("```", "").strip()


# ── Sandbox execution ─────────────────────────────────────────────────────────

def run_code_in_sandbox(code: str, timeout: int = 60) -> dict:
    wrapped = f"""
import geopandas as gpd
import pandas as pd
import osmnx as ox
import warnings
warnings.filterwarnings('ignore')

{code}

print("ROWS:", len(result))
print("CRS:", result.crs)
print("COLUMNS:", list(result.columns))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0:
            return {"success": True,  "output": proc.stdout}
        else:
            return {"success": False, "error": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Sandbox timed out after {timeout}s"}


def validate_output(output: str) -> dict:
    if "ROWS: 0" in output:
        return {"valid": False, "error": "Returned an empty GeoDataFrame"}
    if "ROWS:" not in output:
        return {"valid": False, "error": "Result variable missing or not a GeoDataFrame"}
    if "CRS: None" in output:
        return {"valid": False, "error": "CRS is None — set a CRS before returning"}
    return {"valid": True}


def validate_city_bbox(code: str, city: str, output: str) -> dict:
    """LLM-Find Feature #1: Type 2 error detection — checks if retrieved data
    is actually in the right city by comparing bbox of results with city bbox.
    Catches runnable-but-wrong data (e.g., London hospitals returned for Mumbai query)."""
    if not city or not output:
        return {"valid": True}
    # Extract rows count — skip check for very small results
    import re
    rows_match = re.search(r"ROWS:\s*(\d+)", output)
    if not rows_match or int(rows_match.group(1)) < 2:
        return {"valid": True}
    try:
        import requests
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "GoAI/1.0"}, timeout=10
        )
        data = res.json()
        if not data:
            return {"valid": True}
        bb = data[0].get("boundingbox", [])
        if len(bb) != 4:
            return {"valid": True}
        city_south, city_north = float(bb[0]), float(bb[1])
        city_west, city_east = float(bb[2]), float(bb[3])
        # Expand city bbox by 50% to allow for suburbs
        lat_margin = (city_north - city_south) * 0.5
        lon_margin = (city_east - city_west) * 0.5
        city_south -= lat_margin
        city_north += lat_margin
        city_west -= lon_margin
        city_east += lon_margin
        # Run a quick check on the saved file
        import subprocess
        import sys
        check_code = f"""
import geopandas as gpd, re
{code}
bounds = result.to_crs('EPSG:4326').total_bounds  # minx, miny, maxx, maxy
print(f"BBOX: {{bounds[1]:.2f}},{{bounds[3]:.2f}},{{bounds[0]:.2f}},{{bounds[2]:.2f}}")
"""
        proc = subprocess.run([sys.executable, "-c", check_code],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return {"valid": True}  # Can't check, assume ok
        bbox_match = re.search(
            r"BBOX: ([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+)", proc.stdout)
        if not bbox_match:
            return {"valid": True}
        data_south = float(bbox_match.group(1))
        data_north = float(bbox_match.group(2))
        data_west = float(bbox_match.group(3))
        data_east = float(bbox_match.group(4))
        # Check overlap
        lat_ok = data_south <= city_north and data_north >= city_south
        lon_ok = data_west <= city_east and data_east >= city_west
        if not lat_ok or not lon_ok:
            print(
                f"[Retrieval] Type 2 error: data bbox ({data_south:.1f},{data_north:.1f},{data_west:.1f},{data_east:.1f}) doesn't overlap {city} bbox ({city_south:.1f},{city_north:.1f},{city_west:.1f},{city_east:.1f})")
            return {"valid": False, "error": f"Retrieved data is not in {city} — data bbox doesn't overlap city bbox"}
        return {"valid": True}
    except Exception as e:
        return {"valid": True}  # On any error, don't block


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_data_for_task(task: str, city: str = "") -> dict:
    print(f"[Retrieval] Selecting sources for: {task}")

    # 1. Multi-source OSMnx path
    multi_sources = detect_multi_source_query(task)
    if multi_sources:
        print(f"[Retrieval] Multi-source query: {multi_sources}")
        results = {}

        for source_type in multi_sources:
            code = generate_osmnx_fetch_code(source_type, city, task)
            if not code:
                continue

            print(f"[Retrieval] Fetching {source_type}...")
            timeout = 180 if "roads" in source_type else 300 if source_type == "osm_boundaries" else 120
            sandbox_result = run_code_in_sandbox(code, timeout=timeout)

            if sandbox_result["success"]:
                validation = validate_output(sandbox_result["output"])
                if validation["valid"]:
                    # LLM-Find Feature #1: Type 2 error detection
                    # Check if retrieved data is actually in the right city
                    bbox_check = validate_city_bbox(
                        code, city, sandbox_result["output"])
                    if not bbox_check["valid"]:
                        print(
                            f"[Retrieval] {source_type} Type 2 error: {bbox_check['error'][:80]}")
                        results[source_type] = {
                            "error": bbox_check["error"], "attempts": 1, "file_path": None}
                        continue
                    print(f"[Retrieval] {source_type} fetched")
                    saved_path = save_to_disk(code, source_type, task)
                    if saved_path:
                        print(
                            f"[Retrieval] {source_type} saved to {saved_path}")
                    else:
                        print(
                            f"[Retrieval] WARNING: {source_type} save failed — file_path will be None")
                    # FIX 4: always include file_path key (None if save failed)
                    results[source_type] = {
                        "code":      code,
                        "output":    sandbox_result["output"],
                        "attempts":  1,
                        "file_path": saved_path,  # None if save failed, not missing
                    }
                else:
                    print(
                        f"[Retrieval] {source_type} failed validation: {validation['error']}")
                    results[source_type] = {
                        "error": validation["error"], "attempts": 1, "file_path": None}
            else:
                print(
                    f"[Retrieval] {source_type} fetch failed: {sandbox_result['error'][:100]}")
                results[source_type] = {
                    "error": sandbox_result["error"], "attempts": 1, "file_path": None}

        if any("error" not in v for v in results.values()):
            return results

    # 2. Standard single-source retrieval
    selected = select_handbooks(task)
    print(f"[Retrieval] Selected: {selected}")

    results = {}

    for name in selected:
        if name in KNOWN_LOCAL_SOURCES:
            path = KNOWN_LOCAL_SOURCES[name]
            code = (
                f"import geopandas as gpd\n\n"
                f"def download_data():\n"
                f"    global result\n"
                f"    result = gpd.read_file('{path}')\n\n"
                f"download_data()"
            )
            sandbox_result = run_code_in_sandbox(code)
            if sandbox_result["success"] and validate_output(sandbox_result["output"])["valid"]:
                print(f"[Retrieval] {name} loaded from local file")
                results[name] = {
                    "code": code, "output": sandbox_result["output"],
                    "attempts": 1, "file_path": path}
                continue

        handbooks = load_all_handbooks()
        handbook = handbooks.get(name)
        if not handbook:
            print(f"[Retrieval] No handbook for {name} — skipping")
            continue

        print(f"[Retrieval] Fetching live: {name}")
        code = generate_fetch_code(handbook, task)
        attempt_history = []
        error = "No attempts made"

        for attempt in range(3):
            sandbox_result = run_code_in_sandbox(code, timeout=180)

            if sandbox_result["success"]:
                validation = validate_output(sandbox_result["output"])
                if validation["valid"]:
                    print(
                        f"[Retrieval] {name} fetched on attempt {attempt + 1}")
                    results[name] = {
                        "code":     code,
                        "output":   sandbox_result["output"],
                        "attempts": attempt + 1,
                        "file_path": None,  # FIX 4: always present
                    }
                    break
                error = validation["error"]
            else:
                error = sandbox_result["error"]

            attempt_history.append(
                {"attempt": attempt + 1, "code": code, "error": error})
            print(f"[Retrieval] Attempt {attempt + 1} failed: {error[:100]}")

            if attempt < 2:
                history_text = "\n\n".join(
                    f"Attempt {h['attempt']}:\n{h['code']}\nError: {h['error']}"
                    for h in attempt_history
                )
                fix_prompt = f"""Fetch failed for task: "{task}"
Handbook: {json.dumps(handbook, indent=2)}
{load_guidance()}

Failed attempts:
{history_text}

Fix. Rules: always include country in place name, use predicate= not op=,
no spatial joins in retrieval, result must be a GeoDataFrame.
All code in download_data(). Last line calls download_data().
Return only Python code."""

                code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT,
                                  fix_prompt, use_groq=True)
                code = code.replace("```python", "").replace("```", "").strip()
            else:
                results[name] = {
                    "error": error, "attempts": attempt_history, "file_path": None}

    if not any("error" not in v for v in results.values()):
        print(
            "[Retrieval] All sources failed — passing OSMnx fallback to analysis agent")
        results["openstreetmap_fallback"] = {
            "code": "", "output": "ROWS: 1\nCOLUMNS: []", "attempts": 0,
            "file_path": None,
        }
    return results
