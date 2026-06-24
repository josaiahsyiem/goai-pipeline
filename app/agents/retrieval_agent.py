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

import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'

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
        "keywords": [
            "flood risk", "flood zone", "flood exposure",
            "flood", "flooding", "flood prone", "flood-prone",
            "inundation", "waterlogging", "waterlogged",
        ],
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
            "public transport accessibility",
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
    # per_capita fetches boundaries only — the analysis agent's per-capita path
    # detects the requested feature from the task and fetches it inline.
    # WorldPop is added by _ensure_per_capita_sources. Feature-specific entries
    # (hospital_proximity, schools, transit_access) match first for their keywords.
    "per_capita": {
        "keywords": [
            "per capita", "per 100k", "per 1000", "per population",
            "per resident", "per person", "per 100000",
        ],
        "sources": ["osm_boundaries"],
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
    "satellite_uhi": {
        "keywords": [
            "heat island", "urban heat", "uhi", "surface temperature",
            "land surface temperature", "lst", "thermal", "hot spot",
            "hotspot", "heat map", "heat stress",
        ],
        "sources": ["osm_boundaries", "satellite_thermal"],
    },
    "satellite_vegetation": {
        "keywords": [
            "ndvi", "vegetation index", "vegetation health",
            "greenness", "vegetation cover", "leaf area",
        ],
        "sources": ["osm_boundaries", "satellite_ndvi"],
    },
    "satellite_worldcover": {
        "keywords": [
            "land cover", "landcover", "land use", "lulc",
            "green cover", "built-up", "built up", "urban area percentage",
            "vegetation percentage", "tree cover", "bare soil",
            "impervious surface", "esa worldcover", "worldcover",
        ],
        "sources": ["osm_boundaries", "satellite_worldcover"],
    },
    # NOTE: satellite_landcover/sentinel2 removed — it fetched only a STAC
    # item_id with no raster and no analysis-side path. Reintroduce as a full
    # feature (band download + classification + analysis path) when needed.
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


# Deliberate prefix stems — match 'pharmacy', 'pharmacies', 'libraries', etc.
_STEM_KEYWORDS = {"pharmac", "librar", "veterinar"}


def _kw_match(kw: str, text: str) -> bool:
    """Word-boundary keyword match. Prevents 'park' matching 'parking',
    'garden' matching 'kindergarten', 'green' matching 'greenness',
    'metro' matching 'metropolitan'. Stems keep open-ended suffixes."""
    import re
    tail = r'' if kw in _STEM_KEYWORDS else r'(?:s|es)?\b'
    return re.search(r'\b' + re.escape(kw) + tail, text) is not None


def detect_multi_source_query(task: str) -> list:
    import json as _json
    task_lower = task.lower()
    print(f"[Retrieval] Multi-source check: {task_lower}")

    # First try keyword matching for speed.
    # Most-specific-wins: check ALL keywords, prefer more words, then longer —
    # so 'forest cover' (landcover) beats 'forest' (greenspace) and
    # 'hospitals per' (hospital_proximity) beats 'per capita' (per_capita),
    # regardless of registry order.
    candidates = []
    for config in MULTI_SOURCE_QUERIES.values():
        for kw in config["keywords"]:
            if _kw_match(kw, task_lower):
                candidates.append(
                    (kw.count(" "), len(kw), kw, config["sources"]))
    if candidates:
        candidates.sort(reverse=True)
        _, _, best_kw, best_sources = candidates[0]
        print(
            f"[Retrieval] Matched keyword: '{best_kw}' -> {best_sources}")
        return _ensure_per_capita_sources(best_sources, task_lower)

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

_PLACE_CACHE: dict = {}


def generate_osmnx_fetch_code(source_type: str, city: str, task: str) -> str | None:
    place = city or "Mumbai"
    if city in _PLACE_CACHE:
        place = _PLACE_CACHE[city]

    # Resolve a clean "City, Country" place string via Nominatim
    try:
        import requests
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1,
                    "accept-language": "en"},
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

    except Exception:
        pass

    # Sanitize place name — remove quotes that break f-string templates
    place = place.replace("'", "").replace('"', "").strip()
    if city:
        _PLACE_CACHE[city] = place
    city_name = place.split(",")[0].strip()
    # Cache slug from the USER's city string — must mirror analysis_agent's
    # fallback: plan city -> split(',')[0].strip().replace(' ', '_')
    city_slug = (city or place).replace("'", "").replace(
        '"', "").split(",")[0].strip().replace(" ", "_")

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
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
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
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
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

    templates = {
        "osm_boundaries": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
        params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en', 'polygon_geojson': 1, 'accept-language': 'en'}},
        headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned no results for {place}')
    bb = data[0].get('boundingbox', [])
    if len(bb) != 4:
        raise ValueError('No bounding box for {place}')
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    # Use Nominatim's actual lat/lon as center, not bbox center.
    # Bbox center is unreliable for cities with non-contiguous territory
    # (e.g. Tokyo includes remote Pacific islands, making bbox span half of Japan).
    # This fix generalizes to any city with outlying territories.
    cy = float(data[0].get('lat', (south + north) / 2))
    cx = float(data[0].get('lon', (west + east) / 2))
    # Cap oversized bboxes to a metro-area span around the actual city center.
    _max_span = 1.5
    if (north - south) > _max_span or (east - west) > _max_span:
        south, north = cy - _max_span / 2, cy + _max_span / 2
        west, east = cx - _max_span / 2, cx + _max_span / 2
        print(f'Bbox capped to {{_max_span}}deg around ({{cy:.3f}},{{cx:.3f}}): s={{south:.3f}} n={{north:.3f}} w={{west:.3f}} e={{east:.3f}}')
    
    zone = int((cx + 180) / 6) + 1
    utm_crs = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    best = None
    best_score = 0
    for level in ['9', '10', '8', '7', '6', '5']:
        try:
            tags = {{'boundary': 'administrative', 'admin_level': level}}
            try:
                candidate = _ffb(north, south, east, west, tags)
                if len(candidate) == 0:
                    raise ValueError('empty bbox result')
            except Exception as _fp_e:
                print(f'bbox fetch lvl {{level}} failed: {{_fp_e}}, trying features_from_place')
                try:
                    candidate = ox.features_from_place('{place}', tags=tags)
                except Exception as _fp_e2:
                    print(f'features_from_place lvl {{level}}: {{_fp_e2}}')
                    continue
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
            areas = candidate_utm.geometry.area / 1e6
            median_area = areas.median()
            candidate = candidate[(areas >= median_area * 0.1) &
                                  (areas <= median_area * 20)].copy().reset_index(drop=True)
            if len(candidate) < 3:
                continue
            areas_filtered = candidate.to_crs(utm_crs).geometry.area / 1e6
            ward_like = ((areas_filtered >= 0.1) & (areas_filtered <= 500)).sum()
            score = int(ward_like)
            print(f'Boundaries: admin_level={{level}}, {{len(candidate)}} features, score={{score}}')
            if score > 0 and len(candidate) > 500:
                keep_idx = areas_filtered[(areas_filtered >= 0.1) & (areas_filtered <= 500)].nlargest(500).index
                candidate = candidate.loc[keep_idx].reset_index(drop=True)
                score = len(candidate)
                print(f'Capped to {{len(candidate)}} ward-sized units')
            # Prefer more granular level: if same score, pick the one with more features
            if score > best_score or (score == best_score and best is not None and len(candidate) > len(best)):
                best_score = score
                best = candidate
        except Exception as e:
            print(f'Boundaries: level {{level}} failed: {{e}}')
            continue
    if best is None or len(best) < 3:
        raise ValueError('Could not find administrative boundaries for {place}')
    print(f'Best level: {{len(best)}} features, score={{best_score}}')
    best['geometry'] = best['geometry'].apply(make_valid)

    # V4: city clip with 3-tier polygon source — NEVER silently skipped.
    # Tier 1: Nominatim polygon. Tier 2: OSMnx geocode_to_gdf polygon.
    # Tier 3: bbox rectangle (coarse, but strictly better than no clip).
    city_poly = None
    try:
        from shapely.geometry import shape as _shape
        geojson = data[0].get('geojson')
        if geojson and geojson.get('type') in ('Polygon', 'MultiPolygon'):
            city_poly = make_valid(_shape(geojson))
            print('City polygon: Nominatim')
    except Exception as _ce:
        print(f'Nominatim polygon failed: {{_ce}}')
    if city_poly is None:
        try:
            _cg = ox.geocode_to_gdf('{place}')
            _g0 = make_valid(_cg.geometry.iloc[0])
            if _g0.geom_type in ('Polygon', 'MultiPolygon'):
                city_poly = _g0
                print('City polygon: OSMnx geocode_to_gdf')
        except Exception as _ce2:
            print(f'geocode_to_gdf polygon failed: {{_ce2}}')
    if city_poly is None:
        from shapely.geometry import box as _bbox_box
        city_poly = _bbox_box(west, south, east, north)
        print('City polygon: bbox fallback (coarse)')
    before = len(best)
    keep = best.geometry.centroid.within(city_poly)
    best = best[keep.values].copy().reset_index(drop=True)
    print(f'City clip: {{before}} -> {{len(best)}} units inside {place}')

    # Step 2: remove area outliers (state/region boundaries mixed in)
    best_utm = best.to_crs(utm_crs)
    best['_area_km2'] = (best_utm.geometry.area / 1e6)
    area_median = best['_area_km2'].median()
    before = len(best)
    best = best[best['_area_km2'] <= area_median * 8].copy().reset_index(drop=True)
    if len(best) < before:
        print(f'Area filter: {{before}} -> {{len(best)}} (removed {{before - len(best)}} oversized units)')

    # Step 3: remove the city-level boundary itself (e.g. "Berlin" row in Berlin query)
    if 'name' in best.columns:
        city_name_lower = '{place}'.split(',')[0].strip().lower()
        best = best[best['name'].str.lower().str.strip() != city_name_lower].copy().reset_index(drop=True)

    best = best.drop(columns=['_area_km2'], errors='ignore')

    if len(best) < 3:
        raise ValueError('Too few boundaries after filtering for {place}')

    result = best.to_crs('EPSG:4326')

download_data()
""",
        "osm_roads": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
        G = ox.graph_from_place('{place}', network_type='drive')
    except Exception as _gp_e:
        print('place geocode failed (' + str(_gp_e)[:80] + ') — bbox graph fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        if int(str(ox.__version__).split('.')[0]) >= 2:
            G = ox.graph_from_bbox((float(_bb[2]), float(_bb[0]), float(_bb[3]), float(_bb[1])), network_type='drive')
        else:
            G = ox.graph_from_bbox(north=float(_bb[1]), south=float(_bb[0]), east=float(_bb[3]), west=float(_bb[2]), network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = edges.reset_index()
    edges['geometry'] = edges['geometry'].apply(make_valid)
    result = edges.to_crs('EPSG:4326')

download_data()
""",
        "osm_hospitals": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
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
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
    gdf  = gdf.reset_index()
    gdf  = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_water": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_greenspace": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
    gdf  = gdf.reset_index()
    gdf  = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_schools": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
            res = requests.get('https://nominatim.openstreetmap.org/search', params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}}, headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
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
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_commercial": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
    gdf  = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf.to_crs('EPSG:4326')

download_data()
""",
        "osm_cycling": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
        try:
            edges = ox.features_from_place('{place}', tags)
        except Exception as _fp_e:
            print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
            import requests as _rq
            _nm = _rq.get('https://nominatim.openstreetmap.org/search',
                params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
                headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
            _bb = _nm[0]['boundingbox']
            edges = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
        edges = edges.reset_index()
    edges['geometry'] = edges['geometry'].apply(make_valid)
    result = edges.to_crs('EPSG:4326')

download_data()
""",
        "osm_parking": f"""
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
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
    try:
        gdf = ox.features_from_place('{place}', tags)
    except Exception as _fp_e:
        print('place geocode failed (' + str(_fp_e)[:80] + ') — bbox fallback')
        import requests as _rq
        _nm = _rq.get('https://nominatim.openstreetmap.org/search',
            params={{'q': '{place}', 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en'}},
            headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
        _bb = _nm[0]['boundingbox']
        gdf = _ffb(float(_bb[1]), float(_bb[0]), float(_bb[3]), float(_bb[2]), tags)
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

    _ISO2_TO_ISO3 = {{
        'in': 'IND', 'gb': 'GBR', 'de': 'DEU', 'fr': 'FRA', 'nl': 'NLD',
        'es': 'ESP', 'it': 'ITA', 'at': 'AUT', 'ch': 'CHE', 'be': 'BEL',
        'se': 'SWE', 'no': 'NOR', 'dk': 'DNK', 'pl': 'POL', 'cz': 'CZE',
        'hu': 'HUN', 'pt': 'PRT', 'gr': 'GRC', 'ie': 'IRL', 'fi': 'FIN',
        'ro': 'ROU', 'hr': 'HRV', 'sg': 'SGP', 'jp': 'JPN', 'kr': 'KOR',
        'th': 'THA', 'id': 'IDN', 'my': 'MYS', 'ph': 'PHL', 'vn': 'VNM',
        'cn': 'CHN', 'tw': 'TWN', 'hk': 'HKG', 'bd': 'BGD', 'pk': 'PAK',
        'lk': 'LKA', 'np': 'NPL', 'mm': 'MMR', 'ae': 'ARE', 'qa': 'QAT',
        'kw': 'KWT', 'om': 'OMN', 'tr': 'TUR', 'sa': 'SAU', 'eg': 'EGY',
        'ir': 'IRN', 'iq': 'IRQ', 'il': 'ISR', 'jo': 'JOR', 'lb': 'LBN',
        'ng': 'NGA', 'ke': 'KEN', 'za': 'ZAF', 'gh': 'GHA', 'tz': 'TZA',
        'et': 'ETH', 'ma': 'MAR', 'tn': 'TUN', 'dz': 'DZA', 'ug': 'UGA',
        'cd': 'COD', 'ao': 'AGO', 'us': 'USA', 'ca': 'CAN', 'br': 'BRA',
        'ar': 'ARG', 'mx': 'MEX', 'co': 'COL', 'pe': 'PER', 'cl': 'CHL',
        've': 'VEN', 'ec': 'ECU', 'bo': 'BOL', 'au': 'AUS', 'nz': 'NZL',
        'ru': 'RUS', 'ua': 'UKR', 'kz': 'KAZ', 'uz': 'UZB',
    }}
    city_lower = '{place}'.lower().split(',')[0].strip()
    iso3 = CITY_ISO3.get(city_lower, None)
    if iso3 is None:
        try:
            _nom = requests.get('https://nominatim.openstreetmap.org/search',
                params={{'q': city_lower, 'format': 'json', 'limit': 1, 'accept-language': 'en', 'accept-language': 'en', 'addressdetails': 1}},
                headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10).json()
            if _nom:
                _cc = _nom[0].get('address', {{}}).get('country_code', '').lower()
                iso3 = _ISO2_TO_ISO3.get(_cc)
                if iso3:
                    print(f"WorldPop: resolved '{{city_lower}}' -> {{iso3}} via Nominatim")
        except Exception as _re:
            print(f"WorldPop: Nominatim resolve failed: {{_re}}")
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

        "satellite_ndvi": """
import planetary_computer
import pystac_client
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings('ignore')

def download_data():
    global result
    place = '__PLACE__'
    nom_params = dict(q=place, format='json', limit=1)
    res = requests.get('https://nominatim.openstreetmap.org/search',
        params=nom_params, headers={'User-Agent': 'GoAI/1.0'}, timeout=10)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned nothing for ' + place)
    bb = data[0]['boundingbox']
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    bbox = [west, south, east, north]
    city_name = '__CITYSLUG__'
    cache_path = '/data/processed/ndvi_' + city_name + '.tif'
    os.makedirs('/data/processed', exist_ok=True)

    def _summer_window(s, n):
        from datetime import date
        lat_mid = (s + n) / 2
        today = date.today()
        if lat_mid >= 0:
            # Northern summer Jun-Sep; use last completed one
            y = today.year if today.month > 9 else today.year - 1
            return str(y) + '-06-01/' + str(y) + '-09-30'
        # Southern summer Dec-Mar; use last completed one
        y = today.year if today.month > 3 else today.year - 1
        return str(y - 1) + '-12-01/' + str(y) + '-03-31'

    if not os.path.exists(cache_path):
        catalog = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com/api/stac/v1',
            modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=['landsat-c2-l2'],
            bbox=bbox,
            datetime=_summer_window(south, north),
            query={'eo:cloud_cover': {'lt': 20},
                   'platform': {'in': ['landsat-8', 'landsat-9']}},
            max_items=10)
        items = list(search.items())
        if not items:
            raise ValueError('No Landsat scenes found for ' + place)
        def _overlap_frac(it):
            try:
                ib = it.bbox
                ix = max(0.0, min(ib[2], bbox[2]) - max(ib[0], bbox[0]))
                iy = max(0.0, min(ib[3], bbox[3]) - max(ib[1], bbox[1]))
                city_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                return (ix * iy) / city_area if city_area > 0 else 0.0
            except Exception:
                return 0.0
        items.sort(key=lambda it: (-_overlap_frac(it),
                   it.properties.get('eo:cloud_cover', 100)))
        item = items[0]
        print('Scene: ' + item.id + ', coverage=' + str(round(_overlap_frac(item), 2))
              + ', cloud=' + str(item.properties.get('eo:cloud_cover')))
        signed = planetary_computer.sign(item)
        nir_href = signed.assets['nir08'].href
        red_href = signed.assets['red'].href
        print('Downloading NIR+Red bands...')
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box as _box
        import os as _os
        dst_crs = 'EPSG:4326'
        city_geom = [_box(*bbox)]
        def _reproject_clip(href, tmp_path):
            try:
                with rasterio.open(href) as src:
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds,
                        dst_width=3000, dst_height=3000)
                    meta = src.meta.copy()
                    meta.update({'crs': dst_crs, 'transform': transform,
                                 'width': width, 'height': height, 'driver': 'GTiff', 'dtype': 'float32'})
                    with rasterio.open(tmp_path, 'w', **meta) as tmp:
                        reproject(source=rasterio.band(src, 1),
                                  destination=rasterio.band(tmp, 1),
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=dst_crs,
                                  resampling=Resampling.nearest)
                with rasterio.open(tmp_path) as tmp:
                    clipped, clipped_transform = rio_mask(tmp, city_geom, crop=True)
            finally:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
            return clipped[0].astype(float), clipped_transform
        nir, clip_transform = _reproject_clip(nir_href, cache_path + '.nir' + str(_os.getpid()) + '.tif')
        red, _red_t = _reproject_clip(red_href, cache_path + '.red' + str(_os.getpid()) + '.tif')
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), np.nan).astype(np.float32)
        _part = cache_path + '.part' + str(_os.getpid()) + '.tif'
        with rasterio.open(_part, 'w', driver='GTiff',
                           height=ndvi.shape[0], width=ndvi.shape[1],
                           count=1, dtype='float32', crs=dst_crs,
                           transform=clip_transform) as dst:
            dst.write(ndvi, 1)
        _os.replace(_part, cache_path)
        print('NDVI raster saved: ' + cache_path)
    else:
        print('NDVI raster cached: ' + cache_path)

    result = gpd.GeoDataFrame(
        pd.DataFrame([{'tif_path': cache_path, 'source': 'landsat_ndvi'}]),
        geometry=gpd.points_from_xy([0], [0]), crs='EPSG:4326')
    print('tif_path=' + cache_path)

download_data()
""".replace("__PLACE__", place).replace("__CITYSLUG__", city_slug),

        "satellite_thermal": """
import planetary_computer
import pystac_client
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings('ignore')

def download_data():
    global result
    place = '__PLACE__'
    nom_params = dict(q=place, format='json', limit=1)
    res = requests.get('https://nominatim.openstreetmap.org/search',
        params=nom_params, headers={'User-Agent': 'GoAI/1.0'}, timeout=10)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned nothing for ' + place)
    bb = data[0]['boundingbox']
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    bbox = [west, south, east, north]
    city_name = '__CITYSLUG__'
    cache_path = '/data/processed/lst_' + city_name + '.tif'
    os.makedirs('/data/processed', exist_ok=True)

    def _summer_window(s, n):
        from datetime import date
        lat_mid = (s + n) / 2
        today = date.today()
        if 0 <= lat_mid <= 30:
            # Monsoon belt (India, SE Asia, Sahel): pre-monsoon Mar-May —
            # peak land surface temps AND clear skies (Jun-Sep = clouds)
            y = today.year if today.month > 5 else today.year - 1
            return str(y) + '-03-01/' + str(y) + '-05-31'
        if lat_mid > 0:
            y = today.year if today.month > 9 else today.year - 1
            return str(y) + '-06-01/' + str(y) + '-09-30'
        y = today.year if today.month > 3 else today.year - 1
        return str(y - 1) + '-12-01/' + str(y) + '-03-31'

    if not os.path.exists(cache_path):
        catalog = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com/api/stac/v1',
            modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=['landsat-c2-l2'],
            bbox=bbox,
            datetime=_summer_window(south, north),
            query={'eo:cloud_cover': {'lt': 20},
                   'platform': {'in': ['landsat-8', 'landsat-9']}},
            max_items=10)
        items = list(search.items())
        if not items:
            raise ValueError('No Landsat scenes found for ' + place)
        def _overlap_frac(it):
            try:
                ib = it.bbox
                ix = max(0.0, min(ib[2], bbox[2]) - max(ib[0], bbox[0]))
                iy = max(0.0, min(ib[3], bbox[3]) - max(ib[1], bbox[1]))
                city_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                return (ix * iy) / city_area if city_area > 0 else 0.0
            except Exception:
                return 0.0
        items.sort(key=lambda it: (-_overlap_frac(it),
                   it.properties.get('eo:cloud_cover', 100)))
        item = items[0]
        print('Scene: ' + item.id + ', coverage=' + str(round(_overlap_frac(item), 2))
              + ', cloud=' + str(item.properties.get('eo:cloud_cover')))
        signed = planetary_computer.sign(item)
        thermal_href = signed.assets['lwir11'].href if 'lwir11' in signed.assets else signed.assets['lwir'].href
        print('Downloading thermal band...')
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box as _box
        import os as _os
        tmp_path = cache_path + '.tmp' + str(_os.getpid()) + '.tif'
        with rasterio.open(thermal_href) as src:
            dst_crs = 'EPSG:4326'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                dst_width=3000, dst_height=3000)
            meta = src.meta.copy()
            meta.update({'crs': dst_crs, 'transform': transform,
                         'width': width, 'height': height, 'driver': 'GTiff'})
            with rasterio.open(tmp_path, 'w', **meta) as tmp:
                reproject(source=rasterio.band(src, 1),
                          destination=rasterio.band(tmp, 1),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=dst_crs,
                          resampling=Resampling.nearest)
        with rasterio.open(tmp_path) as tmp:
            city_geom = [_box(*bbox)]
            clipped, clipped_transform = rio_mask(tmp, city_geom, crop=True)
            clipped_meta = tmp.meta.copy()
            clipped_meta.update({'height': clipped.shape[1],
                                 'width': clipped.shape[2],
                                 'transform': clipped_transform})
        _part = cache_path + '.part' + str(_os.getpid()) + '.tif'
        with rasterio.open(_part, 'w', **clipped_meta) as dst:
            dst.write(clipped)
        _os.replace(_part, cache_path)
        _os.remove(tmp_path)
        print('LST raster saved: ' + cache_path)
    else:
        print('LST raster cached: ' + cache_path)

    result = gpd.GeoDataFrame(
        pd.DataFrame([{'tif_path': cache_path, 'source': 'landsat_lst'}]),
        geometry=gpd.points_from_xy([0], [0]), crs='EPSG:4326')
    print('tif_path=' + cache_path)

download_data()
""".replace("__PLACE__", place).replace("__CITYSLUG__", city_slug),
        "satellite_worldcover": """
import planetary_computer
import pystac_client
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings('ignore')

def download_data():
    global result
    place = '__PLACE__'
    nom_params = dict(q=place, format='json', limit=1)
    res = requests.get('https://nominatim.openstreetmap.org/search',
        params=nom_params, headers={'User-Agent': 'GoAI/1.0'}, timeout=10)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned nothing for ' + place)
    bb = data[0]['boundingbox']
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    # Cap oversized bbox
    cy, cx = float(data[0]['lat']), float(data[0]['lon'])
    _ms = 1.5
    if (north - south) > _ms or (east - west) > _ms:
        south, north = cy - _ms/2, cy + _ms/2
        west, east = cx - _ms/2, cx + _ms/2
    bbox = [west, south, east, north]
    city_slug = '__CITYSLUG__'
    cache_path = '/data/processed/worldcover_' + city_slug + '.tif'
    os.makedirs('/data/processed', exist_ok=True)

    if not os.path.exists(cache_path):
        catalog = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com/api/stac/v1',
            modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=['io-lulc-annual-v02'],
            bbox=bbox,
            datetime='2022-01-01/2022-12-31',
            max_items=5)
        items = list(search.items())
        if not items:
            raise ValueError('No ESA WorldCover data found for ' + place)
        item = items[0]
        print('WorldCover scene: ' + item.id)
        signed = planetary_computer.sign(item)
        band_href = signed.assets['data'].href
        print('Downloading WorldCover band...')
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box as _box
        import os as _os
        tmp_path = cache_path + '.tmp' + str(_os.getpid()) + '.tif'
        with rasterio.open(band_href) as src:
            dst_crs = 'EPSG:4326'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                dst_width=2000, dst_height=2000)
            meta = src.meta.copy()
            meta.update({'crs': dst_crs, 'transform': transform,
                         'width': width, 'height': height, 'driver': 'GTiff'})
            with rasterio.open(tmp_path, 'w', **meta) as tmp:
                reproject(source=rasterio.band(src, 1),
                          destination=rasterio.band(tmp, 1),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=dst_crs,
                          resampling=Resampling.nearest)
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box as _box
        with rasterio.open(tmp_path) as tmp:
            city_geom = [_box(*bbox)]
            clipped, clipped_transform = rio_mask(tmp, city_geom, crop=True)
            clipped_meta = tmp.meta.copy()
            clipped_meta.update({'height': clipped.shape[1],
                                 'width': clipped.shape[2],
                                 'transform': clipped_transform})
            with rasterio.open(cache_path, 'w', **clipped_meta) as dst:
                dst.write(clipped)
        _os.remove(tmp_path)
        print('WorldCover saved: ' + cache_path)
    else:
        print('WorldCover cache hit: ' + cache_path)

    result = gpd.GeoDataFrame(
        pd.DataFrame([{'tif_path': cache_path, 'source': 'esa_worldcover'}]),
        geometry=gpd.points_from_xy([0], [0]), crs='EPSG:4326')
    print('tif_path=' + cache_path)

download_data()
""".replace("__PLACE__", place).replace("__CITYSLUG__", city_slug),
    }

    return templates.get(source_type)


# ── Disk persistence ──────────────────────────────────────────────────────────
# Saving now happens INSIDE the fetch sandbox run (run_code_in_sandbox save_path
# parameter) — one download instead of three (fetch / bbox-check / save).


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

def run_code_in_sandbox(code: str, timeout: int = 60, save_path: str = None) -> dict:
    save_section = ""
    if save_path:
        save_section = f"""
import os as _os_save
_os_save.makedirs('/data/processed', exist_ok=True)
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
    wrapped = f"""
import geopandas as gpd
import pandas as pd
import osmnx as ox
ox.settings.overpass_url = 'http://overpass-api.de/api/interpreter'
import warnings
warnings.filterwarnings('ignore')

result = None

{code}

if result is None:
    raise ValueError("result is None — download_data() never assigned a GeoDataFrame")
print("ROWS:", len(result))
print("CRS:", result.crs)
print("COLUMNS:", list(result.columns))
try:
    _b = result.to_crs('EPSG:4326').total_bounds
    print(f"BBOX: {{_b[1]:.4f}},{{_b[3]:.4f}},{{_b[0]:.4f}},{{_b[2]:.4f}}")
except Exception:
    print("BBOX: NA")
{save_section}
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


def validate_city_bbox(city: str, output: str) -> dict:
    """LLM-Find Feature #1: Type 2 error detection — checks if retrieved data
    is actually in the right city by comparing the BBOX line printed by the
    sandbox against the city's Nominatim bbox. No re-execution of fetch code."""
    if not city or not output:
        return {"valid": True}
    import re
    rows_match = re.search(r"ROWS:\s*(\d+)", output)
    if not rows_match or int(rows_match.group(1)) < 2:
        return {"valid": True}
    bbox_match = re.search(
        r"BBOX: (-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)", output)
    if not bbox_match:
        return {"valid": True}
    data_south = float(bbox_match.group(1))
    data_north = float(bbox_match.group(2))
    data_west = float(bbox_match.group(3))
    data_east = float(bbox_match.group(4))
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
        lat_ok = data_south <= city_north and data_north >= city_south
        lon_ok = data_west <= city_east and data_east >= city_west
        if not lat_ok or not lon_ok:
            print(
                f"[Retrieval] Type 2 error: data bbox ({data_south:.1f},{data_north:.1f},{data_west:.1f},{data_east:.1f}) doesn't overlap {city} bbox ({city_south:.1f},{city_north:.1f},{city_west:.1f},{city_east:.1f})")
            return {"valid": False, "error": f"Retrieved data is not in {city} — data bbox doesn't overlap city bbox"}
        return {"valid": True}
    except Exception:
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

            # ── Persistent cache for boundaries (city-level, survives restarts) ──
            # Implements LLM-dCache-style key-value caching (dataset:city slug).
            # First fetch for any city takes 2-10 min; all subsequent are instant.
            if source_type == "osm_boundaries":
                import re as _re
                _slug = _re.sub(r'[^a-z0-9]', '_', city.lower()).strip('_')
                _cache_dir = "/data/cache"
                _cache_path = f"{_cache_dir}/boundaries_{_slug}.geojson"
                os.makedirs(_cache_dir, exist_ok=True)
                if os.path.exists(_cache_path):
                    print(
                        f"[Retrieval] osm_boundaries cache HIT: {_cache_path}")
                    results[source_type] = {
                        "code": code, "output": "SAVED_OK",
                        "attempts": 0, "file_path": _cache_path,
                    }
                    continue
            if source_type.startswith("satellite_") or source_type == "worldpop_population":
                timeout = 900
            elif "roads" in source_type or source_type == "osm_boundaries":
                timeout = 600
            else:
                timeout = 120
            import hashlib as _hl
            task_hash = int(_hl.md5(task.encode()).hexdigest()
                            [:8], 16) % 1_000_000
            save_path = f"/data/processed/{source_type}_{task_hash}.geojson"
            sandbox_result = run_code_in_sandbox(
                code, timeout=timeout, save_path=save_path)

            if not sandbox_result["success"] and source_type == "osm_boundaries":
                print(
                    f"[Retrieval] Retrying osm_boundaries (first attempt failed)...")
                sandbox_result = run_code_in_sandbox(
                    code, timeout=timeout, save_path=save_path)

            if sandbox_result["success"]:
                validation = validate_output(sandbox_result["output"])
                if validation["valid"]:
                    # LLM-Find Feature #1: Type 2 error detection
                    # Check if retrieved data is actually in the right city
                    bbox_check = validate_city_bbox(
                        city, sandbox_result["output"])
                    if not bbox_check["valid"]:
                        print(
                            f"[Retrieval] {source_type} Type 2 error: {bbox_check['error'][:80]}")
                        # Wrong-city data — remove the file saved in-run
                        try:
                            if os.path.exists(save_path):
                                os.remove(save_path)
                        except Exception:
                            pass
                        results[source_type] = {
                            "error": bbox_check["error"], "attempts": 1, "file_path": None}
                        continue
                    saved_path = save_path if (
                        "SAVED_OK" in sandbox_result["output"] and os.path.exists(save_path)) else None
                    if saved_path:
                        print(
                            f"[Retrieval] {source_type} fetched + saved to {saved_path}")
                        # Write to persistent cache for future queries
                        if source_type == "osm_boundaries" and saved_path:
                            try:
                                import re as _re
                                import shutil as _sh
                                _slug = _re.sub(
                                    r'[^a-z0-9]', '_', city.lower()).strip('_')
                                _cache_path = f"/data/cache/boundaries_{_slug}.geojson"
                                os.makedirs("/data/cache", exist_ok=True)
                                _sh.copy2(saved_path, _cache_path)
                                print(
                                    f"[Retrieval] osm_boundaries cached: {_cache_path}")
                            except Exception as _ce:
                                print(f"[Retrieval] cache write failed: {_ce}")
                    else:
                        print(
                            f"[Retrieval] WARNING: {source_type} fetched but save failed — file_path will be None")
                    results[source_type] = {
                        "code":      code,
                        "output":    sandbox_result["output"],
                        "attempts":  1,
                        "file_path": saved_path,
                    }
                else:
                    print(
                        f"[Retrieval] {source_type} failed validation: {validation['error']}")
                    results[source_type] = {
                        "error": validation["error"], "attempts": 1, "file_path": None}
            else:
                print(
                    f"[Retrieval] {source_type} fetch failed: ...{sandbox_result['error'][-300:]}")
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

        import hashlib as _hl2
        _ss_hash = int(_hl2.md5(task.encode()).hexdigest()[:8], 16) % 1_000_000
        _ss_path = f"/data/processed/{name}_{_ss_hash}.geojson"
        for attempt in range(3):
            sandbox_result = run_code_in_sandbox(
                code, timeout=180, save_path=_ss_path)

            if sandbox_result["success"]:
                validation = validate_output(sandbox_result["output"])
                if validation["valid"]:
                    _saved = _ss_path if (
                        "SAVED_OK" in sandbox_result["output"] and os.path.exists(_ss_path)) else None
                    print(
                        f"[Retrieval] {name} fetched on attempt {attempt + 1}"
                        + (f", saved to {_saved}" if _saved else ", save failed — file_path None"))
                    results[name] = {
                        "code":     code,
                        "output":   sandbox_result["output"],
                        "attempts": attempt + 1,
                        "file_path": _saved,
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
