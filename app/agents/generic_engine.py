"""
agents/generic_engine.py
------------------------
A single deterministic engine that answers ANY "feature per area" question:

    "<feature> [count|density|area|length|per capita] by <unit> in <city>"

CHANGES in this version:
  - FEATURE_FETCH now uses Overpass API instead of OSMnx
    * No StringDtype issues (builds GeoDataFrame from plain Python dicts)
    * Faster for sparse features (pharmacies, hospitals, schools)
    * Falls back to second Overpass endpoint on failure
  - Cross-validation added to ANALYSIS_TEMPLATE
    * Re-counts using 'within' predicate vs main 'intersects' predicate
    * Prints CROSS_CORR:<r>:N:<n> for parsing
  - run_generic_analysis() returns cross_correlation in result dict
"""

import os
import re
import sys
import json
import subprocess

try:
    from tools.llm_client import smart_chat
except Exception:
    def smart_chat(system, user, use_groq=True):
        return ""

PROCESSED = "/data/processed"

# ── Compact ISO3 map for WorldPop (per-capita only) ──────────────────────────
CITY_ISO3 = {
    'mumbai': 'IND', 'greater mumbai': 'IND', 'delhi': 'IND', 'new delhi': 'IND',
    'bengaluru': 'IND', 'bangalore': 'IND', 'kolkata': 'IND', 'pune': 'IND',
    'hyderabad': 'IND', 'chennai': 'IND', 'ahmedabad': 'IND', 'jaipur': 'IND',
    'london': 'GBR', 'greater london': 'GBR', 'manchester': 'GBR', 'birmingham': 'GBR',
    'leeds': 'GBR', 'glasgow': 'GBR', 'edinburgh': 'GBR', 'bristol': 'GBR', 'liverpool': 'GBR',
    'berlin': 'DEU', 'munich': 'DEU', 'hamburg': 'DEU', 'frankfurt': 'DEU', 'cologne': 'DEU',
    'paris': 'FRA', 'lyon': 'FRA', 'marseille': 'FRA', 'toulouse': 'FRA',
    'amsterdam': 'NLD', 'rotterdam': 'NLD', 'madrid': 'ESP', 'barcelona': 'ESP',
    'rome': 'ITA', 'milan': 'ITA', 'vienna': 'AUT', 'zurich': 'CHE', 'brussels': 'BEL',
    'stockholm': 'SWE', 'oslo': 'NOR', 'copenhagen': 'DNK', 'warsaw': 'POL',
    'prague': 'CZE', 'budapest': 'HUN', 'lisbon': 'PRT', 'athens': 'GRC',
    'singapore': 'SGP', 'tokyo': 'JPN', 'osaka': 'JPN', 'seoul': 'KOR', 'busan': 'KOR',
    'bangkok': 'THA', 'jakarta': 'IDN', 'kuala lumpur': 'MYS', 'manila': 'PHL',
    'hanoi': 'VNM', 'beijing': 'CHN', 'shanghai': 'CHN', 'shenzhen': 'CHN',
    'guangzhou': 'CHN', 'hong kong': 'HKG', 'taipei': 'TWN', 'karachi': 'PAK',
    'lahore': 'PAK', 'dhaka': 'BGD', 'colombo': 'LKA', 'kathmandu': 'NPL',
    'dubai': 'ARE', 'abu dhabi': 'ARE', 'riyadh': 'SAU', 'jeddah': 'SAU',
    'istanbul': 'TUR', 'ankara': 'TUR', 'tehran': 'IRN', 'cairo': 'EGY',
    'alexandria': 'EGY', 'lagos': 'NGA', 'abuja': 'NGA', 'nairobi': 'KEN',
    'johannesburg': 'ZAF', 'cape town': 'ZAF', 'durban': 'ZAF', 'accra': 'GHA',
    'addis ababa': 'ETH', 'casablanca': 'MAR', 'new york': 'USA', 'los angeles': 'USA',
    'chicago': 'USA', 'houston': 'USA', 'phoenix': 'USA', 'philadelphia': 'USA',
    'san francisco': 'USA', 'seattle': 'USA', 'boston': 'USA', 'toronto': 'CAN',
    'montreal': 'CAN', 'vancouver': 'CAN', 'sao paulo': 'BRA', 'rio de janeiro': 'BRA',
    'buenos aires': 'ARG', 'mexico city': 'MEX', 'bogota': 'COL', 'lima': 'PER',
    'santiago': 'CHL', 'sydney': 'AUS', 'melbourne': 'AUS', 'brisbane': 'AUS',
    'perth': 'AUS', 'auckland': 'NZL', 'moscow': 'RUS', 'kyiv': 'UKR',
}

COUNTRY_HINT = {
    'IND': 'India', 'GBR': 'United Kingdom', 'DEU': 'Germany', 'FRA': 'France',
    'NLD': 'Netherlands', 'ESP': 'Spain', 'ITA': 'Italy', 'AUT': 'Austria',
    'CHE': 'Switzerland', 'BEL': 'Belgium', 'SWE': 'Sweden', 'NOR': 'Norway',
    'DNK': 'Denmark', 'POL': 'Poland', 'CZE': 'Czechia', 'HUN': 'Hungary',
    'PRT': 'Portugal', 'GRC': 'Greece', 'SGP': 'Singapore', 'JPN': 'Japan',
    'KOR': 'South Korea', 'THA': 'Thailand', 'IDN': 'Indonesia', 'MYS': 'Malaysia',
    'PHL': 'Philippines', 'VNM': 'Vietnam', 'CHN': 'China', 'HKG': 'Hong Kong',
    'TWN': 'Taiwan', 'PAK': 'Pakistan', 'BGD': 'Bangladesh', 'LKA': 'Sri Lanka',
    'NPL': 'Nepal', 'ARE': 'United Arab Emirates', 'SAU': 'Saudi Arabia',
    'TUR': 'Turkey', 'IRN': 'Iran', 'EGY': 'Egypt', 'NGA': 'Nigeria',
    'KEN': 'Kenya', 'ZAF': 'South Africa', 'GHA': 'Ghana', 'ETH': 'Ethiopia',
    'MAR': 'Morocco', 'USA': 'USA', 'CAN': 'Canada', 'BRA': 'Brazil',
    'ARG': 'Argentina', 'MEX': 'Mexico', 'COL': 'Colombia', 'PER': 'Peru',
    'CHL': 'Chile', 'AUS': 'Australia', 'NZL': 'New Zealand', 'RUS': 'Russia',
    'UKR': 'Ukraine',
}

# ── Feature keyword -> OSM tag spec ──────────────────────────────────────────
FEATURE_TAGS = {
    "hospital": ({"amenity": ["hospital"]}, "point"),
    "clinic": ({"amenity": ["clinic", "doctors"]}, "point"),
    "pharmac": ({"amenity": ["pharmacy"]}, "point"),
    "dentist": ({"amenity": ["dentist"]}, "point"),
    "veterinar": ({"amenity": ["veterinary"]}, "point"),
    "school": ({"amenity": ["school"]}, "point"),
    "kindergarten": ({"amenity": ["kindergarten"]}, "point"),
    "nursery": ({"amenity": ["kindergarten"]}, "point"),
    "university": ({"amenity": ["university"]}, "point"),
    "college": ({"amenity": ["college"]}, "point"),
    "librar": ({"amenity": ["library"]}, "point"),
    "bank": ({"amenity": ["bank"]}, "point"),
    "atm": ({"amenity": ["atm"]}, "point"),
    "restaurant": ({"amenity": ["restaurant"]}, "point"),
    "cafe": ({"amenity": ["cafe"]}, "point"),
    "coffee": ({"amenity": ["cafe"]}, "point"),
    "fast food": ({"amenity": ["fast_food"]}, "point"),
    "bar": ({"amenity": ["bar", "pub"]}, "point"),
    "pub": ({"amenity": ["pub"]}, "point"),
    "fuel": ({"amenity": ["fuel"]}, "point"),
    "petrol": ({"amenity": ["fuel"]}, "point"),
    "gas station": ({"amenity": ["fuel"]}, "point"),
    "charging station": ({"amenity": ["charging_station"]}, "point"),
    "ev charger": ({"amenity": ["charging_station"]}, "point"),
    "police": ({"amenity": ["police"]}, "point"),
    "fire station": ({"amenity": ["fire_station"]}, "point"),
    "post office": ({"amenity": ["post_office"]}, "point"),
    "place of worship": ({"amenity": ["place_of_worship"]}, "point"),
    "church": ({"amenity": ["place_of_worship"]}, "point"),
    "temple": ({"amenity": ["place_of_worship"]}, "point"),
    "mosque": ({"amenity": ["place_of_worship"]}, "point"),
    "cinema": ({"amenity": ["cinema"]}, "point"),
    "theatre": ({"amenity": ["theatre"]}, "point"),
    "marketplace": ({"amenity": ["marketplace"]}, "point"),
    "toilet": ({"amenity": ["toilets"]}, "point"),
    "drinking water": ({"amenity": ["drinking_water"]}, "point"),
    "bench": ({"amenity": ["bench"]}, "point"),
    "playground": ({"leisure": ["playground"]}, "point"),
    "gym": ({"leisure": ["fitness_centre", "sports_centre"]}, "point"),
    "fitness": ({"leisure": ["fitness_centre"]}, "point"),
    "swimming pool": ({"leisure": ["swimming_pool"]}, "area"),
    "stadium": ({"leisure": ["stadium"]}, "area"),
    "golf": ({"leisure": ["golf_course"]}, "area"),
    "sports pitch": ({"leisure": ["pitch"]}, "area"),
    "nature reserve": ({"leisure": ["nature_reserve"]}, "area"),
    "supermarket": ({"shop": ["supermarket"]}, "point"),
    "convenience": ({"shop": ["convenience"]}, "point"),
    "bakery": ({"shop": ["bakery"]}, "point"),
    "butcher": ({"shop": ["butcher"]}, "point"),
    "mall": ({"shop": ["mall", "department_store"]}, "point"),
    "shopping": ({"shop": ["mall", "supermarket", "department_store"]}, "point"),
    "shop": ({"shop": True}, "point"),
    "store": ({"shop": True}, "point"),
    "hotel": ({"tourism": ["hotel", "hostel", "guest_house"]}, "point"),
    "museum": ({"tourism": ["museum"]}, "point"),
    "hospital bed": ({"amenity": ["hospital"]}, "point"),
    "bus stop": ({"highway": ["bus_stop"], "public_transport": ["platform", "station"]}, "point"),
    "metro": ({"railway": ["station", "subway_entrance"], "station": ["subway"]}, "point"),
    "subway": ({"railway": ["station", "subway_entrance"], "station": ["subway"]}, "point"),
    "tram": ({"railway": ["tram_stop"]}, "point"),
    "train station": ({"railway": ["station", "halt"]}, "point"),
    "railway station": ({"railway": ["station", "halt"]}, "point"),
    "transit": ({"public_transport": ["station", "platform", "stop_position"], "highway": ["bus_stop"]}, "point"),
    "bike lane": ({"highway": ["cycleway"]}, "line"),
    "cycle lane": ({"highway": ["cycleway"]}, "line"),
    "cycle path": ({"highway": ["cycleway", "path"]}, "line"),
    "cycling": ({"highway": ["cycleway"]}, "line"),
    "footpath": ({"highway": ["footway", "path", "pedestrian"]}, "line"),
    "sidewalk": ({"highway": ["footway"]}, "line"),
    "road": ({"highway": ["primary", "secondary", "tertiary", "residential", "trunk", "motorway"]}, "line"),
    "street": ({"highway": ["primary", "secondary", "tertiary", "residential"]}, "line"),
    "parking": ({"amenity": ["parking"]}, "point"),
    "bicycle parking": ({"amenity": ["bicycle_parking"]}, "point"),
    "park": ({"leisure": ["park"], "landuse": ["recreation_ground"]}, "area"),
    "garden": ({"leisure": ["garden"]}, "area"),
    "forest": ({"landuse": ["forest"], "natural": ["wood"]}, "area"),
    "green space": ({"leisure": ["park", "garden"], "landuse": ["forest", "grass", "recreation_ground", "meadow"]}, "area"),
    "greenspace": ({"leisure": ["park", "garden"], "landuse": ["forest", "grass", "recreation_ground"]}, "area"),
    "water": ({"natural": ["water"], "waterway": ["river", "stream", "canal"]}, "area"),
    "tree": ({"natural": ["tree"]}, "point"),
    "building": ({"building": True}, "area"),
    "industrial": ({"landuse": ["industrial"]}, "area"),
    "commercial area": ({"landuse": ["commercial", "retail"]}, "area"),
    "residential area": ({"landuse": ["residential"]}, "area"),
}

PER_CAPITA_KW = [
    "per capita", "per 100k", "per 100000", "per 100,000", "per 1000",
    "per 1,000", "per 10000", "per population", "per resident", "per person",
    "per inhabitant",
]
DENSITY_KW = ["density", "per km", "per square km",
              "per sq km", "per km2", "per sq.km"]
ASC_KW = ["least", "lowest", "fewest", "worst", "minimum", "safest", "poorest"]

# ── Composite query keywords ──────────────────────────────────────────────────
DEPRIVATION_KW = [
    "underserved", "under-served", "underserved by", "lack of", "lacking",
    "desert", "deprived", "deprivation", "insufficient", "shortage of",
    "poor access", "limited access", "no access", "without", "gap in",
    "deficit", "missing", "absent"
]
CONTRAST_KW = [
    "but poor", "but low", "but lack", "but no", "but without",
    "yet poor", "yet low", "yet no",
    "high.*but", "good.*but", "many.*but", "lots of.*but"
]
OPTIMIZATION_KW = [
    "where should", "best location", "optimal location", "where to place",
    "where to build", "where to open", "where to put", "best place for",
    "where would", "ideal location", "site selection", "underserved area"
]
COMBINED_KW = [
    "and high", "and low", "combined", "together", "both",
    "correlation between", "relationship between", "overlap"
]

PROXIMITY_KW = [
    "within", "near", "closest", "nearest", "km of", "miles of",
    "radius", "distance from", "close to", "around"
]


def _country_for(city):
    iso = CITY_ISO3.get(city.lower().split(",")[0].strip())
    return COUNTRY_HINT.get(iso) if iso else None


def _place(city):
    hint = _country_for(city)
    base = city.split(",")[0].strip()
    return f"{base}, {hint}" if hint and hint.lower() not in city.lower() else city


def _clean(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "feature"


def parse_query(task):
    t = task.lower()
    ascending = any(k in t for k in ASC_KW)

    # Detect composite pattern FIRST before mode detection
    if any(k in t for k in OPTIMIZATION_KW):
        composite_pattern = "optimization"
    elif any(k in t for k in DEPRIVATION_KW):
        composite_pattern = "deprivation"
    elif any(re.search(k, t) for k in CONTRAST_KW):
        composite_pattern = "contrast"
    elif any(k in t for k in COMBINED_KW):
        composite_pattern = "combined"
    else:
        composite_pattern = None

    if any(k in t for k in PROXIMITY_KW):
        mode = "proximity"
    elif any(k in t for k in PER_CAPITA_KW):
        mode = "percapita"
    elif any(k in t for k in DENSITY_KW):
        mode = "density"
    elif composite_pattern in ("deprivation", "optimization"):
        mode = "percapita"   # deprivation always relative to population
        ascending = True      # worst-served first
    else:
        mode = "raw"

    # longest stem first
    for stem in sorted(FEATURE_TAGS.keys(), key=len, reverse=True):
        if stem in t:
            tags, kind = FEATURE_TAGS[stem]
            label = stem.replace(" ", "_")
            return tags, kind, mode, ascending, label, composite_pattern

    return None, None, mode, ascending, None, composite_pattern


def llm_resolve_feature(task):
    prompt = (
        'Translate this geographic question into OpenStreetMap query parameters.\n'
        f'Question: "{task}"\n\n'
        'Return ONLY a JSON object, no prose:\n'
        '{"tags": {"<osm_key>": ["<value>", ...]}, "kind": "point|area|line", "label": "<short noun>"}\n'
        'Rules: kind=point for things you count; kind=area for polygons; kind=line for linear features.\n'
        'Use real OSM keys (amenity, shop, leisure, landuse, natural, highway, railway, tourism).\n'
        'Example: {"tags": {"amenity": ["pharmacy"]}, "kind": "point", "label": "pharmacy"}'
    )
    try:
        resp = smart_chat(
            "You are an OpenStreetMap tagging expert. Return only JSON.", prompt, use_groq=True)
        resp = resp.replace("```json", "").replace("```", "").strip()
        s, e = resp.find("{"), resp.rfind("}") + 1
        obj = json.loads(resp[s:e])
        tags = obj.get("tags") or {}
        kind = obj.get("kind", "point")
        label = obj.get("label", "feature")
        if tags and kind in ("point", "area", "line"):
            print(
                f"[Generic] LLM resolved feature -> tags={tags}, kind={kind}")
            return tags, kind, label
    except Exception as ex:
        print(f"[Generic] LLM feature resolution failed: {ex}")
    return None, None, None


# ── Version-safe OSMnx bbox helper (used only for boundaries) ────────────────
_FFB = (
    "def _ffb(_n, _s, _e, _w, _t):\n"
    "    if int(str(ox.__version__).split('.')[0]) >= 2:\n"
    "        return ox.features_from_bbox((_w, _s, _e, _n), _t)\n"
    "    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_t)\n"
)

# ── Boundary fetch (OSMnx — fine for polygons, no dtype issues) ───────────────
BOUNDARY_FETCH = '''
import osmnx as ox, geopandas as gpd, requests
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')
''' + _FFB + '''
def go():
    global result
    res = requests.get('https://nominatim.openstreetmap.org/search',
                       params={'q': '__PLACE__', 'format': 'json', 'limit': 1},
                       headers={'User-Agent': 'GoAI/1.0'}, timeout=20)
    data = res.json()
    if not data:
        raise ValueError('Nominatim returned nothing for __PLACE__')
    bb = data[0]['boundingbox']
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    cx = (west + east) / 2; cy = (south + north) / 2
    zone = int((cx + 180) / 6) + 1
    utm = 'EPSG:%d' % (32600 + zone if cy >= 0 else 32700 + zone)
    best = None
    for lvl in ['10', '9', '8', '7', '6']:
        try:
            cand = _ffb(north, south, east, west, {'boundary': 'administrative', 'admin_level': lvl}).reset_index(drop=True)
            if 'boundary' in cand.columns:
                cand = cand[cand['boundary'] == 'administrative'].copy()
            cand = cand[cand.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
            for col in ['leisure', 'landuse', 'natural']:
                if col in cand.columns:
                    cand = cand[cand[col].isna()].copy()
            if 'name' in cand.columns:
                cand = cand[cand['name'].notna()].copy()
            cand = cand.reset_index(drop=True)
            if len(cand) < 3:
                continue
            cand['geometry'] = cand['geometry'].apply(make_valid)
            cu = cand.to_crs(utm)
            med = cu.geometry.area.median()
            cand = cand[cu.geometry.area.values >= med * 0.1].copy().reset_index(drop=True)
            if len(cand) >= 3:
                best = cand
                print('boundaries admin_level=%s n=%d' % (lvl, len(best)))
                break
        except Exception as e:
            print('boundary level %s failed: %s' % (lvl, e))
            continue
    if best is None or len(best) < 3:
        best = ox.geocode_to_gdf('__PLACE__').reset_index(drop=True)
        print('boundaries fallback to city outline')
    best['geometry'] = best['geometry'].apply(make_valid)
    result = best.to_crs('EPSG:4326')

go()
'''

# ── Feature fetch via Overpass API ────────────────────────────────────────────
# Overpass returns clean JSON → we build GeoDataFrame from plain Python dicts
# → no StringDtype issues. Two endpoint fallbacks for reliability.
FEATURE_FETCH = '''
import requests, geopandas as gpd
from shapely.geometry import Point
from shapely.validation import make_valid

def _tags_to_overpass(tags):
    """Convert OSM tags dict to Overpass QL filter string."""
    parts = []
    for key, values in tags.items():
        if values is True:
            parts.append('["%s"]' % key)
        elif isinstance(values, list):
            val_str = '|'.join(str(v) for v in values)
            parts.append('["%s"~"^(%s)$"]' % (key, val_str))
        elif isinstance(values, str):
            parts.append('["%s"="%s"]' % (key, values))
    return ''.join(parts)

def go():
    global result
    tags = __TAGS__
    tag_str = _tags_to_overpass(tags)

    # Get bounding box from Nominatim
    nom = requests.get('https://nominatim.openstreetmap.org/search',
                       params={'q': '__PLACE__', 'format': 'json', 'limit': 1},
                       headers={'User-Agent': 'GoAI/1.0'}, timeout=20).json()
    if not nom:
        raise ValueError('Nominatim: no result for __PLACE__')
    bb = nom[0]['boundingbox']
    # Overpass bbox order: south, west, north, east
    s, w, n, e = bb[0], bb[2], bb[1], bb[3]

    # Build Overpass QL query — nodes + ways (with center point for ways)
    query = (
        '[out:json][timeout:90];'
        '(node%(t)s(%(s)s,%(w)s,%(n)s,%(e)s);'
        'way%(t)s(%(s)s,%(w)s,%(n)s,%(e)s););'
        'out center;'
    ) % {'t': tag_str, 's': s, 'w': w, 'n': n, 'e': e}

    data = None
    for ep in [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
    ]:
        try:
            resp = requests.post(ep, data={'data': query},
                                 headers={'User-Agent': 'GoAI/1.0'}, timeout=120)
            d = resp.json()
            if d.get('elements'):
                data = d
                print('Overpass fetch from %s' % ep)
                break
        except Exception as ep_err:
            print('Overpass %s failed: %s' % (ep, ep_err))

    if not data or not data.get('elements'):
        raise ValueError('Overpass: no features found for __PLACE__')

    rows = []
    for el in data['elements']:
        if el['type'] == 'node':
            lat, lon = el.get('lat'), el.get('lon')
        elif 'center' in el:
            lat, lon = el['center']['lat'], el['center']['lon']
        else:
            continue
        if lat is None or lon is None:
            continue
        t = el.get('tags', {})
        rows.append({
            'geometry': Point(float(lon), float(lat)),
            'name':    str(t.get('name', '')),
            'amenity': str(t.get('amenity', '')),
            'shop':    str(t.get('shop', '')),
            'leisure': str(t.get('leisure', '')),
        })

    if not rows:
        raise ValueError('Overpass: no valid point geometries for __PLACE__')

    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326')
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf
    print('Overpass fetched %d features' % len(result))

go()
'''


PROXIMITY_FETCH = '''
import requests, geopandas as gpd
from shapely.geometry import Point
from shapely.validation import make_valid

def _tags_to_overpass(tags):
    parts = []
    for key, values in tags.items():
        if values is True:
            parts.append('["%s"]' % key)
        elif isinstance(values, list):
            val_str = '|'.join(str(v) for v in values)
            parts.append('["%s"~"^(%s)$"]' % (key, val_str))
        elif isinstance(values, str):
            parts.append('["%s"="%s"]' % (key, values))
    return ''.join(parts)

def go():
    global result
    tags = __TAGS__
    tag_str = _tags_to_overpass(tags)
    nom = requests.get('https://nominatim.openstreetmap.org/search',
                       params={'q': '__PLACE__', 'format': 'json', 'limit': 1},
                       headers={'User-Agent': 'GoAI/1.0'}, timeout=20).json()
    if not nom:
        raise ValueError('Nominatim: no result for __PLACE__')
    cx = float(nom[0]['lon'])
    cy = float(nom[0]['lat'])
    bb = nom[0]['boundingbox']
    s, w, n, e = bb[0], bb[2], bb[1], bb[3]
    query = (
        '[out:json][timeout:90];'
        '(node%(t)s(%(s)s,%(w)s,%(n)s,%(e)s);'
        'way%(t)s(%(s)s,%(w)s,%(n)s,%(e)s););'
        'out center;'
    ) % {'t': tag_str, 's': s, 'w': w, 'n': n, 'e': e}
    data = None
    for ep in ['https://overpass-api.de/api/interpreter',
               'https://overpass.kumi.systems/api/interpreter']:
        try:
            resp = requests.post(ep, data={'data': query},
                                 headers={'User-Agent': 'GoAI/1.0'}, timeout=120)
            d = resp.json()
            if d.get('elements'):
                data = d; break
        except Exception as ep_err:
            print('Overpass %s failed: %s' % (ep, ep_err))
    if not data or not data.get('elements'):
        raise ValueError('Overpass: no features found for __PLACE__')
    rows = []
    for el in data['elements']:
        if el['type'] == 'node':
            lat, lon = el.get('lat'), el.get('lon')
        elif 'center' in el:
            lat, lon = el['center']['lat'], el['center']['lon']
        else:
            continue
        if lat is None or lon is None:
            continue
        t = el.get('tags', {})
        rows.append({'geometry': Point(float(lon), float(lat)),
                     'name': str(t.get('name', 'Unknown')),
                     'amenity': str(t.get('amenity', '')),
                     'center_lon': cx, 'center_lat': cy})
    if not rows:
        raise ValueError('Overpass: no valid features for __PLACE__')
    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326')
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    result = gdf
    print('Proximity fetch: %d features' % len(result))

go()
'''


# ── Save suffix (used by _fetch_to_file) ─────────────────────────────────────
SAVE_SUFFIX = '''
import os as _os
_os.makedirs('/data/processed', exist_ok=True)
_out = result.reset_index(drop=True).copy()
_keep = ['geometry'] + [c for c in ['name', 'Name', 'NAME', 'ward', 'ward_name', 'label',
         'title', 'amenity', 'leisure', 'landuse', 'shop', 'natural', 'highway', 'railway',
         'public_transport', 'tourism', 'boundary', 'admin_level', 'area_name', 'localname',
         'center_lon', 'center_lat']
         if c in _out.columns]
_out = _out[_keep].copy()
for _c in _keep:
    if _c != 'geometry':
        try:
            _out[_c] = [str(v) if str(v) not in ('<NA>', 'None', 'nan') else '' for v in _out[_c]]
        except Exception:
            _out = _out.drop(columns=[_c])
_out = _out[_out.geometry.notna() & _out.geometry.is_valid].copy()
import json as _json
_geojson_str = _out.to_json()
with open('__OUT__', 'w') as _wf:
    _wf.write(_geojson_str)
print('SAVED_OK rows=%d' % len(_out))
'''

# ── Analysis template ─────────────────────────────────────────────────────────
# Cross-validation section added at the end (point features only):
# Re-counts using 'within' predicate vs main 'intersects' predicate.
# High Spearman r = result is robust to boundary-treatment method.
ANALYSIS_TEMPLATE = '''
import geopandas as gpd, pandas as pd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    WGS84 = 'EPSG:4326'
    UTM = "__UTM__"
    RASTER_CRS = "EPSG:__RASTER_EPSG__"
    MODE = "__MODE__"
    KIND = "__KIND__"
    METRIC = "__METRIC_COL__"
    MEASURE = "__MEASURE_COL__"
    ASCENDING = __ASCENDING__
    CITY = "__CITY__".lower().strip()

    wards = gpd.read_file("__B_PATH__")
    feats = gpd.read_file("__F_PATH__")
    wards['geometry'] = wards['geometry'].apply(make_valid)
    feats['geometry'] = feats['geometry'].apply(make_valid)
    wards = wards.set_crs(WGS84) if wards.crs is None else wards.to_crs(WGS84)
    feats = feats.set_crs(WGS84) if feats.crs is None else feats.to_crs(WGS84)

    name_col = next((c for c in wards.columns if c.lower() in
                    ['ward_full', 'ward_name', 'name', 'ward', 'label', 'title',
                     'area_name', 'localname', 'name:en', 'borough', 'district',
                     'neighbourhood']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(
        lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()

    wards_utm = wards.to_crs(UTM).copy()
    feats_utm = feats.to_crs(UTM).copy()
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(4)
    # Drop state-level boundary if mixed in with ward polygons
    _area_med = wards_utm['area_km2'].median()
    wards_utm = wards_utm[wards_utm['area_km2'] <= _area_med * 15].copy().reset_index(drop=True)

    if KIND == 'area':
        fpoly = feats_utm[feats_utm.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
        if len(fpoly) == 0:
            result = gpd.GeoDataFrame(); return
        inter = gpd.overlay(fpoly[['geometry']], wards_utm[['geometry', 'ward_name']], how='intersection')
        inter['v'] = inter.geometry.area / 1e6
        base = inter.groupby('ward_name')['v'].sum().reset_index(name='base_value')
    elif KIND == 'line':
        fline = feats_utm[feats_utm.geometry.geom_type.isin(['LineString', 'MultiLineString'])].copy()
        if len(fline) == 0:
            result = gpd.GeoDataFrame(); return
        fline['v'] = fline.geometry.length
        j = gpd.sjoin(fline[['geometry', 'v']], wards_utm[['geometry', 'ward_name']],
                      how='left', predicate='intersects')
        j = j.drop(columns='geometry')
        base = j.groupby('ward_name')['v'].sum().reset_index(name='base_value')
        base['base_value'] = base['base_value'] / 1000.0
    else:
        feats_utm['geometry'] = feats_utm.geometry.apply(
            lambda g: g.centroid if g.geom_type in ['Polygon', 'MultiPolygon'] else g)
        j = gpd.sjoin(feats_utm[['geometry']], wards_utm[['geometry', 'ward_name']],
                      how='left', predicate='intersects')
        j = j.drop(columns='geometry')
        base = j.groupby('ward_name').size().reset_index(name='base_value')

    merged = wards_utm.merge(base, on='ward_name', how='left')
    merged['base_value'] = merged['base_value'].fillna(0)

    if MODE == 'percapita':
        try:
            from rasterstats import zonal_stats
            wfs = merged.to_crs(RASTER_CRS)
            stats = zonal_stats(wfs, "__POP_TIF__", stats=['sum'], nodata=-99999, all_touched=True)
            pop = [s['sum'] if s is not None and s['sum'] is not None else 0 for s in stats]
            merged['population'] = pop
            if merged['population'].gt(0).sum() == 0:
                merged['population'] = (merged['area_km2'] * 1000).clip(lower=1)
            else:
                merged['population'] = merged['population'].clip(lower=1)
        except Exception as _e:
            print('zonal_stats failed: %s' % _e)
            merged['population'] = (merged['area_km2'] * 1000).clip(lower=1)
        merged['population'] = merged['population'].round(0).astype(int)
        merged[METRIC] = (merged['base_value'] / merged['population'] * 100000).round(2)
    elif MODE == 'density':
        merged[METRIC] = (merged['base_value'] / merged['area_km2'].replace(0, float('nan'))).round(4).fillna(0)
    else:
        merged[METRIC] = merged['base_value'].round(4)

    if KIND == 'point':
        merged[MEASURE] = merged['base_value'].round(0).astype(int)
    else:
        merged[MEASURE] = merged['base_value'].round(4)

    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != CITY]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    merged = merged.sort_values(METRIC, ascending=ASCENDING).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep, seen = [], set()
    for c in ['rank', 'ward_name', MEASURE, METRIC, 'area_km2', 'population', 'geometry']:
        if c in merged.columns and c not in seen:
            keep.append(c); seen.add(c)
    result = merged[keep].to_crs(WGS84).reset_index(drop=True)
    top = result.iloc[0]['ward_name'] if len(result) else 'NONE'
    print('GENERIC: %d units mode=%s kind=%s top=%s' % (len(result), MODE, KIND, top))
    for _, _row in result.head(50).iterrows():
        try:
            _val = int(_row[MEASURE]) if KIND == 'point' else round(float(_row[MEASURE]), 2)
        except Exception:
            _val = _row.get(MEASURE, 0)
        print('#%d %s: %s (%s)' % (int(_row['rank']), str(_row['ward_name']), _val, MEASURE))

    # ── Cross-validation (point features only) ────────────────────────────────
    # Re-count using strict 'within' predicate vs main 'intersects' predicate.
    # Spearman r between the two counts = spatial robustness score.
    # High r (>0.8) means ward assignments are stable regardless of boundary treatment.
    if KIND == 'point' and len(result) >= 5:
        try:
            from scipy.stats import spearmanr as _sr
            _feats2 = feats_utm.copy()
            _feats2['geometry'] = _feats2.geometry.apply(
                lambda g: g.centroid if g.geom_type not in ('Point', 'MultiPoint') else g)
            _j2 = gpd.sjoin(
                _feats2[['geometry']],
                wards_utm[['geometry', 'ward_name']],
                how='left', predicate='within')
            _c2 = _j2.groupby('ward_name').size().reset_index(name='_cnt2')
            _m2 = result[['ward_name', MEASURE]].merge(_c2, on='ward_name', how='inner')
            if len(_m2) >= 5 and _m2[MEASURE].std() > 0 and _m2['_cnt2'].std() > 0:
                _r, _ = _sr(_m2[MEASURE].values, _m2['_cnt2'].values)
                print('CROSS_CORR:%.4f:N:%d' % (float(_r), len(_m2)))
            else:
                print('CROSS_CORR:None:N:0')
        except Exception as _cve:
            print('CROSS_CORR:None:ERR:%s' % str(_cve)[:80])

run_analysis()
'''


PROXIMITY_ANALYSIS_TEMPLATE = '''
import geopandas as gpd
from shapely.geometry import Point
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_proximity():
    global result
    WGS84 = 'EPSG:4326'
    RADIUS_KM = __RADIUS_KM__
    LABEL = "__LABEL__"
    CITY = "__CITY__"
    feats = gpd.read_file("__F_PATH__")
    feats['geometry'] = feats['geometry'].apply(make_valid)
    feats = feats.set_crs(WGS84) if feats.crs is None else feats.to_crs(WGS84)
    cx = float(feats['center_lon'].iloc[0])
    cy = float(feats['center_lat'].iloc[0])
    zone = int((cx + 180) / 6) + 1
    utm = 'EPSG:%d' % (32600 + zone if cy >= 0 else 32700 + zone)
    center_utm = gpd.GeoDataFrame([{'geometry': Point(cx, cy)}],
                                   crs=WGS84).to_crs(utm).geometry.iloc[0]
    feats_utm = feats.to_crs(utm).copy()
    feats_utm['distance_km'] = feats_utm.geometry.apply(
        lambda g: round(g.distance(center_utm) / 1000, 3))
    within = feats_utm[feats_utm['distance_km'] <= RADIUS_KM].copy()
    if len(within) == 0:
        within = feats_utm[feats_utm['distance_km'] <= RADIUS_KM * 3].copy()
        print('Expanded radius: %d features' % len(within))
    within = within.sort_values('distance_km').reset_index(drop=True)
    within['rank'] = range(1, len(within) + 1)
    keep = [c for c in ['rank', 'name', 'distance_km', 'amenity', 'geometry'] if c in within.columns]
    result = within[keep].to_crs(WGS84).reset_index(drop=True)
    print('PROXIMITY: %d %s within %dkm of %s' % (len(result), LABEL, RADIUS_KM, CITY))
    for _, row in result.head(20).iterrows():
        print('#%d %s: %.2fkm' % (int(row['rank']), str(row['name']), float(row['distance_km'])))

run_proximity()
'''


COMPOSITE_ANALYSIS_TEMPLATE = '''
import geopandas as gpd, pandas as pd, numpy as np
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_composite():
    global result
    WGS84 = 'EPSG:4326'
    UTM = "__UTM__"
    PATTERN = "__PATTERN__"
    LABEL_A = "__LABEL_A__"
    LABEL_B = "__LABEL_B__"
    CITY = "__CITY__"

    wards = gpd.read_file("__B_PATH__")
    feats_a = gpd.read_file("__FA_PATH__")
    wards['geometry'] = wards['geometry'].apply(make_valid)
    feats_a['geometry'] = feats_a['geometry'].apply(make_valid)
    wards = wards.set_crs(WGS84) if wards.crs is None else wards.to_crs(WGS84)
    feats_a = feats_a.set_crs(WGS84) if feats_a.crs is None else feats_a.to_crs(WGS84)

    name_col = next((c for c in wards.columns if c.lower() in
                    ['ward_full','ward_name','name','ward','label','title',
                     'area_name','localname','name:en','borough','district',
                     'neighbourhood']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(
        lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()

    wards_utm = wards.to_crs(UTM).copy()
    feats_a_utm = feats_a.to_crs(UTM).copy()
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(4)

    # Count feature A per ward
    feats_a_utm['geometry'] = feats_a_utm.geometry.apply(
        lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    j = gpd.sjoin(feats_a_utm[['geometry']], wards_utm[['geometry','ward_name']],
                  how='left', predicate='intersects')
    count_a = j.groupby('ward_name').size().reset_index(name='count_a')
    merged = wards_utm.merge(count_a, on='ward_name', how='left')
    merged['count_a'] = merged['count_a'].fillna(0)

    # Try to get population for per-capita
    try:
        from rasterstats import zonal_stats
        _stats = zonal_stats(merged.to_crs(WGS84), "__POP_TIF__",
                             stats=['sum'], nodata=-99999, all_touched=True)
        merged['population'] = [s['sum'] if s and s['sum'] else 0 for s in _stats]
        merged['population'] = merged['population'].clip(lower=1).round(0).astype(int)
    except Exception:
        merged['population'] = (merged['area_km2'] * 1000).clip(lower=1).astype(int)

    # Feature B (if contrast/combined pattern)
    if PATTERN in ('contrast', 'combined') and "__FB_PATH__" != "":
        feats_b = gpd.read_file("__FB_PATH__")
        feats_b['geometry'] = feats_b['geometry'].apply(make_valid)
        feats_b = feats_b.set_crs(WGS84) if feats_b.crs is None else feats_b.to_crs(WGS84)
        feats_b_utm = feats_b.to_crs(UTM).copy()
        feats_b_utm['geometry'] = feats_b_utm.geometry.apply(
            lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
        j2 = gpd.sjoin(feats_b_utm[['geometry']], wards_utm[['geometry','ward_name']],
                       how='left', predicate='intersects')
        count_b = j2.groupby('ward_name').size().reset_index(name='count_b')
        merged = merged.merge(count_b, on='ward_name', how='left')
        merged['count_b'] = merged['count_b'].fillna(0)
    else:
        merged['count_b'] = 0

    # Normalise to [0,1]
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series([0.5]*len(s), index=s.index)

    merged['norm_a'] = _norm(merged['count_a'])
    merged['norm_b'] = _norm(merged['count_b'])
    merged['density_a'] = (merged['count_a'] / merged['area_km2'].replace(0, float('nan'))).fillna(0)
    merged['per_capita_a'] = (merged['count_a'] / merged['population'] * 100000).round(2)

    # Apply pattern-specific scoring
    if PATTERN == 'deprivation':
        # Low density relative to population = most deprived
        merged['composite_score'] = (1 - _norm(merged['per_capita_a'])).round(4)
        score_label = LABEL_A + '_deprivation_score'
        ascending = False  # highest deprivation first

    elif PATTERN == 'optimization':
        # Low existing coverage × high population = best location
        merged['composite_score'] = (
            (1 - _norm(merged['density_a'])) * _norm(merged['population'])
        ).round(4)
        score_label = 'location_score_for_' + LABEL_A
        ascending = False

    elif PATTERN == 'contrast':
        # High A but low B = biggest gap
        merged['composite_score'] = (
            _norm(merged['count_a']) - _norm(merged['count_b'])
        ).round(4)
        score_label = LABEL_A + '_vs_' + LABEL_B + '_gap'
        ascending = False

    elif PATTERN == 'combined':
        # High A AND high B
        merged['composite_score'] = (
            _norm(merged['count_a']) * _norm(merged['count_b'])
        ).round(4)
        score_label = LABEL_A + '_and_' + LABEL_B + '_combined'
        ascending = False

    else:
        merged['composite_score'] = _norm(merged['count_a'])
        score_label = LABEL_A + '_score'
        ascending = False

    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != CITY.lower().strip()]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    merged = merged.sort_values('composite_score', ascending=ascending).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)

    keep = [c for c in ['rank','ward_name','composite_score','count_a','count_b',
                         'per_capita_a','density_a','population','area_km2','geometry']
            if c in merged.columns]
    result = merged[keep].rename(columns={'composite_score': score_label}).to_crs(WGS84).reset_index(drop=True)

    top = result.iloc[0]['ward_name'] if len(result) else 'NONE'
    print('COMPOSITE: %d units pattern=%s top=%s' % (len(result), PATTERN, top))
    for _, row in result.head(20).iterrows():
        val = round(float(row[score_label]), 3)
        print('#%d %s: %.3f (%s)' % (int(row['rank']), str(row['ward_name']), val, score_label))

run_composite()
'''


def _run(code, timeout):
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout after %ds" % timeout


def _fetch_to_file(code, out_path, timeout):
    rc, out, err = _run(
        code + SAVE_SUFFIX.replace("__OUT__", out_path), timeout)
    if rc == 0 and "SAVED_OK" in out and os.path.exists(out_path):
        return out_path
    print("[Generic] fetch failed: %s" %
          (err.strip()[-300:] or out.strip()[-200:]))
    return None


def _ensure_worldpop(city):
    import requests
    iso = CITY_ISO3.get(city.lower().split(",")[0].strip(), "IND")
    try:
        os.makedirs(PROCESSED, exist_ok=True)
        data = requests.get(
            "https://hub.worldpop.org/rest/data/pop/wpgp?iso3=%s" % iso, timeout=30).json()["data"]
        latest = sorted(data, key=lambda x: x["popyear"], reverse=True)[0]
        url, year = latest["files"][0], latest["popyear"]
        tif = "%s/worldpop_%s_%s.tif" % (PROCESSED, iso, year)
        if not os.path.exists(tif):
            print("[Generic] downloading WorldPop %s..." % iso)
            r = requests.get(url, timeout=600, stream=True)
            with open(tif, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        epsg = 4326
        try:
            import rasterio
            with rasterio.open(tif) as src:
                epsg = src.crs.to_epsg() or 4326
        except Exception:
            pass
        return tif, epsg
    except Exception as e:
        print("[Generic] WorldPop failed: %s" % e)
        return "", 4326


def _build_analysis_code(b_path, f_path, tags_kind, mode, ascending, label, city, pop_tif, raster_epsg):
    if mode == "percapita":
        metric = "%s_per_100k" % _clean(label)
    elif mode == "density":
        metric = "%s_density" % _clean(label)
    else:
        metric = ("%s_count" if tags_kind == "point" else
                  "%s_area_km2" if tags_kind == "area" else "%s_length_km") % _clean(label)
    measure = ("%s_count" if tags_kind == "point" else
               "%s_area_km2" if tags_kind == "area" else "%s_length_km") % _clean(label)
    code = ANALYSIS_TEMPLATE
    code = code.replace("__B_PATH__", b_path).replace("__F_PATH__", f_path)
    code = code.replace("__POP_TIF__", pop_tif or "")
    code = code.replace("__UTM__", "EPSG:32643")
    code = code.replace("__RASTER_EPSG__", str(raster_epsg))
    code = code.replace("__MODE__", mode).replace("__KIND__", tags_kind)
    code = code.replace("__METRIC_COL__", metric).replace(
        "__MEASURE_COL__", measure)
    code = code.replace("__ASCENDING__", "True" if ascending else "False")
    code = code.replace("__CITY__", city.replace('"', ''))
    return code, metric


def _utm_for_file(path):
    code = (
        "import geopandas as gpd\n"
        "g = gpd.read_file(%r).to_crs('EPSG:4326')\n"
        "c = g.geometry.unary_union.centroid\n"
        "z = int((c.x + 180) / 6) + 1\n"
        "print('UTM:EPSG:%%d' %% (32600 + z if c.y >= 0 else 32700 + z))\n" % path
    )
    rc, out, err = _run(code, 60)
    m = re.search(r"UTM:(EPSG:\d+)", out)
    return m.group(1) if m else "EPSG:32643"


def _extract_second_feature(task):
    """Extract second feature from contrast queries like 'high X but poor Y'."""
    t = task.lower()
    import re
    contrast_match = re.search(
        r'but\s+(?:low|poor|lack(?:ing)?|no|without|limited)\s+(\w[\w\s]*?)(?:\s+(?:in|by|per|access|coverage)|$)',
        t)
    if contrast_match:
        phrase = contrast_match.group(1).strip()
        for stem in sorted(FEATURE_TAGS.keys(), key=len, reverse=True):
            if stem in phrase:
                return FEATURE_TAGS[stem][0], FEATURE_TAGS[stem][1], stem.replace(' ', '_')
        tags, kind, label = llm_resolve_feature(phrase)
        if tags:
            return tags, kind, label
    return None, None, None

# ── Public entry point ───────────────────────────────────────────────────────


# ── Public entry point ───────────────────────────────────────────────────────

def run_generic_analysis(task, plan, retrieved_data):
    city = plan.get("city", "") or ""
    if not city:
        return {"success": False, "error": "Generic engine needs a city"}

    tags, kind, mode, ascending, label, composite_pattern = parse_query(task)
    if tags is None:
        tags, kind, label = llm_resolve_feature(task)
        if tags is None:
            return {"success": False, "error": "Could not map query to an OSM feature"}

    place = _place(city)
    h = abs(hash(task + city)) % 1_000_000

    # 1) boundaries
    b_path = None
    rb = (retrieved_data or {}).get("osm_boundaries", {})
    if isinstance(rb, dict):
        rp = rb.get("file_path")
        if rp and os.path.exists(rp):
            b_path = rp
    if b_path is None and "mumbai" in city.lower():
        mw = "/data/mumbai_ward_shapefile/Mumbai_wards.geojson"
        if os.path.exists(mw):
            b_path = mw
    if b_path is None:
        print("[Generic] fetching boundaries for %s" % place)
        b_path = _fetch_to_file(
            BOUNDARY_FETCH.replace("__PLACE__", place),
            "%s/generic_bound_%d.geojson" % (PROCESSED, h), 180)
    if b_path is None:
        return {"success": False, "error": "Could not fetch boundaries for %s" % city}

    # 2) feature — Overpass API (no StringDtype, faster for sparse features)
    # 2) feature — proximity uses PROXIMITY_FETCH (stores city center coords)
    #            — all other modes use FEATURE_FETCH (Overpass bbox)
    print("[Generic] fetching feature %s (%s) for %s" % (label, kind, place))
    fetch_template = PROXIMITY_FETCH if mode == "proximity" else FEATURE_FETCH
    f_path = _fetch_to_file(
        fetch_template.replace("__PLACE__", place).replace(
            "__TAGS__", repr(tags)),
        "%s/generic_feat_%d.geojson" % (PROCESSED, h), 180)
    if f_path is None:
        nice = label.replace("_", " ")
        return {"success": False,
                "error": "No %s found in %s (no matching OpenStreetMap data)" % (nice, city),
                "no_data": True}

    # 2b) proximity mode — different analysis path
    if mode == "proximity":
        import re as _re
        radius_match = _re.search(r'(\d+)\s*km', task.lower())
        radius_km = int(radius_match.group(1)) if radius_match else 5
        prox_code = (PROXIMITY_ANALYSIS_TEMPLATE
                     .replace("__F_PATH__", f_path)
                     .replace("__RADIUS_KM__", str(radius_km))
                     .replace("__LABEL__", label)
                     .replace("__CITY__", city))
        rc, out, err = _run(prox_code + "\nprint('ROWS:', len(result))\n", 300)
        if rc != 0:
            return {"success": False, "error": "Proximity analysis failed: %s" % err[-200:]}
        m = re.search(r"ROWS:\s*(\d+)", out)
        if not m or int(m.group(1)) < 1:
            return {"success": False, "error": "No features found within radius"}
        print("[Generic] proximity success: %s features" % m.group(1))
        return {"success": True, "code": prox_code, "output": out,
                "attempts": 1, "cross_correlation": None}

    # 2c) composite mode — different analysis path
    if composite_pattern and composite_pattern != "proximity":
        print("[Generic] composite pattern=%s for %s" %
              (composite_pattern, place))

        # Fetch feature A
        fa_path = _fetch_to_file(
            FEATURE_FETCH.replace("__PLACE__", place).replace(
                "__TAGS__", repr(tags)),
            "%s/generic_feat_a_%d.geojson" % (PROCESSED, h), 180)
        if fa_path is None:
            return {"success": False, "error": "Could not fetch %s for composite analysis" % label}

        # Fetch feature B if contrast/combined
        fb_path = ""
        label_b = ""
        if composite_pattern in ("contrast", "combined"):
            tags_b, kind_b, label_b = _extract_second_feature(task)
            if tags_b:
                fb_path = _fetch_to_file(
                    FEATURE_FETCH.replace("__PLACE__", place).replace(
                        "__TAGS__", repr(tags_b)),
                    "%s/generic_feat_b_%d.geojson" % (PROCESSED, h), 180)

        # Population raster
        pop_tif, raster_epsg = "", 4326
        pop_tif, raster_epsg = _ensure_worldpop(city)

        # Build composite code
        utm = _utm_for_file(b_path)
        code = (COMPOSITE_ANALYSIS_TEMPLATE
                .replace("__B_PATH__", b_path)
                .replace("__FA_PATH__", fa_path)
                .replace("__FB_PATH__", fb_path or "")
                .replace("__POP_TIF__", pop_tif or "")
                .replace("__UTM__", utm)
                .replace("__PATTERN__", composite_pattern)
                .replace("__LABEL_A__", label)
                .replace("__LABEL_B__", label_b or "feature_b")
                .replace("__CITY__", city.replace('"', '')))

        rc, out, err = _run(code + "\nprint('ROWS:', len(result))\n", 600)
        if rc != 0:
            return {"success": False, "error": "Composite analysis failed: %s" % err[-200:]}
        m = re.search(r"ROWS:\s*(\d+)", out)
        if not m or int(m.group(1)) < 1:
            return {"success": False, "error": "Composite analysis produced no rows"}

        print("[Generic] composite success: %s rows" % m.group(1))
        return {"success": True, "code": code, "output": out,
                "attempts": 1, "cross_correlation": None}

    # 3) population raster
    pop_tif, raster_epsg = "", 4326
    if mode == "percapita":
        pop_tif, raster_epsg = _ensure_worldpop(city)
        if not pop_tif:
            print("[Generic] no population raster — falling back to density")
            mode = "density"

    # 4) build + run analysis (includes cross-validation)
    utm = _utm_for_file(b_path)
    code, metric = _build_analysis_code(
        b_path, f_path, kind, mode, ascending, label, city, pop_tif, raster_epsg)
    code = code.replace('UTM = "EPSG:32643"', 'UTM = "%s"' % utm)

    rc, out, err = _run(code + "\nprint('ROWS:', len(result))\n", 600)
    if rc != 0:
        return {"success": False, "error": "Generic analysis failed: %s" % err.strip()[-300:]}
    m = re.search(r"ROWS:\s*(\d+)", out)
    if not m or int(m.group(1)) < 1:
        return {"success": False, "error": "Generic analysis produced no rows"}

    # Parse cross-validation correlation from output
    cross_correlation = None
    cv_m = re.search(r"CROSS_CORR:([-\d.]+):N:(\d+)", out)
    if cv_m and cv_m.group(1) != 'None':
        try:
            cross_correlation = round(float(cv_m.group(1)), 4)
            print("[Generic] cross-validation r=%.4f n=%s" %
                  (cross_correlation, cv_m.group(2)))
        except ValueError:
            pass

    print("[Generic] success: %s rows, metric=%s" % (m.group(1), metric))
    return {
        "success": True,
        "code": code,
        "output": out,
        "attempts": 1,
        "cross_correlation": cross_correlation,
    }
