"""
agents/analysis_agent.py
------------------------
Runs spatial analysis code in a sandboxed subprocess.

GISclaw features: asymmetric truncation, package constraint, error memory,
                  code deduplication, schema analysis, per-task timeout,
                  sandbox import interception, output format contract,
                  zero-values validation, Type B attentional error detection

Paper 2 features: missing data early flagging, pre-execution syntax validation,
                  fault attribution by error source

FIXES in this version:
  - Bug 1: else: block restored in run_per_capita_analysis (was missing, caused wrong WorldPop raster)
  - Bug 2: UTM auto-detection helper _get_utm_epsg() replaces hardcoded EPSG:32643 everywhere
  - Bug 3: CITY_ISO3 promoted to module-level constant (single source of truth, no duplication)
  - Bug 4: Duplicate empty file_context assignment removed
  - Bug 5: Dead road_filter variable removed from run_flood_risk_analysis
"""

import json
import os
import subprocess
import sys

from tools.handbook_registry import load_guidance
from tools.llm_client import smart_chat
from tools.prompts import GIS_EXPERT_SYSTEM_PROMPT
from tools.rag import retrieve_relevant_docs
from tools.error_memory import store_error_fix, retrieve_similar_fix

try:
    from langfuse_client import langfuse as _lf
except Exception:
    _lf = None


def _lf_analysis_event(name, input=None, output=None):
    try:
        if _lf:
            kwargs = {"name": name}
            if input is not None:
                kwargs["input"] = input
            if output is not None:
                kwargs["output"] = output
            _lf.create_event(**kwargs)
    except Exception:
        pass


# ── Mumbai flood benchmark ────────────────────────────────────────────────────
MUMBAI_FLOOD_CODE = """
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM   = 'EPSG:32643'
    WGS84 = 'EPSG:4326'
    BBOX  = (72.77, 18.85, 73.00, 19.32)

    wards  = gpd.read_file('/data/mumbai_ward_shapefile/Mumbai_wards.geojson')
    lr_raw = gpd.read_file('/data/geojson_files/lakes_and_rivers.geojson')
    rd_raw = gpd.read_file('/data/geojson_files/river_lines_streams_drains.geojson')

    if wards.crs is None: wards = wards.set_crs(WGS84)
    if lr_raw.crs is None: lr_raw = lr_raw.set_crs(WGS84)
    else: lr_raw = lr_raw.to_crs(WGS84)
    if rd_raw.crs is None: rd_raw = rd_raw.set_crs(WGS84)
    else: rd_raw = rd_raw.to_crs(WGS84)

    minx, miny, maxx, maxy = BBOX
    lr_raw = lr_raw.cx[minx:maxx, miny:maxy].copy()
    rd_raw = rd_raw.cx[minx:maxx, miny:maxy].copy()

    lr_raw['geometry'] = lr_raw['geometry'].apply(make_valid)
    rd_raw['geometry'] = rd_raw['geometry'].apply(make_valid)
    wards['geometry']  = wards['geometry'].apply(make_valid)

    is_poly = lr_raw.geom_type.isin(['Polygon', 'MultiPolygon'])
    lakes   = lr_raw[is_poly].copy();  lakes['ftype']  = 'lake'
    rivers  = lr_raw[~is_poly].copy(); rivers['ftype'] = 'river'
    rd_raw['ftype'] = 'drain'

    flood_gdf = gpd.GeoDataFrame(
        pd.concat([lakes[['ftype', 'geometry']],
                   rivers[['ftype', 'geometry']],
                   rd_raw[['ftype', 'geometry']]], ignore_index=True),
        crs=WGS84
    )

    flood_utm   = flood_gdf.to_crs(UTM).copy()
    BUFFER_M    = {'river': 300, 'lake': 200, 'drain': 150}
    flood_utm['buffer_m'] = flood_utm['ftype'].map(BUFFER_M).fillna(150)
    flood_utm['geometry'] = flood_utm.apply(
        lambda r: make_valid(r.geometry.buffer(r['buffer_m'])), axis=1
    )
    flood_utm = flood_utm[flood_utm.geometry.is_valid & ~flood_utm.geometry.is_empty]

    flood_utm['_key']  = 1
    flood_zone_union   = flood_utm.dissolve(by='_key')[['geometry']].reset_index(drop=True)
    flood_union_geom   = make_valid(flood_zone_union.geometry.iloc[0])

    wards_utm = wards.to_crs(UTM).copy()
    wards_utm['geometry']     = wards_utm['geometry'].apply(make_valid)
    wards_utm['ward_area_m2'] = wards_utm.geometry.area

    def compute_overlap(geom):
        try:
            geom  = make_valid(geom)
            inter = geom.intersection(flood_union_geom)
            return inter.area if not inter.is_empty else 0.0
        except Exception:
            return 0.0

    wards_utm['flood_area_m2']            = wards_utm.geometry.apply(compute_overlap)
    wards_utm['flood_overlap_ratio']      = (wards_utm['flood_area_m2'] / wards_utm['ward_area_m2']).clip(0, 1)
    wards_utm['flood_exposed_population'] = (wards_utm['population'] * wards_utm['flood_overlap_ratio']).round(0).astype(int)
    wards_utm['area_km2']                 = (wards_utm['ward_area_m2'] / 1e6).round(3)

    wards_utm = wards_utm.sort_values('flood_exposed_population', ascending=False)
    wards_utm['rank'] = range(1, len(wards_utm) + 1)

    result = wards_utm[['rank', 'ward_full', 'population', 'area_km2',
                         'flood_overlap_ratio', 'flood_exposed_population',
                         'geometry']].to_crs(WGS84).reset_index(drop=True)

run_analysis()
"""

# ── Module-level CITY_ISO3 (FIX Bug 3: single source of truth) ───────────────
CITY_ISO3 = {
    # India
    'mumbai': 'IND', 'greater mumbai': 'IND', 'delhi': 'IND', 'new delhi': 'IND',
    'bengaluru': 'IND', 'bangalore': 'IND', 'kolkata': 'IND', 'calcutta': 'IND',
    'pune': 'IND', 'hyderabad': 'IND', 'chennai': 'IND', 'madras': 'IND',
    'ahmedabad': 'IND', 'jaipur': 'IND', 'surat': 'IND', 'lucknow': 'IND',
    # UK
    'london': 'GBR', 'greater london': 'GBR', 'inner london': 'GBR',
    'outer london': 'GBR', 'manchester': 'GBR', 'birmingham': 'GBR',
    'leeds': 'GBR', 'glasgow': 'GBR', 'edinburgh': 'GBR', 'bristol': 'GBR',
    # Europe
    'berlin': 'DEU', 'greater berlin': 'DEU', 'munich': 'DEU', 'hamburg': 'DEU',
    'frankfurt': 'DEU', 'cologne': 'DEU',
    'paris': 'FRA', 'greater paris': 'FRA', 'ile-de-france': 'FRA',
    'lyon': 'FRA', 'marseille': 'FRA',
    'amsterdam': 'NLD', 'rotterdam': 'NLD', 'the hague': 'NLD',
    'madrid': 'ESP', 'barcelona': 'ESP',
    'rome': 'ITA', 'milan': 'ITA',
    'vienna': 'AUT', 'zurich': 'CHE', 'brussels': 'BEL',
    'stockholm': 'SWE', 'oslo': 'NOR', 'copenhagen': 'DNK',
    'warsaw': 'POL', 'prague': 'CZE', 'budapest': 'HUN',
    # Asia
    'singapore': 'SGP', 'tokyo': 'JPN', 'osaka': 'JPN',
    'seoul': 'KOR', 'busan': 'KOR',
    'bangkok': 'THA', 'jakarta': 'IDN', 'kuala lumpur': 'MYS',
    'manila': 'PHL', 'ho chi minh city': 'VNM', 'hanoi': 'VNM',
    'beijing': 'CHN', 'shanghai': 'CHN', 'shenzhen': 'CHN', 'guangzhou': 'CHN',
    'taipei': 'TWN', 'hong kong': 'HKG',
    'dhaka': 'BGD', 'karachi': 'PAK', 'lahore': 'PAK',
    'colombo': 'LKA', 'kathmandu': 'NPL',
    # Middle East
    'dubai': 'ARE', 'abu dhabi': 'ARE', 'sharjah': 'ARE',
    'istanbul': 'TUR', 'ankara': 'TUR',
    'riyadh': 'SAU', 'jeddah': 'SAU',
    'cairo': 'EGY', 'alexandria': 'EGY',
    'tehran': 'IRN', 'baghdad': 'IRQ',
    # Africa
    'lagos': 'NGA', 'abuja': 'NGA',
    'nairobi': 'KEN', 'mombasa': 'KEN',
    'johannesburg': 'ZAF', 'cape town': 'ZAF', 'durban': 'ZAF',
    'accra': 'GHA', 'dar es salaam': 'TZA', 'addis ababa': 'ETH',
    'casablanca': 'MAR', 'tunis': 'TUN',
    # Americas
    'new york': 'USA', 'los angeles': 'USA', 'chicago': 'USA',
    'houston': 'USA', 'phoenix': 'USA', 'philadelphia': 'USA',
    'san francisco': 'USA', 'seattle': 'USA', 'boston': 'USA',
    'toronto': 'CAN', 'montreal': 'CAN', 'vancouver': 'CAN',
    'sao paulo': 'BRA', 'rio de janeiro': 'BRA', 'brasilia': 'BRA',
    'buenos aires': 'ARG', 'mexico city': 'MEX', 'guadalajara': 'MEX',
    'bogota': 'COL', 'lima': 'PER', 'santiago': 'CHL',
    # Oceania
    'sydney': 'AUS', 'melbourne': 'AUS', 'brisbane': 'AUS',
    'perth': 'AUS', 'auckland': 'NZL',
    # Russia/CIS
    'moscow': 'RUS', 'saint petersburg': 'RUS', 'st petersburg': 'RUS',
    'kiev': 'UKR', 'kyiv': 'UKR',
}


# ── FIX Bug 2: UTM auto-detection helper ─────────────────────────────────────
def _get_utm_epsg(gdf) -> str:
    """
    Auto-detect correct UTM zone from GeoDataFrame centroid.
    Replaces hardcoded EPSG:32643 (India UTM zone 43N) for global city support.
    London=32630, Berlin=32633, Paris=32631, Mumbai=32643, etc.
    """
    try:
        centroid = gdf.to_crs('EPSG:4326').geometry.unary_union.centroid
        lon, lat = centroid.x, centroid.y
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        print(
            f"[UTM] Auto-detected EPSG:{epsg} (lon={lon:.1f}, lat={lat:.1f}, zone={zone})")
        return f'EPSG:{epsg}'
    except Exception as e:
        print(f"[UTM] Auto-detect failed ({e}), falling back to EPSG:32643")
        return 'EPSG:32643'


# ── Query classification ──────────────────────────────────────────────────────

def is_mumbai_flood_query(task: str, plan: dict) -> bool:
    task_lower = task.lower()
    if any(x in task_lower for x in ['vulnerable', 'vulnerability', 'combining', 'composite']):
        return False
    city = plan.get("city", "").lower()
    analysis = plan.get("analysis_type", "").lower()
    metric = plan.get("ranking_metric", "").lower()
    return "mumbai" in city and "flood" in analysis and "flood_exposed_population" in metric


def is_osmnx_query(plan: dict) -> bool:
    sources = plan.get("required_sources", [])
    city = plan.get("city", "").lower()
    has_osm = any(
        any(x in s.lower()
            for x in ['osm', 'openstreet', 'open_street', 'open street'])
        for s in sources
    )
    return has_osm or "mumbai" not in city


# ── Mumbai metric detection ───────────────────────────────────────────────────

def detect_mumbai_metric(task: str, plan: dict) -> dict:
    task_lower = task.lower()
    ascending = any(x in task_lower for x in
                    ['least', 'lowest', 'smallest', 'minimum', 'fewest', 'poorest'])
    if any(x in task_lower for x in ['vulnerable', 'vulnerability', 'combining',
                                     'combined', 'composite', 'risk score']):
        return {"metric": "vulnerability_score", "ascending": ascending}
    if any(x in task_lower for x in ['area', 'size', 'largest', 'biggest', 'smallest']):
        return {"metric": "area_km2", "ascending": ascending}
    if any(x in task_lower for x in ['density', 'densely', 'dense']):
        return {"metric": "pop_density", "ascending": ascending}
    if any(x in task_lower for x in ['population', 'people', 'populous']):
        return {"metric": "population", "ascending": ascending}
    if any(x in task_lower for x in ['flood', 'water', 'risk']):
        return {"metric": "flood_overlap_ratio", "ascending": ascending}
    return {"metric": "area_km2", "ascending": False}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _detect_name_col(gdf) -> str:
    candidates = ['name', 'ward', 'label', 'title', 'ward_name', 'area_name',
                  'localname', 'name:en', 'name:pt', 'name:es', 'name:de',
                  'name:fr', 'name:ja', 'name:zh', 'name:ar', 'unique']
    for c in gdf.columns:
        if c.lower() in candidates:
            return c
    return gdf.columns[0]


def _safe_name_series(series):
    return series.apply(
        lambda x: (x[0] if isinstance(x, list) else x)
    ).fillna('Unknown').astype(str).str.strip().replace('', 'Unknown').replace('nan', 'Unknown')


# ── Fault attribution (Paper 2) ───────────────────────────────────────────────

def _classify_fault_source(error: str) -> str:
    e = error.lower()
    if any(x in e for x in ['no such file', 'drivererror', 'filenotfounderror',
                            'no such file or directory', 'cannot open']):
        return "data_ingestion"
    if any(x in e for x in ['keyerror', 'column', "'name'", 'attribute']):
        return "schema_mismatch"
    if any(x in e for x in ['crs', 'naive geomet', 'epsg', 'projection']):
        return "crs_mismatch"
    if any(x in e for x in ['syntaxerror', 'indentationerror', 'invalid syntax']):
        return "code_generation"
    if any(x in e for x in ['sjoin', 'spatial join', 'empty', 'no features']):
        return "spatial_operation"
    if any(x in e for x in ['typeerror', 'unhashable', "'list'"]):
        return "list_values"
    if 'timeout' in e:
        return "timeout"
    return "code_generation"


def _parse_domain_hint(domain_hint: str) -> dict:
    if not domain_hint:
        return {}
    import re
    config = {}
    hint = domain_hint.lower()
    match = re.search(r'(\d+)\s*m\b', hint)
    if match:
        config["buffer_m"] = int(match.group(1))
    for feature in ["river", "lake", "drain", "water", "stream"]:
        m = re.search(rf'(\d+)\s*m\s+(?:for\s+)?{feature}', hint)
        if m:
            config[f"buffer_{feature}_m"] = int(m.group(1))
    m = re.search(r'admin_level[=\s]+(\d+)', hint)
    if m:
        config["admin_level"] = m.group(1)
    m = re.search(r'epsg[:\s]+(\d+)', hint)
    if m:
        config["epsg"] = m.group(1)
    for road in ["motorway", "trunk", "primary", "secondary", "tertiary"]:
        if road in hint:
            config.setdefault("road_types", []).append(road)
    if any(x in hint for x in ["ascending", "lowest", "least", "worst"]):
        config["ascending"] = True
    if config:
        print(f"[DomainHint] Parsed config: {config}")
    return config


# ── Sandbox execution ─────────────────────────────────────────────────────────

# GISclaw Feature 1: Persistent sandbox cache — stores loaded GeoDataFrames
# between attempts so large files aren't reloaded every retry
_SANDBOX_CACHE: dict = {}


def _get_cache_key(code: str) -> list:
    """Extract file paths from code to check what's already cached."""
    import re
    return re.findall(r"gpd\.read_file\(['\"]([^'\"]+)['\"]\)", code)


def clear_sandbox_cache():
    """Clear the persistent sandbox cache between tasks."""
    global _SANDBOX_CACHE
    _SANDBOX_CACHE = {}
    print("[Sandbox] Cache cleared")


def run_code_in_sandbox(code: str, timeout: int = 300) -> dict:
    import ast
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"success": False, "error": f"SyntaxError before execution: {e}"}

    # GISclaw Feature 2: Type coercion — auto-cast string numerics before analysis
    # Prevents silent failures when numeric columns are stored as strings
    type_coercion_patch = """
# GISclaw type coercion: auto-cast string columns that look numeric
def _auto_cast_numerics(df):
    import pandas as pd
    for col in df.columns:
        if col == 'geometry': continue
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.5:  # >50% convertible
                    df[col] = converted
            except Exception:
                pass
    return df
"""

    # GISclaw Feature 3: Grid size limit — cap raster operations to prevent OOM
    grid_size_patch = """
# GISclaw grid size guard: limit raster window size to prevent OOM
_MAX_RASTER_CELLS = 5_000_000  # 5M pixels max
def _safe_raster_window(src, max_cells=_MAX_RASTER_CELLS):
    import rasterio
    h, w = src.height, src.width
    if h * w > max_cells:
        scale = (max_cells / (h * w)) ** 0.5
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        print(f"[GridGuard] Raster {h}x{w} → {new_h}x{new_w} (OOM prevention)")
        return rasterio.windows.Window(0, 0, new_w, new_h)
    return rasterio.windows.Window(0, 0, w, h)
"""

    wrapped = f"""
import sys
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

# GISclaw Feature: Sandbox-level import interception (Section 4.3.2)
# Stronger than sys.modules — uses import hooks to redirect at the loader level
import sys, builtins
_original_import = builtins.__import__
_BLOCKED = {{'arcpy', 'pykrige', 'skimage', 'arcgis', 'ArcPy'}}
_REDIRECTS = {{'pykrige': 'scipy.interpolate', 'skimage': 'rasterio', 'arcpy': 'geopandas'}}
def _safe_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in _BLOCKED:
        alt = _REDIRECTS.get(base, 'geopandas/rasterio/scipy')
        raise ImportError(f"'{{name}}' is blocked. Use '{{alt}}' instead.")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _safe_import

# GISclaw Feature: list_files — mandatory first action (Section 4.3.1)
# Agent sees actual files on disk before writing any code
import os as _os
def list_files(directory='/data/processed'):
    try:
        files = [f for f in _os.listdir(directory) if f.endswith(('.geojson','.tif','.csv','.shp'))]
        print(f"FILES_IN_{{directory}}: {{files}}")
        return files
    except Exception as e:
        print(f"FILES_IN_{{directory}}: []")
        return []

# GISclaw Feature: search_docs — agent can look up library APIs mid-task
def search_docs(query):
    docs = {{
        'sjoin': 'gpd.sjoin(left, right, how="left", predicate="intersects"). Drop geometry before groupby.',
        'overlay': 'gpd.overlay(df1, df2, how="intersection"). Returns clipped geometries.',
        'zonal_stats': 'rasterstats.zonal_stats(gdf.to_crs("EPSG:4326"), tif_path, stats=["sum"], nodata=-99999)',
        'to_crs': 'gdf.to_crs("EPSG:4326") or gdf.to_crs(utm_crs). Always project before area/distance calcs.',
        'features_from_place': 'ox.features_from_place(place, tags={{...}}). Returns GeoDataFrame.',
        'buffer': 'gdf.to_crs(utm).geometry.buffer(metres). Always buffer in UTM not WGS84.',
        'unary_union': 'gdf.geometry.unary_union. Returns single shapely geometry.',
        'centroid': 'gdf.geometry.centroid. Returns point GeoSeries.',
        'area': 'gdf.to_crs(utm).geometry.area / 1e6 for km2.',
        'make_valid': 'from shapely.validation import make_valid; gdf["geometry"] = gdf.geometry.apply(make_valid)',
    }}
    query_lower = query.lower()
    for key, doc in docs.items():
        if key in query_lower:
            print(f"DOCS[{{key}}]: {{doc}}")
            return doc
    print(f"DOCS: No match for '{{query}}'")
    return ""

{type_coercion_patch}
{grid_size_patch}

# GISclaw Feature: list_files called FIRST before any analysis code
list_files('/data/processed')
list_files('/data/mumbai_ward_shapefile')
list_files('/data/geojson_files')

{code}

if result is None:
    raise ValueError("result is None — run_analysis() never assigned a GeoDataFrame")

# Restore original import
builtins.__import__ = _original_import

# GISclaw Feature 2: apply type coercion to result
result = _auto_cast_numerics(result)

print("ROWS:", len(result))
print("CRS:", result.crs)
print("COLUMNS:", list(result.columns))

# GISclaw Feature 5: variable tracking — report what variables exist
_tracked_vars = {{k: type(v).__name__ for k, v in locals().items()
                  if not k.startswith('_') and k not in ('result', 'sys', 'gpd', 'pd', 'ox', 'warnings')}}
print("VARIABLES:", list(_tracked_vars.keys())[:15])

def get_name(row):
    for col in ['ward_name', 'ward_full', 'name_en', 'area_name', 'name', 'label', 'title']:
        val = row.get(col)
        if val and str(val).strip() not in ['', 'nan', 'unknown', 'None']:
            return str(val)
    return 'Unknown'

def get_rank(row):
    try:    return int(row.get('rank', 0))
    except: return 0

cols = list(result.columns)
skip_cols = {{'geometry', 'rank', 'bbox', 'fill-opacity', 'stroke-opacity', 'stroke',
             'ward_name', 'ward_full', 'name_en', 'name_ka', 'area_name', 'name',
             'label', 'title', 'population'}}
numeric_cols = [
    c for c in cols
    if c not in skip_cols
    and str(result[c].dtype) in ['float64', 'int64', 'int32', 'float32']
]
has_vuln  = 'vulnerability_score' in cols and 'flood_exposed_population' not in cols
has_flood = 'flood_exposed_population' in cols
priority_keywords = ['per_100k', 'per_capita', 'rate', 'density', 'ratio',
                     'overlap', 'coverage', 'score', 'count', 'length', 'area']
metric_col = None
for kw in priority_keywords:
    match = next((c for c in numeric_cols if kw in c.lower()), None)
    if match:
        metric_col = match
        break
if not metric_col:
    metric_col = next(
        (c for c in numeric_cols if c not in ['area_km2', 'ward_area_m2', 'flood_area_m2']),
        numeric_cols[0] if numeric_cols else None
    )
print("TOP 5 RESULTS:")
if has_vuln:
    for _, row in result.head(5).iterrows():
        print(f"  #{{get_rank(row)}} {{get_name(row)}}: {{row.get('vulnerability_score', 0):.3f}} score "
              f"(flood={{row.get('flood_overlap_ratio', 0):.1%}}, density={{row.get('pop_density', 0):.0f}}/km2)")
elif has_flood:
    for _, row in result.head(5).iterrows():
        print(f"  #{{get_rank(row)}} {{get_name(row)}}: {{int(row.get('flood_exposed_population', 0)):,}} people "
              f"(overlap={{row.get('flood_overlap_ratio', 0):.1%}})")
elif metric_col:
    val       = result[metric_col].iloc[0] if len(result) > 0 else 0
    col_lower = metric_col.lower()
    if 'ratio' in col_lower or 'overlap' in col_lower or 'coverage' in col_lower or (isinstance(val, float) and 0 <= float(val) <= 1):
        fmt = lambda v: f"{{float(v):.1%}}"
    elif 'per_100k' in col_lower or 'per_capita' in col_lower:
        fmt = lambda v: f"{{float(v):.2f}} per 100k"
    elif 'rate' in col_lower:
        fmt = lambda v: f"{{float(v):.2f}}"
    elif 'density' in col_lower:
        fmt = lambda v: f"{{float(v):.3f}}"
    elif 'count' in col_lower or str(result[metric_col].dtype) == 'int64':
        fmt = lambda v: f"{{int(v):,}} items"
    elif 'area' in col_lower or 'length' in col_lower or 'km' in col_lower:
        fmt = lambda v: f"{{float(v):.3f}} km"
    else:
        fmt = lambda v: f"{{float(v):.3f}}"
    for _, row in result.head(5).iterrows():
        print(f"  #{{get_rank(row)}} {{get_name(row)}}: {{fmt(row.get(metric_col, 0))}} ({{metric_col}})")
else:
    for _, row in result.head(5).iterrows():
        print(f"  #{{get_rank(row)}} {{get_name(row)}}")
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0:
            # GISclaw Feature 5: extract variable tracking info for retry context
            output = proc.stdout
            return {"success": True, "output": output}
        else:
            stderr = proc.stderr
            # GISclaw asymmetric truncation: preserve tail of stderr (root cause)
            if len(stderr) > 2000:
                stderr = "...[truncated]...\n" + stderr[-2000:]
            return {"success": False, "error": stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Sandbox timed out after {timeout}s"}


def validate_analysis_output(output: str) -> dict:
    if "ROWS: 0" in output:
        return {"valid": False, "error": "Type 1: analysis returned an empty result"}
    if "ROWS:" not in output:
        return {"valid": False, "error": "Type 1: result variable missing or not a GeoDataFrame"}
    import re
    match = re.search(r"ROWS:\s*(\d+)", output)
    if match and int(match.group(1)) < 3:
        return {"valid": False, "error": f"Type 2: result has only {match.group(1)} features — spatial join likely failed silently"}
    if "TOP 5 RESULTS:" in output:
        values = re.findall(
            r':\s*([\d.]+)', output.split("TOP 5 RESULTS:")[-1])
        if values and len(values) >= 3 and all(float(v) == 0.0 for v in values[:5]):
            return {"valid": False, "error": "Type 3: all metric values are zero — spatial join likely failed silently"}
    return {"valid": True}


def validate_code_paths(code: str, retrieved_data: dict) -> dict:
    """GTChain Feature #4 — Self-check mechanism.
    Validates that every gpd.read_file() call in generated code uses a path
    that actually exists on disk. Catches invented filenames like
    'london_boroughs.geojson' before wasting sandbox execution time.
    Based on GTChain paper Section 3.1.3 framework-workflow matching."""
    import re

    # Extract all paths used in gpd.read_file() calls
    read_file_paths = re.findall(
        r"gpd\.read_file\(['\"]([^'\"]+)['\"]\)", code)
    if not read_file_paths:
        # No file reads — let sandbox catch other errors
        return {"valid": True}

    # Build set of valid paths from retrieved_data + known local files
    valid_paths = set()
    for source_data in retrieved_data.values():
        if isinstance(source_data, dict):
            fp = source_data.get("file_path")
            if fp:
                valid_paths.add(fp)

    # Always allow known local Mumbai files
    valid_paths.update([
        "/data/mumbai_ward_shapefile/Mumbai_wards.geojson",
        "/data/geojson_files/lakes_and_rivers.geojson",
        "/data/geojson_files/river_lines_streams_drains.geojson",
    ])

    # Also allow any inline-fetched files (hospitals/schools inline)
    valid_paths.update([
        "/data/processed/osm_hospitals_inline.geojson",
        "/data/processed/osm_schools_inline.geojson",
    ])

    # Check each path used in code
    invented = []
    for path in read_file_paths:
        # Skip WorldPop tif paths — those are handled separately
        if path.endswith(".tif"):
            continue
        # Check if path exists on disk OR is in our valid set
        if not os.path.exists(path) and path not in valid_paths:
            invented.append(path)

    if invented:
        valid_paths_list = [
            p for p in valid_paths if p and "None" not in str(p)]
        return {
            "valid": False,
            "error": f"Invented file paths: {invented}. Use ONLY these valid paths: {valid_paths_list[:8]}"
        }

    return {"valid": True}


def validate_geographic_result(code: str, city: str) -> dict:
    if not city:
        return {"valid": True}
    check_code = code + f"""
import requests, warnings
warnings.filterwarnings('ignore')
try:
    res = requests.get('https://nominatim.openstreetmap.org/search',
        params={{'q': '{city}', 'format': 'json', 'limit': 1}},
        headers={{'User-Agent': 'GoAI/1.0'}}, timeout=10)
    data = res.json()
    if not data:
        print("GEO_CHECK: SKIP")
    else:
        bbox = data[0].get('boundingbox', [])
        if len(bbox) == 4:
            south, north, west, east = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            south -= 2; north += 2; west -= 2; east += 2
            centroid = result.to_crs('EPSG:4326').geometry.centroid
            cx, cy = centroid.x.mean(), centroid.y.mean()
            if west <= cx <= east and south <= cy <= north:
                print("GEO_CHECK: PASS")
            else:
                print(f"GEO_CHECK: FAIL cx={{cx:.2f}} cy={{cy:.2f}} bbox={{west:.1f}},{{south:.1f}},{{east:.1f}},{{north:.1f}}")
        else:
            print("GEO_CHECK: SKIP")
except Exception as e:
    print(f"GEO_CHECK: SKIP {{e}}")
"""
    try:
        proc = subprocess.run([sys.executable, "-c", check_code],
                              capture_output=True, text=True, timeout=30)
        if "GEO_CHECK: FAIL" in proc.stdout:
            return {"valid": False, "error": f"Type 2: result centroid is outside {city} bounding box"}
        return {"valid": True}
    except Exception:
        return {"valid": True}


# ── Mumbai general analysis ───────────────────────────────────────────────────

def run_mumbai_general_analysis(task: str, plan: dict) -> dict:
    info = detect_mumbai_metric(task, plan)
    metric = info["metric"]
    ascending = info["ascending"]
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    WGS84 = 'EPSG:4326'
    UTM   = 'EPSG:32643'
    wards = gpd.read_file('/data/mumbai_ward_shapefile/Mumbai_wards.geojson')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    wards_utm = wards.to_crs(UTM)
    wards_utm['area_km2']    = (wards_utm.geometry.area / 1e6).round(3)
    wards_utm['pop_density'] = (wards_utm['population'] / wards_utm['area_km2']).round(1)
    if 'flood_overlap_ratio' not in wards_utm.columns:
        wards_utm['flood_overlap_ratio'] = 0.0
    max_flood = wards_utm['flood_overlap_ratio'].max()
    max_dens  = wards_utm['pop_density'].max()
    wards_utm['flood_norm']          = (wards_utm['flood_overlap_ratio'] / max_flood).round(4) if max_flood > 0 else 0.0
    wards_utm['pop_density_norm']    = (wards_utm['pop_density'] / max_dens).round(4) if max_dens > 0 else 0.0
    wards_utm['vulnerability_score'] = (0.5 * wards_utm['flood_norm'] + 0.5 * wards_utm['pop_density_norm']).round(4)
    sort_col  = '{metric}'
    ascending = {str(ascending)}
    if sort_col not in wards_utm.columns:
        sort_col = 'area_km2'
    wards_utm = wards_utm.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    wards_utm['rank'] = range(1, len(wards_utm) + 1)
    keep   = ['rank', 'ward_full', 'population', 'area_km2', 'pop_density', 'flood_overlap_ratio', 'vulnerability_score', 'geometry']
    result = wards_utm[[c for c in keep if c in wards_utm.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print(
                f"[Analysis] Mumbai general engine success — metric: {metric}")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
    return {"success": False, "error": sandbox_result.get("error", "Unknown error")}


# ── Deterministic hospital density ───────────────────────────────────────────

def run_hospital_density_analysis(task: str, plan: dict, retrieved_data: dict) -> dict:
    h_path = retrieved_data.get("osm_hospitals", {}).get("file_path")
    if not h_path:
        return {"success": False, "error": "Missing file path for hospitals"}

    # Use osm_boundaries if available, otherwise fall back to mumbai_wards
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    city = plan.get("city", "").lower()
    if not b_path:
        if "mumbai" in city:
            b_path = "/data/mumbai_ward_shapefile/Mumbai_wards.geojson"
            print(
                "[Analysis] osm_boundaries missing — using Mumbai_wards.geojson fallback")
        else:
            return {"success": False, "error": "Missing file path for boundaries and no local fallback"}

    ascending = any(x in task.lower() for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards     = gpd.read_file('{b_path}')
    hospitals = gpd.read_file('{h_path}')
    wards['geometry']     = wards['geometry'].apply(make_valid)
    hospitals['geometry'] = hospitals['geometry'].apply(make_valid)
    if wards.crs is None:     wards     = wards.set_crs(WGS84)
    else:                     wards     = wards.to_crs(WGS84)
    if hospitals.crs is None: hospitals = hospitals.set_crs(WGS84)
    else:                     hospitals = hospitals.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm     = wards.to_crs(UTM).copy()
    hospitals_utm = hospitals.to_crs(UTM).copy()
    hospitals_utm['geometry'] = hospitals_utm.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(3)
    joined = gpd.sjoin(hospitals_utm[['geometry']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    counts = joined.groupby('ward_name').size().reset_index(name='hospital_count')
    merged = wards_utm.merge(counts, on='ward_name', how='left')
    merged['hospital_count']   = merged['hospital_count'].fillna(0).astype(int)
    merged['hospital_density'] = (merged['hospital_count'] / merged['area_km2'].replace(0, float('nan'))).round(4).fillna(0)
    ascending = {str(ascending)}
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged['rank'] = range(1, len(merged) + 1)
    keep   = ['rank','ward_name','hospital_count','hospital_density','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    print(
        f"[Analysis] Running hospital density for {city} (UTM: {utm_epsg})...")
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Hospital density succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Deterministic road density ────────────────────────────────────────────────

def run_road_density_analysis(task: str, plan: dict, retrieved_data: dict) -> dict:
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    r_path = retrieved_data.get("osm_roads", {}).get("file_path")
    if not b_path or not r_path:
        return {"success": False, "error": "Missing file paths for boundaries or roads"}
    city = plan.get("city", "")
    ascending = any(x in task.lower() for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards = gpd.read_file('{b_path}')
    edges = gpd.read_file('{r_path}')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    edges['geometry'] = edges['geometry'].apply(make_valid)
    if wards.crs is None: wards = wards.set_crs(WGS84)
    else:                 wards = wards.to_crs(WGS84)
    if edges.crs is None: edges = edges.set_crs(WGS84)
    else:                 edges = edges.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm = wards.to_crs(UTM).copy()
    edges_utm = edges.to_crs(UTM).copy()
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(3)
    edges_utm['length_m'] = edges_utm.geometry.length
    joined = gpd.sjoin(edges_utm[['geometry','length_m']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    road_lengths = joined.groupby('ward_name')['length_m'].sum().reset_index()
    road_lengths['road_length_km'] = (road_lengths['length_m'] / 1000).round(3)
    merged = wards_utm.merge(road_lengths[['ward_name','road_length_km']], on='ward_name', how='left')
    merged['road_length_km'] = merged['road_length_km'].fillna(0)
    merged['road_density']   = (merged['road_length_km'] / merged['area_km2'].replace(0, float('nan'))).round(3).fillna(0)
    ascending = {str(ascending)}
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged['rank'] = range(1, len(merged) + 1)
    keep   = ['rank','ward_name','road_length_km','road_density','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    print(f"[Analysis] Running road density for {city} (UTM: {utm_epsg})...")
    sandbox_result = run_code_in_sandbox(code, timeout=600)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Road density succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Deterministic flood risk ──────────────────────────────────────────────────

def run_flood_risk_analysis(task: str, plan: dict, retrieved_data: dict, domain_hint: str = "") -> dict:
    hint_config = plan.get("_hint_config", {})
    default_buffer = hint_config.get(
        "buffer_water_m") or hint_config.get("buffer_m") or 100
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    w_path = retrieved_data.get("osm_water", {}).get("file_path")
    if not b_path or not w_path:
        return {"success": False, "error": "Missing file paths for boundaries or water"}
    city = plan.get("city", "")
    ascending = any(x in task.lower()
                    for x in ["least", "lowest", "safest", "minimum", "best"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards = gpd.read_file('{b_path}')
    water = gpd.read_file('{w_path}')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    water['geometry'] = water['geometry'].apply(make_valid)
    if wards.crs is None: wards = wards.set_crs(WGS84)
    else:                 wards = wards.to_crs(WGS84)
    if water.crs is None: water = water.set_crs(WGS84)
    else:                 water = water.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm = wards.to_crs(UTM).copy()
    water_utm = water.to_crs(UTM).copy()
    water_utm['geometry'] = water_utm.geometry.buffer({default_buffer})
    water_utm['geometry'] = water_utm['geometry'].apply(make_valid)
    flood_zone = make_valid(unary_union(water_utm.geometry))
    wards_utm['ward_area_m2'] = wards_utm.geometry.area
    def compute_overlap(geom):
        try:
            geom  = make_valid(geom)
            inter = geom.intersection(flood_zone)
            return inter.area if not inter.is_empty else 0.0
        except Exception:
            return 0.0
    wards_utm['flood_area_m2']       = wards_utm.geometry.apply(compute_overlap)
    wards_utm['flood_overlap_ratio'] = (wards_utm['flood_area_m2'] / wards_utm['ward_area_m2']).clip(0, 1).round(4)
    wards_utm['area_km2']            = (wards_utm['ward_area_m2'] / 1e6).round(3)
    ascending = {str(ascending)}
    wards_utm = wards_utm.drop_duplicates(subset='ward_name', keep='first')
    wards_utm['rank'] = range(1, len(wards_utm) + 1)
    keep   = ['rank','ward_name','flood_overlap_ratio','area_km2','geometry']
    result = wards_utm[[c for c in keep if c in wards_utm.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    print(
        f"[Analysis] Running flood risk for {city} (UTM: {utm_epsg}, buffer: {default_buffer}m)...")
    sandbox_result = run_code_in_sandbox(code, timeout=600)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Flood risk succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Deterministic greenspace ──────────────────────────────────────────────────

def run_greenspace_analysis(task: str, plan: dict, retrieved_data: dict) -> dict:
    g_path = retrieved_data.get("osm_greenspace", {}).get("file_path")
    if not g_path:
        return {"success": False, "error": "Missing file path for greenspace"}
    city = plan.get("city", "")
    ascending = any(x in task.lower()
                    for x in ["least", "lowest", "minimum", "worst", "fewest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(g_path))
    except Exception:
        utm_epsg = 'EPSG:32643'  # fallback to India UTM, _get_utm_epsg handles correct zone
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    green = gpd.read_file('{g_path}')
    green['geometry'] = green['geometry'].apply(make_valid)
    if green.crs is None: green = green.set_crs(WGS84)
    else:                 green = green.to_crs(WGS84)
    green     = green[green.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
    green     = green[green.geometry.notna() & green.geometry.is_valid].copy()
    green_utm = green.to_crs(UTM).copy()
    green_utm['area_km2'] = (green_utm.geometry.area / 1e6).round(4)
    name_col = next((c for c in ['name','Name','NAME','label','title','leisure','landuse','natural'] if c in green_utm.columns), None)
    if name_col:
        green_utm['ward_name'] = green_utm[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unnamed').astype(str).str.strip()
    else:
        green_utm['ward_name'] = 'Unnamed'
    green_utm = green_utm[green_utm['area_km2'] > 0.001].copy()
    ascending = {str(ascending)}
    green_utm = green_utm.sort_values('area_km2', ascending=ascending).reset_index(drop=True)
    green_utm['rank'] = range(1, len(green_utm) + 1)
    keep   = ['rank','ward_name','area_km2','geometry']
    result = green_utm[[c for c in keep if c in green_utm.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    print(f"[Analysis] Running greenspace for {city} (UTM: {utm_epsg})...")
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Greenspace succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Deterministic schools ─────────────────────────────────────────────────────

def run_schools_analysis(task: str, plan: dict, retrieved_data: dict) -> dict:
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    s_path = retrieved_data.get("osm_schools", {}).get("file_path")
    if not b_path or not s_path:
        return {"success": False, "error": "Missing file paths for boundaries or schools"}
    city = plan.get("city", "")
    ascending = any(x in task.lower() for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards   = gpd.read_file('{b_path}')
    schools = gpd.read_file('{s_path}')
    wards['geometry']   = wards['geometry'].apply(make_valid)
    schools['geometry'] = schools['geometry'].apply(make_valid)
    if wards.crs is None:   wards   = wards.set_crs(WGS84)
    else:                   wards   = wards.to_crs(WGS84)
    if schools.crs is None: schools = schools.set_crs(WGS84)
    else:                   schools = schools.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm   = wards.to_crs(UTM).copy()
    schools_utm = schools.to_crs(UTM).copy()
    schools_utm['geometry'] = schools_utm.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(3)
    joined = gpd.sjoin(schools_utm[['geometry']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    counts = joined.groupby('ward_name').size().reset_index(name='school_count')
    merged = wards_utm.merge(counts, on='ward_name', how='left')
    merged['school_count']   = merged['school_count'].fillna(0).astype(int)
    merged['school_density'] = (merged['school_count'] / merged['area_km2'].replace(0, float('nan'))).round(4).fillna(0)
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != '{city}'.lower().strip()]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    ascending = {str(ascending)}
    merged = merged.sort_values('school_count', ascending=ascending).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep   = ['rank','ward_name','school_count','school_density','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    print(f"[Analysis] Running schools for {city} (UTM: {utm_epsg})...")
    sandbox_result = run_code_in_sandbox(code, timeout=600)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Schools succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── OSMnx deterministic templates ────────────────────────────────────────────

def generate_osmnx_analysis_code(task: str, plan: dict) -> str:
    city = plan.get("city", "")
    city_lower = city.lower()
    # Append country hint for known cities to help OSMnx geocoding
    UK_CITIES = ['london', 'birmingham', 'manchester', 'leeds', 'glasgow', 'liverpool',
                 'newcastle', 'sheffield', 'bristol', 'edinburgh', 'greater london']
    if any(x in city_lower for x in UK_CITIES):
        place = f"{city}, United Kingdom"
    elif any(x in city_lower for x in ['mumbai', 'delhi', 'bengaluru', 'kolkata', 'pune', 'hyderabad', 'chennai']):
        place = f"{city}, India"
    elif any(x in city_lower for x in ['berlin', 'munich', 'hamburg', 'frankfurt', 'cologne']):
        place = f"{city}, Germany"
    elif any(x in city_lower for x in ['paris', 'lyon', 'marseille']):
        place = f"{city}, France"
    else:
        place = city
    utm_inline = """
    cx   = gdf.to_crs('EPSG:4326').geometry.unary_union.centroid.x
    cy   = gdf.to_crs('EPSG:4326').geometry.unary_union.centroid.y
    zone = int((cx + 180) / 6) + 1
    UTM  = f'EPSG:{32600 + zone if cy >= 0 else 32700 + zone}'"""

    # Feature #6: proximity queries — "X within Nkm of Y", "nearest X to Y"
    task_lower = task.lower()
    proximity_keywords = ["within ", "km of", "nearest", "closest",
                          "near ", "radius", "distance from", "airport", "station"]
    is_proximity = any(kw in task_lower for kw in proximity_keywords)
    if is_proximity:
        # Detect feature type from task
        if any(x in task_lower for x in ["hospital", "clinic", "medical"]):
            prox_tags = "{'amenity': ['hospital', 'clinic', 'doctors']}"
            prox_name = "hospitals"
        elif any(x in task_lower for x in ["school", "university", "college"]):
            prox_tags = "{'amenity': ['school', 'university', 'college']}"
            prox_name = "schools"
        elif any(x in task_lower for x in ["airport", "airfield"]):
            prox_tags = "{'aeroway': 'aerodrome'}"
            prox_name = "airports"
        elif any(x in task_lower for x in ["station", "railway", "metro", "subway", "train"]):
            prox_tags = "{'railway': ['station', 'halt'], 'public_transport': 'station'}"
            prox_name = "stations"
        elif any(x in task_lower for x in ["park", "garden", "green"]):
            prox_tags = "{'leisure': 'park', 'landuse': ['forest','grass']}"
            prox_name = "parks"
        else:
            prox_tags = "{'amenity': True}"
            prox_name = "amenities"
        # Extract radius if mentioned (default 5km)
        import re as _re
        radius_match = _re.search(r'(\d+)\s*km', task_lower)
        radius_km = int(radius_match.group(1)) if radius_match else 5
        return f"""
import osmnx as ox, geopandas as gpd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    # Geocode the city/location
    location = ox.geocode('{place}')
    lat, lon = location[0], location[1]
    # Get features within radius
    tags = {prox_tags}
    gdf = ox.features_from_point((lat, lon), tags=tags, dist={radius_km * 1000})
    gdf = gdf.reset_index()
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    gdf = gdf.to_crs('EPSG:4326')
    # Convert polygons to centroids for distance calc
    gdf['geometry'] = gdf.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    # Compute distance from city center
    from shapely.geometry import Point
    center = Point(lon, lat)
    cx, cy = lon, lat
    zone = int((cx + 180) / 6) + 1
    UTM = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    gdf_utm = gdf.to_crs(UTM)
    center_utm = gpd.GeoDataFrame(geometry=[center], crs='EPSG:4326').to_crs(UTM).geometry.iloc[0]
    gdf_utm['distance_km'] = (gdf_utm.geometry.distance(center_utm) / 1000).round(2)
    name_col = next((c for c in ['name','Name','amenity','aeroway','railway'] if c in gdf_utm.columns), gdf_utm.columns[0])
    gdf_utm['ward_name'] = gdf_utm[name_col].apply(lambda x: x[0] if isinstance(x,list) else x).fillna('Unknown').astype(str).str.strip()
    gdf_utm = gdf_utm[gdf_utm['ward_name'] != 'Unknown'].copy()
    gdf_utm = gdf_utm.sort_values('distance_km').reset_index(drop=True)
    gdf_utm['rank'] = range(1, len(gdf_utm) + 1)
    keep = ['rank', 'ward_name', 'distance_km', 'geometry']
    result = gdf_utm[[c for c in keep if c in gdf_utm.columns]].to_crs('EPSG:4326').reset_index(drop=True)
    print(f"Found {{len(result)}} {prox_name} within {radius_km}km of {place}")

run_analysis()
"""

    if any(x in task.lower() for x in ['green', 'park', 'forest', 'garden']):
        return f"""
import osmnx as ox, geopandas as gpd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    tags  = {{'leisure': 'park', 'landuse': ['forest', 'grass', 'recreation_ground']}}
    parks = ox.features_from_place('{place}', tags).reset_index()
    parks = parks[parks.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
    parks['geometry'] = parks['geometry'].apply(make_valid)
    gdf = parks
    cx = parks.to_crs('EPSG:4326').geometry.unary_union.centroid.x
    cy = parks.to_crs('EPSG:4326').geometry.unary_union.centroid.y
    zone = int((cx + 180) / 6) + 1
    UTM  = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    parks_utm = parks.to_crs(UTM)
    parks_utm['area_km2'] = (parks_utm.geometry.area / 1_000_000).round(4)
    name_col = next((c for c in ['name','Name','leisure','landuse'] if c in parks_utm.columns), None)
    parks_utm['area_name'] = parks_utm[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unnamed').astype(str) if name_col else 'Unnamed'
    parks_utm = parks_utm.sort_values('area_km2', ascending=False).reset_index(drop=True)
    parks_utm['rank'] = range(1, len(parks_utm) + 1)
    extra  = [c for c in ['leisure','landuse'] if c in parks_utm.columns]
    result = parks_utm[['rank','area_name','area_km2','geometry'] + extra].to_crs('EPSG:4326').reset_index(drop=True)

run_analysis()
"""
    elif any(x in task.lower() for x in ['flood', 'water', 'coastal', 'inundation']):
        return f"""
import osmnx as ox, geopandas as gpd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    wards = ox.geocode_to_gdf('{place}').reset_index(drop=True)
    water = ox.features_from_place('{place}', {{'natural': ['water','coastline'], 'waterway': ['river','stream','drain']}}).reset_index()
    water['geometry'] = water['geometry'].apply(make_valid)
    cx = wards.to_crs('EPSG:4326').geometry.unary_union.centroid.x
    cy = wards.to_crs('EPSG:4326').geometry.unary_union.centroid.y
    zone = int((cx + 180) / 6) + 1
    UTM  = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    wards_utm = wards.to_crs(UTM)
    water_utm = water.to_crs(UTM)
    water_utm['geometry'] = water_utm.geometry.buffer(2000)
    flood_zone = water_utm.dissolve().geometry.iloc[0]
    wards_utm['ward_area_m2']        = wards_utm.geometry.area
    wards_utm['flood_area_m2']       = wards_utm.geometry.apply(lambda g: make_valid(g).intersection(flood_zone).area)
    wards_utm['flood_overlap_ratio'] = (wards_utm['flood_area_m2'] / wards_utm['ward_area_m2']).clip(0, 1)
    wards_utm['area_km2']            = (wards_utm['ward_area_m2'] / 1e6).round(3)
    wards_utm = wards_utm.sort_values('flood_overlap_ratio', ascending=False).reset_index(drop=True)
    wards_utm['rank'] = range(1, len(wards_utm) + 1)
    result = wards_utm[['rank','area_km2','flood_overlap_ratio','geometry']].to_crs('EPSG:4326')

run_analysis()
"""
    else:
        return f"""
import osmnx as ox, geopandas as gpd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    city_gdf = ox.geocode_to_gdf('{place}')
    cx = city_gdf.to_crs('EPSG:4326').geometry.unary_union.centroid.x
    cy = city_gdf.to_crs('EPSG:4326').geometry.unary_union.centroid.y
    zone = int((cx + 180) / 6) + 1
    UTM  = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    city_gdf_utm  = city_gdf.to_crs(UTM)
    city_geom_utm = city_gdf_utm.geometry.iloc[0]
    city_area_utm = city_geom_utm.area
    best = None
    for level in ['10','9','8','7']:
        try:
            candidate = ox.features_from_place('{place}', {{'boundary':'administrative','admin_level':level}}).reset_index(drop=True)
            if 'boundary' in candidate.columns:
                candidate = candidate[candidate['boundary']=='administrative'].copy()
            candidate = candidate[candidate.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
            for col in ['leisure','landuse','natural']:
                if col in candidate.columns:
                    candidate = candidate[candidate[col].isna()].copy()
            if 'name' in candidate.columns:
                candidate = candidate[candidate['name'].notna()].copy()
            candidate = candidate.reset_index(drop=True)
            if len(candidate) == 0: continue
            candidate['geometry'] = candidate['geometry'].apply(make_valid)
            candidate_utm = candidate.to_crs(UTM)
            keep_mask     = candidate_utm.geometry.area < city_area_utm * 1.5
            candidate     = candidate[keep_mask].copy().reset_index(drop=True)
            candidate_utm = candidate_utm[keep_mask].copy().reset_index(drop=True)
            intersects    = candidate_utm.geometry.intersects(city_geom_utm)
            candidate     = candidate[intersects.values].copy().reset_index(drop=True)
            if len(candidate) >= 3:
                best = candidate
                print(f'OSMnx template: admin_level={{level}}, {{len(best)}} neighborhoods')
                break
        except Exception:
            continue
    if best is None or len(best) < 3:
        best = city_gdf.reset_index(drop=True)
        print('OSMnx template: fallback to city outline')
    best['geometry'] = best['geometry'].apply(make_valid)
    gdf_utm = best.to_crs(UTM)
    gdf_utm['area_km2'] = (gdf_utm.geometry.area / 1e6).round(3)
    name_col = next((c for c in ['name','Name','NAME'] if c in gdf_utm.columns), gdf_utm.columns[0])
    gdf_utm['ward_name'] = gdf_utm[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    gdf_utm = gdf_utm.sort_values('area_km2', ascending=False).reset_index(drop=True)
    gdf_utm['rank'] = range(1, len(gdf_utm) + 1)
    result = gdf_utm[['rank','ward_name','area_km2','geometry']].to_crs('EPSG:4326')

run_analysis()
"""


# ── GTChain Feature #2: Tool description templates ───────────────────────────

TOOL_DESCRIPTIONS = {
    "osm_boundaries": {
        "name": "osm_boundaries",
        "function": "Administrative boundary polygons (wards, boroughs, districts). Use as the base layer for all spatial aggregation.",
        "input": "GeoDataFrame, Polygon/MultiPolygon geometries, CRS=EPSG:4326",
        "output": "One row per administrative unit. Key columns: name/ward_name (string), geometry (Polygon).",
        "usage": "Load with gpd.read_file(path). Detect name col dynamically. Project to UTM for area calc.",
        "example": "Compute hospital density per ward: load boundaries, sjoin hospital points, count per ward, divide by area_km2.",
    },
    "osm_hospitals": {
        "name": "osm_hospitals",
        "function": "Hospital and clinic locations from OpenStreetMap. Mix of point and polygon geometries.",
        "input": "GeoDataFrame, Point/Polygon geometries, CRS=EPSG:4326",
        "output": "One row per hospital. Key columns: amenity (string), name (string), geometry.",
        "usage": "Convert polygons to centroids before sjoin. Use predicate='intersects' in gpd.sjoin().",
        "example": "Count hospitals per ward: convert all to centroids, sjoin to boundaries, groupby ward_name.",
    },
    "osm_schools": {
        "name": "osm_schools",
        "function": "School, college, university locations from OpenStreetMap.",
        "input": "GeoDataFrame, Point/Polygon geometries, CRS=EPSG:4326",
        "output": "One row per school. Key columns: amenity (string), name (string), geometry.",
        "usage": "Convert polygons to centroids before sjoin. Group by ward after spatial join.",
        "example": "Count schools per district: centroids → sjoin boundaries → groupby → count.",
    },
    "osm_roads": {
        "name": "osm_roads",
        "function": "Road network edges (LineString geometries) from OpenStreetMap.",
        "input": "GeoDataFrame, LineString geometries, CRS=EPSG:4326",
        "output": "One row per road segment. Key columns: highway (string), name (string), geometry (LineString).",
        "usage": "Project to UTM, compute length_m = geometry.length, sjoin to boundaries, sum per ward.",
        "example": "Road density: sjoin edges to wards → sum length_m per ward → divide by area_km2.",
    },
    "osm_greenspace": {
        "name": "osm_greenspace",
        "function": "Parks, forests, grass areas as polygon geometries from OpenStreetMap.",
        "input": "GeoDataFrame, Polygon/MultiPolygon geometries, CRS=EPSG:4326",
        "output": "One row per green area. Key columns: leisure/landuse (string), name (string), geometry.",
        "usage": "Use gpd.overlay(how='intersection') with boundaries to get green area per ward.",
        "example": "Green space per ward: overlay intersection → sum feature_area_km2 per ward.",
    },
    "osm_parks": {
        "name": "osm_parks",
        "function": "Park polygons from OpenStreetMap. Similar to osm_greenspace but parks only.",
        "input": "GeoDataFrame, Polygon/MultiPolygon geometries, CRS=EPSG:4326",
        "output": "One row per park. Key columns: leisure (string), name (string), geometry.",
        "usage": "Use gpd.overlay(how='intersection') with boundaries. Group by ward_name.",
        "example": "Park area per borough: overlay → groupby ward → sum area → divide by population.",
    },
    "osm_water": {
        "name": "osm_water",
        "function": "Rivers, lakes, streams, drains as Point/LineString/Polygon geometries.",
        "input": "GeoDataFrame, mixed geometries, CRS=EPSG:4326",
        "output": "One row per water feature. Key columns: natural/waterway (string), geometry.",
        "usage": "Buffer water features in UTM, dissolve, then intersect with ward boundaries for flood ratio.",
        "example": "Flood risk: buffer water 300m → dissolve → intersection area / ward area = flood_ratio.",
    },
    "osm_transit": {
        "name": "osm_transit",
        "function": "Bus stops, metro stations, transit platforms as point geometries.",
        "input": "GeoDataFrame, Point/Polygon geometries, CRS=EPSG:4326",
        "output": "One row per stop/station. Key columns: public_transport/highway/railway (string), geometry.",
        "usage": "Convert to centroids, sjoin to boundaries, count per ward.",
        "example": "Transit density: centroids → sjoin → count per ward → divide by area_km2.",
    },
    "osm_commercial": {
        "name": "osm_commercial",
        "function": "Shops, restaurants, banks and commercial amenities as point/polygon geometries.",
        "input": "GeoDataFrame, Point/Polygon geometries, CRS=EPSG:4326",
        "output": "One row per venue. Key columns: shop/amenity (string), name (string), geometry.",
        "usage": "Convert to centroids, sjoin to boundaries, count per ward.",
        "example": "Commercial density: centroids → sjoin → count per ward → divide by area_km2.",
    },
    "osm_cycling": {
        "name": "osm_cycling",
        "function": "Cycling paths and cycle lanes as LineString geometries.",
        "input": "GeoDataFrame, LineString geometries, CRS=EPSG:4326",
        "output": "One row per path segment. Key columns: highway/bicycle (string), geometry.",
        "usage": "Project to UTM, compute length_m, sjoin to boundaries, sum per ward.",
        "example": "Cycling density: sjoin → sum length_km per ward → divide by area_km2.",
    },
    "osm_parking": {
        "name": "osm_parking",
        "function": "Parking facilities as point/polygon geometries.",
        "input": "GeoDataFrame, Point/Polygon geometries, CRS=EPSG:4326",
        "output": "One row per parking facility. Key columns: amenity (string), geometry.",
        "usage": "Convert to centroids, sjoin to boundaries, count per ward.",
        "example": "Parking density: centroids → sjoin → count per ward → divide by area_km2.",
    },
    "worldpop_population": {
        "name": "worldpop_population",
        "function": "WorldPop population raster. 100m resolution. Values = population count per pixel.",
        "input": "GeoTIFF raster, CRS=EPSG:4326, nodata=-99999",
        "output": "tif_path string pointing to cached .tif file.",
        "usage": "Use rasterstats.zonal_stats(wards.to_crs('EPSG:4326'), tif_path, stats=['sum'], nodata=-99999). Sum = total population per ward.",
        "example": "Population per ward: zonal_stats → wards['population'] = [s['sum'] for s in stats] → per_capita = feature/population * 100000.",
    },
    "mumbai_wards": {
        "name": "mumbai_wards",
        "function": "Mumbai ward boundaries with population data. Local file — always available for Mumbai queries.",
        "input": "Local GeoJSON file at /data/mumbai_ward_shapefile/Mumbai_wards.geojson",
        "output": "One row per ward. Key columns: ward_full (string), population (int), area_km2 (float), geometry.",
        "usage": "Load directly with gpd.read_file('/data/mumbai_ward_shapefile/Mumbai_wards.geojson'). No CRS conversion needed.",
        "example": "Hospital density: sjoin hospitals to wards → count per ward_full → divide by area_km2.",
    },
}


def _get_tool_descriptions(data_schema: dict) -> str:
    """GTChain Feature #2: Generate formal tool description templates for all
    data sources present in the query. Based on GTChain paper Section 3.1.1."""
    if not data_schema:
        return ""
    lines = ["TOOL DESCRIPTIONS (use these to understand each data source):"]
    for source_key in data_schema:
        desc = TOOL_DESCRIPTIONS.get(source_key)
        if not desc:
            continue
        lines.append(f"\nTool: {desc['name']}")
        lines.append(f"  Function: {desc['function']}")
        lines.append(f"  Input format: {desc['input']}")
        lines.append(f"  Output: {desc['output']}")
        lines.append(f"  Usage: {desc['usage']}")
        lines.append(f"  Example: {desc['example']}")
    return "\n".join(lines) if len(lines) > 1 else ""


def understand_data(retrieved_data: dict, plan: dict) -> str:
    if not retrieved_data:
        return ""
    import geopandas as gpd
    lines = ["Data overview (auto-extracted):"]
    for source_name, source_data in retrieved_data.items():
        if "output" not in source_data:
            continue
        file_path = source_data.get("file_path") or _extract_path(
            source_data.get("code", ""))
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            gdf = gpd.read_file(file_path)
            cols = list(gdf.columns)
            geom_types = gdf.geometry.geom_type.unique(
            ).tolist() if "geometry" in gdf.columns else []
            bounds = gdf.total_bounds
            extent = f"({bounds[0]:.3f},{bounds[1]:.3f},{bounds[2]:.3f},{bounds[3]:.3f})" if "geometry" in gdf.columns and len(
                gdf) > 0 else "Unknown"
            name_candidates = ['name', 'Name', 'NAME', 'ward', 'ward_name',
                               'label', 'title', 'name:en', 'area_name', 'localname']
            detected_name_col = next(
                (c for c in name_candidates if c in cols), cols[0] if cols else "unknown")
            lines += [f"\n[{source_name}]", f"  File: {os.path.basename(file_path)}", f"  Rows: {len(gdf)}",
                      f"  CRS: {gdf.crs}", f"  Geometry types: {geom_types}", f"  Extent: {extent}",
                      f"  Columns: {cols}", f"  Likely name column: '{detected_name_col}'"]
            sample_vals = {}
            for col in cols[:8]:
                if col == "geometry":
                    continue
                try:
                    val = gdf[col].dropna().iloc[0] if len(
                        gdf[col].dropna()) > 0 else None
                    if val is not None:
                        sample_vals[col] = str(val)[:50]
                except Exception:
                    pass
            if sample_vals:
                lines.append(f"  Sample values: {sample_vals}")
        except Exception as e:
            lines.append(f"\n[{source_name}] Could not read file: {e}")
    upload_files = plan.get("upload_files", [])
    for f in upload_files:
        fpath = f.get("file_path", "")
        ftype = f.get("type", "")
        if not fpath or not os.path.exists(fpath):
            continue
        try:
            if ftype == "csv":
                import pandas as pd
                df = pd.read_csv(fpath)
                lines += [f"\n[uploaded_{ftype}]",
                          f"  File: {os.path.basename(fpath)}, Rows: {len(df)}", f"  Columns: {list(df.columns)}"]
            else:
                gdf = gpd.read_file(fpath)
                lines += [f"\n[uploaded_{ftype}]",
                          f"  File: {os.path.basename(fpath)}, Rows: {len(gdf)}, CRS: {gdf.crs}", f"  Columns: {list(gdf.columns)}"]
        except Exception as e:
            lines.append(f"\n[uploaded_{ftype}] Could not read: {e}")
    return "\n".join(lines) if len(lines) > 1 else ""


def refine_and_plan(task: str, data_schema: dict, plan: dict) -> tuple:
    """Feature #5 — Token efficiency: combines refine_query + generate_analysis_plan
    into a single LLM call instead of two, saving ~2000-3000 tokens per query."""
    city = plan.get("city", "")
    schema_text = "\n".join([
        f"- {name}: {info.get('file_path', '')} | rows={info.get('output_info', '')[:60]}"
        for name, info in data_schema.items()
    ])
    tool_descs = _get_tool_descriptions(data_schema)
    prompt = f"""You are a GIS expert. For this spatial analysis task, provide TWO things:

Task: "{task}", City: {city}
Available data:
{schema_text}
{tool_descs}

1. DESCRIPTION (2-3 sentences): Describe the spatial operation, data layers, output metric, and pitfalls.
2. WORKFLOW (numbered steps): 1.INPUT 2.TRANSFORM 3.SPATIAL_OP 4.AGGREGATE 5.METRIC 6.OUTPUT
   IMPORTANT: Use WorldPop raster for population. Result must have ONE ROW PER WARD/BOROUGH.

Format your response EXACTLY as:
DESCRIPTION: <your description here>
WORKFLOW:
1. INPUT: ...
2. TRANSFORM: ...
3. SPATIAL_OP: ...
4. AGGREGATE: ...
5. METRIC: ...
6. OUTPUT: ...

Write only the description and workflow, no Python code."""
    try:
        response = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
        # Parse description and workflow from combined response
        description = task  # fallback
        workflow = ""
        if "DESCRIPTION:" in response and "WORKFLOW:" in response:
            desc_part = response.split("WORKFLOW:")[0].replace(
                "DESCRIPTION:", "").strip()
            workflow_part = response.split("WORKFLOW:")[-1].strip()
            if desc_part and not any(desc_part.startswith(kw) for kw in ["def ", "import ", "```"]):
                description = desc_part
            workflow = workflow_part
        elif "WORKFLOW:" in response:
            workflow = response.split("WORKFLOW:")[-1].strip()
        else:
            # Whole response is the description
            if not any(response.strip().startswith(kw) for kw in ["def ", "import ", "```"]):
                description = response.strip()
        print(f"[Analysis] Refined+planned: {description[:80]}...")
        if workflow:
            print(f"[GTChain] Workflow generated: {workflow[:150]}...")
        return description, workflow
    except Exception as e:
        print(f"[Analysis] refine_and_plan failed: {e}")
        return task, ""


def refine_query(task: str, plan: dict) -> str:
    """Kept for backward compatibility — calls refine_and_plan internally."""
    description, _ = refine_and_plan(task, {}, plan)
    return description


def generate_analysis_plan(task: str, data_schema: dict, plan: dict) -> str:
    """GTChain Feature #1 — kept for re-planning on retry attempts."""
    city = plan.get("city", "")
    schema_text = "\n".join(
        [f"- {name}: {info.get('file_path', '')} | {info.get('output_info', '')[:100]}" for name, info in data_schema.items()])
    tool_descs = _get_tool_descriptions(data_schema)
    prompt = f"""You are a GIS expert. Plan the steps to answer this spatial analysis question.
Task: "{task}", City: {city}
Available data:\n{schema_text}
{tool_descs}
IMPORTANT: Use WorldPop raster for population. result must have ONE ROW PER WARD/BOROUGH.
Write numbered workflow: 1.INPUT 2.TRANSFORM 3.SPATIAL_OP 4.AGGREGATE 5.METRIC 6.OUTPUT
Write only the plan, no code."""
    try:
        workflow = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
        print(f"[GTChain] Workflow generated: {workflow[:200]}...")
        return workflow
    except Exception as e:
        print(f"[GTChain] Workflow generation failed: {e}")
        return ""


def generate_analysis_code(task: str, data_schema: dict, plan: dict, retrieved_data: dict = None) -> str:
    guidance = load_guidance()
    plan_text = json.dumps(plan, indent=2)
    schema_text = json.dumps(data_schema, indent=2)
    # Feature #5: single LLM call for both refinement and workflow planning
    refined_task, workflow_plan = refine_and_plan(task, data_schema, plan)
    data_overview = ""
    if retrieved_data:
        data_overview = understand_data(retrieved_data, plan)
        if data_overview:
            print(
                f"[Analysis] Data understanding: {len(data_overview.splitlines())} lines")
    city = plan.get("city", "").lower()
    # FIX Bug 4: single file_context assignment, no duplicate
    file_context = ""
    if "mumbai" in city:
        file_context = """
Known local file paths for Mumbai (use EXACTLY):
- Ward boundaries: /data/mumbai_ward_shapefile/Mumbai_wards.geojson (cols: ward_full, population, area_km2, geometry)
- Lakes/rivers: /data/geojson_files/lakes_and_rivers.geojson
- Drains/streams: /data/geojson_files/river_lines_streams_drains.geojson
"""
    per_capita_hint = ""
    if plan.get("_query_type", {}).get("per_capita"):
        city_name = plan.get("city", "")
        tif_path = ensure_worldpop_raster(city_name)
        if not tif_path:
            import glob
            worldpop_files = glob.glob('/data/processed/worldpop_*.tif')
            tif_path = worldpop_files[0] if worldpop_files else '/data/processed/worldpop_IND_2020.tif'
        per_capita_hint = f"""
Per-capita instructions:
- Population raster: {tif_path}
- stats = zonal_stats(wards.to_crs('EPSG:4326'), '{tif_path}', stats=['sum'], nodata=-99999, all_touched=True)
- wards['population'] = [s['sum'] if s and s['sum'] else 1 for s in stats]
- Per-capita formula: (feature_value / population) * 100000
"""
    multi_source_hint = ""
    # Build a complete file path map for ALL retrieved sources so Groq never guesses
    path_lines = []
    for src_key, src_val in (data_schema if data_schema else {}).items():
        fp = src_val.get("file_path") if isinstance(src_val, dict) else None
        if fp:
            path_lines.append(f"  {src_key}: {fp}")
    if path_lines:
        multi_source_hint = "ACTUAL FILE PATHS (use EXACTLY, do NOT re-fetch from OSMnx):\n" + "\n".join(
            path_lines) + "\n"

    if "osm_boundaries" in data_schema and "osm_roads" in data_schema:
        b_path = data_schema["osm_boundaries"].get("file_path", "")
        r_path = data_schema["osm_roads"].get("file_path", "")
        multi_source_hint += f"Task: road density. sjoin edges to wards. road_density=km/area_km2. Auto-detect UTM."
    elif "osm_boundaries" in data_schema and "osm_hospitals" in data_schema:
        b_path = data_schema["osm_boundaries"].get("file_path", "")
        h_path = data_schema["osm_hospitals"].get("file_path", "")
        multi_source_hint += f"Task: hospital density. Centroids. sjoin. count/area_km2. Auto-detect UTM."
    elif "osm_boundaries" in data_schema and ("osm_greenspace" in data_schema or "osm_parks" in data_schema):
        b_path = data_schema["osm_boundaries"].get("file_path", "")
        p_path = data_schema.get("osm_greenspace", data_schema.get(
            "osm_parks", {})).get("file_path", "")
        multi_source_hint += f"Task: greenspace. ONE ROW PER WARD. gpd.overlay intersection. groupby ward_name. Auto-detect UTM."
    elif "osm_boundaries" in data_schema and "osm_water" in data_schema:
        b_path = data_schema["osm_boundaries"].get("file_path", "")
        w_path = data_schema["osm_water"].get("file_path", "")
        multi_source_hint += f"Task: flood risk. Buffer water 300m. flood_ratio=flood_area/ward_area. Auto-detect UTM."
    elif "osm_boundaries" in data_schema and "osm_schools" in data_schema:
        b_path = data_schema["osm_boundaries"].get("file_path", "")
        s_path = data_schema["osm_schools"].get("file_path", "")
        multi_source_hint += f"Task: school density. Centroids. sjoin. count per ward. Auto-detect UTM."
    rag_context = retrieve_relevant_docs(task)
    rag_section = f"\nRelevant tool documentation:\n{rag_context}\n" if rag_context else ""
    data_overview_section = f"\n{data_overview[:1000]}\n" if data_overview else ""
    hint_config = plan.get("_hint_config", {})
    hint_config_section = f"\nExpert hint config:\n{json.dumps(hint_config, indent=2)}\n" if hint_config else ""
    # GTChain Feature #2: formal tool description templates
    tool_desc_section = _get_tool_descriptions(data_schema)
    tool_desc_section = f"\n{tool_desc_section}\n" if tool_desc_section else ""
    # GTChain Feature #3: workflow plan already generated by refine_and_plan above
    workflow_section = f"\nFollow this workflow:\n{workflow_plan}\n" if workflow_plan else ""
    prompt = f"""Write Python GeoPandas code for this spatial analysis.
Task: "{task}"

{per_capita_hint}
Refined task: {refined_task}
Plan: {plan_text}
Data: {schema_text}
{data_overview_section}{file_context}{hint_config_section}{tool_desc_section}{workflow_section}{multi_source_hint}{rag_section}{guidance}
Rules:
- global result as FIRST line inside run_analysis()
- NEVER guess column names — detect dynamically using next((c for c in df.columns if ...), df.columns[0])
- result = final_geodataframe (no return)
- Last line: run_analysis()
- Auto-detect UTM: cx=wards.to_crs('EPSG:4326').geometry.unary_union.centroid.x; cy=wards.to_crs('EPSG:4326').geometry.unary_union.centroid.y; zone=int((cx+180)/6)+1; UTM=f'EPSG:{{32600+zone if cy>=0 else 32700+zone}}'
- Load from provided paths only — do NOT re-fetch from OSMnx
- predicate= in sjoin, never op=; drop geometry before groupby
- range(1,len+1) for rank; make_valid() on all geometries
- NEVER use: arcpy, pykrige, skimage — these are blocked at import level
- NEVER use plt.show() — use plt.savefig('/data/processed/output.png', dpi=150, bbox_inches='tight') if visualization needed
- Call list_files() and search_docs() if unsure about available files or API usage
- GISclaw Output Format Contract: result MUST be a GeoDataFrame with columns: rank(int), ward_name(str), [metric_col](float), geometry(EPSG:4326). No extra nested lists. All numeric columns must be actual numbers not strings.
- GISclaw Type Safety: after any merge/join, cast numeric cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
- GISclaw Grid Safety: never read entire large rasters at once — use windowed reads or rasterstats for population rasters
Return only Python code."""
    code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
    code = code.replace("```python", "").replace("```", "").replace(
        "~~~python", "").replace("~~~", "").strip()
    if code.startswith("python\n") or code.startswith("python "):
        code = code[6:].strip()
    return code


# ── Multi-file spatial join ───────────────────────────────────────────────────

def run_multi_file_analysis(task: str, plan: dict) -> dict:
    upload_files = plan.get("upload_files", [])
    geojson_file = next((f for f in reversed(upload_files)
                        if f["type"] == "geojson"), None)
    csv_file = next((f for f in upload_files if f["type"] == "csv"), None)
    if not geojson_file or not csv_file:
        return {"success": False, "error": "Need both a GeoJSON boundary file and a CSV data file"}
    ward_path = geojson_file["file_path"]
    csv_path = csv_file["file_path"]
    try:
        import geopandas as gpd
        import pandas as pd
        wards = gpd.read_file(ward_path)
        ward_cols = list(wards.columns)
        name_candidates = ['name_en', 'ward_name', 'Ward_Name',
                           'WARD', 'ward', 'NAME', 'Name', 'label', 'title', 'name']
        ward_name_col = next(
            (c for c in name_candidates if c in ward_cols), ward_cols[0])
        pop_candidates = ['population', 'pop',
                          'Population', 'POP', 'total_population']
        ward_pop_col = next(
            (c for c in pop_candidates if c in ward_cols), None)
        df = pd.read_csv(csv_path)
        csv_cols = list(df.columns)
        lat_col = next((c for c in csv_cols if c.lower()
                       in ['latitude', 'lat']), None)
        lon_col = next((c for c in csv_cols if c.lower() in [
                       'longitude', 'lon', 'lng']), None)
        print(
            f"[Analysis] Ward: {len(wards)} rows | CSV: {len(df)} rows, lat={lat_col}, lon={lon_col}")
    except Exception as e:
        return {"success": False, "error": f"Could not inspect uploaded files: {e}"}
    if not lat_col or not lon_col:
        return {"success": False, "error": f"CSV has no lat/lon columns. Found: {csv_cols}"}
    ascending = any(x in task.lower() for x in [
                    'lowest', 'least', 'minimum', 'fewest', 'poorest', 'worst'])
    pop_col_str = ward_pop_col or "population"
    code = f"""
import geopandas as gpd, pandas as pd
from shapely.geometry import Point
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    wards = gpd.read_file('{ward_path}')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    if wards.crs is None: wards = wards.set_crs('EPSG:4326')
    else:                 wards = wards.to_crs('EPSG:4326')
    df = pd.read_csv('{csv_path}')
    df = df.dropna(subset=['{lat_col}','{lon_col}'])
    df['{lat_col}'] = pd.to_numeric(df['{lat_col}'], errors='coerce')
    df['{lon_col}'] = pd.to_numeric(df['{lon_col}'], errors='coerce')
    df = df.dropna(subset=['{lat_col}','{lon_col}'])
    df['geometry'] = df.apply(lambda r: Point(r['{lon_col}'], r['{lat_col}']), axis=1)
    points = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    joined = gpd.sjoin(points, wards[['geometry','{ward_name_col}']], how='left', predicate='within')
    counts = joined.groupby('{ward_name_col}').size().reset_index(name='point_count')
    merged = wards.merge(counts, on='{ward_name_col}', how='left')
    merged['point_count'] = merged['point_count'].fillna(0).astype(int)
    merged['ward_name']   = merged['{ward_name_col}'].fillna('Unknown').astype(str).str.strip()
    if '{pop_col_str}' in merged.columns:
        merged['population']     = pd.to_numeric(merged['{pop_col_str}'], errors='coerce').fillna(1)
        merged['count_per_100k'] = (merged['point_count'] / merged['population'] * 100000).round(2)
        sort_col = 'count_per_100k'
    else:
        merged['population']     = 1
        merged['count_per_100k'] = merged['point_count'].astype(float)
        sort_col = 'point_count'
    merged = merged.sort_values(sort_col, ascending={str(ascending)}).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep   = ['rank','ward_name','point_count','count_per_100k','population','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs('EPSG:4326').reset_index(drop=True)

run_analysis()
"""
    print("[Analysis] Running multi-file spatial join...")
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Multi-file spatial join succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Single uploaded file ──────────────────────────────────────────────────────

def run_uploaded_file_analysis(task: str, plan: dict) -> dict:
    upload_path = plan.get("upload_path", "")
    if not upload_path or not os.path.exists(upload_path):
        return {"success": False, "error": "Uploaded file not found"}
    try:
        import geopandas as gpd
        import pandas as pd
        ext = upload_path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(upload_path)
            columns, rows, file_type = list(df.columns), len(df), "CSV"
        else:
            gdf = gpd.read_file(upload_path)
            columns, rows, file_type = list(gdf.columns), len(gdf), "GeoJSON"
        print(f"[Analysis] Uploaded file: {rows} rows, columns: {columns}")
    except Exception as e:
        return {"success": False, "error": f"Could not read uploaded file: {e}"}
    guidance = load_guidance()
    lat_col = next((c for c in columns if c.lower()
                   in ['latitude', 'lat']), None)
    lon_col = next((c for c in columns if c.lower() in [
                   'longitude', 'lon', 'lng']), None)
    geo_hint = f'Coordinates: lat="{lat_col}", lon="{lon_col}"' if lat_col and lon_col else ""
    group_candidates = ['Ward', 'ward', 'District', 'district', 'Area',
                        'area', 'Region', 'region', 'City', 'city', 'State', 'state']
    group_col = next((c for c in group_candidates if c in columns), columns[0])
    prompt = f"""Write Python code to analyse this uploaded {file_type} file.
Task: "{task}", File: "{upload_path}", Columns: {columns}, {geo_hint}
{guidance}
Group by "{group_col}", count/aggregate, result['ward_name']={group_col}, sort+rank.
Store as GeoDataFrame in global result (EPSG:4326). global result inside run_analysis(). Last line calls run_analysis().
Return only Python code."""
    code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, prompt, use_groq=True)
    code = code.replace("```python", "").replace("```", "").replace(
        "~~~python", "").replace("~~~", "").strip()
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Single file analysis succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    fix_prompt = f'Fix: Task="{task}", File="{upload_path}", Columns={columns}, Group="{group_col}", Error={error[:300]}. global result, GeoDataFrame, rank. Return only Python code.'
    code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT, fix_prompt, use_groq=True)
    code = code.replace("```python", "").replace("```", "").replace(
        "~~~python", "").replace("~~~", "").strip()
    sandbox_result = run_code_in_sandbox(code)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Single file analysis succeeded on retry")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 2}
    return {"success": False, "error": sandbox_result.get("error", error)}


def patch_result_assignment(code: str) -> str:
    if "result" in code:
        return code
    lines = code.strip().splitlines()
    last_gdf_var = None
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in ["gpd.read_file", "gpd.sjoin", ".merge(", "GeoDataFrame"]):
            if "=" in stripped:
                last_gdf_var = stripped.split("=")[0].strip()
    patch = f"\n    result = {last_gdf_var}\n" if last_gdf_var else "\n    result = gpd.GeoDataFrame()\n"
    return code + patch


# ── Hybrid Upload + OSM ───────────────────────────────────────────────────────

def run_hybrid_upload_osm_analysis(task: str, plan: dict) -> dict:
    upload_path = plan.get("upload_path", "")
    if not upload_path or not os.path.exists(upload_path):
        return {"success": False, "error": "No uploaded boundary file"}
    task_lower = task.lower()
    if any(x in task_lower for x in ["hospital", "clinic", "medical", "healthcare"]):
        osm_type, metric, count_col, tags_code = "hospital", "hospital_density", "hospital_count", "{'amenity': ['hospital','clinic','doctors']}"
    elif any(x in task_lower for x in ["school", "education", "college"]):
        osm_type, metric, count_col, tags_code = "school", "school_density", "school_count", "{'amenity': ['school','kindergarten','college','university']}"
    else:
        return {"success": False, "error": "Hybrid path: task type not detected"}
    city = plan.get("city", "")
    ascending = any(x in task_lower for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    print(f"[Analysis] Hybrid Upload+OSM: {osm_type} density for {city}")
    code = f"""
import geopandas as gpd, pandas as pd, osmnx as ox
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def _ffb(_n, _s, _e, _w, _tags):
    if int(str(ox.__version__).split('.')[0]) >= 2:
        return ox.features_from_bbox((_w, _s, _e, _n), _tags)
    return ox.features_from_bbox(north=_n, south=_s, east=_e, west=_w, tags=_tags)

def run_analysis():
    global result
    WGS84 = 'EPSG:4326'
    tags  = {tags_code}
    wards = gpd.read_file('{upload_path}')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    if wards.crs is None: wards = wards.set_crs(WGS84)
    elif str(wards.crs) != 'EPSG:4326': wards = wards.to_crs(WGS84)
    name_col_candidates = ['name','ward','ward_name','label','title','area_name','name_en','localname','neighbourhood','district']
    name_col = next((c for c in wards.columns if c.lower() in [x.lower() for x in name_col_candidates]), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    bounds = wards.total_bounds
    bbox_north, bbox_south, bbox_east, bbox_west = bounds[3], bounds[1], bounds[2], bounds[0]
    try:
        points = _ffb(bbox_north, bbox_south, bbox_east, bbox_west, tags).reset_index()
    except Exception as e:
        print(f"bbox fetch failed: {{e}}"); result = gpd.GeoDataFrame(); return
    if len(points) == 0: result = gpd.GeoDataFrame(); return
    points['geometry'] = points['geometry'].apply(make_valid)
    if points.crs is None: points = points.set_crs(WGS84)
    else:                  points = points.to_crs(WGS84)
    cx = (bbox_west + bbox_east) / 2; cy = (bbox_south + bbox_north) / 2
    zone = int((cx + 180) / 6) + 1
    UTM  = f'EPSG:{{32600 + zone if cy >= 0 else 32700 + zone}}'
    wards_utm  = wards.to_crs(UTM).copy()
    points_utm = points.to_crs(UTM).copy()
    points_utm['geometry'] = points_utm.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    wards_utm['area_km2']  = (wards_utm.geometry.area / 1e6).round(3)
    joined = gpd.sjoin(points_utm[['geometry']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    counts = joined.groupby('ward_name').size().reset_index(name='{count_col}')
    merged = wards_utm.merge(counts, on='ward_name', how='left')
    merged['{count_col}'] = merged['{count_col}'].fillna(0).astype(int)
    merged['{metric}']    = (merged['{count_col}'] / merged['area_km2'].replace(0, float('nan'))).round(4).fillna(0)
    merged = merged.sort_values('{metric}', ascending={str(ascending)}).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep   = ['rank','ward_name','{count_col}','{metric}','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    sandbox_result = run_code_in_sandbox(code, timeout=120)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Hybrid Upload+OSM succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── Generic point/linear density helpers ─────────────────────────────────────

def run_point_density_analysis(task: str, plan: dict, retrieved_data: dict,
                               source_key: str, metric_name: str, count_name: str) -> dict:
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    p_path = retrieved_data.get(source_key, {}).get("file_path")
    if not b_path or not p_path:
        return {"success": False, "error": f"Missing file paths for boundaries or {source_key}"}
    city = plan.get("city", "")
    ascending = any(x in task.lower() for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd, pandas as pd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards  = gpd.read_file('{b_path}')
    points = gpd.read_file('{p_path}')
    wards['geometry']  = wards['geometry'].apply(make_valid)
    points['geometry'] = points['geometry'].apply(make_valid)
    if wards.crs is None:  wards  = wards.set_crs(WGS84)
    else:                  wards  = wards.to_crs(WGS84)
    if points.crs is None: points = points.set_crs(WGS84)
    else:                  points = points.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm  = wards.to_crs(UTM).copy()
    points_utm = points.to_crs(UTM).copy()
    points_utm['geometry'] = points_utm.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
    wards_utm['area_km2']  = (wards_utm.geometry.area / 1e6).round(3)
    joined = gpd.sjoin(points_utm[['geometry']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    counts = joined.groupby('ward_name').size().reset_index(name='{count_name}')
    merged = wards_utm.merge(counts, on='ward_name', how='left')
    merged['{count_name}']  = merged['{count_name}'].fillna(0).astype(int)
    merged['{metric_name}'] = (merged['{count_name}'] / merged['area_km2'].replace(0, float('nan'))).round(4).fillna(0)
    ascending = {str(ascending)}
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != '{city}'.lower().strip()]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    merged = merged.sort_values('{metric_name}', ascending=ascending).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep = ['rank','ward_name','{count_name}','{metric_name}','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    sandbox_result = run_code_in_sandbox(code, timeout=300)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print(f"[Analysis] {metric_name} succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


def run_linear_density_analysis(task: str, plan: dict, retrieved_data: dict,
                                source_key: str, metric_name: str, length_name: str) -> dict:
    b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")
    l_path = retrieved_data.get(source_key, {}).get("file_path")
    if not b_path or not l_path:
        return {"success": False, "error": f"Missing file paths for boundaries or {source_key}"}
    city = plan.get("city", "")
    ascending = any(x in task.lower() for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'
    code = f"""
import geopandas as gpd, pandas as pd
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')

def run_analysis():
    global result
    UTM, WGS84 = '{utm_epsg}', 'EPSG:4326'
    wards = gpd.read_file('{b_path}')
    lines = gpd.read_file('{l_path}')
    wards['geometry'] = wards['geometry'].apply(make_valid)
    lines['geometry'] = lines['geometry'].apply(make_valid)
    if wards.crs is None: wards = wards.set_crs(WGS84)
    else:                 wards = wards.to_crs(WGS84)
    if lines.crs is None: lines = lines.set_crs(WGS84)
    else:                 lines = lines.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    wards_utm = wards.to_crs(UTM).copy()
    lines_utm = lines.to_crs(UTM).copy()
    wards_utm['area_km2'] = (wards_utm.geometry.area / 1e6).round(3)
    lines_utm['length_m'] = lines_utm.geometry.length
    joined = gpd.sjoin(lines_utm[['geometry','length_m']], wards_utm[['geometry','ward_name','area_km2']], how='left', predicate='intersects')
    joined = joined.drop(columns='geometry')
    lengths = joined.groupby('ward_name')['length_m'].sum().reset_index()
    lengths['{length_name}'] = (lengths['length_m'] / 1000).round(3)
    merged = wards_utm.merge(lengths[['ward_name','{length_name}']], on='ward_name', how='left')
    merged['{length_name}'] = merged['{length_name}'].fillna(0)
    merged['{metric_name}'] = (merged['{length_name}'] / merged['area_km2'].replace(0, float('nan'))).round(3).fillna(0)
    ascending = {str(ascending)}
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != '{city}'.lower().strip()]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    merged = merged.sort_values('{metric_name}', ascending=ascending).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep = ['rank','ward_name','{length_name}','{metric_name}','area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)

run_analysis()
"""
    sandbox_result = run_code_in_sandbox(code, timeout=600)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print(f"[Analysis] {metric_name} succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    return {"success": False, "error": error}


# ── WorldPop raster management ────────────────────────────────────────────────

def ensure_worldpop_raster(city: str) -> str:
    """Downloads WorldPop raster. Uses module-level CITY_ISO3 — single source of truth."""
    import requests as _req
    city_lower = city.lower().split(",")[0].strip()
    iso3 = CITY_ISO3.get(city_lower, None)
    if iso3 is None:
        try:
            nom = _req.get(f'https://nominatim.openstreetmap.org/search?q={city_lower}&format=json&limit=1',
                           headers={'User-Agent': 'GoAI/1.0'}, timeout=10).json()
            iso3 = 'IND'
            if nom:
                print(
                    f"[WorldPop] Unknown city '{city_lower}', using fallback ISO3=IND")
        except Exception:
            iso3 = 'IND'
    try:
        r = _req.get(
            f'https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}', timeout=30)
        data = r.json()['data']
        latest = sorted(data, key=lambda x: x['popyear'], reverse=True)[0]
        tif_url = latest['files'][0]
        tif_year = latest['popyear']
        tif_path = f'/data/processed/worldpop_{iso3}_{tif_year}.tif'
        if not os.path.exists(tif_path):
            print(f"[WorldPop] Downloading {iso3} raster ({tif_year})...")
            resp = _req.get(tif_url, timeout=600, stream=True)
            with open(tif_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(
                f"[WorldPop] Downloaded: {tif_path} ({os.path.getsize(tif_path)//1024//1024}MB)")
        else:
            print(f"[WorldPop] Using cached {tif_path}")
        return tif_path
    except Exception as e:
        print(f"[WorldPop] Download failed: {e}")
        return ""


# ── Per-capita analysis ───────────────────────────────────────────────────────

def run_per_capita_analysis(task: str, plan: dict, retrieved_data: dict) -> dict:
    """
    FIX Bug 1: else: block properly restored.
    FIX Bug 2: uses _get_utm_epsg() for correct UTM.
    FIX Bug 3: uses module-level CITY_ISO3.
    """
    import re
    city_lower_check = plan.get("city", "").lower()
    if any(x in city_lower_check for x in ["mumbai", "greater mumbai"]):
        b_path = "/data/mumbai_ward_shapefile/Mumbai_wards.geojson"
    else:
        b_path = retrieved_data.get("osm_boundaries", {}).get("file_path")

    wp_data = retrieved_data.get("worldpop_population", {})
    wp_output = wp_data.get("output", "")
    tif_match = re.search(
        r"tif_path=([^\s]+)", wp_output) if wp_output else None
    tif_path = ""  # initialize to avoid undefined variable risk

    if tif_match:
        tif_path = tif_match.group(1)
    else:
        # FIX Bug 1: else: properly indented and present
        city_name = plan.get("city", "")
        city_lower = city_name.lower().split(",")[0].strip()
        # FIX Bug 3: module-level CITY_ISO3
        iso3 = CITY_ISO3.get(city_lower, None)
        if iso3 is None:
            try:
                import requests as _req2
                nom = _req2.get(f'https://nominatim.openstreetmap.org/search?q={city_lower}&format=json&limit=1',
                                headers={'User-Agent': 'GoAI/1.0'}, timeout=10).json()
                iso3 = 'IND'
                if nom:
                    print(
                        f"[Analysis] Unknown city '{city_lower}', using fallback ISO3=IND")
            except Exception:
                iso3 = 'IND'
        import requests as _req
        try:
            r = _req.get(
                f'https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}', timeout=30)
            data = r.json()['data']
            latest = sorted(data, key=lambda x: x['popyear'], reverse=True)[0]
            tif_url = latest['files'][0]
            tif_year = latest['popyear']
            tif_path = f'/data/processed/worldpop_{iso3}_{tif_year}.tif'
            if not os.path.exists(tif_path):
                print(f"[Analysis] Downloading WorldPop for {iso3}...")
                resp = _req.get(tif_url, timeout=300, stream=True)
                with open(tif_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[Analysis] WorldPop downloaded: {tif_path}")
            else:
                print(f"[Analysis] Using cached WorldPop {tif_path}")
        except Exception as e:
            return {"success": False, "error": f"WorldPop download failed: {e}"}

    if not b_path:
        return {"success": False, "error": "Missing boundary file"}
    if not os.path.exists(tif_path):
        return {"success": False, "error": f"WorldPop raster not found: {tif_path}"}

    # Validate raster is not corrupted — delete and re-download if so
    try:
        import rasterio as _rio
        with _rio.open(tif_path) as _src:
            raster_epsg = _src.crs.to_epsg() or 4326
            _ = _src.read(1, window=_rio.windows.Window(0, 0, 10, 10))
        print(f"[Analysis] WorldPop raster CRS: EPSG:{raster_epsg}")
    except Exception as e:
        print(
            f"[Analysis] Raster corrupted ({e}) — deleting and re-downloading")
        try:
            os.remove(tif_path)
        except Exception:
            pass
        # Re-download
        import requests as _req_retry
        try:
            city_name2 = plan.get("city", "")
            city_lower2 = city_name2.lower().split(",")[0].strip()
            iso3_retry = CITY_ISO3.get(city_lower2, 'IND')
            r2 = _req_retry.get(
                f'https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3_retry}', timeout=30)
            data2 = r2.json()['data']
            latest2 = sorted(
                data2, key=lambda x: x['popyear'], reverse=True)[0]
            print(f"[Analysis] Re-downloading WorldPop for {iso3_retry}...")
            resp2 = _req_retry.get(
                latest2['files'][0], timeout=600, stream=True)
            with open(tif_path, 'wb') as _f2:
                for chunk in resp2.iter_content(chunk_size=8192):
                    _f2.write(chunk)
            with _rio.open(tif_path) as _src2:
                raster_epsg = _src2.crs.to_epsg() or 4326
            print(
                f"[Analysis] Re-download successful, CRS: EPSG:{raster_epsg}")
        except Exception as e2:
            return {"success": False, "error": f"WorldPop raster corrupted and re-download failed: {e2}"}

    # FIX Bug 2: auto-detect UTM from boundaries
    try:
        import geopandas as _gpd
        utm_epsg = _get_utm_epsg(_gpd.read_file(b_path))
    except Exception:
        utm_epsg = 'EPSG:32643'

    task_lower = task.lower()

    # Helper: safely get file_path, skip if None
    def _fp(key):
        v = retrieved_data.get(key, {})
        p = v.get("file_path") if isinstance(v, dict) else None
        return p if p and os.path.exists(p) else None

    # Detect which OSM layer to use; inline_fetch is used when pre-fetch failed
    inline_fetch = ""
    city_for_fetch = plan.get("city", "Mumbai")

    if _fp("osm_hospitals"):
        osm_path, count_col = _fp("osm_hospitals"), "hospital_count"
        rate_col, rate_factor = plan.get(
            "ranking_metric", "hospitals_per_100k"), 100000
    elif any(x in task_lower for x in ["hospital", "clinic", "medical", "healthcare"]):
        # Hospital retrieval failed — fetch inline inside sandbox
        osm_path = "/data/processed/osm_hospitals_inline.geojson"
        count_col, rate_col, rate_factor = "hospital_count", "hospitals_per_100k", 100000
        inline_fetch = f"""
import osmnx as ox
_hosp_tags = {{'amenity': ['hospital', 'clinic', 'doctors']}}
try:
    _hosp = ox.features_from_place('{city_for_fetch}', _hosp_tags).reset_index()
    _hosp['geometry'] = _hosp['geometry'].apply(make_valid)
    _hosp = _hosp.to_crs('EPSG:4326')
    _hosp.to_file('{osm_path}', driver='GeoJSON')
    print(f"Inline hospital fetch: {{len(_hosp)}} features")
except Exception as _e:
    print(f"Inline hospital fetch failed: {{_e}}")
    import geopandas as _gpd; import pandas as _pd
    _hosp = _gpd.GeoDataFrame(_pd.DataFrame(), geometry=[], crs='EPSG:4326')
    _hosp.to_file('{osm_path}', driver='GeoJSON')
"""
    elif _fp("osm_schools"):
        osm_path, count_col = _fp("osm_schools"), "school_count"
        rate_col, rate_factor = plan.get(
            "ranking_metric", "schools_per_100k"), 100000
    elif any(x in task_lower for x in ["school", "education", "university"]):
        osm_path = "/data/processed/osm_schools_inline.geojson"
        count_col, rate_col, rate_factor = "school_count", "schools_per_100k", 100000
        inline_fetch = f"""
import osmnx as ox
_sch_tags = {{'amenity': ['school','kindergarten','college','university']}}
try:
    _sch = ox.features_from_place('{city_for_fetch}', _sch_tags).reset_index()
    _sch['geometry'] = _sch['geometry'].apply(make_valid)
    _sch = _sch.to_crs('EPSG:4326')
    _sch.to_file('{osm_path}', driver='GeoJSON')
    print(f"Inline school fetch: {{len(_sch)}} features")
except Exception as _e:
    print(f"Inline school fetch failed: {{_e}}")
    import geopandas as _gpd; import pandas as _pd
    _gpd.GeoDataFrame(_pd.DataFrame(), geometry=[], crs='EPSG:4326').to_file('{osm_path}', driver='GeoJSON')
"""
    elif _fp("osm_transit"):
        osm_path, count_col = _fp("osm_transit"), "transit_count"
        rate_col, rate_factor = plan.get(
            "ranking_metric", "transit_per_100k"), 100000
    elif _fp("osm_greenspace"):
        osm_path, count_col = _fp("osm_greenspace"), "green_area_km2"
        rate_col, rate_factor = plan.get(
            "ranking_metric", "green_space_per_100k"), 100000
    elif _fp("osm_parks"):
        osm_path, count_col = _fp("osm_parks"), "park_count"
        rate_col, rate_factor = plan.get(
            "ranking_metric", "parks_per_100k"), 100000
    else:
        return {"success": False, "error": "No OSM layer found for per-capita analysis"}

    city = plan.get("city", "")
    ascending = any(x in task_lower for x in [
                    "least", "lowest", "fewest", "worst", "minimum", "poorest"])

    code = f"""
import geopandas as gpd, pandas as pd, rasterio
from rasterstats import zonal_stats
from shapely.validation import make_valid
import warnings; warnings.filterwarnings('ignore')
{inline_fetch}
def run_analysis():
    global result
    WGS84      = 'EPSG:4326'
    UTM        = '{utm_epsg}'
    RASTER_CRS = 'EPSG:{raster_epsg}'
    wards    = gpd.read_file('{b_path}')
    features = gpd.read_file('{osm_path}')
    wards['geometry']    = wards['geometry'].apply(make_valid)
    features['geometry'] = features['geometry'].apply(make_valid)
    if wards.crs is None:    wards    = wards.set_crs(WGS84)
    else:                    wards    = wards.to_crs(WGS84)
    if features.crs is None: features = features.set_crs(WGS84)
    else:                    features = features.to_crs(WGS84)
    name_col = next((c for c in wards.columns if c.lower() in ['ward_full','ward_name','name','ward','label','title','area_name','localname','name:en']), wards.columns[0])
    wards['ward_name'] = wards[name_col].apply(lambda x: x[0] if isinstance(x, list) else x).fillna('Unknown').astype(str).str.strip()
    print(f"Computing zonal stats (raster EPSG:{raster_epsg}, UTM:{utm_epsg})...")
    wards_for_stats = wards.to_crs(RASTER_CRS)
    stats = zonal_stats(wards_for_stats, '{tif_path}', stats=['sum'], nodata=-99999, all_touched=True)
    print(f"Raw stats sample: {{stats[:3]}}")
    pop_values = [s['sum'] if s is not None and s['sum'] is not None else 0 for s in stats]
    wards['population'] = pop_values
    valid_pop = wards[wards['population'] > 0]['population']
    print(f"Wards with valid population: {{len(valid_pop)}}/{{len(wards)}}")
    if len(valid_pop) == 0:
        print("WARNING: WorldPop no overlap — using area proxy")
        wards_utm_temp = wards.to_crs(UTM)
        wards['population'] = (wards_utm_temp.geometry.area / 1e4).round(0).astype(int).clip(lower=1)
    else:
        wards['population'] = wards['population'].clip(lower=1).round(0).astype(int)
    print(f"Population range: {{wards['population'].min()}} - {{wards['population'].max()}}")
    wards_utm    = wards.to_crs(UTM).copy()
    features_utm = features.to_crs(UTM).copy()
    wards_utm['ward_area_km2'] = (wards_utm.geometry.area / 1e6).round(3)
    wards_utm['ward_name']     = wards['ward_name'].values
    wards_utm['population']    = wards['population'].values
    poly_ratio    = features_utm.geometry.geom_type.isin(['Polygon','MultiPolygon']).mean()
    is_area_based = poly_ratio > 0.5
    print(f"Feature type: {{'area' if is_area_based else 'point'}}-based ({{poly_ratio:.0%}} polygons)")
    if is_area_based:
        feat_poly   = features_utm[features_utm.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
        intersected = gpd.overlay(feat_poly[['geometry']], wards_utm[['geometry','ward_name']], how='intersection')
        intersected['feature_area_km2'] = (intersected.geometry.area / 1e6).round(4)
        grouped = intersected.groupby('ward_name')['feature_area_km2'].sum().reset_index(name='metric_value')
        merged  = wards_utm.merge(grouped, on='ward_name', how='left')
        merged['metric_value'] = merged['metric_value'].fillna(0)
        merged['population']   = merged['population'].fillna(1).astype(int)
    else:
        features_utm['geometry'] = features_utm.geometry.apply(lambda g: g.centroid if g.geom_type in ['Polygon','MultiPolygon'] else g)
        joined = gpd.sjoin(features_utm[['geometry']], wards_utm[['geometry','ward_name','ward_area_km2']], how='left', predicate='intersects')
        joined = joined.drop(columns='geometry')
        counts = joined.groupby('ward_name').size().reset_index(name='metric_value')
        merged = wards_utm.merge(counts, on='ward_name', how='left')
        merged['metric_value'] = merged['metric_value'].fillna(0).astype(int)
        merged['population']   = merged['population'].fillna(1).astype(int)
    merged['{rate_col}']  = (merged['metric_value'] / merged['population'] * {rate_factor}).round(2).fillna(0)
    merged['{count_col}'] = merged['metric_value']
    ascending = {str(ascending)}
    merged = merged.drop_duplicates(subset='ward_name', keep='first')
    merged = merged[merged['ward_name'].str.lower().str.strip() != '{city}'.lower().strip()]
    merged = merged[merged['ward_name'].str.lower().str.strip() != '']
    merged = merged.sort_values('{rate_col}', ascending=ascending).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    keep = ['rank','ward_name','{count_col}','population','{rate_col}','ward_area_km2','geometry']
    result = merged[[c for c in keep if c in merged.columns]].to_crs(WGS84).reset_index(drop=True)
    result['geometry'] = result.geometry.simplify(0.0001, preserve_topology=True)
    print(f"Result: {{len(result)}} wards, top: {{result.iloc[0]['ward_name']}} = {{result.iloc[0]['{rate_col}']}}")

run_analysis()
"""
    sandbox_result = run_code_in_sandbox(code, timeout=600)
    if sandbox_result["success"]:
        validation = validate_analysis_output(sandbox_result["output"])
        if validation["valid"]:
            print("[Analysis] Per-capita analysis succeeded")
            return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
        error = validation["error"]
    else:
        error = sandbox_result["error"]
    print(f"[Analysis] Per-capita analysis failed: {error[:200]}")
    return {"success": False, "error": error}


# ── Main entry point ──────────────────────────────────────────────────────────

def run_analysis_for_task(task: str, retrieved_data: dict, plan: dict) -> dict:
    print(f"[Analysis] Starting: {task}")
    # GISclaw Feature 1: clear sandbox cache for each new task
    clear_sandbox_cache()
    domain_hint = plan.get("domain_hint", "") or ""
    hint_config = _parse_domain_hint(domain_hint)
    if hint_config:
        plan["_hint_config"] = hint_config

    for source, data in list(retrieved_data.items()):
        fpath = data.get('file_path', '')
        if fpath and not os.path.exists(fpath):
            print(f"[Analysis] Early flag: {source} file missing: {fpath}")
            retrieved_data[source] = {"error": f"File not found: {fpath}"}

    try:
        from memory.store import retrieve_similar
        similar = retrieve_similar(task, plan.get("city", ""), limit=1)
        stored_code = similar[0].get("working_code", "")
        stored_city = similar[0].get("city", "").lower()
        current_city = plan.get("city", "").lower()

        def _code_paths_valid(code):
            import re
            paths = re.findall(r"['\"](/data/[^'\"]+)['\"]", code)
            return all(os.path.exists(p) for p in paths) if paths else True
        # Skip memory reuse if generic engine handles this query type
        from agents.generic_engine import parse_query as _pq
        _generic_handles = _pq(task)[0] is not None or _pq(task)[5] is not None

        if (not _generic_handles and similar and stored_code and similar[0].get("similarity", 0) > 0.95 and
                stored_city == current_city and current_city in stored_code.lower() and
                _code_paths_valid(stored_code)):
            print(
                f"[Analysis] Reusing code from memory (similarity: {similar[0]['similarity']})")
            sandbox_result = run_code_in_sandbox(stored_code)
            if sandbox_result["success"]:
                validation = validate_analysis_output(sandbox_result["output"])
                if validation["valid"]:
                    print("[Analysis] Reused code succeeded")
                    return {"success": True, "code": stored_code, "output": sandbox_result["output"], "attempts": 1}
            print("[Analysis] Reused code failed — proceeding with normal analysis")
    except Exception as e:
        print(f"[Analysis] Memory lookup failed: {e}")

    task_lower_q = task.lower()
    is_per_capita = any(x in task_lower_q for x in [
        "per 100k", "per capita", "per 1000", "per population", "per resident", "per person", "per 100000"])
    is_vulnerability = any(x in task_lower_q for x in [
        "vulnerability", "vulnerable", "combining", "composite", "combined score", "risk score", "multi-factor"])
    is_flood = any(x in task_lower_q for x in [
                   "flood", "inundation", "water risk", "flood risk"]) and not is_per_capita
    print(
        f"[Analysis] Query type: per_capita={is_per_capita}, flood={is_flood}, vulnerability={is_vulnerability}")

    if is_mumbai_flood_query(task, plan) and not is_per_capita and not is_vulnerability:
        print("[Analysis] Mumbai flood benchmark path")
        sandbox_result = run_code_in_sandbox(MUMBAI_FLOOD_CODE)
        if sandbox_result["success"]:
            validation = validate_analysis_output(sandbox_result["output"])
            if validation["valid"]:
                print("[Analysis] Mumbai flood benchmark succeeded")
                return {"success": True, "code": MUMBAI_FLOOD_CODE, "output": sandbox_result["output"], "attempts": 1}

    upload_files = plan.get("upload_files", [])
    has_geojson = any(f["type"] == "geojson" for f in upload_files)
    has_csv = any(f["type"] == "csv" for f in upload_files)

    if has_geojson and has_csv:
        print("[Analysis] Multi-file path (GeoJSON + CSV)")
        result = run_multi_file_analysis(task, plan)
        if result["success"]:
            geo_check = validate_geographic_result(
                result.get("code", ""), plan.get("city", ""))
            if geo_check["valid"]:
                return result
            print(f"[Analysis] {geo_check['error']}")
        else:
            print(
                f"[Analysis] Multi-file path failed: {result.get('error', '')[:100]}")

    if plan.get("upload_path"):
        task_lower = task.lower()
        needs_osm_points = any(x in task_lower for x in [
                               "hospital", "clinic", "medical", "school", "education"])
        if needs_osm_points:
            print("[Analysis] Hybrid Upload+OSM path")
            result = run_hybrid_upload_osm_analysis(task, plan)
            if result["success"]:
                return result
            print("[Analysis] Hybrid path failed — falling back to single file")
        print("[Analysis] Single file path")
        result = run_uploaded_file_analysis(task, plan)
        if result["success"]:
            return result
        print(
            f"[Analysis] Single file path failed: {result.get('error', '')[:100]}")

    has_boundaries = ("osm_boundaries" in retrieved_data and retrieved_data["osm_boundaries"].get("file_path") and os.path.exists(
        str(retrieved_data["osm_boundaries"].get("file_path", "")))) or "mumbai" in plan.get("city", "").lower()
    if is_per_capita and has_boundaries:
        print("[Analysis] Per-capita analysis path")
        result = run_per_capita_analysis(task, plan, retrieved_data)
        if result["success"]:
            return result
        print(f"[Analysis] Per-capita failed: {result.get('error', '')[:100]}")

    # Mumbai general engine — ONLY fires for pure population/area/density queries
    # Block it for any query that involves specific OSM features
    _OSM_FEATURE_KEYWORDS = [
        'hospital', 'clinic', 'school', 'university', 'college',
        'park', 'green', 'road', 'street', 'transit', 'bus', 'metro',
        'shop', 'commercial', 'cycling', 'parking', 'water', 'flood',
        'per 100k', 'per capita', 'per population'
    ]
    _is_osm_feature_query = any(
        kw in task_lower_q for kw in _OSM_FEATURE_KEYWORDS)
    has_deterministic = any(k in retrieved_data for k in [
        'osm_hospitals', 'osm_roads', 'osm_water', 'osm_schools',
        'osm_greenspace', 'osm_transit', 'osm_commercial', 'osm_cycling', 'osm_parking'])
    if "mumbai" in plan.get("city", "").lower() and not is_per_capita and not has_deterministic and not _is_osm_feature_query:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print(
                "[Analysis] Composite pattern detected — skipping Mumbai general engine")
        else:
            print("[Analysis] Mumbai general engine path")
            result = run_mumbai_general_analysis(task, plan)
            if result["success"]:
                return result
        print("[Analysis] Mumbai general engine failed — falling back to Groq")

    # Hospital density — works with just osm_hospitals (boundaries optional)
    if "osm_hospitals" in retrieved_data and not is_per_capita:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print(
                "[Analysis] Composite pattern detected — skipping deterministic hospital")
        else:
            print("[Analysis] Deterministic hospital density path")
            result = run_hospital_density_analysis(task, plan, retrieved_data)
            if result["success"]:
                return result
            print(
                f"[Analysis] Hospital density failed: {result.get('error', '')[:100]}")

    if "osm_boundaries" in retrieved_data and "osm_roads" in retrieved_data and not is_per_capita:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print("[Analysis] Composite pattern detected — skipping deterministic road")
        else:
            print("[Analysis] Deterministic road density path")
            result = run_road_density_analysis(task, plan, retrieved_data)
            if result["success"]:
                return result
            print(
                f"[Analysis] Road density failed: {result.get('error', '')[:100]}")

    if "osm_boundaries" in retrieved_data and "osm_water" in retrieved_data and not is_per_capita:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print(
                "[Analysis] Composite pattern detected — skipping deterministic flood")
        else:
            print("[Analysis] Deterministic flood risk path")
            result = run_flood_risk_analysis(
                task, plan, retrieved_data, domain_hint=plan.get("domain_hint", ""))
            if result["success"]:
                return result
            print(
                f"[Analysis] Flood risk failed: {result.get('error', '')[:100]}")

    if "osm_boundaries" in retrieved_data and "osm_schools" in retrieved_data and not is_per_capita:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print(
                "[Analysis] Composite pattern detected — skipping deterministic schools")
        else:
            print("[Analysis] Deterministic schools path")
            result = run_schools_analysis(task, plan, retrieved_data)
            if result["success"]:
                return result
            print(
                f"[Analysis] Schools failed: {result.get('error', '')[:100]}")

    if "osm_greenspace" in retrieved_data and not is_per_capita:
        from agents.generic_engine import parse_query as _pq_check
        if _pq_check(task)[5]:
            print(
                "[Analysis] Composite pattern detected — skipping deterministic greenspace")
        else:
            print("[Analysis] Deterministic greenspace path")
            result = run_greenspace_analysis(task, plan, retrieved_data)
            if result["success"]:
                return result
            print(
                f"[Analysis] Greenspace failed: {result.get('error', '')[:100]}")

    # Early generic engine call — fires before the generic point/line density
    # paths so unseen features (pharmacies, cafes, gyms, etc.) are handled
    # deterministically rather than by the wrong osm_commercial/transit data.
    # Skipped when specific deterministic sources are already present so
    # hospital/road/water/schools/greenspace paths run unchanged.
    _has_specific_sources = any(k in retrieved_data for k in [
        "osm_hospitals", "osm_roads", "osm_water", "osm_schools", "osm_greenspace"])
    if not _has_specific_sources:
        try:
            from agents.generic_engine import run_generic_analysis as _run_generic_early
        except ImportError:
            try:
                from generic_engine import run_generic_analysis as _run_generic_early
            except ImportError:
                _run_generic_early = None
        if _run_generic_early is not None:
            print("[Analysis] Generic engine early path")
            _gr_early = _run_generic_early(task, plan, retrieved_data)
            if _gr_early.get("success"):
                return _gr_early
            print(
                f"[Analysis] Generic engine early: {_gr_early.get('error', '')[:100]}")

    if "osm_boundaries" in retrieved_data and "osm_transit" in retrieved_data:
        result = run_point_density_analysis(
            task, plan, retrieved_data, "osm_transit", "transit_density", "transit_count")
        if result["success"]:
            return result
    if "osm_boundaries" in retrieved_data and "osm_commercial" in retrieved_data:
        result = run_point_density_analysis(
            task, plan, retrieved_data, "osm_commercial", "commercial_density", "commercial_count")
        if result["success"]:
            return result
    if "osm_boundaries" in retrieved_data and "osm_cycling" in retrieved_data:
        result = run_linear_density_analysis(
            task, plan, retrieved_data, "osm_cycling", "cycling_density", "cycling_length_km")
        if result["success"]:
            return result
    if "osm_boundaries" in retrieved_data and "osm_parking" in retrieved_data:
        result = run_point_density_analysis(
            task, plan, retrieved_data, "osm_parking", "parking_density", "parking_count")
        if result["success"]:
            return result

    has_multi_source = any(k in retrieved_data for k in
                           ['osm_boundaries', 'osm_roads', 'osm_hospitals', 'osm_parks', 'osm_water', 'osm_schools',
                            'osm_transit', 'osm_commercial', 'osm_cycling', 'osm_parking'])
    if is_osmnx_query(plan) and not has_multi_source:
        print("[Analysis] OSMnx template path")
        code = generate_osmnx_analysis_code(task, plan)
        sandbox_result = run_code_in_sandbox(code)
        if sandbox_result["success"]:
            validation = validate_analysis_output(sandbox_result["output"])
            if validation["valid"]:
                geo_check = validate_geographic_result(
                    code, plan.get("city", ""))
                if geo_check["valid"]:
                    print("[Analysis] OSMnx template succeeded")
                    return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": 1}
                print(f"[Analysis] {geo_check['error']}")
            print(f"[Analysis] OSMnx template failed: {validation['error']}")
        else:
            print(
                f"[Analysis] OSMnx template failed: {sandbox_result['error'][:100]}")

    # Generic OSM engine — handles any feature-per-area query before the LLM loop.
    # Covers 50+ OSM feature types deterministically; 1 LLM call for anything else.
    # Fetches boundaries inline so it works even when retrieval timed out.
    try:
        from agents.generic_engine import run_generic_analysis as _run_generic
    except ImportError:
        try:
            from generic_engine import run_generic_analysis as _run_generic
        except ImportError:
            _run_generic = None
    if _run_generic is not None:
        print("[Analysis] Generic engine path")
        _gr = _run_generic(task, plan, retrieved_data)
        if _gr.get("success"):
            return _gr
        print(f"[Analysis] Generic engine: {_gr.get('error', '')[:100]}")

    data_schema = {
        name: {"description": f"GeoDataFrame for {name}", "output_info": data["output"],
               "file_path": data.get("file_path") or _extract_path(data.get("code", "")),
               "code_used": data.get("code", "")}
        for name, data in retrieved_data.items() if "output" in data
    }

    attempt_history = []
    submitted_codes = set()
    error = "No attempts made"

    for attempt in range(5):
        if attempt == 0:
            code = generate_analysis_code(
                task, data_schema, plan, retrieved_data)
            code = patch_result_assignment(code)
        else:
            history_text = "\n\n".join(
                f"Attempt {h['attempt']}:\n{h['code'][:400]}...\nError: {h['error'][:300]}"
                for h in attempt_history[-2:])
            last_error = attempt_history[-1]['error'] if attempt_history else ''
            fault_source = _classify_fault_source(last_error)
            print(f"[Analysis] Fault attributed to: {fault_source}")
            if len(attempt_history) >= 2 and attempt_history[-1]['error'][:100] == attempt_history[-2]['error'][:100]:
                error_hint = "TYPE B ATTENTIONAL ERROR: Do not repeat the same code. ACTUALLY implement the fix."
            elif fault_source == "data_ingestion":
                actual_paths = [v.get('file_path') for v in retrieved_data.values() if v.get(
                    'file_path') and os.path.exists(str(v.get('file_path', '')))]
                error_hint = f"DATA INGESTION ERROR: ONLY valid paths: {actual_paths}. Use ONLY these."
            elif fault_source == "schema_mismatch":
                error_hint = "SCHEMA MISMATCH: Use name_col = next((c for c in gdf.columns if c.lower() in ['name','ward','label','title']), gdf.columns[0]). Never hardcode column names."
            elif fault_source == "crs_mismatch":
                error_hint = "CRS ERROR: Set CRS after loading. Ensure same CRS before spatial operations."
            elif fault_source == "list_values":
                error_hint = "LIST VALUE ERROR: col = col.apply(lambda x: x[0] if isinstance(x, list) else x)"
            elif fault_source == "spatial_operation":
                error_hint = "SPATIAL OPERATION ERROR: Ensure same CRS. Reset index before sjoin."
            elif fault_source == "timeout":
                error_hint = "TIMEOUT: Use smaller bounding box."
            else:
                error_hint = f"CODE GENERATION ERROR: {last_error[:200]}. Fix this specific error."
            past_fixes = retrieve_similar_fix(last_error)
            memory_hint = ""
            if past_fixes:
                memory_hint = "\nPast fixes:\n" + \
                    "".join(
                        f"- Error: {pf['error_snippet'][:100]}\n  Fix: {pf['fix_code'][:200]}\n" for pf in past_fixes)
                print(
                    f"[ErrorMemory] Injecting {len(past_fixes)} past fix(es)")
            gtchain_section = ""
            if attempt >= 2:
                print(f"[GTChain] Re-planning on attempt {attempt+1}")
                workflow = generate_analysis_plan(task, data_schema, plan)
                if workflow:
                    gtchain_section = f"\nFollow this workflow EXACTLY:\n{workflow}\n"
            # GISclaw Feature 5: extract variable tracking from previous attempt output
            var_context = ""
            if attempt_history:
                prev_output = attempt_history[-1].get("output", "")
                if prev_output and "VARIABLES:" in prev_output:
                    import re as _re
                    var_match = _re.search(r"VARIABLES: (.+)", prev_output)
                    if var_match:
                        var_context = f"\nVariables created in last attempt: {var_match.group(1)}\n"
            fix_prompt = f"""Spatial analysis failed.
Task: "{task}", Plan: {json.dumps(plan, indent=2)}
{load_guidance()}
{_get_tool_descriptions(data_schema)}
{var_context}{gtchain_section}{error_hint}
{memory_hint}
Failed attempts:
{history_text}
Fix: global result first line, result=final_gdf last line, run_analysis() last line.
NEVER hardcode columns. drop geometry before groupby. range(1,len+1) for rank.
Auto-detect UTM: cx=wards.to_crs('EPSG:4326').geometry.unary_union.centroid.x; cy=wards.to_crs('EPSG:4326').geometry.unary_union.centroid.y; zone=int((cx+180)/6)+1; UTM=f'EPSG:{{32600+zone if cy>=0 else 32700+zone}}'
Output format contract: result must have rank(int), ward_name(str), metric(float), geometry(EPSG:4326). Cast numerics: pd.to_numeric(col, errors='coerce').
Return only Python code."""
            code = smart_chat(GIS_EXPERT_SYSTEM_PROMPT,
                              fix_prompt, use_groq=True)
            code = code.replace("```python", "").replace("```", "").replace(
                "~~~python", "").replace("~~~", "").strip()
            if code.startswith("python\n") or code.startswith("python "):
                code = code[6:].strip()

        code_hash = hash(code.strip())
        if code_hash in submitted_codes:
            # GISclaw context-aware deduplication (Section 4.3.2):
            # Allow retry if error changed — same code may succeed after timeout
            last_same = next((h for h in reversed(attempt_history)
                             if hash(h.get('code', '').strip()) == code_hash), None)
            if last_same and last_same.get('error', '')[:50] == error[:50]:
                print(
                    f"[Analysis] Attempt {attempt+1}/5 — duplicate code+error, skipping")
                error = "Duplicate code — LLM repeated a failed attempt"
                attempt_history.append(
                    {"attempt": attempt+1, "code": code, "error": error[:500], "output": ""})
                continue
            else:
                print(
                    f"[Analysis] Attempt {attempt+1}/5 — same code, different context — allowing retry")
        submitted_codes.add(code_hash)

        # GTChain Feature #4: self-check file paths before running sandbox
        path_check = validate_code_paths(code, retrieved_data)
        if not path_check["valid"]:
            print(
                f"[Analysis] Attempt {attempt+1}/5 — self-check failed: {path_check['error'][:100]}")
            error = path_check["error"]
            _lf_analysis_event(f"attempt_{attempt+1}_self_check_failed",
                               input={"code_preview": code[:300]},
                               output={"error": error[:300], "fault": "invented_file_path"})
            attempt_history.append(
                {"attempt": attempt+1, "code": code, "error": error[:500]})
            continue

        print(f"[Analysis] Attempt {attempt+1}/5")
        sandbox_result = run_code_in_sandbox(code)
        if sandbox_result["success"]:
            validation = validate_analysis_output(sandbox_result["output"])
            if validation["valid"]:
                city = plan.get("city", "")
                geo_check = validate_geographic_result(code, city)
                if not geo_check["valid"]:
                    print(f"[Analysis] {geo_check['error']}")
                    error = geo_check["error"]
                    _lf_analysis_event(f"attempt_{attempt+1}_geo_fail",
                                       input={"code_preview": code[:300]},
                                       output={"error": error[:300], "fault": "wrong_geometry"})
                else:
                    print(f"[Analysis] Succeeded on attempt {attempt+1}")
                    _lf_analysis_event(f"attempt_{attempt+1}_success",
                                       input={"code_preview": code[:300]},
                                       output={"rows": sandbox_result["output"][:200]})
                    if attempt > 0 and attempt_history:
                        store_error_fix(
                            attempt_history[-1]['error'], code, task_type=task)
                    return {"success": True, "code": code, "output": sandbox_result["output"], "attempts": attempt+1}
            error = validation["error"]
            _lf_analysis_event(f"attempt_{attempt+1}_validation_fail",
                               input={"code_preview": code[:300]},
                               output={"error": error[:200], "fault": _classify_fault_source(error)})
        else:
            error = sandbox_result["error"]
            _lf_analysis_event(f"attempt_{attempt+1}_sandbox_fail",
                               input={"code_preview": code[:300]},
                               output={"error": error[:300], "fault": _classify_fault_source(error)})
        attempt_history.append({"attempt": attempt+1, "code": code, "error": error[:500], "output": sandbox_result.get(
            "output", "") if isinstance(sandbox_result, dict) else ""})
        print(f"[Analysis] Attempt {attempt+1} failed: {error[:100]}")

    return {"success": False, "error": error, "attempts": attempt_history}


def generate_methodology_explanation(task: str, plan: dict, result: dict) -> str:
    output = result.get("output", "")
    code = result.get("code", "")
    city = plan.get("city", "")
    data_sources = []
    if "worldpop" in code.lower() or "rasterstats" in code.lower():
        data_sources.append("WorldPop 2020 population raster")
    if "osm" in code.lower() or "ox." in code:
        data_sources.append("OpenStreetMap")
    if "gpd.read_file" in code:
        data_sources.append("local boundary files")
    prompt = f"""In 2-3 sentences, explain how this GIS analysis was computed.
Be specific: data sources, spatial operation, metric formula. Plain prose, no bullets.
Task: "{task}", City: {city}, Sources: {data_sources}, Output: {output[:300]}
Write only the explanation."""
    try:
        return smart_chat("You are a GIS expert. Explain analyses concisely.", prompt, use_groq=True)
    except Exception:
        return f"Used OpenStreetMap data for {city} to compute spatial metrics per ward."


# ── Utilities ─────────────────────────────────────────────────────────────────

def _extract_path(code: str) -> str:
    for line in code.split("\n"):
        if "read_file(" in line and "/data/" in line:
            start = line.find("'") + 1
            end = line.rfind("'")
            if 0 < start < end:
                return line[start:end]
    return ""


# ── Feature #7: Startup WorldPop pre-download ────────────────────────────────

# Countries to pre-download on worker startup — covers most common query cities
STARTUP_WORLDPOP_COUNTRIES = ["IND", "GBR", "DEU",
                              "FRA", "NLD", "USA", "BRA", "NGA", "KEN", "AUS"]


def predownload_worldpop() -> None:
    """Feature #7 — Pre-download WorldPop rasters for common countries on worker
    startup so queries never wait for downloads mid-analysis.
    Only downloads if not already cached. Runs in background thread."""
    import threading

    def _download():
        import requests
        os.makedirs("/data/processed", exist_ok=True)
        for iso3 in STARTUP_WORLDPOP_COUNTRIES:
            try:
                # Check if any year already cached
                import glob
                existing = glob.glob(f"/data/processed/worldpop_{iso3}_*.tif")
                if existing:
                    print(
                        f"[Startup] WorldPop {iso3}: already cached ({existing[0]})")
                    continue
                # Download
                r = requests.get(
                    f"https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}",
                    timeout=30
                )
                data = r.json()["data"]
                latest = sorted(
                    data, key=lambda x: x["popyear"], reverse=True)[0]
                tif_url = latest["files"][0]
                tif_year = latest["popyear"]
                tif_path = f"/data/processed/worldpop_{iso3}_{tif_year}.tif"
                print(f"[Startup] Downloading WorldPop {iso3} ({tif_year})...")
                resp = requests.get(tif_url, timeout=600, stream=True)
                with open(tif_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(
                    f"[Startup] WorldPop {iso3} ready: {tif_path} ({os.path.getsize(tif_path)//1024//1024}MB)")
            except Exception as e:
                print(f"[Startup] WorldPop {iso3} download failed: {e}")

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    print(
        f"[Startup] WorldPop pre-download started for: {STARTUP_WORLDPOP_COUNTRIES}")
