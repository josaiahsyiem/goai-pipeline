"""
add_population.py
-----------------
One-time data preparation script.
Adds census population estimates to the Mumbai ward shapefile.
Run once from the project root: python add_population.py
"""

import sys
import geopandas as gpd

WARD_POPULATION = {
    "Colaba":          94000,
    "Dongri":         147000,
    "Pydhonie":        90000,
    "Girgaon":         94000,
    "Byculla":        160000,
    "Sion North":     196000,
    "Sion South":     138000,
    "Dharavi":        166000,
    "Matunga":         85000,
    "Kurla":          450000,
    "Govandi":        441000,
    "Chembur West":   298000,
    "Ghatkopar":      388000,
    "Bhandup":        273000,
    "Mulund":         200000,
    "Bandra East":    100000,
    "Bandra West":     77000,
    "Andheri East":   451000,
    "Andheri West":   384000,
    "Malad":          410000,
    "Goregaon":       363000,
    "Borivali Central": 310000,
    "Dahisar":        349000,
    "Borivali West":  285000,
}

WARD_FILE = "data/mumbai_ward_shapefile/Mumbai_wards.geojson"
FALLBACK_POPULATION = 200000


def main():
    gdf = gpd.read_file(WARD_FILE)
    print(f"Loaded {len(gdf)} wards from {WARD_FILE}")
    print(f"Ward names: {gdf['ward_full'].tolist()}\n")

    gdf["population"] = gdf["ward_full"].map(WARD_POPULATION)

    unmatched = gdf[gdf["population"].isna()]["ward_full"].tolist()
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} ward(s) not found in population table:")
        for ward in unmatched:
            print(f"  - {ward}")
        print(f"Filling with fallback population: {FALLBACK_POPULATION:,}\n")
        gdf["population"] = gdf["population"].fillna(FALLBACK_POPULATION)
    else:
        print(f"All {len(gdf)} wards matched successfully.\n")

    gdf["population"] = gdf["population"].astype(int)
    gdf.to_file(WARD_FILE, driver="GeoJSON")

    print(f"Saved to {WARD_FILE}")
    print(f"Columns: {list(gdf.columns)}\n")
    print(gdf[["ward_full", "population"]].to_string(index=False))


if __name__ == "__main__":
    main()
