from pystac_client import Client
import planetary_computer

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("Connected!")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[72.8, 18.9, 73.0, 19.2],  # Mumbai
    limit=1,
)

items = list(search.items())

print("Items found:", len(items))
if items:
    print(items[0].id)
