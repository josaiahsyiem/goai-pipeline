import requests
r = requests.post(
    'https://overpass-api.de/api/interpreter',
    data={
        'data': '[out:json];node[amenity=hospital](12.83,77.46,13.14,77.78);out 3;'},
    headers={'User-Agent': 'GoAI/1.0'},
    timeout=30
)
print('status:', r.status_code)
print('elements:', len(r.json().get('elements', [])))
