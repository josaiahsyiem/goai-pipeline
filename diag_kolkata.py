import requests
H = {'User-Agent': 'GoAI/1.0'}
# Use wider bbox
s, n, w, e = 22.45, 22.63, 88.23, 88.47
for lvl in ['8', '9', '10']:
    q = '[out:json][timeout:90];relation["boundary"="administrative"]["admin_level"="%s"](%.3f,%.3f,%.3f,%.3f);out tags;' % (
        lvl, s, w, n, e)
    r = requests.post('https://overpass-api.de/api/interpreter',
                      data={'data': q}, headers=H, timeout=120)
    if r.status_code != 200:
        print('lvl=%s status=%d' % (lvl, r.status_code))
        continue
    elems = r.json().get('elements', [])
    names = [e.get('tags', {}).get('name', '?') for e in elems[:5]]
    print('lvl=%s count=%d sample=%s' % (lvl, len(elems), names))
