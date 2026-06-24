import requests
bbox = '35.077,138.964,36.277,140.564'
H = {'User-Agent': 'GoAI/1.0', 'Accept': 'application/json'}
for lvl in ['5', '6', '7', '8', '9', '10', '11']:
    q = '[out:json][timeout:60];relation["boundary"="administrative"]["admin_level"="%s"](%s);out tags;' % (
        lvl, bbox)
    try:
        r = requests.post('https://overpass-api.de/api/interpreter',
                          data={'data': q}, headers=H, timeout=120)
        if r.status_code != 200:
            print('lvl=%s status=%d first=%r' %
                  (lvl, r.status_code, r.text[:150]))
            continue
        d = r.json()
        elems = d.get('elements', [])
        names = [e.get('tags', {}).get('name:en') or e.get(
            'tags', {}).get('name', '?') for e in elems[:8]]
        print('lvl=%s count=%d sample=%s' % (lvl, len(elems), names))
    except Exception as ex:
        print('lvl=%s failed: %s' % (lvl, str(ex)[:120]))
