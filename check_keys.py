import os
import requests
keys = [os.getenv('GROQ_API_KEY', '')] + \
    [os.getenv(f'GROQ_API_KEY_{i}', '') for i in range(2, 8)]
keys = [k.strip() for k in keys if k.strip()]
print('Total keys:', len(keys))
for k in keys:
    try:
        r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization': 'Bearer '+k, 'Content-Type': 'application/json'}, json={
                          'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1}, timeout=10)
        d = r.json()
        if 'error' in d:
            print(k[-6:], 'EXHAUSTED')
        else:
            print(k[-6:], 'OK')
    except Exception as e:
        print(k[-6:], 'ERROR:', str(e)[:40])
