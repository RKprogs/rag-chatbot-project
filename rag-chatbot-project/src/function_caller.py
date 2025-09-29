import requests

def wiki_summary(topic: str) -> str:
    """Fetch a short Wikipedia summary using the REST summary endpoint."""
    safe_topic = topic.replace(' ', '_')
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get('extract')
        else:
            return None
    except Exception as e:
        return None
