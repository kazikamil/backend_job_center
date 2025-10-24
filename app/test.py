import requests

BONS_AI_URL = "https://51719bdd00:2af2d024007b80966f1f@courteous-plum-1hr1qtcp.us-east-1.bonsaisearch.net"
res = requests.get(f"{BONS_AI_URL}/jobs2/_search", headers={"Content-Type": "application/json"})
print(res.json())
