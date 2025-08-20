import json
from server.app import app

def test_api_predict_smoke():
    client = app.test_client()
    r = client.post("/api/predict", json={"text": "I absolutely love this app!"})
    assert r.status_code == 200
    data = r.get_json()
    assert "label" in data and "probability" in data