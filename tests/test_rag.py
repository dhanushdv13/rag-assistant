"""Basic smoke tests"""

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_ask_without_ingest():
    resp = client.post("/ask", json={"query": "What is this?"})
    assert resp.status_code == 400
