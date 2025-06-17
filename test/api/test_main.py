"""
Tests for the FastAPI application endpoints.
"""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_readiness_check(client: TestClient, mock_model):
    """Test the readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
