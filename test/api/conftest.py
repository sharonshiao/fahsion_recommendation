"""
Shared test fixtures for API tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="function")
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Fixture to mock the model loading."""

    class MockModel:
        def predict_scores(self, data):
            return [0.5]  # Return a dummy prediction

    def mock_load_model(*args, **kwargs):
        model = MockModel()
        # Set the model directly on the app instance
        app.state.model = model
        return model

    # Mock the model loading function
    monkeypatch.setattr("app.main.load_ranker_model", mock_load_model)

    # Ensure model is loaded before each test
    mock_load_model()

    yield

    # Clean up after each test
    if hasattr(app.state, "model"):
        del app.state.model
