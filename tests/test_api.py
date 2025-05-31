import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Attempt to import the app from the correct location
# This assumes that the tests will be run from the project root directory,
# and PYTHONPATH is set up (e.g. by using `pip install -e .` or `pytest` discovering the src module)
try:
    from src.api import app
    from src.agent import DocumentResearchAgent # Needed for isinstance checks or specific mocking
except ImportError:
    # Fallback for cases where src might not be directly in PYTHONPATH during linting/static analysis
    # or if the execution environment is different.
    # For actual test runs, the above should work if the project structure is standard.
    app = None
    DocumentResearchAgent = None


# Fixture for the TestClient
@pytest.fixture(scope="module")
def client():
    if app is None:
        pytest.fail("FastAPI app could not be imported. Ensure 'src' is in PYTHONPATH or project is installed correctly.")
    with TestClient(app) as c:
        yield c

def test_research_successful_with_filenames(client):
    mock_agent_instance = MagicMock(spec=DocumentResearchAgent)
    mock_agent_instance.run.return_value = "Successful research result with files."

    with patch('src.api.DocumentResearchAgent', return_value=mock_agent_instance) as mock_agent_class:
        response = client.post("/research", json={"query": "Test query", "filenames": ["file1.txt", "file2.txt"]})

        assert response.status_code == 200
        assert response.json() == {"response": "Successful research result with files."}
        mock_agent_class.assert_called_once() # Check if agent was initialized
        mock_agent_instance.run.assert_called_once_with("Test query", ["file1.txt", "file2.txt"])

def test_research_successful_without_filenames(client):
    mock_agent_instance = MagicMock(spec=DocumentResearchAgent)
    mock_agent_instance.run.return_value = "Successful research result without files."

    with patch('src.api.DocumentResearchAgent', return_value=mock_agent_instance) as mock_agent_class:
        response = client.post("/research", json={"query": "Test query without files"})

        assert response.status_code == 200
        assert response.json() == {"response": "Successful research result without files."}
        mock_agent_class.assert_called_once()
        mock_agent_instance.run.assert_called_once_with("Test query without files")

def test_research_missing_query(client):
    response = client.post("/research", json={"filenames": ["file1.txt"]})
    # FastAPI returns 422 for Pydantic validation errors
    assert response.status_code == 422
    # Optionally, check the detail of the error
    assert "query" in response.json()["detail"][0]["loc"]
    assert response.json()["detail"][0]["type"] == "missing"


def test_research_agent_run_exception(client):
    mock_agent_instance = MagicMock(spec=DocumentResearchAgent)
    mock_agent_instance.run.side_effect = Exception("Simulated agent error during run")

    with patch('src.api.DocumentResearchAgent', return_value=mock_agent_instance) as mock_agent_class:
        response = client.post("/research", json={"query": "Test query for agent error"})

        assert response.status_code == 500
        assert response.json() == {"detail": "An error occurred during the research process."}
        mock_agent_class.assert_called_once()
        mock_agent_instance.run.assert_called_once_with("Test query for agent error")

def test_research_agent_init_exception(client):
    with patch('src.api.DocumentResearchAgent', side_effect=Exception("Simulated agent initialization error")) as mock_agent_class:
        response = client.post("/research", json={"query": "Test query for init error"})

        assert response.status_code == 500
        assert response.json() == {"detail": "An error occurred during the research process."}
        mock_agent_class.assert_called_once()

# Example of how to run these tests (from project root):
# Ensure pytest and fastapi[all] (for TestClient) are installed
# PYTHONPATH=. pytest tests/test_api.py
