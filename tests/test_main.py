import joblib
import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app

CHAT_URL = "/api/v1/chatbot/"


@pytest.fixture
def model_file(tmp_path, predictor):
    """A bundle in the same format ml/train.py writes."""
    path = tmp_path / "model.joblib"
    joblib.dump({"model": predictor._model, "symptoms": predictor.symptoms}, path)
    return path


def settings_for(model_file, **overrides) -> Settings:
    return Settings(_env_file=None, model_path=model_file, **overrides)


@pytest.fixture
def client(model_file):
    # The context manager runs the lifespan, which loads the model.
    with TestClient(create_app(settings_for(model_file))) as client:
        yield client


def test_chat_works_after_startup(client):
    response = client.post(CHAT_URL, json={"message": "itching and a rash"})
    assert response.status_code == 200
    assert response.json()["specialization"] == "Dermatologist"


def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ok"}


def test_not_ready_before_model_is_loaded(model_file):
    # No `with`: the lifespan has not run, so nothing is loaded.
    client = TestClient(create_app(settings_for(model_file)))
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503


def test_missing_model_stops_startup(tmp_path):
    app = create_app(settings_for(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_unhandled_error_returns_generic_500(model_file):
    app = create_app(settings_for(model_file))

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")
    assert response.status_code == 500
    assert "secret" not in response.text


def test_cors_preflight_allows_listed_origin(model_file):
    settings = settings_for(model_file, allowed_origins=["https://app.example"])
    with TestClient(create_app(settings)) as client:
        response = client.options(
            CHAT_URL,
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        blocked = client.options(
            CHAT_URL,
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in blocked.headers


def test_docs_are_hidden_in_production(model_file):
    with TestClient(
        create_app(settings_for(model_file, environment="production"))
    ) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
