from pathlib import Path

from src.config import PROJECT_ROOT, Settings


def make(**env: str) -> Settings:
    return Settings(_env_file=None, **env)


def test_defaults(monkeypatch):
    for key in ("ENVIRONMENT", "MODEL_PATH", "ALLOWED_ORIGINS"):
        monkeypatch.delenv(key, raising=False)
    settings = make()
    assert settings.environment == "local"
    assert settings.allowed_origins == ["*"]
    assert settings.docs_enabled


def test_relative_model_path_resolves_from_project_root(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "artifacts/model.joblib")
    assert make().model_path == PROJECT_ROOT / "artifacts" / "model.joblib"


def test_absolute_model_path_is_kept(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/srv/model.joblib")
    assert make().model_path == Path("/srv/model.joblib")


def test_origins_are_split_on_commas(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example ,")
    assert make().allowed_origins == ["https://a.example", "https://b.example"]


def test_production_turns_docs_off(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert not make().docs_enabled
