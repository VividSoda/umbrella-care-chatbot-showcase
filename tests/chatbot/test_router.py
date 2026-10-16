import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.chatbot.router import router

URL = "/api/v1/chatbot/"


@pytest.fixture
def client(predictor) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.predictor = predictor
    return TestClient(app)


def test_returns_reply_in_documented_shape(client):
    response = client.post(URL, json={"message": "itching and a rash"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "message",
        "suggested",
        "specialization",
        "symptoms",
        "urgency",
        "ask",
    }
    assert body["symptoms"] == ["itching", "skin_rash"]


def test_symptoms_field_is_optional(client):
    assert client.post(URL, json={"message": "hello"}).status_code == 200


def test_previous_symptoms_round_trip(client):
    first = client.post(URL, json={"message": "I have itching"}).json()
    second = client.post(
        URL, json={"message": "and a rash", "symptoms": first["symptoms"]}
    ).json()
    assert second["symptoms"] == ["itching", "skin_rash"]
    assert second["specialization"] is not None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "x" * 501},
        {"message": "hi", "symptoms": "fever"},
        {"message": "hi", "symptoms": ["x" * 65]},
        {"message": "hi", "duration_days": -1},
        {"message": "hi", "duration_days": 3651},
        {"message": "hi", "severity": 0},
        {"message": "hi", "severity": 11},
        {"message": "hi", "age": -1},
        {"message": "hi", "age": 121},
        {"message": "hi", "sex": "robot"},
        {"message": "hi", "pregnant": "maybe"},
        {"message": "hi", "chronic_conditions": "asthma"},
        {"message": "hi", "chronic_conditions": ["x" * 65]},
        {"message": "hi", "chronic_conditions": [""]},
        {"message": "hi", "chronic_conditions": ["a"] * 21},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post(URL, json=payload).status_code == 422


def test_triage_fields_are_optional_and_accepted(client):
    payload = {
        "message": "itching and a rash",
        "duration_days": 3,
        "severity": 5,
        "age": 30,
        "sex": "female",
        "pregnant": False,
        "chronic_conditions": ["asthma"],
    }
    assert client.post(URL, json=payload).status_code == 200


def test_urgency_is_sent_as_text(client):
    body = client.post(URL, json={"message": "itching and a rash"}).json()
    assert body["urgency"] == "self_care"
    assert body["ask"] is None
