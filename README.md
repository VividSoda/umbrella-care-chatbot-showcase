# UmbrellaCare Chatbot Backend

A small, stateless FastAPI service behind the UmbrellaCare Flutter app's chatbot. It takes symptoms the user typed, predicts the most likely condition with a scikit-learn model, and returns a reply, follow-up questions, and the medical specialization to recommend. It uses no external API keys.

> This is not a medical diagnosis tool. Every reply carries a disclaimer telling the user to consult a doctor.

## How it works

```
Flutter app ──POST {message, symptoms}──▶ FastAPI ──▶ extract symptoms
                                                     └▶ model predicts disease
                                                     └▶ disease → specialization
Flutter app ◀── {message, suggested, specialization} ──┘
Flutter app ──▶ queries Firestore for doctors with that specialization
```

- **Stateless.** The server stores nothing between requests. The app sends the symptoms collected so far with each request.
- **Doctor lookup and chat history live in the app and Firestore**, not here. The server needs no Firebase credentials.
- **Training happens offline** on your machine. The server only loads the saved model and predicts.

## Project layout

```
src/
  main.py            app factory, lifespan (loads the model once), router includes
  config.py          settings from environment variables
  exceptions.py      app-wide exceptions and handlers
  models.py          shared Pydantic base model
  chatbot/
    router.py        POST /api/v1/chatbot/ (HTTP only)
    schemas.py       request / response models
    service.py       extract symptoms → predict → build reply
    predictor.py     wraps the model: symptoms in, probabilities out
    symptoms.py      free text → symptom names
    constants.py     disease → specialization map, thresholds
    dependencies.py  Depends() providing the loaded predictor
    exceptions.py    chatbot-specific errors
ml/
  train.py           trains the model (run locally, not on the server)
  data/              training CSVs
artifacts/
  model.joblib       output of train.py, deployed with the code
tests/               mirrors src/
```

Layout follows `.claude/skills/fast-review/checklist.md`: organized by domain, thin `main.py` and routers, logic in services.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Train the model

1. Download `Training.csv` and `Testing.csv` from the Kaggle dataset
   [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
   (132 symptom columns plus `prognosis`, 41 diseases) and place them in `ml/data/`.
   The CSVs are git-ignored, so each developer downloads them. Check the dataset's license
   on its Kaggle page. The data is synthetic, so predictions are a demo, not medical advice.
2. Run the training script:

```bash
python ml/train.py
```

This writes `artifacts/model.joblib`. Retrain only when the data changes, and redeploy afterwards.

The scikit-learn version used for training must match the one the server runs, or loading the model may warn or fail. Keep versions pinned in `requirements.txt`.

## Run locally

```bash
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

On a physical phone, use your computer's LAN IP in the app's `.env` instead of `127.0.0.1`:

```
CHATBOT_API_URL=http://<your-lan-ip>:8000/api/v1/chatbot/
```

## API

`POST /api/v1/chatbot/`

Request:

```json
{ "message": "I have a headache and feel sick", "symptoms": ["mild_fever"] }
```

- `message` is required, 1–500 characters.
- `symptoms` is optional. Send back the `symptoms` array from the previous response unchanged, since the server keeps no history. These are model symptom names such as `mild_fever`, not free text. Names the model does not know are ignored.

Response:

```json
{
  "message": "Based on your symptoms this could be Migraine. Other possibilities: Hypertension or Vertigo. This is not a diagnosis, please consult a doctor.",
  "suggested": ["dizziness", "nausea", "vomiting"],
  "specialization": "Neurologist",
  "symptoms": ["mild_fever", "headache", "nausea"]
}
```

- `suggested` are quick-reply chips: symptoms common in the predicted disease that the user has not mentioned. Sending a chip back as the next `message` is understood as-is.
- `specialization` is one of the Flutter app's doctor categories, or `null` while there is not enough to go on.
- `symptoms` is everything recognized so far. Store it and send it with the next request.

If fewer than the minimum number of symptoms are recognized, or the best guess is below the confidence threshold, `specialization` is `null` and the message asks for more detail.

## Deploy

The server needs only the code and `artifacts/model.joblib`. Any host that runs a Python web service works (Render, Google Cloud Run, Hugging Face Spaces).

Start command:

```bash
uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

Set the host's Python version to match your local one. Free tiers may sleep when idle, so the first request after a pause can take up to a minute. Set a generous timeout in the app's HTTP client. Check each host's current free-tier limits before relying on it.

Then set `CHATBOT_API_URL` in the Flutter app's `.env` to the deployed URL.

## Development

- Lint and format: `ruff check . && ruff format .`
- Tests: `pip install -r requirements-dev.txt`, then `pytest`
- Health probes for the host: `GET /health/live` (process up) and `GET /health/ready` (model loaded).
- Set `ENVIRONMENT=production` on the host to turn off `/docs` and `/openapi.json`.
- Review against the FastAPI checklist: run `/fast-review` in Claude Code from this directory.
- Keep `.env` out of git. List every variable in `.env.example`.
