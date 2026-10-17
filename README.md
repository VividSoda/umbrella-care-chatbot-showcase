# UmbrellaCare Chatbot Backend

A small, stateless FastAPI service behind the UmbrellaCare Flutter app's chatbot. It takes symptoms the user typed, predicts the most likely condition with a scikit-learn model, and returns a reply, follow-up questions, and the medical specialization to recommend. It uses no external API keys.

> This is not a medical diagnosis tool. Every reply carries a disclaimer telling the user to consult a doctor.

**About this project.** A personal portfolio project: a symptom chatbot backend with a Naive Bayes model, information-gain follow-up questions, rule-based urgency triage and an honest evaluation of the model (see [Evaluation](#evaluation)). The design decisions, the evaluation and the safety rules are mine; I built it with AI coding assistance (Claude Code) and reviewed and tested what it produced. The training data is synthetic, so this is a demonstration, not a clinical tool.

**Status.** Working end to end with the [Flutter app](https://github.com/VividSoda/UmbrellaCare). Next steps are in the [roadmap](docs/roadmap.md): better doctor ranking and booking from the chat, and a clinician review of the urgency rules.

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
  evaluate_model.py  honest accuracy and confidence evaluation
  evaluate_questions.py  compares follow-up question strategies
  data/              training CSVs
artifacts/
  model.joblib       output of train.py, deployed with the code
tests/               mirrors src/
docs/
  roadmap.md         planned improvements
  model_evaluation.md  how the model was evaluated and why
  learning/          my study notes (not needed to run the project)
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
{
  "message": "I have a headache and feel sick",
  "symptoms": ["mild_fever"],
  "duration_days": 3,
  "severity": 6,
  "age": 34,
  "sex": "female",
  "pregnant": false,
  "chronic_conditions": ["asthma"],
  "asked": ["duration"],
  "answering": "duration"
}
```

- `message` is required, 1–500 characters.
- `symptoms` is optional. Send back the `symptoms` array from the previous response unchanged, since the server keeps no history. These are model symptom names such as `mild_fever`, not free text. Names the model does not know are ignored.
- Triage fields, all optional. Send only what the user answered or agreed to share; a missing field never makes the result safer.
  - `duration_days`: 0–3650.
  - `severity`: 1–10.
  - `age`: 0–120.
  - `sex`: `male`, `female` or `other`.
  - `pregnant`: boolean.
  - `chronic_conditions`: up to 20 short strings.
- `asked` lists the triage questions (`duration`, `severity`) already asked, copied from earlier responses' `ask`. The server uses it to avoid asking twice, and a skipped question stays skipped.
- `answering` is set to `duration` or `severity` when this message answers (or skips) that question. The message then needs no symptom in it, and the reply is refreshed with the new values.

Response:

```json
{
  "message": "Based on your symptoms this could be Migraine. Other possibilities: Hypertension or Vertigo. This can often be managed at home, but see a doctor if it gets worse. On a scale of 1 to 10, how severe is it? This is not a diagnosis, please consult a doctor.",
  "suggested": ["dizziness", "nausea", "vomiting"],
  "specialization": "Neurologist",
  "symptoms": ["mild_fever", "headache", "nausea"],
  "urgency": "self_care",
  "ask": "severity"
}
```

- `suggested` are quick-reply chips: the symptoms whose answer would best tell the likely diseases apart. Sending a chip back as the next `message` is understood as-is. It is empty once the top disease is clearly ahead.
- `specialization` is one of the Flutter app's doctor categories, or `null` while there is not enough to go on.
- `symptoms` is everything recognized so far. Store it and send it with the next request.
- `urgency` is `self_care`, `see_doctor_soon` or `emergency`.
- `ask` is `duration`, `severity` or `null`: the triage question this reply asks. Show a matching input, then send the value in the next request and add the question to `asked`.

If fewer than the minimum number of symptoms are recognized, or the best guess is below the confidence threshold, `specialization` is `null` and the message asks for more detail.

### Urgency

Urgency comes from plain rules in `src/chatbot/triage.py`, never from a model or an LLM. The highest level any rule calls for wins, and nothing lowers it:

- An emergency phrase in the message (for example "seizure", "not breathing") gives `emergency`. It is checked first, on the message alone, and the model never runs.
- The predicted disease sets a floor (`DISEASE_URGENCY` in `constants.py`). A disease not listed there defaults to `see_doctor_soon`.
- A red-flag phrase ("chest pain", "blood in stool"), severity of 7 or more, or symptoms lasting 7 days or more give at least `see_doctor_soon`.
- Age under 5 or over 65, pregnancy, or a chronic condition also give at least `see_doctor_soon`, but never `emergency` on their own.

The questions are asked at most once each, after a prediction the bot trusts, and never during an emergency. Answering them is optional and never blocks the reply. The disease floors, red-flag phrases and thresholds are a first draft and need clinician review before release.

## Deploy

The server needs only `src/` and `artifacts/model.joblib`. The `Dockerfile` packages exactly that, so any container host works. These steps use Google Cloud Run.

### Test the image locally

```bash
docker build --platform linux/amd64 -t umbrella-chatbot .
docker run --rm -e PORT=8080 -p 8080:8080 umbrella-chatbot
curl localhost:8080/health/ready
```

### Deploy to Cloud Run

One-time setup:

1. In the [Cloud console](https://console.cloud.google.com), create a project and attach a billing account. Cloud Run needs one even for the free tier. Add a budget alert (Billing → Budgets) so a surprise never costs more than you expect.
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Deploy (and redeploy after any change) from this directory:

```bash
gcloud run deploy umbrella-care-chatbot \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 30 \
  --set-env-vars ENVIRONMENT=production
```

The command prints a `Service URL`. Put it in the Flutter app's `.env`, keeping the path and trailing slash:

```
CHATBOT_API_URL=https://umbrella-care-chatbot-xxxxx-el.a.run.app/api/v1/chatbot/
```

Notes:

- `--min-instances 0` scales to zero when idle, which keeps it inside the free tier. The first request after a pause waits for a cold start (a few seconds; the model is tiny). `--min-instances 1` removes that wait but is billed continuously, so it is not free.
- `--max-instances 2` caps cost and abuse, since the endpoint is public.
- Region: `asia-south1` (Mumbai) is the closest region to Nepal. It is on Tier 2 pricing, which only matters beyond the monthly free allowance; the free allowance is reported to apply there too, so confirm the current numbers on the [pricing page](https://cloud.google.com/run/pricing). `us-central1` is a Tier 1 alternative with higher latency. A service cannot change region: deploy a new one instead.
- `ENVIRONMENT=production` hides `/docs`. Add `ALLOWED_ORIGINS` (comma-separated) only if you serve Flutter web.
- Keep the app's HTTP timeout generous anyway (the app uses 60 s).
- After retraining, commit the new `artifacts/model.joblib` and redeploy. scikit-learn must stay pinned to the version that trained it.

### Deploy on merge (CI/CD)

`.github/workflows/ci-cd.yml` lints and tests every pull request. A merge to `main` also builds the image, pushes it to Artifact Registry tagged with the commit SHA, deploys it to Cloud Run, and smoke-tests `/health/ready`. To roll back, redeploy an earlier SHA's image from the Cloud Run console.

GitHub authenticates to Google with Workload Identity Federation, so no key or secret is stored in GitHub. Do this setup once (the values match the workflow's `env` block):

```bash
P=umbrellacare-d6eb0
N=707329620333                                  # project number
REGION=asia-south1
REPO=VividSoda/umbrella-care-chatbot-backend    # GitHub owner/repo
SA=github-deployer@$P.iam.gserviceaccount.com

# 1. A service account that only the CI job can impersonate
gcloud iam service-accounts create github-deployer --project $P \
  --display-name "GitHub Actions deployer"

# 2. Trust GitHub's tokens, but only from this one repository
gcloud iam workload-identity-pools create github --project $P \
  --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc umbrella-care-chatbot-backend \
  --project $P --location global --workload-identity-pool github \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository == '$REPO'"
gcloud iam service-accounts add-iam-policy-binding $SA --project $P \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$N/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

# 3. Least privilege: push images, update this one service, act as its runtime account
gcloud artifacts repositories add-iam-policy-binding cloud-run-source-deploy \
  --project $P --location $REGION --role roles/artifactregistry.writer --member serviceAccount:$SA
gcloud run services add-iam-policy-binding umbrella-care-chatbot \
  --project $P --region $REGION --role roles/run.developer --member serviceAccount:$SA
gcloud iam service-accounts add-iam-policy-binding $N-compute@developer.gserviceaccount.com \
  --project $P --role roles/iam.serviceAccountUser --member serviceAccount:$SA
```

Then, in GitHub, protect `main` (Settings → Branches → require the `test` check to pass before merging) so a failing PR can't reach the deploy.

If the first deploy fails with "Permission denied to use service", grant the account `roles/serviceusage.serviceUsageConsumer` on the project.

## Development

- Lint and format: `ruff check . && ruff format .`
- Tests: `pip install -r requirements-dev.txt`, then `pytest`
- Health probes for the host: `GET /health/live` (process up) and `GET /health/ready` (model loaded).
- Set `ENVIRONMENT=production` on the host to turn off `/docs` and `/openapi.json`.
- Review against the FastAPI checklist: run `/fast-review` in Claude Code from this directory.
- Keep `.env` out of git. List every variable in `.env.example`.

## Evaluation

How good is the model when a user names only a few symptoms? The usual 98%
figure gives the model all 132 symptoms of a patient, which a chat user never
does. `ml/evaluate_model.py` hides all but `k` of each test patient's symptoms,
using 5-fold cross-validation so the model never sees the patient it is tested
on. Run it with `cd ml && python evaluate_model.py`.

**Accuracy by symptoms named.** "Current" treated every unmentioned symptom as
absent (the old `Predictor.predict`). "Partial" treats them as unknown, which is
what the bot does now.

| Symptoms named | Current | Partial |
|---|---|---|
| 2 | 43.6% | 69.1% |
| 3 | 62.5% | 83.8% |
| 5 | 80.7% | 93.2% |

**Can the confidence be trusted?** With two symptoms, "partial" is
underconfident but ranks cases well: guesses it rated 0.3 to 0.5 were right 92%
of the time. "Current" was overconfident: rated 0.7 to 0.9, right 65%.

**Thresholds** (two symptoms, partial):

| Answer only if confidence is at least | Bot answers | Right when it answers |
|---|---|---|
| 0.2 | 78% | 81% |
| 0.3 | 47% | 93% |
| 0.4 | 22% | 99% |
| 0.5 | 6% | 100% |

`MIN_CONFIDENCE = 0.3`: favours accuracy while still answering about half the
time on two symptoms. `STOP_CONFIDENCE = 0.6`: with three or more symptoms the
top guess was right every time it reached this level.

Model comparison, 5-fold CV on all rows: BernoulliNB 97.7%, random forest
100%. Both are near the ceiling, so the data is the limit, and the forest would
break the question picker, which reads Naive Bayes internals.

> The dataset is synthetic, with only 304 unique rows, so these numbers
> describe this data, not real patients. Trust the gaps between methods more
> than the absolute percentages. The full method is in
> [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Roadmap

Planned improvements, in the order to tackle them, are in [`docs/roadmap.md`](docs/roadmap.md).

## License

[MIT](LICENSE)
