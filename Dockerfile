# Same Python minor as development: scikit-learn must match the version that
# trained artifacts/model.joblib (pinned in requirements.txt).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY artifacts/model.joblib ./artifacts/model.joblib

RUN useradd --system --no-create-home app
USER app

# Cloud Run sets $PORT (8080 by default). `exec` makes uvicorn PID 1 so it
# receives SIGTERM and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
