FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        pydantic-settings \
        python-dotenv \
        joblib \
        numpy \
        onnxruntime

COPY app/backend/__init__.py /app/backend/__init__.py
COPY app/backend/api /app/backend/api
COPY app/backend/models /app/backend/models
COPY app/src /app/src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8080}/healthz || exit 1

CMD ["sh", "-c", "python -m uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
