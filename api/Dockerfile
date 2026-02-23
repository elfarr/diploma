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
        "uvicorn[standard]" \
        pydantic \
        pydantic-settings \
        python-dotenv \
        joblib \
        numpy \
        pandas \
        onnxruntime

COPY backend/__init__.py /app/backend/__init__.py
COPY backend/api /app/backend/api
COPY backend/models /app/backend/models
COPY src /app/src

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
