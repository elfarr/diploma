FROM debian:bookworm-slim AS frontend-builder

ARG FLUTTER_VERSION=3.41.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git unzip xz-utils \
    libstdc++6 libglib2.0-0 libnss3 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 flutter
USER flutter
WORKDIR /home/flutter

RUN curl -fsSL -o flutter.tar.xz \
    "https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz" \
    && tar -xf flutter.tar.xz \
    && rm flutter.tar.xz

ENV PATH="/home/flutter/flutter/bin:/home/flutter/flutter/bin/cache/dart-sdk/bin:${PATH}"
ENV HOME=/home/flutter
ENV PUB_CACHE=/home/flutter/.pub-cache

WORKDIR /build/frontend
COPY --chown=flutter:flutter frontend/pubspec.yaml frontend/pubspec.lock* ./
RUN flutter pub get
COPY --chown=flutter:flutter frontend/ ./
RUN flutter build web --release --pwa-strategy=none


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY api/app/backend/__init__.py /app/backend/__init__.py
COPY api/app/backend/api /app/backend/api
COPY api/app/backend/models /app/backend/models
COPY api/app/src /app/src

COPY --from=frontend-builder /build/frontend/build/web /app/static

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

CMD ["python", "-m", "uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
