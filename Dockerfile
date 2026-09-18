FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# No secrets are baked in: LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_BASE_URL
# must be supplied at runtime via --env-file or the hosting platform's env vars.
# Respects a platform-provided $PORT (Render/Railway/etc.), defaulting to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
