FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Masks for the built-in campaigns are precomputed and committed, so there is
# no segmentation model to download or run here — see backend/precomputed_masks.py.

# The host injects PORT (Render, Railway, Fly); default to 8000 for `docker run`.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
