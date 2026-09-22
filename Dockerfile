FROM python:3.11-slim

# rembg/onnxruntime need these at runtime; Pillow needs libjpeg/zlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/app/.u2net

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ~180MB u2net_human_seg weights into the image. Without this, the
# first user to hit /api/segment after every deploy waits for the download.
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

COPY . .

# Railway injects PORT; default to 8000 for plain `docker run`.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
