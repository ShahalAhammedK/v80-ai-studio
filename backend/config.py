import os

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2.5-sunburst")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5.6-luna")

# Optional gateway for CHAT_MODEL calls (OpenAI-compatible Chat Completions API).
# When EXPLABS_API_KEY is unset, gender detection is simply disabled (soft-fail) —
# this feature never falls back to calling OPENAI_API_KEY directly.
EXPLABS_API_KEY = os.environ.get("EXPLABS_API_KEY", "")
EXPLABS_BASE_URL = os.environ.get("EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1")

# Image editing provider: "openai" (images.edit, mask-based) or "gemini"
# (generate_content, prompt-driven — no mask param, so fidelity outside the
# person relies entirely on the server-side pixel composite in image_utils.py).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "").strip().lower()
if IMAGE_PROVIDER not in ("openai", "gemini"):
    IMAGE_PROVIDER = "gemini" if GEMINI_API_KEY else "openai"

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "12"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10"))

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

MIN_DIMENSION = 256
MAX_DIMENSION = 4096

# gpt-image-* only accepts these output sizes.
SUPPORTED_SIZES = ["1024x1024", "1024x1536", "1536x1024"]
