"""
Best-effort gender classification of the uploaded person photo, used only to
suggest which built-in default campaign image (woman/man) matches. This is a
convenience default, not a verdict — the user can always override it by
picking either default manually or uploading their own campaign image.

CHAT_MODEL calls are routed through the Experiential Labs gateway (an
OpenAI-compatible Chat Completions API) rather than calling OpenAI directly —
see EXPLABS_API_KEY / EXPLABS_BASE_URL in backend/config.py.
"""

import base64
import io
import logging

from PIL import Image

from .config import CHAT_MODEL, EXPLABS_API_KEY, EXPLABS_BASE_URL

logger = logging.getLogger("v80.gender_detect")

_client = None
_warned_missing_key = False


def _get_client():
    global _client, _warned_missing_key
    if not EXPLABS_API_KEY:
        if not _warned_missing_key:
            logger.warning(
                "EXPLABS_API_KEY is not set — gender detection is disabled. Create a key under "
                "Settings -> API Keys on platform.experientiallabs.ai and export it as "
                "EXPLABS_API_KEY (or add it to .env.local) to enable this feature."
            )
            _warned_missing_key = True
        return None
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=EXPLABS_API_KEY, base_url=EXPLABS_BASE_URL)
    return _client


def detect_gender(person_image: Image.Image) -> str:
    """Returns 'male', 'female', or 'unknown'. Never raises."""
    client = _get_client()
    if client is None:
        return "unknown"

    try:
        thumb = person_image.copy()
        thumb.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at the person in this photo. Reply with exactly one word: "
                                "'male', 'female', or 'unknown' if you cannot tell. No punctuation, "
                                "no explanation."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=5,
        )

        text = (response.choices[0].message.content or "").strip().lower()
        if "female" in text:
            return "female"
        if "male" in text:
            return "male"
        return "unknown"
    except Exception:  # noqa: BLE001 - this is a soft suggestion feature only
        logger.exception("Gender detection failed")
        return "unknown"
