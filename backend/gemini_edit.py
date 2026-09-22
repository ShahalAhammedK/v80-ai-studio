import io
import logging

from PIL import Image

from .ai_edit import build_prompt
from .config import GEMINI_API_KEY, GEMINI_IMAGE_MODEL
from .errors import UserFacingError
from .image_utils import composite_result

logger = logging.getLogger("v80.gemini_edit")

_client = None


def _get_client():
    global _client
    if not GEMINI_API_KEY:
        raise UserFacingError(
            "The AI service isn't configured yet. Add GEMINI_API_KEY to your .env file.", 503
        )
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def replace_person_gemini(
    campaign_image: Image.Image,
    person_image: Image.Image,
    mask_l: Image.Image,
    settings: dict,
) -> Image.Image:
    """
    Gemini's image models edit purely from a text prompt + reference images —
    there is no mask parameter like OpenAI's images.edit. We still compute and
    pass the mask conceptually through the prompt, but the real fidelity
    guarantee is the pixel composite below: whatever Gemini returns, only the
    masked (person) region of it is allowed to reach the final output.
    """
    client = _get_client()

    prompt = build_prompt(
        settings.get("identityPreservation", "high"),
        settings.get("scenePreservation", "maximum"),
        settings.get("keepPose", True),
        settings.get("matchLighting", True),
    )
    prompt = (
        "The FIRST image is the campaign photo (the master image). The SECOND image is the "
        "reference person to insert. " + prompt
    )

    from google.genai import types

    try:
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[prompt, campaign_image.convert("RGB"), person_image.convert("RGB")],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
    except Exception as exc:  # noqa: BLE001
        _raise_friendly(exc)

    generated_bytes = None
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        parts = getattr(candidate.content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                generated_bytes = inline.data
                break
        if generated_bytes:
            break

    if not generated_bytes:
        raise UserFacingError("The AI service didn't return an image. Please try again.", 502)

    generated = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
    return composite_result(campaign_image, generated, mask_l)


def _raise_friendly(exc: Exception):
    from google.genai import errors as genai_errors

    logger.exception("Gemini image generation failed")

    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code == 429:
            raise UserFacingError(
                "The AI service is busy or out of quota right now. Please try again later.", 429
            )
        if code in (401, 403):
            raise UserFacingError("The AI service rejected our credentials. Please check the server configuration.", 502)
        raise UserFacingError("We couldn't process this image. Try a clearer reference photo or adjust the mask.", 502)

    if isinstance(exc, genai_errors.ServerError):
        raise UserFacingError("The AI service is temporarily unavailable. Please try again in a moment.", 502)

    raise UserFacingError("Something went wrong while generating your image. Please try again.", 500)
