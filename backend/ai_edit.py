import base64
import io
import logging

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
from PIL import Image

from .config import GEMINI_API_KEY, IMAGE_MODEL, IMAGE_PROVIDER, OPENAI_API_KEY, SUPPORTED_SIZES
from .errors import UserFacingError
from .image_utils import (
    adaptive_composite,
    build_openai_mask,
    expand_mask,
    fit_to_size,
    image_to_png_bytes,
    pick_supported_size,
    remove_protected,
    unfit_from_size,
)

logger = logging.getLogger("v80.ai_edit")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if not OPENAI_API_KEY:
        raise UserFacingError(
            "The AI service isn't configured yet. Set OPENAI_API_KEY in the server's "
            "environment variables (or .env.local when running locally).",
            503,
        )
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


IDENTITY_INSTRUCTIONS = {
    "high": "Prioritize an exact, highly recognizable match to the reference person's face and features above all else.",
    "balanced": "Balance a recognizable likeness of the reference person with a natural, photorealistic blend into the scene.",
    "creative": "Use the reference person's general likeness as inspiration while allowing natural photographic variation.",
}

SCENE_INSTRUCTIONS = {
    "maximum": "Preserve every non-person pixel of the campaign image exactly as-is: background, logo, text, product, lighting and composition must be untouched.",
    "balanced": "Preserve the campaign's background, logo, text, product and composition, allowing only minimal, natural adjustments directly around the replaced person.",
}


GENDER_WORDS = {
    "male": ("man", "his"),
    "female": ("woman", "her"),
}


def build_gender_line(person_gender: str | None) -> str:
    """
    The single campaign image shows a man, so the model tends to reproduce a
    man regardless of the reference photo. When gender detection gives us a
    confident answer we say it outright; otherwise we just point at the
    reference and let the model read it.
    """
    words = GENDER_WORDS.get((person_gender or "").lower())
    if words is None:
        return (
            "CRITICAL — GENDER: Take the replacement person's gender from the REFERENCE photo, "
            "not from the original campaign subject. If the reference person is a woman, the result "
            "must clearly be a woman; if a man, clearly a man."
        )

    noun, possessive = words
    line = (
        f"CRITICAL — GENDER: The reference person is a {noun}. The result MUST clearly be a {noun}, "
        f"with {possessive} face, hair, body build and silhouette."
    )
    if noun == "woman":
        # Only worth saying when it contradicts the campaign shot, which is a man.
        line += (
            " The original campaign subject is a man — do NOT carry that over. Replacing him with a "
            "woman is the intent, not an error to correct."
        )
    return line


def build_prompt(
    identity_preservation: str,
    scene_preservation: str,
    keep_pose: bool,
    match_lighting: bool,
    person_gender: str | None = None,
) -> str:
    identity_line = IDENTITY_INSTRUCTIONS.get(identity_preservation, IDENTITY_INSTRUCTIONS["high"])
    scene_line = SCENE_INSTRUCTIONS.get(scene_preservation, SCENE_INSTRUCTIONS["maximum"])
    pose_line = (
        "Copy the original subject's exact posture and limb placement precisely — do not substitute a different, "
        "more generic, or more common pose. Only re-fit that exact posture to the reference person's own body "
        "build and scale."
        if keep_pose
        else "You may adjust the pose slightly if needed to accommodate the new person naturally."
    )
    lighting_line = (
        "Match the original scene's lighting, shadows, highlights, skin illumination, color temperature and depth of field."
        if match_lighting
        else "Light the replacement person naturally, staying close to the original mood."
    )

    gender_line = build_gender_line(person_gender)

    # Written for the V80 Lite campaign (static/assets/campaigns/v80-lite.png):
    # a chest-up shot holding the phone out toward the camera, with the slogan
    # and battery panel layered over the person. The POSE, FRAMING and GRAPHICS
    # blocks describe that specific image — revisit them if the campaign changes.
    # Keep in step with lib/prompt.js in the Cloudflare version.
    return f"""Replace ONLY the human subject in the first image (the campaign image) with the person shown in the second image (the reference image).

{gender_line}

CRITICAL — POSE AND THE PHONE: The original subject is shown from the chest up, facing the camera, holding the green smartphone out toward the viewer at arm's length with the back of the phone facing the camera. Reproduce this EXACT pose: the same arm reaching toward the camera, the same hand gripping the phone with the fingers curled around its left edge and the thumb on its right edge, the same slight head tilt and the same direct, confident gaze into the lens. The phone is a fixed part of the design: it must not move, rotate, resize or change in any way — the new hand holds it exactly where it already is.

The reference image defines the replacement person's identity: face, facial structure, skin tone, hair, and general body type (e.g. build/frame). Keep their exact hairstyle — the same cut, length, texture, curl pattern, hairline and volume as in the reference photo; the hair is part of who they are, not something to restyle. {identity_line}

CRITICAL — FRAMING AND PROPORTION: Keep the original framing exactly. The head must be the same size and in the same position as the original subject's head, with the top of the hair just below the "ALWAYS POWERED" slogan; the shoulders at the same height and width; the torso filling the same area behind the battery panel. Do not zoom in or out and do not shift the person sideways. Take ONLY the person's likeness from the reference photo, never its framing — a close-up reference must not produce a larger head, and a full-body reference must not produce a smaller one. The head must be correctly proportioned to the shoulders and the extended arm.

CRITICAL — COMPLETENESS: Render a complete, natural head with a full hairstyle and normal hair volume above the forehead; never flatten the top of the head or cut the hair off at a straight line, and never render the head as bald unless the reference person is clearly bald. The hand holding the phone must be anatomically natural: a thumb and four fingers in the same grip as the original, with skin matching the person's face.

CRITICAL — GRAPHICS IN FRONT: The "ALWAYS POWERED" slogan, the green battery panel ("32H 37M", "CONTINUOUS LIVESTREAM ENDURANCE", the Guinness World Records "RECORD HOLDER" badge) and the phone are layers IN FRONT of the person. They stay exactly as they are, on top; the person continues naturally behind them.

Do NOT copy the reference photo's clothing, outfit, accessories, background, or setting — ignore what the reference person is wearing and ignore where the reference photo was taken entirely. Clothing comes from the instruction below, not from the reference photo.

Dress the replacement person in the same outfit the original subject wears: a dark green casual jacket worn open over a plain white crew-neck T-shirt, cut to fit the replacement person's own gender and body build. The same outfit in every case. Not the clothing from the reference photo.

Preserve the original campaign image everywhere outside the person. {scene_line}

Do not change: background, product, logos, text, numbers, badges, typography, any graphic elements, camera framing, perspective, composition, image dimensions, lighting environment, or color grading.

Keep the original subject's exact position and camera angle, and the hand's exact contact with the phone. {pose_line}

{lighting_line} Keep the soft, even studio light and the plain light-grey backdrop; the edges of the person, especially the hair, should blend into that backdrop naturally.

The final result should look like the original campaign photograph was genuinely photographed with the reference person — their own face and body build, in the campaign's outfit and pose, holding the same phone. Do not redesign the campaign, move or alter the phone, alter logos or text, or add people. ONLY replace the human subject — everything else in the scene stays as it was."""


# Growth of the person mask sent to OpenAI (px at the 1024-wide working size).
API_MASK_RADIUS = 24
API_MASK_EXTRA_UP = 16


def _provider_order() -> list[str]:
    """Preferred provider first, then any other configured providers as fallback."""
    order = [IMAGE_PROVIDER]
    if "openai" != IMAGE_PROVIDER and OPENAI_API_KEY:
        order.append("openai")
    if "gemini" != IMAGE_PROVIDER and GEMINI_API_KEY:
        order.append("gemini")
    return order


def _run_provider(provider: str, campaign_image, person_image, mask_l, settings, protect_l, noedit_l) -> Image.Image:
    if provider == "gemini":
        from .gemini_edit import replace_person_gemini

        return replace_person_gemini(campaign_image, person_image, mask_l, settings, protect_l)
    return _replace_person_openai(campaign_image, person_image, mask_l, settings, protect_l, noedit_l)


def replace_person(
    campaign_image: Image.Image,
    person_image: Image.Image,
    mask_l: Image.Image,
    settings: dict,
    protect_l: Image.Image | None = None,
    noedit_l: Image.Image | None = None,
) -> Image.Image:
    """
    Dispatches to the configured image provider (see IMAGE_PROVIDER in config.py).
    If it fails (quota, auth, upstream error) and a second provider is configured,
    automatically retries with that one before giving up.

    Campaign graphics that overlap the person (see scripts/make_protect_*.py):
    `protect_l` — 255 = always the original pixel in the final image;
    `noedit_l`  — 255 = tell the AI not to edit (protect plus e.g. a held phone).
    """
    providers = _provider_order()
    primary_error: UserFacingError | None = None

    for i, provider in enumerate(providers):
        try:
            result = _run_provider(provider, campaign_image, person_image, mask_l, settings, protect_l, noedit_l)
            if i > 0:
                logger.info("Image provider '%s' failed; '%s' succeeded instead.", providers[0], provider)
            return result
        except UserFacingError as exc:
            logger.warning("Image provider '%s' failed (%s): %s", provider, exc.status_code, exc.message)
            if primary_error is None:
                primary_error = exc

    # Report the PRIMARY provider's failure, not the last fallback's. Reporting
    # the last one meant a configured-but-exhausted fallback masked the real
    # cause: with IMAGE_PROVIDER=openai and a dead Gemini key, every OpenAI
    # failure surfaced as Gemini's "out of quota" message instead.
    raise primary_error


def _replace_person_openai(
    campaign_image: Image.Image,
    person_image: Image.Image,
    mask_l: Image.Image,
    settings: dict,
    protect_l: Image.Image | None = None,
    noedit_l: Image.Image | None = None,
) -> Image.Image:
    client = _get_client()

    original_size = campaign_image.size
    target_size = pick_supported_size(*original_size)
    if target_size not in SUPPORTED_SIZES:
        target_size = "1024x1024"

    fitted_campaign, offset, fitted_dims = fit_to_size(campaign_image, target_size)
    fitted_mask, _, _ = fit_to_size(mask_l.convert("L"), target_size)

    # The mask sent to OpenAI says where it may edit. It is grown well past the
    # old person, so a new person with a bigger head, fuller hair or a different
    # grip has room to be drawn whole instead of squeezed into the old outline.
    # (What we finally keep is decided afterwards by adaptive_composite, from
    # what the model actually drew.) Growth runs into graphics that overlap the
    # person, so the do-not-edit area is removed after growing, never before.
    api_mask = expand_mask(fitted_mask, radius=API_MASK_RADIUS, extra_up=API_MASK_EXTRA_UP)
    no_edit = noedit_l or protect_l
    if no_edit is not None:
        fitted_no_edit, _, _ = fit_to_size(no_edit.convert("L"), target_size)
        api_mask = remove_protected(api_mask, fitted_no_edit)

    openai_mask = build_openai_mask(api_mask)

    person_square = person_image.copy()
    person_square.thumbnail((1024, 1024), Image.LANCZOS)

    prompt = build_prompt(
        settings.get("identityPreservation", "high"),
        settings.get("scenePreservation", "maximum"),
        settings.get("keepPose", True),
        settings.get("matchLighting", True),
        settings.get("personGender"),
    )

    campaign_bytes = image_to_png_bytes(fitted_campaign)
    person_bytes = image_to_png_bytes(person_square)
    mask_bytes = image_to_png_bytes(openai_mask)

    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=[
                ("campaign.png", campaign_bytes, "image/png"),
                ("person.png", person_bytes, "image/png"),
            ],
            mask=("mask.png", mask_bytes, "image/png"),
            prompt=prompt,
            size=target_size,
            quality="high",
            n=1,
        )
    except AuthenticationError:
        logger.exception("OpenAI authentication failed")
        raise UserFacingError("The AI service rejected our credentials. Please check the server configuration.", 502)
    except RateLimitError:
        logger.exception("OpenAI rate limit hit")
        raise UserFacingError("The AI service is busy right now. Please try again in a moment.", 429)
    except APIConnectionError:
        logger.exception("OpenAI connection error")
        raise UserFacingError("We couldn't reach the AI service. Please check your connection and try again.", 502)
    except APIStatusError:
        logger.exception("OpenAI API returned an error status")
        raise UserFacingError("We couldn't process this image. Try a clearer reference photo or adjust the mask.", 502)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error calling OpenAI images.edit")
        raise UserFacingError("Something went wrong while generating your image. Please try again.", 500)

    if not result.data or not result.data[0].b64_json:
        raise UserFacingError("The AI service didn't return an image. Please try again.", 502)

    generated_bytes = base64.b64decode(result.data[0].b64_json)
    generated_fitted = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
    generated_full = unfit_from_size(generated_fitted, offset, fitted_dims, original_size)

    # Keep the new person wherever the model drew them; everything it left
    # alone goes back to the exact original pixel. See adaptive_composite.
    return adaptive_composite(campaign_image, generated_full, mask_l, protect_l)
