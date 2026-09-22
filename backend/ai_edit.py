import base64
import io
import logging

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
from PIL import Image

from .config import GEMINI_API_KEY, IMAGE_MODEL, IMAGE_PROVIDER, OPENAI_API_KEY, SUPPORTED_SIZES
from .errors import UserFacingError
from .image_utils import (
    build_openai_mask,
    composite_result,
    expand_mask,
    fit_to_size,
    image_to_png_bytes,
    pick_supported_size,
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

    return f"""Replace ONLY the human subject in the first image (the campaign image) with the person shown in the second image (the reference image).

{gender_line}

CRITICAL — POSTURE: Before anything else, look closely at exactly how the original subject is posed: which arm (if any) is crossed over the body, which hand (if any) is in a pocket, exactly where a hand or arm touches the product, the angle of the head, the stance of the legs. Reproduce that EXACT posture and limb arrangement in the result. A common failure is defaulting to a generic "leaning casually with one hand in a pocket" pose — do NOT do this unless that is literally what the original image shows. If the original subject's arms are crossed/folded, the result's arms must also be crossed/folded in the same way.

The reference image defines the replacement person's identity: face, facial structure, skin tone, hair, and general body type (e.g. build/frame). {identity_line}

CRITICAL — HEAD PROPORTION: Take ONLY the person's likeness from the reference photo, never its framing. The reference is often a close-up or half-body shot where the head fills much of the frame — do not carry that ratio over. In the result the head must be correctly proportioned to the full standing body, matching the head-to-body ratio of the ORIGINAL campaign subject (an adult standing figure is roughly seven-and-a-half head-heights tall). A head that looks too large for its shoulders and torso is a failure.

CRITICAL — SCALE: This is a fixed graphic-design composition, not a candid photo. The product is the hero of this image and is deliberately oversized: the person is a smaller supporting element beside it. The top of the replacement person's head must sit at the SAME height as the original subject's head — never higher. The person must clearly read as shorter than the product, exactly as in the original. Match the original subject's footprint: same head height, same shoulder position, same reach and contact point with the product, standing on the same spot. If any size adjustment is needed, scale the ENTIRE figure uniformly (head, torso, legs together, keeping normal human proportions) — never achieve it by shrinking, flattening or trimming the head, hair or any single body part.

CRITICAL — COMPLETENESS: The person must be rendered complete and anatomically natural. The full head must be drawn including the complete top of the skull and a full, natural hairstyle with normal hair volume above the forehead. Never flatten the top of the head, never cut the hair off at a straight line, never render the crown of the head as bald or shaved unless the reference person is clearly bald. Both shoulders, both arms, hands, legs and feet must be fully visible.

Do NOT copy the reference photo's clothing, outfit, accessories, background, or setting — ignore what the reference person is wearing and ignore where the reference photo was taken entirely. Clothing comes from the instruction below, not from the reference photo.

Dress the replacement person in a tailored black suit — the same premium, formal style the original campaign subject wears, tailored to the replacement person's own gender and body build (a men's cut for a man, a women's cut for a woman). Black suit in both cases. Not the casual clothing from the reference photo.

Preserve the original campaign image everywhere outside the person. {scene_line}

Do not change: background, architecture, objects, product, logos, text, typography, any large graphic elements, camera framing, perspective, composition, image dimensions, floor, reflections, lighting environment, or color grading.

Keep the original subject's exact position, camera angle, and physical interaction/contact point with surrounding objects (such as where a hand or arm touches the product). {pose_line}

{lighting_line} Pay special attention to the original person's contact shadow, floor shadow and reflections — the replacement person should cast realistic shadows and reflections consistent with the original environment, without altering the floor or background itself.

The final result should look like the original campaign photograph was genuinely photographed with the reference person — their own face and body build, dressed in the campaign's premium outfit style, in the campaign's pose. Do not redesign the campaign, regenerate the background, move objects, alter logos or text, or add people. ONLY replace the human subject — everything else in the scene stays as it was."""


def _provider_order() -> list[str]:
    """Preferred provider first, then any other configured providers as fallback."""
    order = [IMAGE_PROVIDER]
    if "openai" != IMAGE_PROVIDER and OPENAI_API_KEY:
        order.append("openai")
    if "gemini" != IMAGE_PROVIDER and GEMINI_API_KEY:
        order.append("gemini")
    return order


def _run_provider(provider: str, campaign_image, person_image, mask_l, settings) -> Image.Image:
    if provider == "gemini":
        from .gemini_edit import replace_person_gemini

        return replace_person_gemini(campaign_image, person_image, mask_l, settings)
    return _replace_person_openai(campaign_image, person_image, mask_l, settings)


def replace_person(
    campaign_image: Image.Image,
    person_image: Image.Image,
    mask_l: Image.Image,
    settings: dict,
) -> Image.Image:
    """
    Dispatches to the configured image provider (see IMAGE_PROVIDER in config.py).
    If it fails (quota, auth, upstream error) and a second provider is configured,
    automatically retries with that one before giving up.
    """
    providers = _provider_order()
    last_error: UserFacingError | None = None

    for i, provider in enumerate(providers):
        try:
            result = _run_provider(provider, campaign_image, person_image, mask_l, settings)
            if i > 0:
                logger.info("Image provider '%s' failed; '%s' succeeded instead.", providers[0], provider)
            return result
        except UserFacingError as exc:
            logger.warning("Image provider '%s' failed (%s): %s", provider, exc.status_code, exc.message)
            last_error = exc

    raise last_error


def _replace_person_openai(
    campaign_image: Image.Image,
    person_image: Image.Image,
    mask_l: Image.Image,
    settings: dict,
) -> Image.Image:
    client = _get_client()

    original_size = campaign_image.size
    target_size = pick_supported_size(*original_size)
    if target_size not in SUPPORTED_SIZES:
        target_size = "1024x1024"

    fitted_campaign, offset, fitted_dims = fit_to_size(campaign_image, target_size)
    fitted_mask, _, _ = fit_to_size(mask_l.convert("L"), target_size)

    # Two masks with different jobs, so they get grown by different amounts:
    #
    # 1. The mask sent to OpenAI defines where it may edit. Sideways growth stays
    #    small so the model can't drift out of the original pose, but it gets a lot
    #    of headroom upward — otherwise a replacement person with more hair than the
    #    original has nowhere to put it and the model flattens their hairstyle.
    # 2. The composite mask defines how much of OpenAI's output we keep. It is grown
    #    further still, so nothing the model legitimately drew (hair, shoulders) gets
    #    hard-reset back to the original background and sliced off. The product,
    #    logo and text are far from the person, so they stay pixel-exact regardless.
    # The model reliably draws the person as tall as the mask permits, so the mask's
    # top edge effectively sets their height. Measured: 70px of headroom made them
    # ~100px taller than the original, 22px made them 34px taller. Keeping it minimal
    # pins their head to the original subject's head height, which is what keeps them
    # correctly scaled against the (deliberately oversized) product. The composite
    # mask stays looser so nothing actually drawn gets clipped.
    api_mask = expand_mask(fitted_mask, radius=10, extra_up=6)
    composite_mask_fitted = expand_mask(fitted_mask, radius=30, extra_up=40)

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

    # These unified image-gen models subtly re-grade color/brightness across the
    # WHOLE returned image during generation — even the "untouched" background and
    # product drift slightly from the original. So we still composite back onto the
    # pristine original outside the mask. Using the *dilated* mask (not the tight
    # auto-detected one) is what avoids reintroducing the earlier hair-clipping ghost
    # edge: the dilation gives a margin where OpenAI's own naturally-blended pixels
    # are kept, and only genuinely untouched areas (background, product) get hard-
    # reasserted to the original.
    composite_mask = unfit_from_size(composite_mask_fitted, offset, fitted_dims, original_size)
    return composite_result(campaign_image, generated_full, composite_mask, feather=3)
