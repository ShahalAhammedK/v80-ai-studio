import io
import os

from PIL import Image, ImageFile, UnidentifiedImageError

from .config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_DIMENSION,
    MAX_UPLOAD_BYTES,
    MIN_DIMENSION,
    SUPPORTED_SIZES,
)
from .errors import UserFacingError

# Reject truncated/corrupt files instead of silently loading partial data.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def validate_upload(filename: str, content_type: str, data: bytes) -> Image.Image:
    """Validate an uploaded file and return a decoded, EXIF-corrected RGB image."""
    if not data:
        raise UserFacingError("Please choose an image file to upload.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise UserFacingError(
            f"That image is too large. Please upload a file under {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
        )

    ext = os.path.splitext(filename or "")[1].lower()
    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise UserFacingError("Please upload a JPG, PNG or WebP image.")

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        # verify() invalidates the image object for further use, so reopen it.
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise UserFacingError("We couldn't read that image. Please try a different file.")

    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    if image is None:
        raise UserFacingError("We couldn't read that image. Please try a different file.")
    image = image.convert("RGB")

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise UserFacingError(
            f"This image is too small. Please upload an image at least {MIN_DIMENSION}x{MIN_DIMENSION}px."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    return image


def pick_supported_size(width: int, height: int) -> str:
    """Pick the closest OpenAI-supported output size for the given aspect ratio."""
    aspect = width / height
    if aspect >= 1.15:
        return "1536x1024"
    if aspect <= 0.87:
        return "1024x1536"
    return "1024x1024"


def fit_to_size(image: Image.Image, size: str, fill=(255, 255, 255)) -> Image.Image:
    """Letterbox-fit an image into the exact target size, preserving aspect ratio."""
    target_w, target_h = (int(v) for v in size.split("x"))
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new(image.mode if image.mode != "L" else "L", (target_w, target_h), fill if image.mode != "L" else 0)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas, offset, (new_w, new_h)


def unfit_from_size(image: Image.Image, offset, fitted_dims, original_size: tuple) -> Image.Image:
    """Reverse fit_to_size: crop the letterboxed region back out and resize to original dimensions."""
    x, y = offset
    w, h = fitted_dims
    cropped = image.crop((x, y, x + w, y + h))
    return cropped.resize(original_size, Image.LANCZOS)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def expand_mask(mask_l: Image.Image, radius: int, extra_up: int = 0) -> Image.Image:
    """
    Grow a person mask by `radius` px in every direction, plus `extra_up` px
    upward only.

    The extra headroom matters because the mask is traced from the *original*
    subject's silhouette: a replacement person with more hair volume needs room
    above the old hairline, or the model is forced to flatten their hair to fit.
    Sideways growth is kept small separately, since a wide mask also lets the
    model drift away from the original pose.
    """
    import numpy as np
    from PIL import ImageFilter

    mask = mask_l.convert("L")
    if radius > 0:
        mask = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))

    if extra_up > 0:
        base = np.array(mask)
        out = base.copy()
        step = 4
        for shift in range(step, extra_up + 1, step):
            out[:-shift] = np.maximum(out[:-shift], base[shift:])
        # Smooth over the stepping so the upward extension has no visible stairs.
        mask = Image.fromarray(out).filter(ImageFilter.MaxFilter(step * 2 + 1))

    return mask


def build_openai_mask(person_mask_l: Image.Image) -> Image.Image:
    """
    Convert an internal mask (white=255 -> replace/edit, black=0 -> preserve)
    into the RGBA mask OpenAI's images.edit expects, where alpha=0 marks the
    editable region and alpha=255 marks pixels that must stay untouched.
    """
    if person_mask_l.mode != "L":
        person_mask_l = person_mask_l.convert("L")

    size = person_mask_l.size
    rgba = Image.new("RGBA", size, (0, 0, 0, 255))
    # Alpha = 255 - mask value: mask 255 (edit) -> alpha 0, mask 0 (preserve) -> alpha 255.
    alpha = person_mask_l.point(lambda v: 255 - v)
    rgba.putalpha(alpha)
    return rgba


def composite_result(
    original: Image.Image,
    generated: Image.Image,
    mask_l: Image.Image,
    feather: int = 2,
) -> Image.Image:
    """
    Guarantee pixel-perfect preservation outside the mask by compositing the
    AI result back onto the untouched original everywhere the mask is 0.
    """
    from PIL import ImageFilter

    mask = mask_l.convert("L").resize(original.size, Image.LANCZOS)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    generated = generated.convert("RGB").resize(original.size, Image.LANCZOS)
    return Image.composite(generated, original.convert("RGB"), mask)
