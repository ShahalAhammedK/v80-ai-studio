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


def _grow(mask, radius: int):
    """Square dilation of a boolean array by `radius` px (separable, numpy only)."""
    import numpy as np

    out = mask.astype(np.uint8)
    for axis in (0, 1):
        src = out.copy()
        n = src.shape[axis]
        for d in range(1, min(radius, n - 1) + 1):
            head = [slice(None)] * 2
            tail = [slice(None)] * 2
            head[axis] = slice(d, None)
            tail[axis] = slice(None, -d)
            np.maximum(out[tuple(head)], src[tuple(tail)], out=out[tuple(head)])
            np.maximum(out[tuple(tail)], src[tuple(head)], out=out[tuple(tail)])
    return out > 0


def _shrink(mask, radius: int):
    return ~_grow(~mask, radius)


def _connected(seed, allowed, block: int):
    """
    The parts of `allowed` that are 8-connected to `seed` (through `allowed`).

    Worked out on a grid of `block`x`block` cells, so it is a few hundred cheap
    steps on a small array rather than a flood fill over every pixel. Two
    blobs within about one cell of each other count as connected.
    """
    import numpy as np

    h, w = allowed.shape
    gh, gw = -(-h // block), -(-w // block)

    def cells(mask):
        padded = np.zeros((gh * block, gw * block), bool)
        padded[:h, :w] = mask
        return padded.reshape(gh, block, gw, block).any(axis=(1, 3))

    reach = cells(seed)
    passable = cells(allowed) | reach
    while True:
        grown = _grow(reach, 1) & passable
        if not (grown != reach).any():
            break
        reach = grown
    return allowed & np.repeat(np.repeat(reach, block, axis=0), block, axis=1)[:h, :w]


# Tuned against real generations of the V80 Lite campaign.
ADAPTIVE = {
    "link_block": 4,       # cell size (px) for tracing what is connected to the person
    "sample_margin": 120,  # colour-match only on pixels at least this far away
    "backdrop_min": 150,   # ...that are bright backdrop in both images (mean RGB)
    "threshold": 35,       # max per-channel change (0-255) that counts as "the AI changed this"
    "speck": 2,            # remove isolated changed specks smaller than this
    "close": 10,           # bridge small gaps inside the new person
    "pad": 6,              # grow before feathering, so the blur doesn't thin fingers or hair
    "feather": 4,          # Gaussian blur on the final edge
}


def adaptive_composite(
    original: Image.Image,
    generated: Image.Image,
    person_mask_l: Image.Image,
    protect_l: Image.Image | None = None,
) -> Image.Image:
    """
    Merge the AI's image into the original, keeping the AI's new person whole.

    The earlier approach kept AI pixels only inside the OLD person's outline.
    A new person rarely has the same outline — a larger head, a different
    hairline, a hand that grips the phone slightly differently — so the
    result was cut off along that outline: a cheek sliced in a straight line,
    fingertips ending at the phone's edge.

    Instead, find where the AI actually changed the picture and keep that:

    1. Colour-match. These models re-grade colour across the whole frame, so
       measure a per-channel brightness gain on the backdrop far from the
       person (where the AI should have changed nothing) and apply it to the
       AI image. Its backdrop then matches the original, and edges blend.
       Gain only, no offset: a fitted line (orig = a * gen + b) was skewed by
       dark areas the AI legitimately re-rendered, and its offset lifted
       every black by ~20 levels, laying a white haze over dark hair.
    2. Changed region = pixels that differ noticeably after colour-matching
       AND are connected to the old person — the new person, however far
       they extend (a broader shoulder reaching the frame edge, fuller
       hair), while stray changes elsewhere are ignored. Plus the old
       person's own area, which must be replaced either way. (A fixed
       search radius here cut off a shoulder that reached further than it.)
    3. Clean up, feather, and composite. Everything else — backdrop, the
       phone where the AI left it alone — stays the exact original pixel.
       Protected graphics (panel, headline) always stay original.
    """
    import numpy as np
    from PIL import ImageFilter

    size = original.size
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32)
    old = np.asarray(person_mask_l.convert("L").resize(size, Image.LANCZOS)) > 127
    # Protect is soft-edged (0-255): fully protected pixels are exact
    # originals, and its 1px rim blends so graphics edges stay anti-aliased.
    protect_f = (
        np.asarray(protect_l.convert("L").resize(size, Image.NEAREST), dtype=np.float32) / 255.0
        if protect_l is not None
        else np.zeros(old.shape, np.float32)
    )
    protect = protect_f > 0.5
    p = ADAPTIVE

    sample = ~_grow(old, p["sample_margin"]) & ~protect
    sample[1::2, :] = False  # every other row is plenty for three gains
    backdrop = sample & (orig.mean(axis=2) > p["backdrop_min"]) & (gen.mean(axis=2) > p["backdrop_min"])
    if backdrop.sum() > 1000:
        gen *= orig[backdrop].sum(axis=0) / gen[backdrop].sum(axis=0)
    gen = np.clip(gen, 0, 255)

    diff = np.abs(gen - orig).max(axis=2)
    changed = (diff > p["threshold"]) & ~protect
    changed = _grow(_shrink(changed, p["speck"]), p["speck"])
    changed = _connected(old, changed, p["link_block"])

    region = changed | old
    region = _shrink(_grow(region, p["close"]), p["close"])
    region = _grow(region, p["pad"])

    alpha = Image.fromarray((region * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(p["feather"]))
    alpha = np.asarray(alpha, dtype=np.float32) / 255.0 * (1.0 - protect_f)

    out = gen * alpha[..., None] + orig * (1.0 - alpha[..., None])
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def remove_protected(mask_l: Image.Image, protect_l: Image.Image | None) -> Image.Image:
    """Zero the mask wherever the protect mask is set (255 = always keep original)."""
    if protect_l is None:
        return mask_l
    from PIL import ImageChops

    protect = protect_l.convert("L")
    if protect.size != mask_l.size:
        protect = protect.resize(mask_l.size, Image.NEAREST)
    return ImageChops.subtract(mask_l.convert("L"), protect)

