"""
Automatic person mask generation.

Uses rembg's human-segmentation model (u2net_human_seg) rather than the
general salient-object model, since a generic model would happily grab the
giant blue V or logo instead of (or in addition to) the person. If rembg or
its model weights aren't available, we fail soft: the frontend simply opens
the manual mask editor with a blank mask so the user can paint one by hand.
"""

import importlib.util
import io
import logging

from PIL import Image, ImageFilter

logger = logging.getLogger("v80.segmentation")

_session = None
_session_failed = False


def _get_session():
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    try:
        from rembg import new_session

        # First use downloads the ~180MB u2net_human_seg weights; subsequent
        # calls reuse the cached session.
        _session = new_session("u2net_human_seg")
    except Exception:  # noqa: BLE001 - any failure here should just disable auto-mask
        logger.exception("Failed to initialize rembg human segmentation model")
        _session_failed = True
        _session = None
    return _session


def is_installed() -> bool:
    """Cheap, non-blocking check for whether the rembg package is present at all."""
    return importlib.util.find_spec("rembg") is not None


def is_available() -> bool:
    """Whether a segmentation session is actually ready (may trigger a model download)."""
    return _get_session() is not None


def generate_person_mask(image: Image.Image) -> Image.Image | None:
    """Return a single-channel mask (255 = person, 0 = background) or None on failure."""
    session = _get_session()
    if session is None:
        return None

    try:
        from rembg import remove

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        result_bytes = remove(buf.getvalue(), session=session, only_mask=True)
        mask = Image.open(io.BytesIO(result_bytes)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        # Clean up small stray specks and smooth jagged edges from the raw mask.
        mask = mask.point(lambda v: 255 if v > 40 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        return mask
    except Exception:  # noqa: BLE001
        logger.exception("Person segmentation failed")
        return None
