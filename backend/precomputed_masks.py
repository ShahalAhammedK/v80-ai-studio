"""
Precomputed person masks for the built-in campaign images.

The app ships a fixed set of campaign images (see static/assets/campaigns) and
the custom-upload option was removed, so every mask the app will ever need is
known ahead of time. Running rembg to rediscover them at runtime would cost
~320MB of dependencies (scipy/numba/llvmlite/skimage/onnxruntime) plus ~176MB
of model weights and ~1GB of peak RAM — enough to OOM a small host — to produce
a result that never changes.

So the masks are generated once, committed next to their campaign image as
"<name>.mask.png", and looked up here by a SHA-256 of the campaign file's
bytes. The frontend fetches those exact files to build its upload, so the
bytes (and therefore the hash) match exactly.

Hashes are computed from disk at import time rather than hardcoded, so
replacing a campaign image can't silently leave a stale mask behind — swap the
pair and the lookup keeps working.

Falls back to live rembg segmentation when a campaign isn't in this set, which
keeps custom images working in any environment where rembg is installed.
"""

import hashlib
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger("v80.precomputed_masks")

CAMPAIGN_DIR = Path(__file__).resolve().parent.parent / "static" / "assets" / "campaigns"

# sha256(campaign bytes) -> path of its mask PNG
_index: dict[str, Path] | None = None


def _build_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not CAMPAIGN_DIR.is_dir():
        logger.warning("Campaign directory not found at %s", CAMPAIGN_DIR)
        return index

    for image_path in sorted(CAMPAIGN_DIR.glob("*.png")):
        if image_path.name.endswith(".mask.png"):
            continue
        mask_path = image_path.with_suffix(".mask.png")
        if not mask_path.exists():
            logger.warning("No precomputed mask for %s", image_path.name)
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        index[digest] = mask_path

    logger.info("Loaded %d precomputed campaign mask(s)", len(index))
    return index


def get_index() -> dict[str, Path]:
    global _index
    if _index is None:
        _index = _build_index()
    return _index


def lookup(campaign_bytes: bytes, size: tuple[int, int]) -> Image.Image | None:
    """Return the precomputed mask for these exact campaign bytes, or None."""
    digest = hashlib.sha256(campaign_bytes).hexdigest()
    mask_path = get_index().get(digest)
    if mask_path is None:
        return None

    try:
        mask = Image.open(mask_path).convert("L")
    except Exception:  # noqa: BLE001 - a bad mask file should fall back, not 500
        logger.exception("Failed to read precomputed mask %s", mask_path)
        return None

    if mask.size != size:
        mask = mask.resize(size, Image.LANCZOS)
    return mask


def count() -> int:
    return len(get_index())
