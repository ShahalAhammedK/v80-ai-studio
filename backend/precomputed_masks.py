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

A campaign may also ship a "<name>.protect.png": graphics that overlap the
person (a phone held in front of the body, a text panel over the torso) and
must always come from the original, no matter how far the person mask is
grown. See scripts/make_protect_v80_lite.py.

Hashes are computed from disk at import time rather than hardcoded, so
replacing a campaign image can't silently leave a stale mask behind — swap the
files and the lookup keeps working.

Falls back to live rembg segmentation when a campaign isn't in this set, which
keeps custom images working in any environment where rembg is installed.
"""

import hashlib
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger("v80.precomputed_masks")

CAMPAIGN_DIR = Path(__file__).resolve().parent.parent / "static" / "assets" / "campaigns"

_DERIVED_SUFFIXES = (".mask.png", ".protect.png")

# sha256(campaign bytes) -> {"mask": Path, "protect": Path | None}
_index: dict[str, dict[str, Path | None]] | None = None


def is_campaign_image(path: Path) -> bool:
    """A campaign image, as opposed to one of its derived mask files."""
    return not path.name.endswith(_DERIVED_SUFFIXES)


def protect_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".protect.png")


def _build_index() -> dict[str, dict[str, Path | None]]:
    index: dict[str, dict[str, Path | None]] = {}
    if not CAMPAIGN_DIR.is_dir():
        logger.warning("Campaign directory not found at %s", CAMPAIGN_DIR)
        return index

    for image_path in sorted(CAMPAIGN_DIR.glob("*.png")):
        if not is_campaign_image(image_path):
            continue
        mask_path = image_path.with_suffix(".mask.png")
        if not mask_path.exists():
            logger.warning("No precomputed mask for %s", image_path.name)
            continue
        protect_path = protect_path_for(image_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        index[digest] = {"mask": mask_path, "protect": protect_path if protect_path.exists() else None}

    logger.info("Loaded %d precomputed campaign mask(s)", len(index))
    return index


def get_index() -> dict[str, dict[str, Path | None]]:
    global _index
    if _index is None:
        _index = _build_index()
    return _index


def _load(campaign_bytes: bytes, kind: str, size: tuple[int, int]) -> Image.Image | None:
    entry = get_index().get(hashlib.sha256(campaign_bytes).hexdigest())
    path = entry and entry.get(kind)
    if path is None:
        return None

    try:
        image = Image.open(path).convert("L")
    except Exception:  # noqa: BLE001 - a bad mask file should fall back, not 500
        logger.exception("Failed to read precomputed %s %s", kind, path)
        return None

    if image.size != size:
        image = image.resize(size, Image.NEAREST if kind == "protect" else Image.LANCZOS)
    return image


def lookup(campaign_bytes: bytes, size: tuple[int, int]) -> Image.Image | None:
    """Return the precomputed person mask for these exact campaign bytes, or None."""
    return _load(campaign_bytes, "mask", size)


def lookup_protect(campaign_bytes: bytes, size: tuple[int, int]) -> Image.Image | None:
    """Return the protect mask (255 = always keep original) for this campaign, or None."""
    return _load(campaign_bytes, "protect", size)


def count() -> int:
    return len(get_index())
