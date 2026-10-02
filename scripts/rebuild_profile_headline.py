"""
Prepare the Profile campaign (static/assets/campaigns/v80-lite-profile.png)
from the designer's file.

1. Save it as plain RGB without its colour profile (Pillow ignores embedded
   profiles but browsers apply them, which would shift the Cloudflare
   version's colours). It shows the model IN FRONT of the big "10000" — his
   hair covers the bottom of two digits, and that is the intended look.
2. Make "v80-lite-profile.plate.png": the same picture with the digits
   completed where the model's hair covered them. A new person's head is a
   different shape, so wherever the old hair covered the digits and the new
   hair doesn't, the result needs the digits there — not the old man's hair,
   and not the AI's attempt at drawing digits, which came out broken. The
   final composite puts the new person over this plate (see
   adaptive_composite in backend/image_utils.py). The hidden parts of the
   digits are rebuilt from an unobstructed "0" — they are identical glyphs on
   a 235.33px pitch. Hair showing through the hole of a "0" is replaced with
   that hole's own backdrop colour (a tuft there is cut off from the head by
   the digit, so it would otherwise survive as a stray patch of old hair).

    python scripts/rebuild_profile_headline.py <designer-file.png>

Then rebuild the masks (see scripts/make_protect_v80_lite.py).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_fill_holes

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "assets" / "campaigns" / "v80-lite-profile.png"
PLATE = OUT.with_suffix(".plate.png")

# Measured on the designer's 1607x1607 file.
PITCH = 706 / 3              # digit spacing: the "0"s start at x 413, 648, 883, 1119
TILE = (405, 125, 641, 362)  # the second digit, a "0" the hair never touches
# Where the hair covered digits, and which digits sit there (multiples of PITCH
# from the tile): the fourth and fifth "0".
REPAIR = (900, 300, 1195, 362)  # inside the "behind" box of make_protect_v80_lite.py
SHIFTS = (2, 3)


def shifted(img: Image.Image, dx: float) -> np.ndarray:
    """img moved right by dx pixels (sub-pixel, bilinear)."""
    moved = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, 0), resample=Image.BILINEAR)
    return np.asarray(moved).astype(float)


def main() -> None:
    src = Image.open(sys.argv[1]).convert("RGB")
    src.save(OUT, optimize=True, icc_profile=None)
    print(f"ok  {OUT.name} {src.size}")

    a = np.asarray(src).astype(float)
    h, w = a.shape[:2]

    # The tile: the clean "0", on a blank canvas.
    x0, y0, x1, y1 = TILE
    tile_rgb = np.zeros_like(a)
    tile_rgb[y0:y1, x0:x1] = a[y0:y1, x0:x1]
    r, g, b = (a[y0:y1, x0:x1, c] for c in range(3))
    # The glyph: its green pixels (hair, skin and backdrop are never green),
    # and the hole it encloses.
    green = (g > r + 12) & (g > b + 3)
    glyph = np.zeros((h, w), np.uint8)
    glyph[y0:y1, x0:x1] = green * 255
    hole = np.zeros((h, w), np.uint8)
    hole[y0:y1, x0:x1] = (binary_fill_holes(green) & ~green) * 255
    tile_img = Image.fromarray(tile_rgb.astype(np.uint8))
    glyph_img = Image.fromarray(glyph)

    out = a.copy()
    rx0, ry0, rx1, ry1 = REPAIR
    region = np.zeros((h, w), bool)
    region[ry0:ry1, rx0:rx1] = True
    hole_img = Image.fromarray(hole)
    for k in SHIFTS:
        rgb = shifted(tile_img, k * PITCH)
        alpha = shifted(glyph_img.filter(ImageFilter.GaussianBlur(0.5)), k * PITCH) / 255.0
        alpha = np.where(region, alpha, 0.0)[..., None]
        out = rgb * alpha + out * (1 - alpha)
        # Inside the hole: hair (anything not bright) becomes the hole's own
        # backdrop colour — the tile's backdrop is a shade off this one's.
        inside = (shifted(hole_img, k * PITCH) > 127) & region
        hair = inside & (a.mean(axis=2) < 215)
        clear = (shifted(hole_img, k * PITCH) > 127) & (a.mean(axis=2) >= 215)
        if hair.any() and clear.any():
            out[hair] = np.median(a[clear], axis=0)

    Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8)).save(PLATE, optimize=True)
    print(f"ok  {PLATE.name}")


if __name__ == "__main__":
    main()
