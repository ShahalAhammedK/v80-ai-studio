"""
Build the protect mask for the V80 Lite campaign (static/assets/campaigns/
v80-lite.png).

A protect mask marks pixels that must always come from the original campaign,
whatever the AI returns and however far the person mask is grown. The V80
Lite layout needs one because graphics overlap the person:

  - the green battery panel ("32H 37M", Guinness badge) sits over the torso,
    and person segmentation swallows it whole;
  - the phone is held out in front of the body, gripped by the hand;
  - the "ALWAYS POWERED" slogan overlaps the top of the hair.

Without it, the AI would redraw the panel text, the phone and the slogan.

The regions are measured from this specific image, so this script is
campaign-specific. If the campaign image changes, re-measure the coordinates
and rerun, then run scripts/make_masks.py.

    python scripts/make_protect_v80_lite.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "static" / "assets" / "campaigns" / "v80-lite.png"


def shape_mask(size, draw):
    mask = Image.new("L", size, 0)
    draw(ImageDraw.Draw(mask))
    return np.array(mask) > 0


def main() -> None:
    image = Image.open(CAMPAIGN).convert("RGB")
    rgb = np.array(image).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = rgb.shape[0]
    size = image.size

    # Battery panel (x 110-886, y 1139-1544) plus its nub on the right
    # (to x 927, y 1233-1456). Outset 3px with a small corner radius, so the
    # corners are over-protected rather than under-protected.
    panel = shape_mask(size, lambda d: (
        d.rounded_rectangle([107, 1136, 889, 1547], radius=20, fill=255),
        d.rounded_rectangle([884, 1230, 930, 1459], radius=8, fill=255),
    ))

    # Phone: an outset quadrilateral around the tilted handset, minus skin, so
    # the fingers and thumb wrapped over its edges stay replaceable.
    quad = shape_mask(size, lambda d: d.polygon([(152, 528), (452, 498), (560, 1100), (284, 1170)], fill=255))
    skin = binary_dilation((r > g + 12) & (r > b + 5) & (r > 60), iterations=1)
    phone = quad & ~skin

    # Slogan letters (saturated green) — protected letter by letter, so hair
    # can still grow into the gaps between them — and everything above.
    band = np.zeros(rgb.shape[:2], bool)
    band[320:400] = True
    letters = binary_dilation(band & (g > r + 30) & (g > b + 10), iterations=2)
    above = np.zeros(rgb.shape[:2], bool)
    above[:330] = True

    protect = panel | phone | letters | above
    out = CAMPAIGN.with_suffix(".protect.png")
    Image.fromarray((protect * 255).astype(np.uint8)).save(out, optimize=True)
    print(f"ok  {out.name}: {protect.mean() * 100:.1f}% of the image protected ({h} rows)")


if __name__ == "__main__":
    main()
