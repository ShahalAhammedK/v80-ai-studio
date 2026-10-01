"""
Build the two graphics masks for the V80 Lite campaign (static/assets/
campaigns/v80-lite.png). Graphics overlap the person in this layout:

  - the green battery panel ("32H 37M", Guinness badge) sits over the torso,
    and person segmentation swallows it whole;
  - the "ALWAYS POWERED" slogan overlaps the top of the hair;
  - the phone is held out in front of the body, gripped by the hand.

v80-lite.protect.png — always the original pixel in the final image: the
    panel, the slogan letters and the headline above them. These are layered
    ON TOP of the person, so restoring them exactly is also the natural look.

v80-lite.noedit.png — protect plus the phone; sent to the AI as "do not
    edit". The phone is deliberately NOT in protect: the hand wraps around it,
    and locking it in the final image cut the new person's fingertips off at
    its edge. The final composite keeps the original phone wherever the AI
    left it unchanged (see adaptive_composite in backend/image_utils.py).

v80-lite.hand.png — the original man's hand gripping the phone (fingers and
    thumb). For a woman it is erased from the picture sent to the AI: an
    image-edit model traces what is already there, so with the man's hand in
    place every woman came out holding the phone with his large, broad hand,
    whatever the prompt said. With it erased she gets her own slender hand.

The regions are measured from this specific image, so this script is
campaign-specific. If the campaign image changes, re-measure the coordinates
and rerun, then run scripts/make_masks.py (the hand mask reads the person
mask, so if that changes, rerun this script afterwards).

    python scripts/make_protect_v80_lite.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
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

    # Battery panel: a rounded rectangle, x 110-886, y 1139-1544, corner
    # radius 46 (fitted to the rim to within ~1px on every corner), drawn
    # exactly - no outset. An outset restores slivers of whatever was beside
    # the panel in the ORIGINAL (old jacket, old backdrop), which show as
    # notches once the new person is drawn differently there.
    panel = shape_mask(size, lambda d: d.rounded_rectangle([110, 1139, 886, 1544], radius=46, fill=255))
    # The battery nub on the right edge: traced row by row from its bright
    # green rim, which is clean on that side.
    rim = (g - r > 55) & (g > 90)
    for y in range(1139, 1545):
        right = np.where(rim[y, 700:])[0]
        if len(right):
            panel[y, 886:700 + right.max() + 1] = True

    # Phone: an outset quadrilateral around the tilted handset, minus skin, so
    # the fingers and thumb wrapped over its edges stay replaceable.
    quad = shape_mask(size, lambda d: d.polygon([(152, 528), (452, 498), (560, 1100), (284, 1170)], fill=255))
    skin = binary_dilation((r > g + 12) & (r > b + 5) & (r > 60), iterations=1)
    phone = quad & ~skin

    # Slogan letters (saturated green) — protected letter by letter, so hair
    # can still grow into the gaps between them — and everything above. Only
    # the letter pixels themselves: growing them pulls in the ORIGINAL hair
    # that touched the letters, which shows as dark specks over a new hairline.
    band = np.zeros(rgb.shape[:2], bool)
    band[320:400] = True
    letters = band & (g > r + 40) & (g > b + 15)
    above = np.zeros(rgb.shape[:2], bool)
    above[:330] = True

    protect = panel | letters | above
    noedit = protect | phone

    # The hand: skin within the person, in two boxes (the fingers curled over
    # the phone's left edge, the thumb on its right edge), grown a little to
    # take the skin's soft edge too. Never the phone itself.
    person = np.array(Image.open(CAMPAIGN.with_suffix(".mask.png")).convert("L")) > 127
    boxes = shape_mask(size, lambda d: (d.rectangle([60, 700, 300, 1115], fill=255),
                                        d.rectangle([455, 790, 580, 1010], fill=255)))
    raw_skin = (r > g + 12) & (r > b + 5) & (r > 60)
    hand = binary_dilation(raw_skin & person & boxes, iterations=8) & boxes & ~noedit
    # protect gets a sub-pixel soft edge: a hard on/off edge leaves a stair-
    # stepped seam where the panel's anti-aliased rim meets the new person.
    # noedit only guides the AI, so it stays hard.
    outputs = (
        (".protect.png", Image.fromarray((protect * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))),
        (".noedit.png", Image.fromarray((noedit * 255).astype(np.uint8))),
        (".hand.png", Image.fromarray((hand * 255).astype(np.uint8))),
    )
    for suffix, image in outputs:
        out = CAMPAIGN.with_suffix(suffix)
        image.save(out, optimize=True)
        print(f"ok  {out.name}: {(np.asarray(image) > 127).mean() * 100:.1f}% of the image ({h} rows)")


if __name__ == "__main__":
    main()
