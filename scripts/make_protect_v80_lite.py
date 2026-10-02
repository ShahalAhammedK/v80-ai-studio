"""
Build the graphics masks for the V80 Lite campaigns in static/assets/campaigns/:

  v80-lite.png          — "Social Media" (portrait, 1024x1607)
  v80-lite-profile.png  — "Profile" (square, 1607x1607)

Both show the person chest-up, holding the phone out to the camera, with
graphics layered over them:

  - the green battery panel ("32H 37M", Guinness badge) sits over the torso,
    and person segmentation swallows it whole;
  - the phone is held out in front of the body, gripped by the hand;
  - Social Media: the "ALWAYS POWERED" slogan overlaps the top of the hair;
  - Profile: the hair overlaps the bottom of the big "10000" from IN FRONT,
    and two green lightning bolts float beside the person. The slogan sits
    clear of the person, to the left.

<name>.protect.png — always the original pixel in the final image: the
    panel, the slogan letters and the headline above them. These are layered
    ON TOP of the person, so restoring them exactly is also the natural look.

<name>.noedit.png — protect plus the phone; sent to the AI as "do not
    edit". The phone is deliberately NOT in protect: the hand wraps around it,
    and locking it in the final image cut the new person's fingertips off at
    its edge. The final composite keeps the original phone wherever the AI
    left it unchanged (see adaptive_composite in backend/image_utils.py).

<name>.behind.png — (Profile) graphics BEHIND the person: the "10000" digits
    around the head, which the new hair may cover. Taken from the digits-
    completed plate (scripts/rebuild_profile_headline.py), so it includes the
    parts the old hair hid. Where the new person doesn't cover them, the
    final image shows the plate's digits (see adaptive_composite).

<name>.hand.png — the original man's hand gripping the phone (fingers and
    thumb). For a woman it is erased from the picture sent to the AI: an
    image-edit model traces what is already there, so with the man's hand in
    place every woman came out holding the phone with his large, broad hand,
    whatever the prompt said. With it erased she gets her own slender hand.

The regions are measured from each specific image (see CAMPAIGNS). If an
image changes, re-measure its coordinates, then run:

    python scripts/make_protect_v80_lite.py   # protect + noedit
    python scripts/make_masks.py              # person masks (reads noedit)
    python scripts/make_protect_v80_lite.py   # again: the hand mask reads the person mask
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_DIR = ROOT / "static" / "assets" / "campaigns"

# Measured per image, in that image's pixels.
CAMPAIGNS = {
    "v80-lite": {
        # Battery panel: a rounded rectangle fitted to its rim to within ~1px.
        "panel": [110, 1139, 886, 1544],
        "panel_radius": 46,
        # An outset quadrilateral around the tilted handset.
        "phone": [(152, 528), (452, 498), (560, 1100), (284, 1170)],
        # Green graphics protected pixel by pixel (x0, y0, x1, y1, how much
        # greener than red, how much greener than blue): here the slogan.
        "green": [(0, 320, 1024, 400, 40, 15)],
        # Every row above this is protected outright (the headline).
        "above": 330,
        # Boxes holding the hand: the fingers on the phone's left edge, the
        # thumb on its right edge.
        "hand_boxes": [[60, 700, 300, 1115], [455, 790, 580, 1010]],
    },
    "v80-lite-profile": {
        "panel": None,  # fitted from the rim below `panel_from`
        "panel_from": 1080,
        "panel_radius": 59,
        "phone": [(435, 495), (740, 455), (850, 1065), (565, 1155)],
        "green": [
            # The digits left of the head ("1", first two "0"s), pixel by
            # pixel — never the white between them, so nothing there can clip
            # a new person's hair.
            (250, 125, 650, 365, 12, 3),
            (290, 375, 790, 440, 40, 15),     # "ALWAYS POWERED"
            (140, 790, 360, 1030, 30, 10),    # lightning bolt, left
            (1160, 600, 1510, 960, 30, 10),   # lightning bolt, right
        ],
        "above": 125,  # "vivo V80 Lite 5G"
        # "mAh", protected outright. The digits around the head are not
        # protected at all: the person is in front of them, and they are
        # handled as graphics behind the person (see .behind.png). A
        # protected box here clipped any hair that rose higher or reached
        # wider than the original model's, cutting off part of the head.
        "full": [(1135, 225, 1330, 290)],
        "behind": [(650, 125, 1345, 365)],
        "hand_boxes": [[380, 690, 575, 1130], [785, 805, 870, 935]],
    },
}


def shape_mask(size, draw):
    mask = Image.new("L", size, 0)
    draw(ImageDraw.Draw(mask))
    return np.array(mask) > 0


def build(name: str, spec: dict) -> None:
    campaign = CAMPAIGN_DIR / f"{name}.png"
    image = Image.open(campaign).convert("RGB")
    rgb = np.array(image).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = rgb.shape[0]
    size = image.size

    # Battery panel, drawn exactly - no outset. An outset restores slivers of
    # whatever was beside the panel in the ORIGINAL (old jacket, old
    # backdrop), which show as notches once the new person is drawn
    # differently there.
    rim = (g - r > 55) & (g > 90)
    if spec["panel"] is None:
        # Fit the body to the rim: its left/top/bottom edges, and the right
        # edge as the rim's rightmost point on the row 60px below the top.
        below = rim.copy()
        below[: spec["panel_from"]] = False
        ys, xs = np.where(below)
        right = np.where(below[ys.min() + 60])[0].max()
        spec["panel"] = [int(xs.min()), int(ys.min()), int(right), int(ys.max())]
        print(f"    {name}: panel fitted to {spec['panel']}")
    x0, y0, x1, y1 = spec["panel"]
    panel = shape_mask(size, lambda d: d.rounded_rectangle(spec["panel"], radius=spec["panel_radius"], fill=255))
    # The battery nub on the right edge: traced row by row from its bright
    # green rim, which is clean on that side.
    mid = (x0 + x1) // 2
    for y in range(y0, y1 + 1):
        right = np.where(rim[y, mid:])[0]
        if len(right):
            panel[y, x1:mid + right.max() + 1] = True

    # Phone: the quadrilateral minus skin, so the fingers and thumb wrapped
    # over its edges stay replaceable.
    quad = shape_mask(size, lambda d: d.polygon(spec["phone"], fill=255))
    skin = binary_dilation((r > g + 12) & (r > b + 5) & (r > 60), iterations=1)
    phone = quad & ~skin

    # Green graphics near the person (slogan letters, headline digits, bolts)
    # — protected pixel by pixel, so hair can still grow into the gaps between
    # them — and everything above. Only the green pixels themselves: growing
    # them pulls in the ORIGINAL hair that touched them, which shows as dark
    # specks over a new hairline.
    letters = np.zeros(rgb.shape[:2], bool)
    for gx0, gy0, gx1, gy1, over_r, over_b in spec["green"]:
        box = np.zeros(rgb.shape[:2], bool)
        box[gy0:gy1, gx0:gx1] = True
        letters |= box & (g > r + over_r) & (g > b + over_b)
    above = np.zeros(rgb.shape[:2], bool)
    above[: spec["above"]] = True

    full = np.zeros(rgb.shape[:2], bool)
    for bx0, by0, bx1, by1 in spec.get("full", []):
        full[by0:by1, bx0:bx1] = True
    graphics = above | full
    for ox0, oy0, ox1, oy1 in spec.get("open", []):
        graphics[oy0:oy1, ox0:ox1] = False
    graphics |= letters

    protect = panel | graphics
    noedit = protect | phone
    # protect gets a sub-pixel soft edge: a hard on/off edge leaves a stair-
    # stepped seam where the panel's anti-aliased rim meets the new person.
    # noedit only guides the AI, so it stays hard.
    outputs = [
        (".protect.png", Image.fromarray((protect * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))),
        (".noedit.png", Image.fromarray((noedit * 255).astype(np.uint8))),
    ]

    # The hand: skin within the person, in the hand boxes, grown a little to
    # take the skin's soft edge too. Never the phone itself. Needs the person
    # mask, so it's skipped until scripts/make_masks.py has made one.
    person_path = campaign.with_suffix(".mask.png")
    if person_path.exists():
        person = np.array(Image.open(person_path).convert("L")) > 127
        boxes = shape_mask(size, lambda d: [d.rectangle(box, fill=255) for box in spec["hand_boxes"]])
        raw_skin = (r > g + 12) & (r > b + 5) & (r > 60)
        hand = binary_dilation(raw_skin & person & boxes, iterations=8) & boxes & ~noedit
        outputs.append((".hand.png", Image.fromarray((hand * 255).astype(np.uint8))))
    else:
        print(f"--  {name}.hand.png skipped: run scripts/make_masks.py, then this script again")

    # Graphics behind the person: the plate's digits inside the behind boxes,
    # except anything protected (mAh).
    plate_path = campaign.with_suffix(".plate.png")
    if plate_path.exists() and spec.get("behind"):
        prgb = np.array(Image.open(plate_path).convert("RGB")).astype(int)
        pr, pg, pb = prgb[..., 0], prgb[..., 1], prgb[..., 2]
        boxes = np.zeros(rgb.shape[:2], bool)
        for ox0, oy0, ox1, oy1 in spec["behind"]:
            boxes[oy0:oy1, ox0:ox1] = True
        behind = boxes & (pg > pr + 12) & (pg > pb + 3) & ~protect
        outputs.append((".behind.png", Image.fromarray((behind * 255).astype(np.uint8))))

    for suffix, mask in outputs:
        out = campaign.with_suffix(suffix)
        mask.save(out, optimize=True)
        print(f"ok  {out.name}: {(np.asarray(mask) > 127).mean() * 100:.1f}% of the image ({h} rows)")


def main() -> None:
    for name, spec in CAMPAIGNS.items():
        build(name, spec)


if __name__ == "__main__":
    main()
