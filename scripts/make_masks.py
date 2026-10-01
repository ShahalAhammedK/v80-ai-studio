"""
Regenerate the precomputed person masks for the built-in campaign images.

Run this after adding or replacing anything in static/assets/campaigns/, then
commit the resulting "<name>.mask.png" files alongside their campaign image.

If a campaign has graphics masks (scripts/make_protect_v80_lite.py), build
those first: the do-not-edit area (or, failing that, the protect area) is
removed from the person mask here, so e.g. a hand-held phone isn't counted as
part of the person.

    pip install -r requirements-dev.txt
    python scripts/make_masks.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageChops  # noqa: E402

from backend.image_utils import validate_upload  # noqa: E402
from backend.precomputed_masks import CAMPAIGN_DIR, is_campaign_image, noedit_path_for, protect_path_for  # noqa: E402
from backend.segmentation import generate_person_mask  # noqa: E402


def main() -> int:
    sources = [p for p in sorted(CAMPAIGN_DIR.glob("*.png")) if is_campaign_image(p)]
    if not sources:
        print(f"No campaign images found in {CAMPAIGN_DIR}")
        return 1

    failed = False
    for image_path in sources:
        raw = image_path.read_bytes()
        image = validate_upload(image_path.name, "image/png", raw)
        mask = generate_person_mask(image)
        if mask is None:
            print(f"FAILED  {image_path.name} — segmentation unavailable (pip install -r requirements-dev.txt)")
            failed = True
            continue

        mask = mask.convert("L")
        note = ""
        protect_path = noedit_path_for(image_path)
        if not protect_path.exists():
            protect_path = protect_path_for(image_path)
        if protect_path.exists():
            protect = Image.open(protect_path).convert("L").resize(mask.size)
            mask = ImageChops.subtract(mask, protect)
            note = f" (minus {protect_path.name})"

        out_path = image_path.with_suffix(".mask.png")
        mask.save(out_path, optimize=True)
        print(f"ok      {image_path.name} {image.size} -> {out_path.name}{note}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
