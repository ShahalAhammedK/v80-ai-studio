"""
Regenerate the precomputed person masks for the built-in campaign images.

Run this after adding or replacing anything in static/assets/campaigns/, then
commit the resulting "<name>.mask.png" files alongside their campaign image.

    pip install -r requirements-dev.txt
    python scripts/make_masks.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.image_utils import validate_upload  # noqa: E402
from backend.precomputed_masks import CAMPAIGN_DIR  # noqa: E402
from backend.segmentation import generate_person_mask  # noqa: E402


def main() -> int:
    sources = [p for p in sorted(CAMPAIGN_DIR.glob("*.png")) if not p.name.endswith(".mask.png")]
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

        out_path = image_path.with_suffix(".mask.png")
        mask.convert("L").save(out_path, optimize=True)
        print(f"ok      {image_path.name} {image.size} -> {out_path.name}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
