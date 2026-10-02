# V80 AI Studio — AI Person Replacement

Upload a campaign/advertising image and a photo of a different person; the app
replaces only the person in the campaign image with the new person's identity
while preserving everything else — logo, giant V, text, background, pose,
lighting, shadows and composition.

This build is **pure Python** (FastAPI backend + a no-build HTML/CSS/vanilla-JS
frontend) so it runs without installing Node.js.

## How it works

1. **Choose a format** and upload a person reference photo. Each format is a
   fixed V80 Lite campaign image in `static/assets/campaigns/`:
   - **Social Media** — `v80-lite.png`, a 1024x1607 portrait poster (default);
   - **Profile** — `v80-lite-profile.png`, a 1254x1254 square picture.
2. The backend looks up that campaign's **person mask**, precomputed and
   committed alongside the image (`backend/precomputed_masks.py`). Anything not
   in that set falls back to live `rembg` segmentation
   (`backend/segmentation.py`) where it's installed.
3. Clicking **Replace Person with AI** sends the campaign image, the person
   photo, and the mask to `POST /api/replace-person`
   (`app.py` → `backend/ai_edit.py`), which calls OpenAI's `images.edit` with
   both images and the mask.
4. The backend **composites** the AI result back onto the original
   (`adaptive_composite` in `backend/image_utils.py`): it keeps the new person
   wherever the AI drew them and restores every other pixel to the exact
   original. Graphics over the person (the slogan, the battery panel) always
   stay original.
5. The result is shown, downloadable, and shareable via the Web Share API
   (with manual-upload fallbacks for platforms without a direct image-attach
   API).

**The OpenAI API key is only ever read server-side** (`backend/config.py`,
loaded from `.env.local`/`.env`) and is never sent to the browser or returned
in any API response.

## Setup

1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key to `.env.local` (already created for you):

```
OPENAI_API_KEY=your_api_key_here
IMAGE_MODEL=gpt-image-2.5-sunburst
```

`IMAGE_MODEL` is configurable — check the
[OpenAI image generation docs](https://platform.openai.com/docs/guides/image-generation)
for the current model list before changing it.

4. Run the app:

```bash
python app.py
```

5. Open <http://localhost:8000>.

Masks for the built-in campaigns are committed, so nothing is downloaded at
runtime and `requirements.txt` stays slim (no `rembg`/`onnxruntime`, which
together cost ~500MB and ~1GB of peak RAM).

### Changing a campaign image

Each campaign image has derived masks next to it: `.mask.png` (the person),
`.protect.png` (graphics that always stay original), `.noedit.png` (what the
AI may not edit) and `.hand.png` (the original man's hand, erased from the
AI's copy for a woman). The Profile image also has `.plate.png` (the picture
with the "10000" digits completed where the model's hair covered them) and
`.behind.png` (those digits): the new person is placed in front of the
completed digits, so they show wherever the new hair doesn't reach. The
Profile image and its plate come from the designer's file via
`scripts/rebuild_profile_headline.py`. The graphics masks are measured per
image in `scripts/make_protect_v80_lite.py`; add or re-measure the image
there, then:

```bash
pip install -r requirements-dev.txt
python scripts/make_protect_v80_lite.py
python scripts/make_masks.py
python scripts/make_protect_v80_lite.py
```

The AI prompt (`build_prompt` in `backend/ai_edit.py`) describes the shared
layout of both images; revisit it if a new image changes the pose or graphics.

Masks are keyed by a SHA-256 of the campaign file's bytes, computed from disk
at import — so a stale mask can't silently survive a swapped image. If a
campaign has no matching mask and `rembg` isn't installed, automatic masking is
simply disabled and the app falls back to the manual mask editor — it never
fakes a mask.

## Deploying

A `Dockerfile` and `render.yaml` are included; the image is ~250MB and runs in
~250MB of RAM, so it fits a free/small tier. Set `OPENAI_API_KEY` (and
optionally `EXPLABS_API_KEY`, `GEMINI_API_KEY`) as secrets in the host's
dashboard — never in the repo.

## Project structure

```
app.py                     FastAPI app: routes, validation wiring, error handling
backend/
  config.py                Env vars (API key, model name, limits)
  errors.py                UserFacingError — safe messages returned to the browser
  image_utils.py           Validation, resizing, mask conversion, final compositing
  precomputed_masks.py     Committed masks for the built-in campaigns (no model needed)
  segmentation.py          Fallback person mask via rembg (dev/custom images only)
  ai_edit.py                Prompt construction + OpenAI images.edit call
  rate_limit.py             Simple in-memory per-IP rate limiting
static/
  index.html                Single-page UI (all 6 workflow steps)
  css/style.css              Premium vivo-blue design system
  js/
    api.js                   fetch wrappers for the backend API
    app.js                   Main workflow controller
    maskEditor.js             Canvas-based brush/eraser mask editor
    slider.js                 Before/after drag slider
    share.js                  Download + Web Share/clipboard helpers
    toast.js                  Error/info/success toasts
```

## Notes on fidelity

Priority order, matching the product requirement that this is an editing tool
and not a creative generator:

1. Preserve campaign composition, background, logo, text, product
2. Preserve original pose, perspective, lighting
3. Preserve the reference person's identity
4. Photorealism

Steps 1–2 are enforced twice: once through the dynamically-built edit prompt
(`backend/ai_edit.py:build_prompt`), and again mechanically through the final
pixel composite, which is the real guarantee — the prompt guides the model,
but the composite is what actually makes non-person pixels byte-identical to
the original.

## Security

- All uploads are validated server-side (type, size, dimensions, decodability)
  regardless of what the browser already checked.
- Uploaded/generated images are processed in memory for the duration of a
  single request and are not written to disk or persisted anywhere.
- The OpenAI key never leaves the server; API errors are caught and mapped to
  generic, user-safe messages — no stack traces or internal details are ever
  returned to the browser (see `backend/errors.py` and the exception handlers
  in `app.py`).
- `/api/replace-person` is rate-limited per IP in-memory (`backend/rate_limit.py`).

## Known limitations

- The in-memory rate limiter resets on restart and doesn't share state across
  multiple worker processes — fine for local/single-instance use, not for a
  multi-instance production deployment.
- Automatic segmentation quality depends on `rembg`/`onnxruntime` installing
  successfully on your platform; the manual mask editor is always available
  as a fallback.
- X/Twitter's web share intent cannot attach an image directly — the UI is
  explicit about this and offers a download instead, per platform's actual
  capabilities.
