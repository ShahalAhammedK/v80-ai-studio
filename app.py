import base64
import io
import json
import logging

import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from backend.ai_edit import replace_person
from backend.config import GEMINI_API_KEY, IMAGE_PROVIDER, OPENAI_API_KEY
from backend.errors import UserFacingError
from backend.gender_detect import detect_gender
from backend.image_utils import image_to_png_bytes, validate_upload
from backend.precomputed_masks import count as precomputed_mask_count, lookup as lookup_precomputed_mask
from backend.rate_limit import check_rate_limit
from backend.segmentation import generate_person_mask, is_installed as segmentation_installed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v80.app")

app = FastAPI(title="V80 AI Studio")

app.mount("/static", StaticFiles(directory="static"), name="static")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.exception_handler(UserFacingError)
async def user_facing_error_handler(_request: Request, exc: UserFacingError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "Something went wrong on our end. Please try again."})


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "aiReady": bool(GEMINI_API_KEY) or bool(OPENAI_API_KEY),
        "imageProvider": IMAGE_PROVIDER,
        "autoMaskAvailable": precomputed_mask_count() > 0 or segmentation_installed(),
    }


@app.post("/api/segment")
async def segment(campaign: UploadFile):
    data = await campaign.read()
    image = validate_upload(campaign.filename, campaign.content_type, data)

    # The built-in campaigns have committed masks, so the common path needs no
    # model, no download and no meaningful RAM. Anything else falls through to
    # live segmentation, which is a no-op when rembg isn't installed.
    mask = await run_in_threadpool(lookup_precomputed_mask, data, image.size)
    if mask is None:
        mask = await run_in_threadpool(generate_person_mask, image)
    if mask is None:
        return {"available": False, "mask": None}

    mask_png = image_to_png_bytes(mask.convert("L"))
    return {"available": True, "mask": base64.b64encode(mask_png).decode("ascii")}


@app.post("/api/detect-gender")
async def detect_gender_endpoint(person: UploadFile):
    data = await person.read()
    image = validate_upload(person.filename, person.content_type, data)
    gender = await run_in_threadpool(detect_gender, image)
    return {"gender": gender}


@app.post("/api/replace-person")
async def replace_person_endpoint(
    request: Request,
    campaign: UploadFile,
    person: UploadFile,
    mask: UploadFile,
    settings: str = Form("{}"),
):
    check_rate_limit(client_ip(request))

    try:
        parsed_settings = json.loads(settings) if settings else {}
    except json.JSONDecodeError:
        parsed_settings = {}

    campaign_data = await campaign.read()
    person_data = await person.read()
    mask_data = await mask.read()

    campaign_image = validate_upload(campaign.filename, campaign.content_type, campaign_data)
    person_image = validate_upload(person.filename, person.content_type, person_data)

    if not mask_data:
        raise UserFacingError("A person mask is required. Please review or paint the mask before continuing.")

    try:
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
    except Exception:  # noqa: BLE001
        raise UserFacingError("We couldn't read the mask. Please try editing the mask again.")

    if mask_image.size != campaign_image.size:
        mask_image = mask_image.resize(campaign_image.size, Image.LANCZOS)

    coverage = sum(mask_image.point(lambda v: 1 if v > 127 else 0).getdata())
    if coverage < 50:
        raise UserFacingError(
            "The mask doesn't cover a person yet. Please use Edit Mask to paint over the subject."
        )

    # CPU-bound (especially the local provider, which can take minutes) — run off
    # the event loop so the rest of the site stays responsive during generation.
    result_image = await run_in_threadpool(
        replace_person, campaign_image, person_image, mask_image, parsed_settings
    )

    result_png = image_to_png_bytes(result_image)
    return {"image": base64.b64encode(result_png).decode("ascii")}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
