import os
import io
import uuid
import base64
import tempfile
from typing import Dict, Any
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import cv2

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = os.path.join('static', 'results')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

MODEL_PATH = os.environ.get('MODEL_PATH', './best.pt')
DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API",
    description=(
        "A FastAPI service that runs emotion detection on uploaded images or base64-encoded frames. "
        "Supports real-time inference via `/predict_frame` endpoint."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model once at startup
print('Loading model from', MODEL_PATH)
model = YOLO(MODEL_PATH)
print('Model loaded')


class FrameRequest(BaseModel):
    """
    Request body for /predict_frame endpoint.
    
    Attributes:
        image (str): Base64-encoded image string. May optionally include a data URL prefix 
                     (e.g., `data:image/jpeg;base64,...`). The actual base64 payload will be extracted.
    """
    image: str


class PredictionResponse(BaseModel):
    """
    Response model for successful prediction.
    
    Attributes:
        result_url (str): URL to the annotated result image (relative to server root).
    """
    result_url: str


@app.get("/", response_class=HTMLResponse, summary="Serve frontend page")
def index(request: Request):
    """
    Serve the main HTML page with the upload interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post(
    '/predict_frame',
    response_model=PredictionResponse,
    summary="Run emotion detection on a base64-encoded image",
    description=(
        "Accepts a JSON payload containing a base64-encoded image (optionally with data URL prefix). "
        "Runs YOLO object detection, saves the annotated result, and returns a URL to the output image."
    ),
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {"result_url": "/static/results/abc123_pred.png"}
                }
            }
        },
        400: {
            "description": "Bad Request – missing or invalid image data"
        }
    }
)
async def predict_frame(request: Request, body: FrameRequest):
    """
    Process a base64-encoded image frame, run inference, and return the result image URL.
    """
    img_b64 = body.image

    # Strip data URL prefix if present
    if img_b64.startswith('data:'):
        img_b64 = img_b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding") from e

    # Save to temporary file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd)
    try:
        with open(tmp_path, 'wb') as f:
            f.write(img_bytes)

        # Run YOLO inference
        results = model.predict(source=tmp_path, device=DEVICE, save=False)
        r = results[0]
        plotted = r.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        # Save result
        out_name = f"{uuid.uuid4().hex}_pred.png"
        out_path = os.path.join(RESULT_FOLDER, out_name)
        Image.fromarray(plotted_rgb).save(out_path)
        result_url = f"/static/results/{out_name}"

        return PredictionResponse(result_url=result_url)

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 30018))
    uvicorn.run(app, host='0.0.0.0', port=port)
