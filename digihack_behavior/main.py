import base64
import os
import uuid
from tempfile import NamedTemporaryFile

import cv2
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO
from typing import List, Dict, Any

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    RESULT_FOLDER = os.path.join('static', 'results')
    ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = os.environ.get('MODEL_PATH', './best.pt')
    DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')
    PORT = int(os.environ.get('PORT', 8080))  # Default to 8080 for Cloud Run

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.RESULT_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Behaviour Detection API",
    description=(
        "A FastAPI service for behaviour detection using YOLO. "
        "Upload images or base64-encoded frames for real-time inference."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLO model at startup
print(f"Loading model from {Config.MODEL_PATH}")
model = YOLO(Config.MODEL_PATH)
print("Model loaded successfully")


class FrameRequest(BaseModel):
    """
    Request body for the `/predict_frame` endpoint.

    Attributes:
        image (str): Base64-encoded image string. Can include a data URL prefix.
    """
    image: str = Field(..., description="Base64-encoded image string (with or without data URL prefix).")


class Detection(BaseModel):
    """
    Model for a single detection result.

    Attributes:
        label (str): Detected behaviour label.
        confidence (float): Confidence score of the detection.
    """
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """
    Response model for successful predictions.

    Attributes:
        result_url (str): URL to the annotated result image.
        detections (list[Detection]): List of detected behaviours with labels and confidence scores.
    """
    result_url: str
    detections: List[Dict[str, Any]] = []


@app.get("/", response_class=HTMLResponse, summary="Serve frontend page")
def index(request: Request):
    """
    Serve the main HTML page with the upload interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post(
    "/predict_frame",
    response_model=PredictionResponse,
    summary="Run behaviour detection on a base64-encoded image",
    description=(
        "Accepts a JSON payload containing a base64-encoded image. "
        "Runs YOLO object detection, saves the annotated result, and returns a URL to the output image."
    ),
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "result_url": "/static/results/abc123_pred.png",
                        "detections": [
                            {"label": "Avoid_Eye_Contact", "confidence": 0.98},
                            {"label": "Jumping", "confidence": 0.87}
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Bad Request – missing or invalid image data",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid base64 image encoding"}
                }
            }
        }
    }
)
async def predict_frame(body: FrameRequest):
    """
    Process a base64-encoded image frame, run inference, and return the result image URL.
    """
    img_b64 = body.image

    # Strip data URL prefix if present
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding")

    # Save to a temporary file
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(img_bytes)
        tmp_path = tmp_file.name

    try:
        # Run YOLO inference
        results = model.predict(source=tmp_path, device=Config.DEVICE, save=False)
        r = results[0]
        plotted = r.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        # Extract detection information
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

        # Save result
        out_name = f"{uuid.uuid4().hex}_pred.png"
        out_path = os.path.join(Config.RESULT_FOLDER, out_name)
        Image.fromarray(plotted_rgb).save(out_path)
        result_url = f"/static/results/{out_name}"

        return PredictionResponse(result_url=result_url, detections=detections)

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
