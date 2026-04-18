import io
import gc
import logging
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from helpers import Models, crop_normalized_bbox_square
from classifier.classifier import Deepfaune, class_names
from megadetector.detection import run_detector


models = Models()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This replaces @app.on_event("startup") and @app.on_event("shutdown").
    It runs once at startup and once at shutdown.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models onto {device}...")
    # Detector
    models.detector = run_detector.load_detector("MDV5A")  # type: ignore
    # Classifier
    model_wrapper = Deepfaune("classifier/deepfaune_polish_lr4_checkpoint.pt")
    classifier = model_wrapper.model
    classifier.to(device)
    models.classifier = classifier
    models.transforms = model_wrapper.transforms
    logger.info("Models loaded.")

    yield
    gc.collect()
    logger.info("Shutting down inference server...")


app = FastAPI(
    title="Animal Detection API",
    description=("API for detecting and classifying animals in "
                 "images using MegaDetector and Deepfaune."),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    """
    Predict the animal species in the uploaded image.

    Args:
        image: The uploaded image file.

    Returns:
        JSONResponse with detection and classification results.
    """
    # Read image bytes
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # 1. Run detector
    detections = models.detector.generate_detections_one_image(pil_image)
    if not detections:
        return JSONResponse(content={"category": 0})
    detections = detections.get("detections")
    detection = max(detections, key=lambda d: d["conf"])
    if detection.get("conf") < 0.05:
        results = {
            "category": 0,
            "bbox": [0, 0, 0, 0],
            "detected_animal": "empty",
            "confidence": 0,
        }
        return JSONResponse(content=results)

    bbox = detection.get("bbox")
    category = detection.get("category")
    if int(category) != 1:
        return JSONResponse(content={"category": 0})

    # 2. Crop, transform and classify
    cropped_image = crop_normalized_bbox_square(pil_image, bbox)
    cropped_image_tensor = models.transforms(cropped_image).unsqueeze(0)
    logits = models.classifier.predict(cropped_image_tensor, withsoftmax=False)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

    # 3. Send predictions
    top2 = np.argsort(probabilities[0])[-2:][::-1]
    idx1, idx2 = top2
    species = class_names[idx1]
    species_2 = class_names[idx2]
    confidence = float(probabilities[0, idx1])
    confidence_2 = float(probabilities[0, idx2])

    results = {
        "category": 1,
        "bbox": bbox,
        "detected_animal": species,
        "confidence": confidence,
        "detected_animal_2": species_2,
        "confidence_2": confidence_2,
    }

    return JSONResponse(content=results)
