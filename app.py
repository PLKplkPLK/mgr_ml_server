import io
import gc
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from helpers import Models
# from models.detector import Detector
from models.classifier import Deepfaune


models = Models()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This replaces @app.on_event("startup") and @app.on_event("shutdown").
    It runs once at startup and once at shutdown.
    """
    print("Loading models...")
    models.detector = None #Detector()
    models.classifier = Deepfaune('models/deepfaune_polish_lr4_checkpoint.pt')
    print("Models loaded.")

    yield
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Shutting down inference server...")


app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read image bytes
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # 1. Run detector
    detections = models.detector.predict(pil_image)

    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # 2. Crop and classify
        crop = pil_image.crop((x1, y1, x2, y2))
        classification = models.classifier.classify(crop)

        results.append({
            "bbox": det["bbox"],
            "detection_confidence": det["confidence"],
            "class": classification["label"],
            "class_confidence": classification["confidence"]
        })

    return JSONResponse(content={"detections": results})
