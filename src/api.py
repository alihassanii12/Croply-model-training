import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from src.utils import (
    load_checkpoint,
    predict_image,
    prettify_class_name,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = ROOT_DIR / "models" / "checkpoints" / "best_model.pth"

TOP_K = 5


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# Model State
# --------------------------------------------------

state = {
    "model": None,
    "classes": [],
}


# --------------------------------------------------
# Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once at startup.
    """

    print("Loading model...")

    model, classes = load_checkpoint(
        CHECKPOINT_PATH,
        device
    )

    state["model"] = model
    state["classes"] = classes

    print(f"Model loaded with {len(classes)} classes.")

    yield

    state["model"] = None
    state["classes"] = []


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="Croply - Plant Disease Detection API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
async def health():
    """
    Report API and model status.
    """

    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "num_classes": len(state["classes"]),
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded plant leaf image.
    """

    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    result = predict_image(
        state["model"],
        state["classes"],
        image,
        device,
        top_k=TOP_K
    )

    plant, disease = prettify_class_name(
        result["predicted_class"]
    )

    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top_predictions": result["top_predictions"],
        "plant": plant,
        "disease": disease,
    }
