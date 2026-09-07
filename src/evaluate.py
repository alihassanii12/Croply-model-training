from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import create_dataloaders
from src.model import create_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = ROOT_DIR / "outputs" / "metrics"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH = OUTPUT_DIR / "metrics.json"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Croply - Model Evaluation")
print("=" * 60)

print(f"Device: {device}")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

_, _, test_loader, classes = create_dataloaders()

print(f"Classes: {len(classes)}")
print(f"Test images: {len(test_loader.dataset)}")


# --------------------------------------------------
# Create model
# --------------------------------------------------

print("\nCreating model...")

model = create_model(
    num_classes=len(classes)
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_predictions = []
all_labels = []

print("\nEvaluating test dataset...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# --------------------------------------------------
# Print overall results
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    zero_division=0
)

print(report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = {
    "model": "EfficientNet-B0",
    "num_classes": len(classes),
    "test_images": len(test_loader.dataset),
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "training_epoch": checkpoint.get(
        "epoch",
        None
    ),
    "validation_accuracy": checkpoint.get(
        "validation_accuracy",
        None
    ),
    "validation_loss": checkpoint.get(
        "validation_loss",
        None
    ),
    "classes": classes,
}


with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\nMetrics saved to:")

print(METRICS_PATH)


# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

import numpy as np

cm_path = OUTPUT_DIR / "confusion_matrix.npy"

np.save(
    cm_path,
    cm
)

print("\nConfusion matrix saved to:")

print(cm_path)

print("\nEvaluation complete! 🌱")
