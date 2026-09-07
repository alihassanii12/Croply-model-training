import sys
from pathlib import Path

import torch
from torchvision import transforms

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from src.model import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------
# Checkpoint Loading
# --------------------------------------------------

def load_checkpoint(path, device):
    """
    Load a trained checkpoint and return the model
    (in eval mode) together with the class names.
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    classes = checkpoint.get(
        "classes",
        checkpoint.get("class_names")
    )

    num_classes = checkpoint.get(
        "num_classes",
        len(classes)
    )

    model = create_model(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, classes


# --------------------------------------------------
# Inference Transform
# --------------------------------------------------

def get_inference_transform():
    """
    Transformations applied to images at inference time.
    Mirrors the validation/test transforms in dataset.py.
    """

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


# --------------------------------------------------
# Class Name Prettifying
# --------------------------------------------------

def prettify_class_name(class_name):
    """
    Convert a raw class directory name like
    "Tomato___Late_blight" into a readable
    ("Tomato", "Late blight") tuple.
    """

    parts = class_name.split("___")

    plant = parts[0].replace("_", " ").strip()

    if len(parts) > 1:
        disease = parts[1].replace("_", " ").strip()
    else:
        disease = "Unknown"

    return plant, disease


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(model, classes, pil_image, device, top_k=5):
    """
    Run inference on a single PIL image and return
    the predicted class, its confidence, and the
    top-k predictions.
    """

    transform = get_inference_transform()

    image = pil_image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)[0]

    top_k = min(top_k, len(classes))

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k
    )

    top_predictions = [
        {
            "class_name": classes[index],
            "confidence": float(probability)
        }
        for probability, index in zip(top_probabilities, top_indices)
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }
