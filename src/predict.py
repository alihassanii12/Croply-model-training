import argparse
import sys
from pathlib import Path

import torch
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

DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "checkpoints" / "best_model.pth"


# --------------------------------------------------
# Argument Parsing
# --------------------------------------------------

def parse_arguments():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Croply - Plant Disease Prediction"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to classify"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained model checkpoint"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display"
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    arguments = parse_arguments()

    # ----------------------------------------------
    # Device
    # ----------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Croply - Plant Disease Prediction")
    print("=" * 60)

    print(f"Device: {device}")

    # ----------------------------------------------
    # Load Model
    # ----------------------------------------------

    print("\nLoading model...")

    model, classes = load_checkpoint(
        arguments.checkpoint,
        device
    )

    print(f"Checkpoint: {arguments.checkpoint}")
    print(f"Classes: {len(classes)}")

    # ----------------------------------------------
    # Load Image
    # ----------------------------------------------

    image_path = Path(arguments.image_path)

    if not image_path.exists():
        print(f"\nError: image not found: {image_path}")
        sys.exit(1)

    print(f"\nImage: {image_path}")

    image = Image.open(image_path)

    # ----------------------------------------------
    # Predict
    # ----------------------------------------------

    result = predict_image(
        model,
        classes,
        image,
        device,
        top_k=arguments.top_k
    )

    plant, disease = prettify_class_name(
        result["predicted_class"]
    )

    print("\nPrediction:")
    print(f"Plant:      {plant}")
    print(f"Disease:    {disease}")
    print(f"Class:      {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")

    # ----------------------------------------------
    # Top-K Table
    # ----------------------------------------------

    print(f"\nTop {len(result['top_predictions'])} predictions:")

    print("-" * 60)
    print(f"{'Rank':<6}{'Class':<44}{'Confidence':>10}")
    print("-" * 60)

    for rank, prediction in enumerate(result["top_predictions"], start=1):
        print(
            f"{rank:<6}"
            f"{prediction['class_name']:<44}"
            f"{prediction['confidence'] * 100:>9.2f}%"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()
