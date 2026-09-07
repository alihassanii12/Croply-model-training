import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


NUM_CLASSES = 38


def create_model(num_classes=NUM_CLASSES):
    """
    Create EfficientNet-B0 model with a custom
    classifier for Croply's 38 disease classes.
    """

    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # Freeze pretrained feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace original classifier
    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, num_classes)
    )

    return model


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("Croply Disease Detection Model")
    print("=" * 50)

    print(f"Device: {device}")

    model = create_model()
    model = model.to(device)

    print("\nModel created successfully!")
    print(f"Number of classes: {NUM_CLASSES}")

    # Test with dummy image
    dummy_input = torch.randn(
        2, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\nExpected output:")
    print("[batch_size, 38]")

    print("\nModel is ready for training 🌱")
