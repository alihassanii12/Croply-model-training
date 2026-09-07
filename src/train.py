import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from src.dataset import create_dataloaders
from src.model import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 1

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Croply - Plant Disease Detection Training")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: Not available")
    print("Training will run on CPU.")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading dataset...")

train_loader, val_loader, test_loader, classes = create_dataloaders()

num_classes = len(classes)

print(f"Classes: {num_classes}")
print(f"Training images: {len(train_loader.dataset)}")
print(f"Validation images: {len(val_loader.dataset)}")
print(f"Test images: {len(test_loader.dataset)}")


# --------------------------------------------------
# Create Model
# --------------------------------------------------

print("\nCreating model...")

model = create_model(
    num_classes=num_classes
)

model = model.to(device)


# --------------------------------------------------
# Loss Function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.AdamW(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# --------------------------------------------------
# Learning Rate Scheduler
# --------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# --------------------------------------------------
# Training Function
# --------------------------------------------------

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=True
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        current_accuracy = (
            correct / total
        ) * 100

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{current_accuracy:.2f}%"
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation Function
# --------------------------------------------------

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=True
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    ) * 100

    return (
        validation_loss,
        validation_accuracy
    )


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

best_validation_accuracy = 0.0

print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 60)

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = validate()

    # Update scheduler
    scheduler.step(val_loss)

    print("\nResults:")
    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss:  {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate:    {current_lr:.6f}"
    )

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if val_accuracy > best_validation_accuracy:

        best_validation_accuracy = val_accuracy

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_accuracy,
            "validation_loss": val_loss,
            "classes": classes,
            "num_classes": num_classes,
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print(
            f"\n✓ Best model saved:"
        )

        print(
            BEST_MODEL_PATH
        )


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n" + "=" * 60)
print("Training Finished")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"Model saved at:\n"
    f"{BEST_MODEL_PATH}"
)
