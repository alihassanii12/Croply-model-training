from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Dataset path
DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "plantvillage dataset"
    / "segmented"
)

# Image size
IMAGE_SIZE = 224

# Batch size
BATCH_SIZE = 32

# Reproducibility
SEED = 42


# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Validation/Test transformations
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def create_datasets():
    """
    Creates train, validation and test datasets.
    """

    # Load dataset once to get class information
    full_dataset = datasets.ImageFolder(
        root=str(DATASET_DIR)
    )

    total_size = len(full_dataset)

    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=generator
    )

    # Separate datasets with different transforms
    train_dataset_full = datasets.ImageFolder(
        root=str(DATASET_DIR),
        transform=train_transform
    )

    eval_dataset = datasets.ImageFolder(
        root=str(DATASET_DIR),
        transform=val_test_transform
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset_full,
        train_indices.indices
    )

    val_dataset = torch.utils.data.Subset(
        eval_dataset,
        val_indices.indices
    )

    test_dataset = torch.utils.data.Subset(
        eval_dataset,
        test_indices.indices
    )

    return train_dataset, val_dataset, test_dataset, full_dataset.classes


def create_dataloaders():
    """
    Creates PyTorch DataLoaders.
    """

    train_dataset, val_dataset, test_dataset, classes = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":

    print("=" * 50)
    print("Croply Dataset")
    print("=" * 50)

    print(f"Dataset path: {DATASET_DIR}")

    train_loader, val_loader, test_loader, classes = create_dataloaders()

    print(f"\nNumber of classes: {len(classes)}")
    print(f"Total images: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    print("\nClasses:")

    for index, class_name in enumerate(classes):
        print(f"{index:02d}: {class_name}")

    images, labels = next(iter(train_loader))

    print("\nBatch information:")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
