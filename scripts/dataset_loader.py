from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 1. DATASET PATH
# =========================================================

DATASET_PATH = Path(
    r"D:\Project\PulmoVerse_Preprocessed"
)


# =========================================================
# 2. IMAGE SIZE
# =========================================================

IMAGE_SIZE = 224


# =========================================================
# 3. BATCH SIZE
# =========================================================

BATCH_SIZE = 32


# =========================================================
# 4. NORMALIZATION VALUES
# =========================================================
# ImageNet pretrained models use these standard values.

NORMALIZE_MEAN = [
    0.485,
    0.456,
    0.406
]

NORMALIZE_STD = [
    0.229,
    0.224,
    0.225
]


# =========================================================
# 5. TRAIN TRANSFORMS
# =========================================================
# Augmentation is applied ONLY to training images.

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])


# =========================================================
# 6. VALIDATION TRANSFORMS
# =========================================================
# NO augmentation for validation.

validation_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])


# =========================================================
# 7. TEST TRANSFORMS
# =========================================================
# NO augmentation for test.

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])


# =========================================================
# 8. LOAD TRAIN DATASET
# =========================================================

train_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "train",
    transform=train_transform
)


# =========================================================
# 9. LOAD VALIDATION DATASET
# =========================================================

validation_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "validation",
    transform=validation_transform
)


# =========================================================
# 10. LOAD TEST DATASET
# =========================================================

test_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "test",
    transform=test_transform
)


# =========================================================
# 11. CREATE DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# 12. DISPLAY DATASET INFORMATION
# =========================================================

print("\n========================================")
print("PULMOVerse PyTorch Dataset")
print("========================================")

print("\nClasses:")
print(train_dataset.classes)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

print("\nDataset Size:")
print(
    "Train      :",
    len(train_dataset)
)

print(
    "Validation :",
    len(validation_dataset)
)

print(
    "Test       :",
    len(test_dataset)
)


# =========================================================
# 13. CHECK ONE BATCH
# =========================================================

images, labels = next(
    iter(train_loader)
)

print("\n========================================")
print("BATCH TEST")
print("========================================")

print(
    "Image Shape :",
    images.shape
)

print(
    "Label Shape :",
    labels.shape
)

print(
    "Image Data Type :",
    images.dtype
)

print(
    "Label Data Type :",
    labels.dtype
)


# =========================================================
# 14. CHECK NORMALIZATION
# =========================================================

print("\n========================================")
print("NORMALIZATION TEST")
print("========================================")

print(
    "Minimum pixel value :",
    images.min().item()
)

print(
    "Maximum pixel value :",
    images.max().item()
)


# =========================================================
# 15. FINAL TEST
# =========================================================

print("\n========================================")
print("DATASET LOADER TEST COMPLETED")
print("========================================")

print(
    "5 Classes Loaded :",
    len(train_dataset.classes) == 5
)

print(
    "Expected Image Shape :",
    "(Batch Size, 3, 224, 224)"
)

print("\nReady for WeightedRandomSampler and Model Training.")