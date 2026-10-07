from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# =========================================================
# 1. DATASET PATH
# =========================================================

DATASET_PATH = Path(
    r"D:\Project\PulmoVerse_Preprocessed"
)

BATCH_SIZE = 32


# =========================================================
# 2. NORMALIZATION
# =========================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =========================================================
# 3. TRAIN TRANSFORM
#    Augmentation + Normalization
# =========================================================

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================================================
# 4. VALIDATION TRANSFORM
#    No Augmentation
# =========================================================

validation_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================================================
# 5. TEST TRANSFORM
#    No Augmentation
# =========================================================

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================================================
# 6. LOAD DATASETS
# =========================================================

train_dataset = datasets.ImageFolder(
    DATASET_PATH / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_PATH / "validation",
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    DATASET_PATH / "test",
    transform=test_transform
)


# =========================================================
# 7. CHECK CLASSES
# =========================================================

print("\n========================================")
print("CLASSES")
print("========================================")

print(train_dataset.classes)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)


# =========================================================
# 8. COUNT TRAINING IMAGES PER CLASS
# =========================================================

class_counts = torch.zeros(
    len(train_dataset.classes),
    dtype=torch.long
)

for _, label in train_dataset.samples:
    class_counts[label] += 1


print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):
    print(
        f"{class_name:20s}: {count.item()}"
    )


# =========================================================
# 9. CALCULATE CLASS WEIGHTS
# =========================================================

class_weights = 1.0 / class_counts.float()


print("\n========================================")
print("CLASS WEIGHTS")
print("========================================")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(
        f"{class_name:20s}: {weight.item():.6f}"
    )


# =========================================================
# 10. CREATE SAMPLE WEIGHTS
# =========================================================

sample_weights = torch.tensor(
    [
        class_weights[label].item()
        for _, label in train_dataset.samples
    ],
    dtype=torch.double
)


# =========================================================
# 11. WEIGHTED RANDOM SAMPLER
# =========================================================

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# =========================================================
# 12. CREATE DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
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
# 13. DATASET SIZE
# =========================================================

print("\n========================================")
print("DATASET SIZE")
print("========================================")

print("Train      :", len(train_dataset))
print("Validation :", len(validation_dataset))
print("Test       :", len(test_dataset))


# =========================================================
# 14. TEST ONE TRAINING BATCH
# =========================================================

images, labels = next(iter(train_loader))


print("\n========================================")
print("TRAINING BATCH TEST")
print("========================================")

print("Image Shape :", images.shape)
print("Label Shape :", labels.shape)

print(
    "Image Minimum :",
    images.min().item()
)

print(
    "Image Maximum :",
    images.max().item()
)


# =========================================================
# 15. FINAL STATUS
# =========================================================

print("\n========================================")
print("TRAINING PIPELINE READY")
print("========================================")

print("Train Augmentation       : ENABLED")
print("Validation Augmentation  : DISABLED")
print("Test Augmentation        : DISABLED")
print("Normalization            : ENABLED")
print("WeightedRandomSampler    : ENABLED")

print("\nReady for model training.")