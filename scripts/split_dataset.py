from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ==============================
# Paths
# ==============================

SOURCE_DIR = Path(r"D:\Project\PulmoVerse_Dataset")
OUTPUT_DIR = Path(r"D:\Project\PulmoVerse_Split")

# Dataset classes
CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

# ==============================
# Create output folders
# ==============================

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# ==============================
# Stratified Split
# ==============================

for class_name in CLASSES:

    class_path = SOURCE_DIR / class_name

    images = [
        image for image in class_path.iterdir()
        if image.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    # First split: 70% Train, 30% Temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Second split: Temporary → 15% Validation, 15% Test
    validation_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    # Copy images
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in validation_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "validation" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

    print(f"\n{class_name}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(validation_images)}")
    print(f"Test       : {len(test_images)}")

print("\n========================================")
print("Stratified Dataset Split Completed")
print("========================================")
print("Train      : 70%")
print("Validation : 15%")
print("Test       : 15%")
print("========================================")