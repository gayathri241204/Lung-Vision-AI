from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"D:\Project\PulmoVerse_Dataset")

CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

SAMPLES_PER_CLASS = 3

fig, axes = plt.subplots(
    len(CLASSES),
    SAMPLES_PER_CLASS,
    figsize=(10, 14)
)

for row, class_name in enumerate(CLASSES):

    class_path = DATASET_PATH / class_name

    images = [
        p for p in class_path.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    # Same images every time for easy comparison
    images = sorted(images)[:SAMPLES_PER_CLASS]

    for col, image_path in enumerate(images):

        image = Image.open(image_path).convert("L")

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()