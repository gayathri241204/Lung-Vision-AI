from PIL import Image
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(r"D:\Project\PulmoVerse_Dataset")

classes = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

print("\nPulmoVerse Dataset Resolution Check")
print("=" * 55)

total_images = 0
all_resolutions = Counter()

for class_name in classes:

    class_path = DATASET_PATH / class_name
    resolutions = Counter()

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolutions[(width, height)] += 1
                    all_resolutions[(width, height)] += 1
                    total_images += 1

            except Exception:
                pass

    print(f"\n{class_name}")
    print("-" * 40)

    for resolution, count in resolutions.most_common(10):
        print(f"{resolution[0]} x {resolution[1]} : {count}")

print("\n" + "=" * 55)
print("Overall Resolution Distribution")
print("=" * 55)

for resolution, count in all_resolutions.most_common(15):
    print(f"{resolution[0]} x {resolution[1]} : {count}")

print("=" * 55)
print(f"Total Images Checked : {total_images}")
print(f"Different Resolutions: {len(all_resolutions)}")
print("=" * 55)