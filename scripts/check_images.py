import os
from PIL import Image

# Dataset path
dataset_path = r"D:\Project\PulmoVerse_Dataset"

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0
valid_images = 0
corrupted_images = 0

print("\nPulmoVerse Dataset Image Checking")
print("=" * 40)

for class_name in sorted(os.listdir(dataset_path)):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    class_total = 0
    class_valid = 0
    class_corrupted = 0

    for file_name in os.listdir(class_path):

        if file_name.lower().endswith(image_extensions):

            class_total += 1
            total_images += 1

            image_path = os.path.join(class_path, file_name)

            try:
                with Image.open(image_path) as img:
                    img.verify()

                class_valid += 1
                valid_images += 1

            except Exception:
                class_corrupted += 1
                corrupted_images += 1

                print("Corrupted:", image_path)

    print(f"{class_name:20} Total: {class_total:5} | Valid: {class_valid:5} | Corrupted: {class_corrupted:5}")

print("=" * 40)
print("Total Images     :", total_images)
print("Valid Images     :", valid_images)
print("Corrupted Images :", corrupted_images)
print("=" * 40)