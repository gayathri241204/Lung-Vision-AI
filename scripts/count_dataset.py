import os

dataset_path = r"D:\Project\PulmoVerse_Dataset"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0

print("\nPulmoVerse Dataset Image Count")
print("=" * 40)

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        count = 0

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(image_extensions):
                count += 1

        print(f"{class_name:<20}: {count}")
        total_images += count

print("=" * 40)
print(f"Total Images          : {total_images}")