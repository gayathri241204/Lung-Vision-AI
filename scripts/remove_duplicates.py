import os
import hashlib

DATASET_PATH = r"D:\Project\PulmoVerse_Dataset"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

hashes = {}
duplicates = []

print("PulmoVerse Dataset Duplicate Removal")
print("=" * 55)

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[file_hash] = file_path

        except Exception as e:
            print(f"Error: {file_path}")
            print(e)

print(f"\nDuplicate Images Found: {len(duplicates)}")

if duplicates:

    print("\nRemoving duplicate images...")

    for file_path in duplicates:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Could not remove: {file_path}")
            print(e)

    print("\nDuplicate removal completed.")

else:
    print("\nNo duplicate images found.")

print("=" * 55)