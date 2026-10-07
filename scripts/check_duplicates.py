import os
import hashlib

# Dataset path
DATASET_PATH = r"D:\Project\PulmoVerse_Dataset"

# Image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Store image hashes
hashes = {}
duplicates = []

print("PulmoVerse Dataset Duplicate Image Check")
print("=" * 55)

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    total = 0

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        total += 1

        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(
                    (file_path, hashes[file_hash])
                )
            else:
                hashes[file_hash] = file_path

        except Exception as e:
            print(f"Error: {file_path}")
            print(e)

    print(f"{class_name:<20} : {total}")

print("=" * 55)

print(f"Unique Images       : {len(hashes)}")
print(f"Duplicate Images    : {len(duplicates)}")

if duplicates:
    print("\nDuplicate Images Found:")
    print("-" * 55)

    for duplicate, original in duplicates:
        print(f"Duplicate : {duplicate}")
        print(f"Original  : {original}")
        print()

else:
    print("\nNo duplicate images found.")

print("=" * 55)