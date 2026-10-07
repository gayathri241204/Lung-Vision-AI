import os

DATASET_PATH = r"D:\Project\PulmoVerse_Dataset"

VALID_FORMATS = {".jpg", ".jpeg", ".png"}

print("\nPulmoVerse Dataset Image Format Check")
print("=" * 50)

total_images = 0
valid_format = 0
invalid_format = 0

format_counts = {}

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    class_total = 0

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(file_name)[1].lower()

        total_images += 1
        class_total += 1

        format_counts[extension] = format_counts.get(extension, 0) + 1

        if extension in VALID_FORMATS:
            valid_format += 1
        else:
            invalid_format += 1

    print(f"{class_name:<20} : {class_total}")

print("=" * 50)

print("\nImage Format Distribution")
print("=" * 50)

for extension, count in sorted(format_counts.items()):
    print(f"{extension:<10} : {count}")

print("=" * 50)

print(f"Total Images        : {total_images}")
print(f"Valid Formats       : {valid_format}")
print(f"Invalid Formats     : {invalid_format}")
print("=" * 50)