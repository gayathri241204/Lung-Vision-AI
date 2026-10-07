import os
import matplotlib.pyplot as plt

# Dataset path
dataset_path = r"D:\Project\PulmoVerse_Dataset"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

class_counts = {}

# Count images in each class
for class_name in sorted(os.listdir(dataset_path)):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    count = 0

    for file_name in os.listdir(class_path):

        if file_name.lower().endswith(image_extensions):
            count += 1

    class_counts[class_name] = count


# Total images
total_images = sum(class_counts.values())

print("\nPulmoVerse Dataset Analysis")
print("=" * 45)

for class_name, count in class_counts.items():

    percentage = (count / total_images) * 100

    print(
        f"{class_name:20} "
        f"Images: {count:5} "
        f"Percentage: {percentage:6.2f}%"
    )

print("=" * 45)
print("Total Images:", total_images)


# Bar chart
plt.figure(figsize=(10, 6))

plt.bar(class_counts.keys(), class_counts.values())

plt.title("PulmoVerse Dataset Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()