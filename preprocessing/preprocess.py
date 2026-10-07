from pathlib import Path
import cv2
import numpy as np
from PIL import Image


# =========================================================
# 1. PATH SETTINGS
# =========================================================

INPUT_PATH = Path(r"D:\Project\PulmoVerse_Split")
OUTPUT_PATH = Path(r"D:\Project\PulmoVerse_Preprocessed")


# =========================================================
# 2. DATASET SETTINGS
# =========================================================

SPLITS = ["train", "validation", "test"]

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}

IMAGE_SIZE = (224, 224)


# =========================================================
# 3. IMAGE LOADING
# =========================================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Cannot read image: {image_path.name}")
        return None

    return image


# =========================================================
# 4. BASIC IMAGE QUALITY CHECK
# =========================================================

def quality_check(image, image_name):

    if image is None:
        return False

    height, width = image.shape[:2]

    # Reject extremely small images
    if height < 50 or width < 50:
        print(f"Poor quality / very small image: {image_name}")
        return False

    return True


# =========================================================
# 5. NOISE ASSESSMENT
# =========================================================

def needs_median_filter(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian_variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Conservative threshold
    # Very low variance may indicate excessive smoothness/noise.
    if laplacian_variance < 50:
        return True

    return False


# =========================================================
# 6. CONDITIONAL MEDIAN FILTER
# =========================================================

def apply_median_filter(image, image_name):

    if needs_median_filter(image):

        print(
            f"Median filter applied: {image_name}"
        )

        return cv2.medianBlur(
            image,
            3
        )

    else:

        print(
            f"Median filter skipped: {image_name}"
        )

        return image


# =========================================================
# 7. RESIZE
# =========================================================

def resize_image(image):

    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# 8. RGB CONVERSION
# =========================================================

def convert_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# =========================================================
# 9. COMPLETE PREPROCESSING PIPELINE
# =========================================================

def preprocess_image(image_path):

    # Image Loading
    image = load_image(image_path)

    if image is None:
        return None

    # Quality Check
    if not quality_check(
        image,
        image_path.name
    ):
        return None

    # Conditional Noise Reduction
    image = apply_median_filter(
        image,
        image_path.name
    )

    # Resize 224 × 224
    image = resize_image(image)

    # BGR → RGB
    image = convert_to_rgb(image)

    return image


# =========================================================
# 10. PROCESS ONE SPLIT
# =========================================================

def process_split(split_name):

    input_split_path = (
        INPUT_PATH / split_name
    )

    output_split_path = (
        OUTPUT_PATH / split_name
    )

    if not input_split_path.exists():

        print(
            f"\nWARNING: {split_name} folder not found!"
        )

        return 0, 0

    total_processed = 0
    total_skipped = 0

    print("\n")
    print("========================================")
    print(f"PROCESSING: {split_name.upper()}")
    print("========================================")

    # Find class folders
    class_folders = [
        folder
        for folder in input_split_path.iterdir()
        if folder.is_dir()
    ]

    for class_folder in sorted(class_folders):

        class_name = class_folder.name

        output_class_folder = (
            output_split_path / class_name
        )

        output_class_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in VALID_EXTENSIONS
        ]

        print(
            f"\nClass: {class_name}"
        )

        print(
            f"Images found: {len(image_files)}"
        )

        class_processed = 0
        class_skipped = 0

        for image_path in image_files:

            print(
                f"\nProcessing: {image_path.name}"
            )

            processed_image = preprocess_image(
                image_path
            )

            # Skip invalid/poor images
            if processed_image is None:

                print(
                    f"SKIPPED: {image_path.name}"
                )

                class_skipped += 1
                total_skipped += 1

                continue

            # Output file path
            output_path = (
                output_class_folder
                / f"{image_path.stem}.png"
            )

            # Save processed image
            Image.fromarray(
                processed_image
            ).save(output_path)

            height, width, channels = (
                processed_image.shape
            )

            print(
                f"Output size: "
                f"{width} x {height}"
            )

            print(
                f"Channels: {channels}"
            )

            print(
                f"Saved to: {output_path}"
            )

            class_processed += 1
            total_processed += 1

        print(
            f"\n{class_name} summary:"
        )

        print(
            f"Processed: {class_processed}"
        )

        print(
            f"Skipped: {class_skipped}"
        )

    return total_processed, total_skipped


# =========================================================
# 11. PROCESS COMPLETE DATASET
# =========================================================

def process_dataset():

    print("\n")
    print("========================================")
    print("PULMOVerse FULL DATASET PREPROCESSING")
    print("========================================")

    grand_total_processed = 0
    grand_total_skipped = 0

    for split in SPLITS:

        processed, skipped = process_split(
            split
        )

        grand_total_processed += processed
        grand_total_skipped += skipped

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n")
    print("========================================")
    print("FULL PREPROCESSING COMPLETED")
    print("========================================")

    print(
        f"Total images processed: "
        f"{grand_total_processed}"
    )

    print(
        f"Total images skipped: "
        f"{grand_total_skipped}"
    )

    print(
        f"\nOutput folder:"
    )

    print(OUTPUT_PATH)


# =========================================================
# 12. START FULL DATASET PREPROCESSING
# =========================================================

if __name__ == "__main__":

    process_dataset()
