import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------------------------------------
# 1. Hardware & Class Configuration
# -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = ["COVID", "Lung_Opacity", "Normal", "Tuberculosis", "Viral Pneumonia"]
CLASS_DISPLAY = {
    "COVID": "COVID-19",
    "Lung_Opacity": "Lung Opacity",
    "Normal": "Normal",
    "Tuberculosis": "Tuberculosis",
    "Viral Pneumonia": "Pneumonia"
}

# Standard CXR Normalization Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# 2. DenseNet121 Model Checkpoint Locator
# -------------------------------------------------------------
model_filename = 'densenet121_pulmoverse_final.pth'
possible_paths = [
    os.path.join('models', model_filename),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_filename),
    model_filename,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename),
    rf"D:\Project\PulmoVerse\models\{model_filename}",
    rf"D:\Project\PulmoVerse\{model_filename}"
]

model_path = None
for p in possible_paths:
    if os.path.exists(p):
        model_path = p
        break

if not model_path:
    raise FileNotFoundError(f"Model weight '{model_filename}' kidaikkavillai. Models folder-il irukkadhaa endru paarungal.")

print(f"[INFO] Loading DenseNet121 from: {model_path}")

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(MODEL_CLASSES))

checkpoint = torch.load(model_path, map_location=DEVICE)
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
else:
    state_dict = checkpoint

cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(cleaned_state, strict=False)
model.to(DEVICE).eval()

# -------------------------------------------------------------
# 3. Ungaloda Exact Test Dataset Path
# -------------------------------------------------------------
COVID_TEST_DIR = r"D:\Project\pulmoverse_preprocessed\test\COVID"

if not os.path.exists(COVID_TEST_DIR):
    raise FileNotFoundError(f"Given directory not found: {COVID_TEST_DIR}")

correct_covid_files = []
misclassified_files = []

image_files = [f for f in os.listdir(COVID_TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(image_files)

print(f"\n[INFO] Found {total_images} test images in: {COVID_TEST_DIR}")
print("[INFO] Evaluating test scans with DenseNet121... Please wait a moment.\n")

with torch.no_grad():
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(COVID_TEST_DIR, fname)
        try:
            raw_img = Image.open(img_path).convert('RGB')
            tensor = transform_pipeline(raw_img).unsqueeze(0).to(DEVICE)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            pred_class = CLASS_DISPLAY[MODEL_CLASSES[top_idx]]
            conf = round(float(probs[top_idx]) * 100.0, 2)

            if pred_class == "COVID-19":
                correct_covid_files.append((fname, conf))
            else:
                misclassified_files.append((fname, pred_class, conf))

            if idx % 100 == 0 or idx == total_images:
                print(f"Processed: {idx}/{total_images} scans...")
        except Exception as e:
            print(f"Error reading image {fname}: {e}")

# -------------------------------------------------------------
# 4. Summary & Evaluation Metrics
# -------------------------------------------------------------
tp_count = len(correct_covid_files)
fn_count = len(misclassified_files)
accuracy_pct = round((tp_count / total_images) * 100.0, 2) if total_images > 0 else 0.0

print("\n======================= TEST SUMMARY =======================")
print(f"Total COVID Scans Evaluated : {total_images}")
print(f"Correctly Predicted (TP)     : {tp_count} ({accuracy_pct}%)")
print(f"Misclassified Scans (FN)    : {fn_count} ({round(100.0 - accuracy_pct, 2)}%)")
print("============================================================\n")

print("--- SAMPLE CORRECTLY IDENTIFIED SCANS (TRUE POSITIVES) ---")
for name, conf in correct_covid_files[:5]:
    print(f"✓ {name} -> COVID-19 ({conf}%)")

if misclassified_files:
    print("\n--- SAMPLE MISCLASSIFIED SCANS (FALSE NEGATIVES) ---")
    for name, pred, conf in misclassified_files[:5]:
        print(f"✗ {name} -> Wrongly Predicted As: {pred} ({conf}%)")

# -------------------------------------------------------------
# 5. Save Complete Filenames to CSV File
# -------------------------------------------------------------
output_csv = "densenet_covid_detailed_results.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Ground_Truth", "Predicted_Class", "Confidence", "Status"])
    for name, conf in correct_covid_files:
        writer.writerow([name, "COVID-19", "COVID-19", f"{conf}%", "Correct (TP)"])
    for name, pred, conf in misclassified_files:
        writer.writerow([name, "COVID-19", pred, f"{conf}%", "Misclassified (FN)"])

print(f"\n[SUCCESS] All {total_images} image predictions saved to: {output_csv}")