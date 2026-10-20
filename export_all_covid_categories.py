import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import timm

# -------------------------------------------------------------
# 1. Hardware, Mapping & Transforms
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

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def find_checkpoint(filename):
    possible = [
        os.path.join("models", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename),
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        rf"D:\Project\PulmoVerse\models\{filename}",
        rf"D:\Project\PulmoVerse\{filename}"
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    return None

# -------------------------------------------------------------
# 2. Moondru Models-aiyum Load Seidhal
# -------------------------------------------------------------
print("\n[INFO] Loading DenseNet121, ResNet50, EfficientNetB0 into memory...")

# DenseNet121
dense_path = find_checkpoint("densenet121_pulmoverse_final.pth")
dense_model = models.densenet121(weights=None)
dense_model.classifier = nn.Linear(dense_model.classifier.in_features, 5)
dense_model.load_state_dict(extract_state_dict(torch.load(dense_path, map_location=DEVICE)), strict=False)
dense_model.to(DEVICE).eval()

# ResNet50
res_path = find_checkpoint("resnet50_pulmoverse_final.pth")
res_model = models.resnet50(weights=None)
res_model.fc = nn.Linear(res_model.fc.in_features, 5)
res_model.load_state_dict(extract_state_dict(torch.load(res_path, map_location=DEVICE)), strict=False)
res_model.to(DEVICE).eval()

# EfficientNetB0
eff_path = find_checkpoint("efficientnetb0_pulmoverse_final_best.pth")
eff_state = extract_state_dict(torch.load(eff_path, map_location=DEVICE))
is_timm = any("conv_stem" in k or "blocks" in k for k in eff_state.keys())
if is_timm:
    eff_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
else:
    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 5)
eff_model.load_state_dict(eff_state, strict=False)
eff_model.to(DEVICE).eval()

print("[INFO] All 3 models loaded successfully!\n")

# -------------------------------------------------------------
# 3. Scan 536 COVID Test Images
# -------------------------------------------------------------
COVID_TEST_DIR = r"D:\Project\pulmoverse_preprocessed\test\COVID"
image_files = [f for f in os.listdir(COVID_TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total = len(image_files)

unanimous_428 = []
majority_98 = []
misclassified_10 = []

print(f"[INFO] Scanning all {total} COVID test images... Please wait.\n")

with torch.no_grad():
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(COVID_TEST_DIR, fname)
        try:
            raw_img = Image.open(img_path).convert("RGB")
            tensor = transform_pipeline(raw_img).unsqueeze(0).to(DEVICE)

            # Predictions
            out_d = torch.softmax(dense_model(tensor), dim=1)[0].cpu().numpy()
            pred_d = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_d))]]
            conf_d = round(float(np.max(out_d)) * 100, 2)

            out_r = torch.softmax(res_model(tensor), dim=1)[0].cpu().numpy()
            pred_r = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_r))]]
            conf_r = round(float(np.max(out_r)) * 100, 2)

            out_e = torch.softmax(eff_model(tensor), dim=1)[0].cpu().numpy()
            pred_e = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_e))]]
            conf_e = round(float(np.max(out_e)) * 100, 2)

            covid_votes = [pred_d, pred_r, pred_e].count("COVID-19")

            if covid_votes == 3:
                unanimous_428.append((fname, conf_d, conf_r, conf_e))
            elif covid_votes == 2:
                majority_98.append((fname, pred_d, conf_d, pred_r, conf_r, pred_e, conf_e))
            else:
                misclassified_10.append((fname, pred_d, conf_d, pred_r, conf_r, pred_e, conf_e))

            if idx % 100 == 0 or idx == total:
                print(f"Processed: {idx}/{total} scans...")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

# -------------------------------------------------------------
# 4. Save 3 Separate Clean CSV Files
# -------------------------------------------------------------
# File 1: 3/3 Unanimous Agreement
csv_1 = "covid_3_of_3_unanimous_428.csv"
with open(csv_1, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Category", "DenseNet121_Conf", "ResNet50_Conf", "EfficientNetB0_Conf"])
    for fname, cd, cr, ce in unanimous_428:
        writer.writerow([fname, "3/3 Unanimous COVID-19", f"{cd}%", f"{cr}%", f"{ce}%"])

# File 2: 2/3 Majority Agreement
csv_2 = "covid_2_of_3_majority_98.csv"
with open(csv_2, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Category", "DenseNet121_Pred", "DenseNet121_Conf", "ResNet50_Pred", "ResNet50_Conf", "EfficientNetB0_Pred", "EfficientNetB0_Conf"])
    for fname, pd, cd, pr, cr, pe, ce in majority_98:
        writer.writerow([fname, "2/3 Majority COVID-19", pd, f"{cd}%", pr, f"{cr}%", pe, f"{ce}%"])

# File 3: Misclassified / Borderline
csv_3 = "covid_misclassified_10.csv"
with open(csv_3, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Category", "DenseNet121_Pred", "DenseNet121_Conf", "ResNet50_Pred", "ResNet50_Conf", "EfficientNetB0_Pred", "EfficientNetB0_Conf"])
    for fname, pd, cd, pr, cr, pe, ce in misclassified_10:
        writer.writerow([fname, "Misclassified (< 2 Votes)", pd, f"{cd}%", pr, f"{cr}%", pe, f"{ce}%"])

# -------------------------------------------------------------
# 5. Print Final Verified Summary
# -------------------------------------------------------------
print("\n================ FINAL VERIFIED SUMMARY ================")
print(f"Total COVID Test Scans Evaluated      : {total}")
print(f"1. 3/3 Unanimous COVID-19 Scans (CSV 1): {len(unanimous_428)} ({round((len(unanimous_428)/total)*100, 2)}%)")
print(f"2. 2/3 Majority COVID-19 Scans (CSV 2) : {len(majority_98)} ({round((len(majority_98)/total)*100, 2)}%)")
print(f"3. Misclassified Scans (CSV 3)         : {len(misclassified_10)} ({round((len(misclassified_10)/total)*100, 2)}%)")
print(f"Calculated Sum                         : {len(unanimous_428) + len(majority_98) + len(misclassified_10)} / {total}")
print("========================================================\n")
print(f"[SUCCESS] CSV 1 Generated -> {csv_1}")
print(f"[SUCCESS] CSV 2 Generated -> {csv_2}")
print(f"[SUCCESS] CSV 3 Generated -> {csv_3}")