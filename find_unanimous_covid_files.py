import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import timm

# -------------------------------------------------------------
# 1. Device, Classes & Transform Setup
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
# 2. Moondru Models-aiyum Memory-il Load Seidhal
# -------------------------------------------------------------
print("\n[INFO] Moondru Deep Learning Models load aagudhu... Konjam wait pannunga.")

# 1. DenseNet121
dense_path = find_checkpoint("densenet121_pulmoverse_final.pth")
dense_model = models.densenet121(weights=None)
dense_model.classifier = nn.Linear(dense_model.classifier.in_features, 5)
dense_model.load_state_dict(extract_state_dict(torch.load(dense_path, map_location=DEVICE)), strict=False)
dense_model.to(DEVICE).eval()

# 2. ResNet50
res_path = find_checkpoint("resnet50_pulmoverse_final.pth")
res_model = models.resnet50(weights=None)
res_model.fc = nn.Linear(res_model.fc.in_features, 5)
res_model.load_state_dict(extract_state_dict(torch.load(res_path, map_location=DEVICE)), strict=False)
res_model.to(DEVICE).eval()

# 3. EfficientNetB0
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

print("[INFO] DenseNet121, ResNet50, EfficientNetB0 ellame successfully load aagiduchu!\n")

# -------------------------------------------------------------
# 3. COVID Test Folder Scanning (536 Images)
# -------------------------------------------------------------
COVID_TEST_DIR = r"D:\Project\pulmoverse_preprocessed\test\COVID"
image_files = [f for f in os.listdir(COVID_TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total = len(image_files)

unanimous_3_models = []  # 3/3 Models Agree on COVID-19
majority_2_models = []   # 2/3 Models Agree on COVID-19
other_cases = []         # < 2 Models

print(f"[INFO] Total {total} COVID test images-ai 3 models vaitthu scan pandrom...\n")

with torch.no_grad():
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(COVID_TEST_DIR, fname)
        try:
            raw_img = Image.open(img_path).convert("RGB")
            tensor = transform_pipeline(raw_img).unsqueeze(0).to(DEVICE)

            # DenseNet121 Prediction
            out_d = torch.softmax(dense_model(tensor), dim=1)[0].cpu().numpy()
            pred_d = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_d))]]
            conf_d = round(float(np.max(out_d)) * 100, 2)

            # ResNet50 Prediction
            out_r = torch.softmax(res_model(tensor), dim=1)[0].cpu().numpy()
            pred_r = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_r))]]
            conf_r = round(float(np.max(out_r)) * 100, 2)

            # EfficientNetB0 Prediction
            out_e = torch.softmax(eff_model(tensor), dim=1)[0].cpu().numpy()
            pred_e = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_e))]]
            conf_e = round(float(np.max(out_e)) * 100, 2)

            # Unanimous 3/3 Consensus Verification
            if pred_d == "COVID-19" and pred_r == "COVID-19" and pred_e == "COVID-19":
                unanimous_3_models.append((fname, conf_d, conf_r, conf_e))
            elif [pred_d, pred_r, pred_e].count("COVID-19") == 2:
                majority_2_models.append((fname, pred_d, pred_r, pred_e))
            else:
                other_cases.append((fname, pred_d, pred_r, pred_e))

            if idx % 100 == 0 or idx == total:
                print(f"Processed: {idx}/{total} scans...")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

# -------------------------------------------------------------
# 4. Results Summary
# -------------------------------------------------------------
pct_3 = round((len(unanimous_3_models) / total) * 100, 2)
pct_2 = round((len(majority_2_models) / total) * 100, 2)

print("\n================== CONSENSUS EVALUATION SUMMARY ==================")
print(f"Motha COVID Scans Evaluated              : {total}")
print(f"3/3 Models Unanimous COVID-19 (All Agree): {len(unanimous_3_models)} ({pct_3}%)")
print(f"2/3 Models Majority COVID-19 (Partial)   : {len(majority_2_models)} ({pct_2}%)")
print("==================================================================\n")

print("--- 3/3 MODELS UNANIMOUS COVID-19 SOLRA SAMPLE FILES ---")
for fname, cd, cr, ce in unanimous_3_models[:10]:
    print(f"★ {fname} -> DenseNet: {cd}%, ResNet: {cr}%, EfficientNet: {ce}%")

# -------------------------------------------------------------
# 5. Extract Seidhu CSV File-il Save Seidhal
# -------------------------------------------------------------
out_csv = "covid_unanimous_3_models_files.csv"
with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Consensus", "DenseNet121_Conf", "ResNet50_Conf", "EfficientNetB0_Conf"])
    for fname, cd, cr, ce in unanimous_3_models:
        writer.writerow([fname, "3/3 Unanimous COVID-19", f"{cd}%", f"{cr}%", f"{ce}%"])

print(f"\n[SUCCESS] 3/3 Unanimous files mattum thaniyaa save aagiduchu: {out_csv}")

# -------------------------------------------------------------
# 6. Balance 37 Misclassified Scans - Screen Print & CSV Export
# -------------------------------------------------------------
print(f"\n--- BALANCE {len(other_cases)} MISCLASSIFIED SCANS (< 2 COVID Votes) ---")
for fname, pd, pr, pe in other_cases:
    print(f"✗ {fname} -> DenseNet: {pd}, ResNet: {pr}, EfficientNet: {pe}")

out_misclassified_csv = "covid_misclassified_37_files.csv"
with open(out_misclassified_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Status", "DenseNet121_Prediction", "ResNet50_Prediction", "EfficientNetB0_Prediction"])
    for fname, pd, pr, pe in other_cases:
        writer.writerow([fname, "Misclassified / Borderline", pd, pr, pe])

print(f"\n[SUCCESS] 37 Misclassified files list saved to: {out_misclassified_csv}")