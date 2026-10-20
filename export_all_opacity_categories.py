import csv
import os
import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------------------------------------
# 1. Hardware, Mapping & Transforms Setup
# -------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CLASSES = [
    'COVID',
    'Lung_Opacity',
    'Normal',
    'Tuberculosis',
    'Viral Pneumonia',
]
CLASS_DISPLAY = {
    'COVID': 'COVID-19',
    'Lung_Opacity': 'Lung Opacity',
    'Normal': 'Normal',
    'Tuberculosis': 'Tuberculosis',
    'Viral Pneumonia': 'Pneumonia',
}

# Image Preprocessing Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


def extract_state_dict(checkpoint):
  if isinstance(checkpoint, dict):
    state_dict = (
        checkpoint.get('model_state_dict')
        or checkpoint.get('state_dict')
        or checkpoint
    )
  else:
    state_dict = checkpoint
  return {k.replace('module.', ''): v for k, v in state_dict.items()}


def find_checkpoint(filename):
  possible = [
      os.path.join('models', filename),
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)), 'models', filename
      ),
      filename,
      os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
      rf'D:\Project\PulmoVerse\models\{filename}',
      rf'D:\Project\PulmoVerse\{filename}',
  ]
  for p in possible:
    if os.path.exists(p):
      return p
  return None


# -------------------------------------------------------------
# 2. Triple Deep Learning Models Loading
# -------------------------------------------------------------
print('\n[INFO] Loading DenseNet121, ResNet50, EfficientNetB0 into memory...')

# 1. DenseNet121
dense_path = find_checkpoint('densenet121_pulmoverse_final.pth')
if not dense_path:
  raise FileNotFoundError('densenet121_pulmoverse_final.pth not found!')
dense_model = models.densenet121(weights=None)
dense_model.classifier = nn.Linear(dense_model.classifier.in_features, 5)
dense_model.load_state_dict(
    extract_state_dict(torch.load(dense_path, map_location=DEVICE)),
    strict=False,
)
dense_model.to(DEVICE).eval()

# 2. ResNet50
res_path = find_checkpoint('resnet50_pulmoverse_final.pth')
if not res_path:
  raise FileNotFoundError('resnet50_pulmoverse_final.pth not found!')
res_model = models.resnet50(weights=None)
res_model.fc = nn.Linear(res_model.fc.in_features, 5)
res_model.load_state_dict(
    extract_state_dict(torch.load(res_path, map_location=DEVICE)), strict=False
)
res_model.to(DEVICE).eval()

# 3. EfficientNetB0 (Dynamic timm/torchvision detector)
eff_path = find_checkpoint('efficientnetb0_pulmoverse_final_best.pth')
if not eff_path:
  raise FileNotFoundError('efficientnetb0_pulmoverse_final_best.pth not found!')
eff_state = extract_state_dict(torch.load(eff_path, map_location=DEVICE))
is_timm = any('conv_stem' in k or 'blocks' in k for k in eff_state.keys())

if is_timm:
  eff_model = timm.create_model(
      'efficientnet_b0', pretrained=False, num_classes=5
  )
else:
  eff_model = models.efficientnet_b0(weights=None)
  eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 5)

eff_model.load_state_dict(eff_state, strict=False)
eff_model.to(DEVICE).eval()

print('[INFO] All 3 models verified and loaded successfully!\n')

# -------------------------------------------------------------
# 3. Lung Opacity Test Directory Path Locator
# -------------------------------------------------------------
OPACITY_TEST_DIR = r'D:\Project\pulmoverse_preprocessed\test\Lung_Opacity'
if not os.path.exists(OPACITY_TEST_DIR):
  OPACITY_TEST_DIR = r'D:\Project\pulmoverse_preprocessed\test\Lung Opacity'

if not os.path.exists(OPACITY_TEST_DIR):
  raise FileNotFoundError(
      f'Lung Opacity test folder kidaikkavillai: {OPACITY_TEST_DIR}'
  )

image_files = sorted([
    f
    for f in os.listdir(OPACITY_TEST_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
total = len(image_files)

unanimous_3_models = []
majority_2_models = []
misclassified_scans = []

print(f'[INFO] Found {total} Lung Opacity test scans in: {OPACITY_TEST_DIR}')
print('[INFO] Executing multi-model evaluation across all 3 models...\n')

with torch.no_grad():
  for idx, fname in enumerate(image_files, 1):
    img_path = os.path.join(OPACITY_TEST_DIR, fname)
    try:
      raw_img = Image.open(img_path).convert('RGB')
      tensor = transform_pipeline(raw_img).unsqueeze(0).to(DEVICE)

      # 1. DenseNet121 Inference
      out_d = torch.softmax(dense_model(tensor), dim=1)[0].cpu().numpy()
      pred_d = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_d))]]
      conf_d = round(float(np.max(out_d)) * 100, 2)

      # 2. ResNet50 Inference
      out_r = torch.softmax(res_model(tensor), dim=1)[0].cpu().numpy()
      pred_r = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_r))]]
      conf_r = round(float(np.max(out_r)) * 100, 2)

      # 3. EfficientNetB0 Inference
      out_e = torch.softmax(eff_model(tensor), dim=1)[0].cpu().numpy()
      pred_e = CLASS_DISPLAY[MODEL_CLASSES[int(np.argmax(out_e))]]
      conf_e = round(float(np.max(out_e)) * 100, 2)

      # Vote Calculation for Lung Opacity
      opacity_votes = [pred_d, pred_r, pred_e].count('Lung Opacity')

      if opacity_votes == 3:
        unanimous_3_models.append((fname, conf_d, conf_r, conf_e))
      elif opacity_votes == 2:
        majority_2_models.append(
            (fname, pred_d, conf_d, pred_r, conf_r, pred_e, conf_e)
        )
      else:
        misclassified_scans.append(
            (fname, pred_d, conf_d, pred_r, conf_r, pred_e, conf_e)
        )

      if idx % 150 == 0 or idx == total:
        print(f'Processed: {idx}/{total} scans...')
    except Exception as e:
      print(f'Error reading {fname}: {e}')

# -------------------------------------------------------------
# 4. Save 3 Clean Segregated CSV Files
# -------------------------------------------------------------
# File 1: 3/3 Unanimous Agreement
csv_1 = 'opacity_3_of_3_unanimous.csv'
with open(csv_1, mode='w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  writer.writerow([
      'Filename',
      'Consensus_Category',
      'DenseNet121_Conf',
      'ResNet50_Conf',
      'EfficientNetB0_Conf',
  ])
  for fname, cd, cr, ce in unanimous_3_models:
    writer.writerow(
        [fname, '3/3 Unanimous Lung Opacity', f'{cd}%', f'{cr}%', f'{ce}%']
    )

# File 2: 2/3 Majority Agreement
csv_2 = 'opacity_2_of_3_majority.csv'
with open(csv_2, mode='w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  writer.writerow([
      'Filename',
      'Consensus_Category',
      'DenseNet121_Pred',
      'DenseNet121_Conf',
      'ResNet50_Pred',
      'ResNet50_Conf',
      'EfficientNetB0_Pred',
      'EfficientNetB0_Conf',
  ])
  for fname, pd, cd, pr, cr, pe, ce in majority_2_models:
    writer.writerow([
        fname,
        '2/3 Majority Lung Opacity',
        pd,
        f'{cd}%',
        pr,
        f'{cr}%',
        pe,
        f'{ce}%',
    ])

# File 3: Misclassified / Borderline Scans
csv_3 = 'opacity_misclassified.csv'
with open(csv_3, mode='w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  writer.writerow([
      'Filename',
      'Consensus_Category',
      'DenseNet121_Pred',
      'DenseNet121_Conf',
      'ResNet50_Pred',
      'ResNet50_Conf',
      'EfficientNetB0_Pred',
      'EfficientNetB0_Conf',
  ])
  for fname, pd, cd, pr, cr, pe, ce in misclassified_scans:
    writer.writerow([
        fname,
        'Misclassified (< 2 Opacity Votes)',
        pd,
        f'{cd}%',
        pr,
        f'{cr}%',
        pe,
        f'{ce}%',
    ])

# -------------------------------------------------------------
# 5. Final Verified Terminal Summary
# -------------------------------------------------------------
c1 = len(unanimous_3_models)
c2 = len(majority_2_models)
c3 = len(misclassified_scans)
calc_sum = c1 + c2 + c3

print('\n================ FINAL LUNG OPACITY VERIFIED SUMMARY ================')
print(f'Total Lung Opacity Scans Evaluated      : {total}')
print(
    f'1. 3/3 Unanimous Lung Opacity (CSV 1)  : {c1}'
    f' ({round((c1 / total) * 100, 2)}%)'
)
print(
    f'2. 2/3 Majority Lung Opacity  (CSV 2)  : {c2}'
    f' ({round((c2 / total) * 100, 2)}%)'
)
print(
    f'3. Misclassified Borderline   (CSV 3)  : {c3}'
    f' ({round((c3 / total) * 100, 2)}%)'
)
print(f'Calculated Verification Sum            : {calc_sum} / {total}')
print('====================================================================\n')
print(f'[SUCCESS] Unanimous CSV Generated   -> {csv_1} ({c1} rows)')
print(f'[SUCCESS] Majority CSV Generated    -> {csv_2} ({c2} rows)')
print(f'[SUCCESS] Misclassified Generated   -> {csv_3} ({c3} rows)')