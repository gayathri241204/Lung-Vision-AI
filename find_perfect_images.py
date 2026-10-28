import os
from scripts.inference import predict_lung_disease

# Ungaloda exact test dataset path
TEST_DIR = r"D:\Project\pulmoverse_preprocessed\test"

# 5 Exact class folders
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

print("=" * 65)
print("🔍 LUNG VISION: Scanning for 3/3 Unanimous Perfect Images...")
print("=" * 65)

# Smart disease matching function (Fixes Pneumonia vs Viral Pneumonia discrepancy)
def is_match(target_name, pred_name):
    t = target_name.lower().replace('_', '').replace('-', '').replace(' ', '')
    p = pred_name.lower().replace('_', '').replace('-', '').replace(' ', '')
    
    # Matches both "Pneumonia" and "Viral Pneumonia"
    if 'pneumonia' in t and 'pneumonia' in p:
        return True
    if 'covid' in t and 'covid' in p:
        return True
    if 'opacity' in t and 'opacity' in p:
        return True
    if 'normal' in t and 'normal' in p:
        return True
    if ('tb' in t or 'tuberculosis' in t) and ('tb' in p or 'tuberculosis' in p):
        return True
    return t in p or p in t

perfect_images = {cls_name: [] for cls_name in CLASSES}
MAX_PER_CLASS = 5  # Ovvoru disease-kum 5 safe demo images

for cls_name in CLASSES:
    cls_folder = os.path.join(TEST_DIR, cls_name)
    
    # Case-insensitive folder matching
    if not os.path.exists(cls_folder):
        for f in os.listdir(TEST_DIR):
            if cls_name.lower().replace(' ', '').replace('_', '') == f.lower().replace(' ', '').replace('_', ''):
                cls_folder = os.path.join(TEST_DIR, f)
                break

    print(f"\n📂 Scanning: {cls_name} ...")
    if not os.path.exists(cls_folder):
        print(f"  ⚠️ Folder not found: {cls_folder}")
        continue

    images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in images:
        if len(perfect_images[cls_name]) >= MAX_PER_CLASS:
            break

        img_path = os.path.join(cls_folder, img_name)

        try:
            # 3 models inference
            res = predict_lung_disease(img_path)
            if not res:
                continue

            model_preds = {}
            for model_name, preds in res.items():
                if preds:
                    top_pred = max(preds, key=preds.get)
                    model_preds[model_name] = top_pred

            # Check 3/3 consensus
            all_correct = True
            for m_name, pred in model_preds.items():
                if not is_match(cls_name, pred):
                    all_correct = False
                    break

            if all_correct and len(model_preds) >= 3:
                perfect_images[cls_name].append(img_name)
                print(f"  ✅ 3/3 Match: {img_name} -> {list(model_preds.values())}")

        except Exception as e:
            continue

# viva_safe_images.txt file-il save seigirom
with open("viva_safe_images.txt", "w") as f:
    f.write("=== LUNG VISION VIVA SAFE DEMO IMAGES (3/3 CONSENSUS) ===\n\n")
    for cls_name, img_list in perfect_images.items():
        print(f"\n👉 {cls_name} ({len(img_list)} images found):")
        f.write(f"[{cls_name}]\n")
        for img in img_list:
            print(f"   • {img}")
            f.write(f" - {img}\n")
        if not img_list:
            f.write(" - No unanimous image found\n")
        f.write("\n")

print("\n" + "=" * 65)
print("📁 All 5 categories successfully saved in 'viva_safe_images.txt'!")
print("=" * 65)