import os
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from pathlib import Path

# Thư mục
input_dir = "../img_align_celeba/"     # ảnh gốc
output_dir = "../celeba_resized"   # nơi lưu ảnh resize

os.makedirs(output_dir, exist_ok=True)

# Pipeline: center crop 178×178 → resize 224×224
transform = T.Compose([
    T.CenterCrop(178),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.ToPILImage()
])

count = 0

for fname in tqdm(os.listdir(input_dir), desc="Resuming Resize CelebA"):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        continue

    output_path = os.path.join(output_dir, fname)
    if os.path.exists(output_path):
        continue  # skip nếu ảnh đã tồn tại

    try:
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_out = transform(img)
        img_out.save(output_path)
        count += 1
    except Exception as e:
        print(f"❌ Error {fname}: {e}")


print(f"✅ Done! Resized {count} images to 224×224 (centered) in '{output_dir}'")
