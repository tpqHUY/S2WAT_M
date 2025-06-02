from PIL import Image
import os
from torchvision import transforms as T
from tqdm import tqdm

# Configuration
input_dir    = "./../SsPM/B"   # Thư mục chứa tile gốc
output_dir   = "./../SsPM/D"     # Thư mục lưu ảnh augment
num_augments = 3                            # Số bản augment / ảnh

os.makedirs(output_dir, exist_ok=True)

# Chỉ color augmentation
augment_transform = T.Compose([
    T.ColorJitter(
        brightness=0.05,  # ±5%
        saturation=0.05,  # ±5%
        hue=0.05          # ±5%
    ),
    T.ToTensor(),
    T.ToPILImage()
])

count = 0

for fname in tqdm(os.listdir(input_dir), desc="Color-Augmenting"):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
        continue
    img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
    base = os.path.splitext(fname)[0]

    # Lưu ảnh gốc (nếu cần)
    img.save(os.path.join(output_dir, f"{base}_orig.png"))

    # Tạo các bản color-jitter
    for i in range(1, num_augments+1):
        aug = augment_transform(img)
        aug.save(os.path.join(output_dir, f"{base}_aug{i}.png"))
        count += 1

print(f"✅ Generated {count} color-augmented images in '{output_dir}'")
