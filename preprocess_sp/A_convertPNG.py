from PIL import Image
import os
from tqdm import tqdm

# Cấu hình
input_dir = "../orhsspm"     # Đường dẫn thư mục chứa ảnh gốc
output_dir = "../SsPM"       # Thư mục chứa ảnh đã chuyển thành PNG

os.makedirs(output_dir, exist_ok=True)

count = 0

# Lặp qua từng ảnh
for fname in tqdm(os.listdir(input_dir), desc="Converting to PNG"):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif', '.png')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('RGB')
    
    # Tên mới giữ nguyên phần tên gốc nhưng đổi đuôi thành .png
    base = os.path.splitext(fname)[0]
    img.save(os.path.join(output_dir, f"{base}.png"))
    count += 1

print(f"✅ Done! Converted {count} images to PNG format in '{output_dir}'")
