import os
from PIL import Image
from tqdm import tqdm

input_dir = "../../Datasets/SP_v1v2s"
invalid_count = 0

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        continue
    fpath = os.path.join(input_dir, fname)
    
    try:
        img = Image.open(fpath)
        w, h = img.size
        img.close()  # ĐÓNG ẢNH TRƯỚC KHI XÓA
        
        if (w, h) != (224, 224):
            os.remove(fpath)
            invalid_count += 1
    except Exception as e:
        print(f"❌ Error with {fname}: {e}")

print(f"\n🗑️ Deleted {invalid_count} image(s) not 224x224 in: {input_dir}")
