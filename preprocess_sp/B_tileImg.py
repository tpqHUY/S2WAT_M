import os
import argparse
from PIL import Image

def tile_image(img, tile_size, stride):
    """
    Generate tiles of size tile_size x tile_size with given stride,
    ensuring full coverage by adding edge tiles if needed.
    """
    w, h = img.size
    xs = list(range(0, w - tile_size + 1, stride))
    ys = list(range(0, h - tile_size + 1, stride))
    
    # Ensure coverage of the right and bottom edges
    if xs[-1] != w - tile_size:
        xs.append(w - tile_size)
    if ys[-1] != h - tile_size:
        ys.append(h - tile_size)
    
    for y in ys:
        for x in xs:
            yield x, y, img.crop((x, y, x + tile_size, y + tile_size))

def main():
    parser = argparse.ArgumentParser(description="Tile images with overlap to generate 224x224 patches.")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory of style images.')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save tiles.')
    parser.add_argument('--tile_size', type=int, default=224, help='Size of each tile.')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio between tiles (0-1).')
    
    args = parser.parse_args()
    stride = int(args.tile_size * (1 - args.overlap))
    if stride < 1:
        stride = 1
    
    os.makedirs(args.target_dir, exist_ok=True)
    count = 0
    
    for fname in os.listdir(args.source_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')):
            img_path = os.path.join(args.source_dir, fname)
            img = Image.open(img_path).convert('RGB')
            base = os.path.splitext(fname)[0]
            
            for x, y, tile in tile_image(img, args.tile_size, stride):
                out_name = f"{base}_{x}_{y}.png"
                tile.save(os.path.join(args.target_dir, out_name))
                count += 1
    
    print(f"✅ Created {count} tiles in '{args.target_dir}'")

if __name__ == "__main__":
    main()

# Usage example:
# python tile_style_images.py --source_dir ./style_raw --target_dir ./style_tiled --tile_size 224 --overlap 0.5
