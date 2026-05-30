import numpy as np
from PIL import Image

def analyze():
    img = Image.open("test_render.png").convert("RGB")
    arr = np.array(img)
    # Background is (255, 255, 255)
    mask = np.any(arr < 250, axis=-1)
    if not np.any(mask):
        print("No non-white pixels found!")
        return
    
    y, x = np.where(mask)
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    print(f"Bounding box: Y[{y_min}..{y_max}], X[{x_min}..{x_max}]")
    print(f"Width: {x_max - x_min}, Height: {y_max - y_min}")
    print(f"Coverage: {mask.mean() * 100:.1f}%")

if __name__ == "__main__":
    analyze()
