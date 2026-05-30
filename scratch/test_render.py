import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
from avatar_renderer import AvatarRenderer
from PIL import Image

def main():
    ckpt = "./output/demo_character/extracted/biden_001"
    renderer = AvatarRenderer(ckpt, resolution=512)
    frame = renderer.render_idle()
    print(f"Rendered frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Mean value: {frame.mean()}")
    Image.fromarray(frame).save("test_render.png")
    print("Saved test_render.png")

if __name__ == "__main__":
    main()
