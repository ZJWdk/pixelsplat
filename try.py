import os
from pathlib import Path

src = "/media/disk4/zjwang/pixelsplat/outputs/test/re10k/2024-09-22/13-42-03"
dst = Path("/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/13-42-03") / 'images'

os.system(f"ln -s {src} {dst}")