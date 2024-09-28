import json
from pathlib import Path

if __name__ == "__main__":
    transforms_path = Path("/media/disk4/zjwang/stereo_data/transforms.json")
    new_transforms_path = Path("/media/disk4/zjwang/stereo_data/relative_transforms.json")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)
    for i in range(len(transforms['frames'])):
        transforms['frames'][i]["file_path"] = str(Path(*Path(transforms['frames'][i]["file_path"]).parts[-3:]))
    with open(new_transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)