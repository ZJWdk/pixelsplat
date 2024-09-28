import json
from pathlib import Path

if __name__ == "__main__":
    name_map_path = Path("/media/disk4/zjwang/stereo_data/undistorted/name_map.json")
    new_name_map_path = Path("/media/disk4/zjwang/stereo_data/undistorted/relative_name_map.json")
    with open(name_map_path, "r") as f:
        name_map = json.load(f)
    relative_name_map = {}
    for key, value in name_map.items():
        key = Path(*Path(key).parts[-3:])
        value = Path(*Path(value).parts[-3:])
        relative_name_map[str(key)] = str(value)
    with open(new_name_map_path, "w") as f:
        json.dump(relative_name_map, f, indent=4)