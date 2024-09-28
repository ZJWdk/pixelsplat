from __future__ import print_function
import os, json
import torch
import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import uuid
from pympler import asizeof

DATASET_DIR = Path("/media/disk4/zjwang/stereo_data")
OUTPUT_DIR = Path("/media/disk4/zjwang/pixelsplat/datasets/stereo_data")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


# def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
#     image_keys = set(
#         example.name
#         for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing images")
#     )
#     metadata_keys = set(
#         example.stem
#         for example in tqdm(
#             (INPUT_METADATA_DIR / stage).iterdir(), desc="Indexing metadata"
#         )
#     )

#     missing_image_keys = metadata_keys - image_keys
#     if len(missing_image_keys) > 0:
#         print(
#             f"Found metadata but no images for {len(missing_image_keys)} examples.",
#             file=sys.stderr,
#         )
#     missing_metadata_keys = image_keys - metadata_keys
#     if len(missing_metadata_keys) > 0:
#         print(
#             f"Found images but no metadata for {len(missing_metadata_keys)} examples.",
#             file=sys.stderr,
#         )

#     keys = image_keys & metadata_keys
#     print(f"Found {len(keys)} keys.")
#     return keys

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {int(path.stem): load_raw(path) for path in example_path.iterdir()}


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    with example_path.open("r") as f:
        lines = f.read().splitlines()

    url = lines[0]

    timestamps = []
    cameras = []

    for line in lines[1:]:
        timestamp, *camera = line.split(" ")
        timestamps.append(int(timestamp))
        cameras.append(np.fromstring(",".join(camera), sep=","))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    for stage in ["test"]:
        name_map_path = DATASET_DIR / "undistorted" / "relative_name_map.json"
        with open(name_map_path, "r") as f:
            name_map = json.load(f)
        transforms_path = DATASET_DIR / "relative_transforms.json"
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        fx = transforms["fl_x"] / transforms["w"]
        fy = transforms["fl_y"] / transforms["h"]
        cx = transforms["cx"] / transforms["w"]
        cy = transforms["cy"] / transforms["h"]
        # keys = get_example_keys(stage)
        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        original_image_paths = [
            key for key in name_map.keys() if Path(key).name.startswith("left")
        ]
        original_image_paths = sorted(
            original_image_paths, key=lambda x: int(str(Path(x).stem)[4:])
        )
        frames = [
            frame
            for frame in transforms["frames"]
            if Path(frame["file_path"]).name.startswith("left")
        ]
        frames = sorted(frames, key=lambda x: int(str(Path(x["file_path"]).stem)[4:]))
        assert len(original_image_paths) == len(
            frames
        ), f"Number of images {len(original_image_paths)} does not match number of frames {len(frames)}"
        count = 0
        prefix = "StereoScene"
        name_index = 0
        images = []
        cameras = []
        key = f"{prefix}{name_index:0>4}"
        intrinsic = [fx, fy, cx, cy]
        while len(original_image_paths) > 0:
            if count % 30 == 0 and count != 0:
                name_index += 1
                assert len(images) == len(
                    cameras
                ), f"Number of images {len(images)} does not match number of cameras {len(cameras)}"
                example = Example()
                example["url"] = ""
                example["timestamps"] = torch.zeros(30, dtype=torch.int64)
                example["images"] = images
                images = []
                example["cameras"] = torch.tensor(
                    np.stack(cameras), dtype=torch.float32
                )
                cameras = []
                example["key"] = key
                chunk.append(example)
                print(
                    f"    Added {key} to chunk ({total_size(example['images']) / 1e6:.2f} MB)."
                )
                key = f"{prefix}{name_index:0>4}"

            count += 1
            image_key = original_image_paths.pop(0)
            images.append(load_raw(DATASET_DIR / Path(*image_key.split("/")[1:])))
            extrinsic = np.array(frames.pop(0)["transform_matrix"])
            extrinsic = np.linalg.inv(extrinsic)[:3, :]
            cameras.append(intrinsic + [0, 0] + extrinsic.flatten().tolist())
        assert (
            len(frames) == len(original_image_paths) == 0
        ), f"Number of images {len(original_image_paths)} does not match number of frames {len(frames)}"
        if len(images) > 0:
            example = Example()
            example["url"] = ""
            example["timestamps"] = torch.zeros(len(images), dtype=torch.int64)
            example["images"] = images
            example["cameras"] = torch.tensor(np.stack(cameras), dtype=torch.float32)
            example["key"] = key
            print(
                f"    Added {key} to chunk ({total_size(example['images']) / 1e6:.2f} MB)."
            )
            chunk.append(example)
        save_chunk()
