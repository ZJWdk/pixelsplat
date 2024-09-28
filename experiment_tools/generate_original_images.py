import torch
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from create_gt_video import create_gt_video


def convert_images(data, save_path):
    for scene_data in data:
        scene_path = save_path / scene_data["key"]
        scene_path.mkdir(exist_ok=True, parents=True)
        # 确保 images 的长度已知，并传递给 tqdm
        total_images = len(scene_data["images"])
        for i, image in tqdm(
            enumerate(scene_data["images"]),
            desc="Image",
            total=total_images,  # 添加 total 参数
            ncols=100,
            leave=False,
            bar_format="{l_bar}{bar:30} | {n_fmt}/{total_fmt} [{percentage:.0f}%]",  # 添加百分比和计数格式
        ):
            image = Image.open(BytesIO(image.numpy().tobytes()))
            image.save(scene_path / f"{i:04d}.png")


if __name__ == "__main__":
    torch_path = Path("/media/disk4/zjwang/pixelsplat/datasets/re10k")
    output_path = Path("/media/disk4/zjwang/pixelsplat/datasets_visible/re10k")

    for subset in torch_path.iterdir():
        print(f"Processing {subset.name}...")
        if subset.is_dir():
            subset_files = list(subset.iterdir())  # 获取所有子文件，以便确定总数
            total_chunks = len(subset_files)
            for chunk in tqdm(
                subset_files,
                desc="Chunk",
                total=total_chunks,  # 添加 total 参数
                ncols=100,
                bar_format="{l_bar}{bar:30} | {n_fmt}/{total_fmt} [{percentage:.0f}%]",  # 显示总数和百分比
            ):
                if chunk.suffix == ".torch":
                    data = torch.load(chunk)
                    convert_images(data, output_path / subset.name)
    create_gt_video(output_path)
