from pathlib import Path
from create_video import create_video, video_trans_size
from tqdm import tqdm


def create_gt_video(base_path):
    for stage in base_path.iterdir():
        print(f"Processing {stage.name}...")
        current_path = base_path / stage
        total_chunks = len(list(current_path.iterdir()))  # 获取所有子文件，以便确定总数
        for scene in tqdm(
            current_path.iterdir(),
            desc="Scenes",
            total=total_chunks,  # 添加 total 参数
            ncols=100,
            bar_format="{l_bar}{bar:30} | {n_fmt}/{total_fmt} [{percentage:.0f}%]",
            # leave=False,
        ):
            current_path = base_path / stage / scene
            images = []
            for img in current_path.iterdir():
                if img.suffix == ".png" or img.suffix == ".jpg":
                    images.append(str(img))
            images = sorted(images, key=lambda x: int((x.split(".")[0].split("/")[-1])))
            create_video(images, str(current_path / f"_video.mp4"), 30)
            video_trans_size(str(current_path / f"_video.mp4"), 30)


if __name__ == "__main__":
    base_path = Path("/media/disk4/zjwang/pixelsplat/datasets_visible/re10k")
    create_gt_video(base_path)
