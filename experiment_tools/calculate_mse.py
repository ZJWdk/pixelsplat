import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def calculate_mse(image1, image2):
    """计算两张图像的 MSE 损失。"""
    return np.mean((image1 - image2) ** 2)


def load_image(path):
    """加载图像并转换为灰度。"""
    return np.array(Image.open(path), dtype=np.float32)


def process_images(
    base_path,
    output_path,
    flag,
):
    """处理指定路径下的所有图像对。"""
    # 获取子目录列表，忽略文件
    base_path += "/images"
    image_dirs = [
        dir_name
        for dir_name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, dir_name))
    ]
    psnrs = {}
    for dir_name in image_dirs:
        current_path = os.path.join(base_path, dir_name)
        gt_path = os.path.join(current_path, "gt")
        color_path = os.path.join(current_path, "color")
        # 检查gt和color路径是否存在
        if not os.path.exists(gt_path) or not os.path.exists(color_path):
            print(f"Skipping {dir_name}, gt or color folder not found.")
            continue

        # 获取gt和color图像列表
        gt_images = {
            img: os.path.join(gt_path, img)
            for img in os.listdir(gt_path)
            if img.endswith(".png")
        }
        color_images = {
            img: os.path.join(color_path, img)
            for img in os.listdir(color_path)
            if img.endswith(".png")
        }
        psnr_list = []
        # 确保每个gt有对应的color图像
        for img_name, gt_img_path in tqdm(
            gt_images.items(),
            desc=f"Processing images in {dir_name}",
            total=len(gt_images),
        ):
            color_img_path = color_images.get(img_name)
            if color_img_path:
                gt_img = load_image(gt_img_path)
                color_img = load_image(color_img_path)

                # 计算MSE
                mse = calculate_mse(gt_img, color_img)
                psnr = 20 * np.log10(255) - 10 * np.log10(mse)
                psnr_list.append(psnr)
                # 生成错误图
                error_map = np.abs(gt_img - color_img)
                error_map = np.mean(error_map, axis=2)
                plt.imshow(error_map, cmap="hot")
                plt.colorbar()
                plt.title(f"MSE: {mse:.2f}, PSNR: {psnr:.2f}")

                # 创建输出目录
                output_dir = os.path.join(output_path, dir_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 保存错误图
                if flag is None:
                    error_map_path = os.path.join(
                        output_dir, f"{img_name.split('.')[0]}.png"
                    )
                else:
                    error_map_path = os.path.join(
                        output_dir, f"{img_name.split('.')[0]}_{flag}.png"
                    )
                plt.savefig(error_map_path)
                plt.close()
                # print(f"Processed {img_name}, Error map saved to {error_map_path}")
            else:
                print(f"No matching color image for {gt_img_path}")
        psnrs[dir_name] = np.mean(psnr_list)

    psnrs = sorted(psnrs.items(), key=lambda x: x[1], reverse=False)
    best_path = (Path(output_path) / psnrs[-1][0]).resolve()
    worst_path = (Path(output_path) / psnrs[0][0]).resolve()

    # 创建软链接之前，确保目标路径与创建位置不同
    best_symlink_path = Path(output_path) / "_best"
    worst_symlink_path = Path(output_path) / "_worst"

    # 检查软链接是否已存在，避免重复创建
    if not best_symlink_path.exists():
        os.symlink(best_path, best_symlink_path)
        print(f"软链接已创建: {best_symlink_path} -> {best_path}")
    else:
        print(f"软链接 {best_symlink_path} 已存在")

    if not worst_symlink_path.exists():
        os.symlink(worst_path, worst_symlink_path)
        print(f"软链接已创建: {worst_symlink_path} -> {worst_path}")
    else:
        print(f"软链接 {worst_symlink_path} 已存在")
    psnr_avg = np.mean([psnr for _, psnr in psnrs])
    psnrs.insert(0, ("average", psnr_avg))
    import json

    json_name = f"psnrs_{flag}.json" if flag else "psnrs.json"
    with open(os.path.join(output_path, json_name), "w") as f:
        json.dump(psnrs, f, indent=4)


if __name__ == "__main__":
    # 示例用法
    base_path = "/media/disk4/zjwang/pixelsplat/outputs/2024-09-24/15-50-23"
    output_path = "/media/disk4/zjwang/pixelsplat/experiments/extrapolate_test"
    process_images(base_path, output_path, None)
    # base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-11-59'
    # output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/acid_with_re10k_ckpt'
    # process_images(base_path, output_path, "right")
    # base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-14-12'
    # output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/re10k_with_acid_ckpt'
    # process_images(base_path, output_path, "wrong")
    # base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-14-56'
    # output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/acid_with_re10k_ckpt'
    # process_images(base_path, output_path, "wrong")
