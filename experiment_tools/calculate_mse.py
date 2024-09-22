import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_mse(image1, image2):
    """ 计算两张图像的 MSE 损失。 """
    return np.mean((image1 - image2) ** 2)

def load_image(path):
    """ 加载图像并转换为灰度。 """
    return np.array(Image.open(path), dtype=np.float32)

def process_images(base_path, output_path, flag):
    """ 处理指定路径下的所有图像对。 """
    # 获取子目录列表，忽略文件
    base_path += "/images"
    image_dirs = [dir_name for dir_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir_name))]
    psnrs = {}
    for dir_name in image_dirs:
        current_path = os.path.join(base_path, dir_name)
        gt_path = os.path.join(current_path, 'gt')
        color_path = os.path.join(current_path, 'color')
        # 检查gt和color路径是否存在
        if not os.path.exists(gt_path) or not os.path.exists(color_path):
            print(f"Skipping {dir_name}, gt or color folder not found.")
            continue

        # 获取gt和color图像列表
        gt_images = {img: os.path.join(gt_path, img) for img in os.listdir(gt_path) if img.endswith('.png')}
        color_images = {img: os.path.join(color_path, img) for img in os.listdir(color_path) if img.endswith('.png')}
        psnr_list = []
        # 确保每个gt有对应的color图像
        for img_name, gt_img_path in gt_images.items():
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
                plt.imshow(error_map, cmap='hot')
                plt.colorbar()
                plt.title(f'MSE: {mse:.2f}, PSNR: {psnr:.2f}')

                # 创建输出目录
                output_dir = os.path.join(output_path, dir_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 保存错误图
                error_map_path = os.path.join(output_dir, f"{img_name}_{flag}.png")
                plt.savefig(error_map_path)
                plt.close()
                print(f"Processed {img_name}, Error map saved to {error_map_path}")
            else:
                print(f"No matching color image for {gt_img_path}")
        psnrs[dir_name] = np.mean(psnr_list)
    psnrs = sorted(psnrs.items(), key=lambda x: x[1], reverse=False)
    psnr_avg = np.mean([psnr for _, psnr in psnrs])
    psnrs.insert(0, ("average", psnr_avg))
    import json
    with open(os.path.join(output_path, f'psnrs_{flag}.json'), 'w') as f:
        json.dump(psnrs, f, indent=4)

# 示例用法
base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-11-15'
output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/re10k_with_acid_ckpt'
process_images(base_path, output_path, "right")
base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-11-59'
output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/acid_with_re10k_ckpt'
process_images(base_path, output_path, "right")
base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-14-12'
output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/re10k_with_acid_ckpt'
process_images(base_path, output_path, "wrong")
base_path = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-14-56'
output_path = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/acid_with_re10k_ckpt'
process_images(base_path, output_path, "wrong")