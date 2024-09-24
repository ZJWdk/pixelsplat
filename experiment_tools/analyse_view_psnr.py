import json
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_psnr_vs_extrapolate_frames(psnr_path, index_path):
    with open(psnr_path, "r") as f:
        psnrs = json.load(f)

    with open(index_path, "r") as f:
        index = json.load(f)

    psnrs.pop(0)  # remove average
    x = []
    y = []
    x_relative = []

    for scene, psnr in psnrs:
        frame_left, frame_right = index[scene]["context"]

        def get_nearest_distance(target, left, right):
            return min(abs(target - left), abs(target - right))

        average_interpolate_frames = np.mean(
            [
                get_nearest_distance(index[scene]["target"][i], frame_left, frame_right)
                for i in range(len(index[scene]["target"]))
            ]
        )
        x.append(average_interpolate_frames)
        x_relative.append(average_interpolate_frames / (frame_right - frame_left))
        y.append(psnr)

    # 绘制图形
    plt.scatter(x, y)
    plt.xlabel("Average_extrapolate_frames")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.savefig(os.path.dirname(psnr_path) + "/analysis.png")
    plt.clf()

    plt.scatter(x_relative, y)
    plt.xlabel("Average_extrapolate_frames_relative")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.savefig(os.path.dirname(psnr_path) + "/analysis_relative.png")


if __name__ == "__main__":
    psnr_path = "/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/re10k_with_acid_ckpt/psnrs_right.json"
    # index_path = (
    #     "/media/disk4/zjwang/pixelsplat/assets/evaluation_extrapolate_index_re10k.json"
    # )
    index_path = "/media/disk4/zjwang/pixelsplat/assets/evaluation_index_re10k.json"
    plot_psnr_vs_extrapolate_frames(psnr_path, index_path)
