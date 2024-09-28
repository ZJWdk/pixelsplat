import os
import cv2
import numpy as np
import imageio as iio
from pathlib import Path


def video_trans_size(input_mp4, fps):
    output_h264 = input_mp4.replace(".mp4", "_h264.mp4")
    cap = cv2.VideoCapture(input_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(width, height)

    # 定义编解码器并创建VideoWriter对象
    out = iio.get_writer(
        output_h264,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    while True:
        ret, frame = cap.read()
        if ret:
            image = frame[:, :, (2, 1, 0)]  # Convert BGR to RGB
            out.append_data(image)
        else:
            break
    cap.release()
    out.close()
    cv2.destroyAllWindows()
    # print("转换完成，输出文件为：", output_h264)
    os.remove(input_mp4)  # 删除原视频文件


def process_images(base_path):
    image_dirs = [
        dir_name
        for dir_name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, dir_name))
    ]
    for dir_name in image_dirs:
        current_path = os.path.join(base_path, dir_name)
        color_path = os.path.join(current_path, "color")
        gt_path = os.path.join(current_path, "gt")
        comparison_path = os.path.join(current_path, "comparison")
        os.makedirs(comparison_path, exist_ok=True)

        if os.path.exists(color_path) and os.path.exists(gt_path):
            color_images = sorted(
                [
                    os.path.join(color_path, f)
                    for f in os.listdir(color_path)
                    if f.endswith(".png")
                ]
            )
            gt_images = sorted(
                [
                    os.path.join(gt_path, f)
                    for f in os.listdir(gt_path)
                    if f.endswith(".png")
                ]
            )

            # 创建视频
            create_video(color_images, os.path.join(color_path, "video.mp4"), 30)
            create_video(gt_images, os.path.join(gt_path, "video.mp4"), 30)

            # 转换视频格式
            video_trans_size(os.path.join(color_path, "video.mp4"), 30)
            video_trans_size(os.path.join(gt_path, "video.mp4"), 30)

            # 处理图像合并和视频生成
            for c_img, g_img in zip(color_images, gt_images):
                img1 = cv2.imread(c_img)
                img2 = cv2.imread(g_img)
                h_concat = np.concatenate((img1, img2), axis=1)
                output_img_path = os.path.join(comparison_path, os.path.basename(c_img))
                cv2.imwrite(output_img_path, h_concat)

            # 生成comparison视频
            comparison_images = sorted(
                [
                    os.path.join(comparison_path, f)
                    for f in os.listdir(comparison_path)
                    if f.endswith(".png")
                ]
            )
            create_video(
                comparison_images,
                os.path.join(comparison_path, "comparison_video.mp4"),
                30,
            )
            video_trans_size(os.path.join(comparison_path, "comparison_video.mp4"), 30)


def create_video(image_paths, output_video_path, fps=5):
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in image_paths:
        video.write(cv2.imread(image))

    video.release()


# 设置基本路径
if __name__ == "__main__":
    base_path = "/media/disk4/zjwang/pixelsplat/datasets_visible/re10k"
    for stage in os.listdir(base_path):
        process_images(os.path.join(base_path, stage))
