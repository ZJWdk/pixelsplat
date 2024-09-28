import os
import shutil


def delete_folders_except_b(root_folder, folder_to_keep):
    # 遍历root_folder中的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 检查是否是一个文件夹，并且不是要保留的文件夹
        if os.path.isdir(folder_path) and folder_name != folder_to_keep:
            print(f"正在删除文件夹: {folder_path}")
            shutil.rmtree(folder_path)  # 删除文件夹及其内容
        else:
            print(f"保留文件夹: {folder_name}")


# 使用示例
root_folder = (
    "/media/disk4/zjwang/pixelsplat/outputs/2024-09-25"  # 替换为文件夹A的实际路径
)
folder_to_keep = "19-36-32"  # 替换为要保留的文件夹名称

delete_folders_except_b(root_folder, folder_to_keep)
