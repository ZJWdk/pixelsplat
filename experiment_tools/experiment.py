import calculate_mse, analyse_view_psnr, compare_images


if __name__ == "__main__":
    exp_path = "/media/disk4/zjwang/pixelsplat/outputs/2024-09-24/19-37-01"
    output_path = "/media/disk4/zjwang/pixelsplat/experiments/extrapolate_test/acid"
    index_path = (
        "/media/disk4/zjwang/pixelsplat/assets/evaluation_extrapolate_index_acid.json"
    )
    flag = None
    compare_images.create_image_grid(exp_path, output_path)
    calculate_mse.process_images(exp_path, output_path, flag)
    analyse_view_psnr.plot_psnr_vs_extrapolate_frames(
        output_path + "/psnrs.json", index_path
    )
