import calculate_mse, analyse_view_psnr, compare_images


if __name__ == "__main__":
    exp_path = "/media/disk4/zjwang/pixelsplat/outputs/2024-09-28/23-42-56"
    output_path = (
        "/media/disk4/zjwang/pixelsplat/experiments/extreme_test/identity_context"
    )
    index_path = (
        "/media/disk4/zjwang/pixelsplat/assets/evaluation_extreme_identity_index.json"
    )
    flag = None

    calculate_mse.process_images(exp_path, output_path, flag)
    compare_images.create_image_grid(exp_path, output_path)
    # analyse_view_psnr.plot_psnr_vs_extrapolate_frames(
    #     output_path + "/psnrs.json", index_path
    # )
