import os
from PIL import Image, ImageDraw, ImageFont

import os
from PIL import Image, ImageDraw, ImageFont


def create_image_grid(source_path, output_folder):
    ids = [
        item
        for item in os.listdir(os.path.join(source_path, "images"))
        if os.path.isdir(os.path.join(source_path, "images", item))
    ]

    for id in ids:
        context_path1 = os.path.join(source_path, "images", id, "context")
        gt_path = os.path.join(source_path, "images", id, "gt")
        color_path = os.path.join(source_path, "images", id, "color")

        context_images = [
            Image.open(os.path.join(context_path1, img))
            for img in sorted(os.listdir(context_path1))
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        gt_images = [
            Image.open(os.path.join(gt_path, img))
            for img in sorted(os.listdir(gt_path))
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        color_images = [
            Image.open(os.path.join(color_path, img))
            for img in sorted(os.listdir(color_path))
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        assert len(gt_images) == len(
            color_images
        ), f"Image counts do not match in {os.path.join(source_path, 'images', id)}"
        img_height = gt_images[0].height
        img_width = gt_images[0].width

        total_width = int(
            100 + (len(gt_images) + 1) * img_width
        )  # adjust total width for labels
        total_height = img_height * 2 + 60

        new_image = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.load_default()

        # Adjust top labels
        top_labels = ["Context"]
        top_labels.extend(map(str, list(range(1, len(gt_images) + 1))))
        x_positions = [
            100 + img_width * i + img_width / 2 for i in range(len(gt_images) + 1)
        ]
        for x_pos, label in zip(x_positions, top_labels):
            draw.text((x_pos, 30), label, font=font, fill="black", anchor="mm")

        # Side labels
        side_labels = ["GT", "Output"]
        for i, label in enumerate(side_labels):
            draw.text(
                (50, img_height * i + img_height / 2 + 60),
                label,
                font=font,
                fill="black",
                anchor="mm",
            )

        y_offset = 60
        x_offset = 100  # Space for row labels

        # Insert context images resized to maintain aspect ratio
        for index, img in enumerate(context_images):
            resized_img = img.resize((img_width, img_height))
            new_image.paste(resized_img, (x_offset, y_offset + img_height * index))

        # Insert GT, color images from source 1 and source 2
        for index, img in enumerate(gt_images + color_images):
            column = index % len(gt_images)
            row = index // len(gt_images)
            new_image.paste(
                img.resize((img_width, img_height)),
                (
                    x_offset + img_width * (column + 1),
                    y_offset + img_height * row,
                ),
            )

        # Save image
        output_path = os.path.join(output_folder, f"{id}/comparison.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    source_path = (
        "/media/disk4/zjwang/pixelsplat/outputs/2024-09-24/15-50-23"  # 更改为实际路径
    )
    output_folder = "/media/disk4/zjwang/pixelsplat/experiments/extrapolate_test"  # 更改为实际输出路径
    create_image_grid(source_path, output_folder)
