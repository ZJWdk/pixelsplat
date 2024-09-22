import os
from PIL import Image, ImageDraw, ImageFont

import os
from PIL import Image, ImageDraw, ImageFont



def create_image_grid(source_path1, source_path2, output_folder, dataset_label, ckpt_label):
    ids = [item for item in os.listdir(os.path.join(source_path1, 'images')) if os.path.isdir(os.path.join(source_path1, 'images', item))]

    for id in ids:
        context_path1 = os.path.join(source_path1, 'images', id, 'context')
        gt_path = os.path.join(source_path1, 'images', id, 'gt')
        color_path1 = os.path.join(source_path1, 'images', id, 'color')
        color_path2 = os.path.join(source_path2, 'images', id, 'color')

        context_images = [Image.open(os.path.join(context_path1, img)) for img in sorted(os.listdir(context_path1)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        gt_images = [Image.open(os.path.join(gt_path, img)) for img in sorted(os.listdir(gt_path)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        color_images1 = [Image.open(os.path.join(color_path1, img)) for img in sorted(os.listdir(color_path1)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        color_images2 = [Image.open(os.path.join(color_path2, img)) for img in sorted(os.listdir(color_path2)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        img_height = gt_images[0].height
        img_width = gt_images[0].width
        context_width = context_images[0].width * (img_height * 1.5 / context_images[0].height)
        
        total_width = int(context_width + img_width * 3 + 100)  # adjust total width for labels
        total_height = img_height * 3 + 60

        new_image = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.load_default()
        
        # Adjust top labels
        top_labels = ['Context', '1', '2', '3']
        x_positions = [100 + context_width / 2] + [100 + context_width + img_width * i + img_width / 2 for i in range(3)]
        for x_pos, label in zip(x_positions, top_labels):
            draw.text((x_pos, 30), label, font=font, fill='black', anchor="mm")

        # Side labels
        side_labels = ['GT', dataset_label, ckpt_label]
        for i, label in enumerate(side_labels):
            draw.text((50, img_height * i + img_height / 2 + 60), label, font=font, fill='black', anchor="mm")

        y_offset = 60
        x_offset = 100  # Space for row labels

        # Insert context images resized to maintain aspect ratio
        for index, img in enumerate(context_images):
            resized_img = img.resize((int(context_width), int(img_height * 1.5)))
            new_image.paste(resized_img, (x_offset, y_offset + int(img_height * 1.5) * index))

        # Insert GT, color images from source 1 and source 2
        for index, img in enumerate(gt_images + color_images1 + color_images2):
            column = index % 3
            row = index // 3
            new_image.paste(img.resize((img_width, img_height)), (x_offset + int(context_width) + img_width * column, y_offset + img_height * row))

        # Save image
        output_path = os.path.join(output_folder, f'{id}.png')
        new_image.save(output_path)
        print(f'Saved: {output_path}')


if __name__ == '__main__':
    source_path1 = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-11-59'  # 更改为实际路径
    source_path2 = '/media/disk4/zjwang/pixelsplat/outputs/2024-09-22/15-14-56'  # 更改为实际路径
    output_folder = '/media/disk4/zjwang/pixelsplat/experiments/cross_dataset_ckpt/acid_with_re10k_ckpt'  # 更改为实际输出路径
    dataset = 'acid'
    ckpt = 're10k'
    create_image_grid(source_path1, source_path2, output_folder, dataset, ckpt)
