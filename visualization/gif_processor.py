from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


def process_gif(gif_path, output_path, font_size=16, title=None):
    with Image.open(gif_path) as gif:
        total_frames = gif.n_frames

        print(f"GIF总帧数: {total_frames}")

        frames_to_extract = [
            0,
            max(1, int(total_frames * 1 / 5)),
            int(total_frames * 2 / 5),
            int(total_frames * 3 / 5),
            int(total_frames * 4 / 5),
            total_frames - 1
        ]

        frames_to_extract = [min(i, total_frames - 1) for i in frames_to_extract]

        frames_with_labels = []
        for i, frame_idx in enumerate(frames_to_extract):
            gif.seek(frame_idx)
            frame = gif.copy()

            label_text = f"Time: {['Start', '1/5', '2/5', '3/5', '4/5', 'End'][i]}"
            frames_with_labels.append((frame, label_text))

        grid_rows = 3
        grid_cols = 2

        max_width = max(frame.size[0] for frame, _ in frames_with_labels)
        max_height = max(frame.size[1] for frame, _ in frames_with_labels)
        label_height = font_size + 10

        title_height = 100

        result_width = max_width * grid_cols
        result_height = (max_height + label_height) * grid_rows + title_height
        result_image = Image.new('RGB', (result_width, result_height), color=(255, 255, 255))

        try:
            font_path = "C:/Windows/Fonts/simheib.ttf"
            if not os.path.exists(font_path):
                font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            font = ImageFont.truetype(font_path, font_size)
            title_font = ImageFont.truetype(font_path, 48)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        draw = ImageDraw.Draw(result_image)

        if title:
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height_text = title_bbox[3] - title_bbox[1]

            title_area_top_margin = 10
            title_area_bottom_margin = 10
            total_title_area_height = title_height_text + title_area_top_margin + title_area_bottom_margin

            title_y_start = title_area_top_margin

            title_y = title_y_start + (total_title_area_height - title_height_text) // 2

            title_x = (result_width - title_width) // 2

            draw.text((title_x, title_y), title, font=title_font, fill=(0, 0, 0))

        for i, (frame, label) in enumerate(frames_with_labels):
            row = i // grid_cols
            col = i % grid_cols

            x_offset = col * max_width + (max_width - frame.size[0]) // 2
            y_offset = row * (max_height + label_height) + title_height

            result_image.paste(frame, (x_offset, y_offset))

            text_x = col * max_width + max_width // 2
            text_y = y_offset + max_height + 5
            draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0), anchor="mt")

        result_image.save(output_path)
        print(f"已保存拼接图片至: {output_path}")

        return result_image
