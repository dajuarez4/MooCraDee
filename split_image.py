#!/usr/bin/env python3
# pip install pillow
# Run: python split_image.py 6
# or:  python split_image.py Mercury.tif 6

import sys
from pathlib import Path
from PIL import Image

# Mercury.tif is a very large image.
# Pillow blocks huge images by default, so we turn that limit off here.
Image.MAX_IMAGE_PIXELS = None

def split_image(image_path, n, output_dir=None):
    # This function divides one big image into smaller parts.
    # Later, run_pipeline.py can use these parts one by one.
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive and even integer, please")

    img_path = Path(image_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # If the user does not give an output folder,
    # we create one automatically.
    if output_dir is None:
        output_dir = img_path.parent / f"{img_path.stem}_split_{n}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the image and convert it to RGB so it is easier to save and reuse.
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    # Divide the image into 2 rows and n/2 columns.
    rows = 2
    cols = n // 2

    part_width = width // cols   # base width of each piece
    part_height = height // rows # base height of each piece

    part_num = 1
    part_info = []

    for row in range(rows):
        for col in range(cols):
            # These values define the borders of the crop box.
            left = col * part_width
            upper = row * part_height
            right = (col + 1) * part_width if col < cols - 1 else width
            lower = (row + 1) * part_height if row < rows - 1 else height

            piece = img.crop((left, upper, right, lower))

            part_folder = output_dir / f"part_{part_num}"
            part_folder.mkdir(parents=True, exist_ok=True)

            # We save each split image as input.png
            # because run_pipeline.py can use that name easily later.
            output_file = part_folder / "input.png"
            piece.save(output_file, format="PNG")

            print(f"Guardado: {output_file}")

            # Save useful information for later.
            # run_pipeline.py will need this.
            info = {
                "part_num": part_num,
                "part_folder": part_folder,
                "output_file": output_file,
                "x_offset": left,
                "y_offset": upper,
                "width": right - left,
                "height": lower - upper
            }

            part_info.append(info)
            part_num += 1

    img.close()

    print(f"\nDone. Created {n} folders in: {output_dir}")

    # This return is what makes the function reusable.
    # If you run the script alone, it still works normally.
    # If run_pipeline.py imports it, it can reuse this information.
    return part_info


def main():
    # Option 1:
    #   python split_image.py 6
    # This uses Mercury.tif by default.
    #
    # Option 2:
    #   python split_image.py Mercury.tif 6
    # This lets the user choose the image name.

    if len(sys.argv) == 2:
        image_path = "Mercury.tif"
        n = int(sys.argv[1])

    elif len(sys.argv) == 3:
        image_path = sys.argv[1]
        n = int(sys.argv[2])

    else:
        print("Uso:")
        print("  python split_image.py <n>")
        print("  python split_image.py <image_path> <n>")
        print("Ejemplo:")
        print("  python split_image.py 6")
        print("  python split_image.py Mercury.tif 6")
        sys.exit(1)

    try:
        split_image(image_path, n)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
