#!/usr/bin/env python3
# pip install pillow RUN python split_image.py N
import sys
from pathlib import Path
from PIL import Image


def split_image(image_path: str, n: int) -> None:
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive and even integer, please")

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(img_path) # abre la imagen y recortarla
    width, height = img.size

    # Divido la img en 2 filas y n/2 columnas
    rows = 2
    cols = n // 2

    part_width = width // cols #BASE
    part_height = height // rows

    output_dir = img_path.parent / f"{img_path.stem}_split_{n}" # crea la ruta Folder Principal
    output_dir.mkdir(exist_ok=True) 

    part_num = 1

    for row in range(rows):
        for col in range(cols):
            left = col * part_width # bordes
            upper = row * part_height
            right = (col + 1) * part_width if col < cols - 1 else width #edgecase
            lower = (row + 1) * part_height if row < rows - 1 else height

            piece = img.crop((left, upper, right, lower))

            part_folder = output_dir / f"part_{part_num}"
            part_folder.mkdir(exist_ok=True)

            output_file = part_folder / f"{img_path.stem}_part_{part_num}.png"
            piece.save(output_file, format="PNG")

            print(f"Guardado: {output_file}")
            part_num += 1

    print(f"\nDone. Created {n} folders in: {output_dir}")


def main():
    if len(sys.argv) != 2:
        print("Uso: python split_image.py <n>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        split_image("mercury.jpg", n)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()