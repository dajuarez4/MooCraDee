#!/usr/bin/env python3
# Run the full MooCraDee workflow:
# 1. Split Mercury.tif into smaller parts.
# 2. Run deep_moocrade.py on each part.
# 3. Save each detected image and CSV.
# 4. Create one combined CSV with all crater detections.

import sys
import csv
import subprocess
from pathlib import Path

from split_image import split_image


def check_files(image_path, ckpt_path, deep_script):
    # This checks that the important files exist before starting the pipeline.
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")

    if not Path(deep_script).exists():
        raise FileNotFoundError(f"Detector script not found: {deep_script}")


def run_detector_on_part(part, ckpt_path, deep_script):
    # This runs deep_moocrade.py on one split image.
    part_num = part["part_num"]
    part_folder = Path(part["part_folder"])
    input_image = Path(part["output_file"])

    detected_image = part_folder / "detected.png"
    csv_file = part_folder / "radii.csv"
    log_file = part_folder / "output.txt"

    command = [
        sys.executable,
        str(deep_script),
        str(input_image),
        "--ckpt", str(ckpt_path),
        "--out", str(detected_image),
        "--csv", str(csv_file),

        # These are the same parameters from the original project example.
        "--min_radius", "20",
        "--max_radius", "260",
        "--min_circularity", "0.35",
        "--min_area", "600",
        "--pps", "64",
        "--pred_iou", "0.80",
        "--stability", "0.85",
        "--iou_dedup", "0.12",
    ]

    print("\n" + "=" * 60)
    print(f"Running detector on part {part_num}")
    print(f"Input: {input_image}")
    print(f"Output image: {detected_image}")
    print(f"CSV: {csv_file}")
    print("=" * 60)

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    print(result.stdout)

    # Save the terminal output so we can check Device: cuda later.
    log_file.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"deep_moocrade.py failed on part {part_num}. Check {log_file}")


def combine_csvs(parts, output_dir):
    # This creates one CSV with the crater detections from all parts.
    # It also converts local coordinates into global coordinates.
    combined_csv = Path(output_dir) / "all_craters.csv"

    with combined_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "part",
            "local_id",
            "local_x_px",
            "local_y_px",
            "global_x_px",
            "global_y_px",
            "radius_px",
            "score"
        ])

        for part in parts:
            part_num = part["part_num"]
            part_folder = Path(part["part_folder"])
            part_csv = part_folder / "radii.csv"

            if not part_csv.exists():
                print(f"Warning: missing CSV for part {part_num}")
                continue

            with part_csv.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    local_x = float(row["x_px"])
                    local_y = float(row["y_px"])

                    global_x = local_x + part["x_offset"]
                    global_y = local_y + part["y_offset"]

                    writer.writerow([
                        part_num,
                        row["id"],
                        f"{local_x:.2f}",
                        f"{local_y:.2f}",
                        f"{global_x:.2f}",
                        f"{global_y:.2f}",
                        row["radius_px"],
                        row["score"]
                    ])

    print(f"\nCombined CSV saved: {combined_csv}")


def main():
    # Default project files.
    image_path = "mercury.jpg"
    ckpt_path = "sam_vit_b_01ec64.pth"
    deep_script = "deep_moocrade.py"

    # Example:
    #   python run_pipeline.py 6
    # This means:
    #   split Mercury.tif into 6 parts and run crater detection on each part.
    if len(sys.argv) != 2:
        print("Uso: python run_pipeline.py <n>")
        print("Ejemplo: python run_pipeline.py 6")
        sys.exit(1)

    try:
        n = int(sys.argv[1])

        check_files(image_path, ckpt_path, deep_script)

        output_dir = Path(image_path).parent / f"{Path(image_path).stem}_split_{n}"

        # First, split the large Mercury.tif image.
        parts = split_image(image_path, n, output_dir)

        # Then, run MooCraDee on every split image.
        for part in parts:
            run_detector_on_part(part, ckpt_path, deep_script)

        # Finally, merge all small CSV files into one global CSV.
        combine_csvs(parts, output_dir)

        print("\nPipeline complete.")
        print(f"Results saved in: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()