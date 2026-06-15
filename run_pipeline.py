#!/usr/bin/env python3
"""
Run the full PCS / MooCraDee split-image pipeline.

This script:
1. Splits a planetary image into smaller parts.
2. Runs deep_moocrade.py on each part.
3. Saves one detected image and CSV per part.
4. Combines all part-level CSV files into one crater-candidate table.
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

from split_image import split_image


def check_required_files(image_path, checkpoint_path, detector_script):
    """Check that the input image, SAM checkpoint, and detector script exist."""
    required_files = {
        "Input image": image_path,
        "SAM checkpoint": checkpoint_path,
        "Detector script": detector_script,
    }

    for label, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def run_detector_on_part(part, args):
    """Run deep_moocrade.py on one split image."""
    part_number = part["part_num"]
    part_folder = Path(part["part_folder"])
    input_image = Path(part["output_file"])

    detected_image = part_folder / "detected.png"
    part_csv = part_folder / "radii.csv"
    log_file = part_folder / "output.txt"

    command = [
        "python",
        str(args.detector_script),
        str(input_image),
        "--ckpt", str(args.checkpoint),
        "--out", str(detected_image),
        "--csv", str(part_csv),
        "--min_radius", str(args.min_radius),
        "--max_radius", str(args.max_radius),
        "--min_circularity", str(args.min_circularity),
        "--min_area", str(args.min_area),
        "--pps", str(args.points_per_side),
        "--pred_iou", str(args.pred_iou),
        "--stability", str(args.stability),
        "--iou_dedup", str(args.iou_dedup),
    ]

    print("\n" + "=" * 60)
    print(f"Running detector on part {part_number}")
    print(f"Input: {input_image}")
    print(f"Output image: {detected_image}")
    print(f"CSV: {part_csv}")
    print("=" * 60)

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(result.stdout)
    log_file.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"deep_moocrade.py failed on part {part_number}. "
            f"Check log file: {log_file}"
        )


def combine_part_csvs(parts, output_dir):
    """Combine part-level CSV files and convert local coordinates into global coordinates."""
    rows = []

    for part in parts:
        part_number = part["part_num"]
        part_csv = Path(part["part_folder"]) / "radii.csv"

        if not part_csv.exists():
            print(f"Warning: missing CSV for part {part_number}")
            continue

        part_df = pd.read_csv(part_csv)

        if part_df.empty:
            continue

        part_df["part"] = part_number
        part_df["local_id"] = part_df["id"]
        part_df["local_x_px"] = part_df["x_px"]
        part_df["local_y_px"] = part_df["y_px"]
        part_df["global_x_px"] = part_df["x_px"] + part["x_offset"]
        part_df["global_y_px"] = part_df["y_px"] + part["y_offset"]

        rows.append(
            part_df[
                [
                    "part",
                    "local_id",
                    "local_x_px",
                    "local_y_px",
                    "global_x_px",
                    "global_y_px",
                    "radius_px",
                    "score",
                ]
            ]
        )

    combined_csv = Path(output_dir) / "all_craters.csv"

    if rows:
        combined_df = pd.concat(rows, ignore_index=True)
    else:
        combined_df = pd.DataFrame(
            columns=[
                "part",
                "local_id",
                "local_x_px",
                "local_y_px",
                "global_x_px",
                "global_y_px",
                "radius_px",
                "score",
            ]
        )

    combined_df.to_csv(combined_csv, index=False)

    print(f"\nCombined CSV saved: {combined_csv}")
    return combined_csv


def parse_arguments():
    """Define command-line options for the PCS split-image pipeline."""
    parser = argparse.ArgumentParser(
        description="Run PCS / MooCraDee on a planetary image using split-image processing."
    )

    # Backward compatibility:
    # Allows the prev command: python run_pipeline.py 6
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        help="Optional positional number of splits for backward compatibility.",
    )

    parser.add_argument(
        "--image",
        default="mercury.jpg",
        help="Input planetary image path. Default: mercury.jpg",
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=6,
        help="Number of image parts. Must be a positive even integer.",
    )

    parser.add_argument(
        "--checkpoint",
        default="sam_vit_b_01ec64.pth",
        help="SAM checkpoint path.",
    )

    parser.add_argument(
        "--detector_script",
        default="deep_moocrade.py",
        help="Detector script path.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. If not provided, one is created automatically.",
    )

    # Detection parameters
    parser.add_argument("--min_radius", type=float, default=20)
    parser.add_argument("--max_radius", type=float, default=260)
    parser.add_argument("--min_circularity", type=float, default=0.35)
    parser.add_argument("--min_area", type=int, default=600)
    parser.add_argument("--points_per_side", type=int, default=64)
    parser.add_argument("--pred_iou", type=float, default=0.80)
    parser.add_argument("--stability", type=float, default=0.85)
    parser.add_argument("--iou_dedup", type=float, default=0.12)

    args = parser.parse_args()

    # If the old positional argument is used, it overrides --splits.
    if args.n is not None:
        args.splits = args.n

    if args.splits <= 0 or args.splits % 2 != 0:
        parser.error("--splits must be a positive even integer, such as 2, 4, 6, or 8.")

    return args


def main():
    args = parse_arguments()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    detector_script = Path(args.detector_script)

    check_required_files(image_path, checkpoint_path, detector_script)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = image_path.parent / f"{image_path.stem}_split_{args.splits}"

    print("PCS split-image pipeline")
    print("Input image:", image_path)
    print("Number of splits:", args.splits)
    print("Output directory:", output_dir)

    # Step 1: Split the selected image.
    parts = split_image(image_path, args.splits, output_dir)

    # Step 2: Run crater-candidate detection on each image part.
    for part in parts:
        run_detector_on_part(part, args)

    # Step 3: Combine all part-level detections into one CSV.
    combined_csv = combine_part_csvs(parts, output_dir)

    print("\nPipeline complete.")
    print("Results saved in:", output_dir)
    print("Combined crater-candidate CSV:", combined_csv)

if __name__ == "__main__":
    main()
