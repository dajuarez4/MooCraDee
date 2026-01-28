# MooCraDee
# Crater Detector (SAM)

Detects craters in an image using Segment Anything (SAM). It fits a circle to each accepted mask and saves:
- an output image with red circles
- a CSV file with (x, y, radius)

## Requirements
- Python 3.10–3.12
- Ubuntu recommended
- Optional: GPU + CUDA for speed

## Setup

```bash
git clone https://github.com/dajuarez4/MooCraDee
cd MooCraDee

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python deep_moocrade.py  find_jackson_crater.png    --ckpt sam_vit_b_01ec64.pth   --out sam_out.png --csv sam_radii.csv   --min_radius 20 --max_radius 260   --min_circularity 0.35   --min_area 600   --pps 64 --pred_iou 0.80 --stability 0.85   --iou_dedup 0.12

