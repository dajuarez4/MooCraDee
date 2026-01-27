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
