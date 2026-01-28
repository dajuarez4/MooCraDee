# MooCraDee — Crater Detector (SAM)

<p align="center">
  <img src="assets/banner.png" width="900" alt="MooCraDee banner"/>
</p>

<p align="center">
  <b>Detect craters in images using Segment Anything (SAM)</b><br/>
  Fits a circle to each accepted mask and exports an annotated image + CSV (x, y, radius).
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/platform-Ubuntu-orange" alt="Platform"/>
  <img src="https://img.shields.io/badge/accelerator-GPU%20(CUDA)%20optional-success" alt="GPU optional"/>
  <img src="https://img.shields.io/badge/status-research%20tool-informational" alt="Status"/>
</p>

---

## What it does

Given a crater image, **MooCraDee** uses **Segment Anything (SAM)** to segment crater-like regions, then:
- **fits a circle** to each accepted mask
- writes an **output image** with **red circles**
- saves a **CSV** with crater parameters: **(x, y, radius)**

---

## Demo

> Add your images here (recommended):
> - `assets/input.png` = input image  
> - `assets/sam_out.png` = annotated output (red circles)

<p align="center">
  <img src="assets/find_jackson_crater.png" width="360" alt="Input image"/>
  <img src="assets/sam_out.png" width="360" alt="Output with circles"/>
</p>

<p align="center"><b>Left:</b> input image. <b>Right:</b> detected craters + circle fits.</p>

---

## Requirements

- **Python:** 3.10–3.12  
- **OS:** Ubuntu recommended (works on other Linux/macOS if deps install)  
- **Hardware:** CPU works; **GPU + CUDA** optional for speed  
- **Model:** SAM checkpoint file (e.g., `sam_vit_b_01ec64.pth`)

---

## Setup

```bash
git clone https://github.com/dajuarez4/MooCraDee.git
cd MooCraDee

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
