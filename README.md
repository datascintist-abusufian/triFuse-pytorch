How to Run
Install Dependencies:
bash
pip install -r requirements.txt
Download and Preprocess Data:
bash
# For ACDC
python scripts/preprocess.py --input /path/to/acdc --output data/acdc --dataset acdc --target_size 256

# For MSCMRseg
python scripts/preprocess.py --input /path/to/mscmrseg --output data/mscmrseg --dataset mscmrseg --target_size 256
Train:
bash
python train.py --config configs/triFuse_srnet.yaml --dataset acdc --gpus 0,1,2,3 --seed 42
Evaluate:
bash
python eval.py --checkpoint checkpoints/acdc_42_best.pth --dataset acdc --output results/acdc/
Statistical Analysis:
bash
python scripts/statistical_analysis.py --method1 results/acdc/ --method2 results/cyclemix/ --dataset acdc --output comparison.csv


# TriFuse-SRNet: Dynamic Multi-Expert Fusion With Structural Recovery for Scribble-Supervised Cardiac MRI Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-TMI%202026-red.svg)](https://ieeexplore.ieee.org/)
