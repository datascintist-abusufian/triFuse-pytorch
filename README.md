# TriFuse-SRNet

**Dynamic multi-expert fusion with structural recovery for scribble-supervised cardiac MRI segmentation.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

PyTorch reference implementation, with training, evaluation and statistical-comparison scripts for ACDC and MSCMRseg.

## Why scribbles

Dense pixel-level annotation of cardiac MRI is expensive and requires expert time that most centres cannot spare. Scribble supervision replaces full masks with a few strokes per structure, which cuts annotation cost by an order of magnitude but leaves the majority of pixels unlabelled. The difficulty is that a network trained on strokes alone tends to produce anatomically implausible boundaries. TriFuse-SRNet addresses this by combining complementary experts and adding an explicit structural-recovery objective.

## Method

The model is assembled in `models/triFuse_srnet.py` from the following components.

- `models/encoder.py` - shared feature encoder.
- `models/cnn_expert.py` - convolutional expert, responsible for local boundary detail.
- `models/transformer_expert.py` - transformer expert, responsible for long-range anatomical context.
- `models/hmna.py` - HMNA attention module used to relate features across scales.
- `models/sr_branch.py` - structural-recovery branch, which reconstructs plausible anatomy in the regions the scribbles never touch.
- `models/dynamic_mix.py` - dynamic mixing, which weights the experts per input rather than with fixed coefficients.

Losses, metrics and statistics live in `utils/losses.py`, `utils/metrics.py` and `utils/stats.py`.

## Data

Two public benchmarks are supported, each with a dataset adapter and fixed splits so that results are comparable across runs.

- **ACDC** - Automated Cardiac Diagnosis Challenge. Adapter: `datasets/acdc.py`. Splits: `splits/acdc_train.txt`, `splits/acdc_val.txt`, `splits/acdc_test.txt`.
- **MSCMRseg** - multi-sequence cardiac MRI segmentation. Adapter: `datasets/mscmrseg.py`.

Neither dataset is redistributed here. Download them from their official sources, then preprocess.

## Usage

Install dependencies.

```bash
pip install -r requirements.txt
```

Preprocess.

```bash
python scripts/preprocess.py --input /path/to/acdc --output data/acdc --dataset acdc --target_size 256
python scripts/preprocess.py --input /path/to/mscmrseg --output data/mscmrseg --dataset mscmrseg --target_size 256
```

Train.

```bash
python train.py --config configs/triFuse_srnet.yaml --dataset acdc --gpus 0,1,2,3 --seed 42
```

Evaluate.

```bash
python eval.py --checkpoint checkpoints/acdc_42_best.pth --dataset acdc --output results/acdc/
```

Compare against a baseline with paired statistics.

```bash
python scripts/statistical_analysis.py --method1 results/acdc/ --method2 results/cyclemix/ --dataset acdc --output comparison.csv
```

All experiments are seeded; `--seed 42` reproduces the reference runs.

## Repository layout

```
configs/        experiment configuration (YAML)
datasets/       ACDC and MSCMRseg adapters
models/         encoder, CNN expert, transformer expert, HMNA, SR branch, dynamic mixing
scripts/        preprocessing and statistical comparison
splits/         fixed train / validation / test splits
utils/          losses, metrics, statistics
train.py        training entry point
```

Note: the current upload stores these files flat, with the folder name encoded in the filename (for example `models:encoder.py` rather than `models/encoder.py`). Reorganising them into real directories is the next housekeeping step; the commands above assume the layout shown here.

## Related work

This implementation sits alongside earlier work on hybrid architectures for cardiac MRI: *Hybrid deep learning for computational precision in cardiac MRI segmentation: integrating autoencoders, CNNs and RNNs for enhanced structural analysis*, Computers in Biology and Medicine 186 (2025) 109597.

See also [Deep-Spatiotemporal-Modelling-of-Cardiomyocyte-Ageing-Dysfunction](https://github.com/datascintist-abusufian/Deep-Spatiotemporal-Modelling-of-Cardiomyocyte-Ageing-Dysfunction) and [medical-image-analysis](https://github.com/datascintist-abusufian/medical-image-analysis), the dashboard used to assess segmentation quality.

## Author

Md Abu Sufian, PhD researcher, School of Architecture, Computing and Engineering, University of East London. [GitHub profile](https://github.com/datascintist-abusufian) | [LinkedIn](https://www.linkedin.com/in/tacticalbusinessintelligence/)
