# TailDart

This repository contains code for zero-shot time series forecasting using PyTorch Lightning and GluonTS.

## Installation

This project uses `mamba` for environment and package management. Ensure you have `mamba` (or `conda`) installed, then run the following commands:

1. Create and configure the environment with PyTorch and CUDA support:

   ```bash
   mamba create -n lightning_gpu_py3p12 python=3.12 lightning pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
   ```

2. Activate the environment:

   ```bash
   mamba activate lightning_gpu_py3p12
   ```

3. Install additional dependencies:

   ```bash
   mamba install pandas matplotlib plotly gluonts=0.14.4 -c conda-forge -y
   ```

## Running the Script

From the repository root, run the zero-shot forecasting script:
```bash
python scripts/taildart_zero_shot_singledataset.py
```


references to the former `taildart` package must be migrated before loading with
this version.

Before running, set `DEVICE` in the script for your machine (`cuda`, `mps`, or
`cpu`). The script defaults to CUDA and demonstrates forecasting on a synthetic
time series; replace that input with your own data as needed.
