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

Once your environment is set up and your checkpoints are in place, you can run the zero-shot forecasting script directly:
```bash
cd scripts
```

```bash
python tail_dart_zero_shot_singledataset.py
```