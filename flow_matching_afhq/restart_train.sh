#!/bin/bash
# Restart training script

# Activate conda environment
source /home/xinhuiqian/.conda/envs/pnp-dm/etc/conda/activate.d/env_vars.sh 2>/dev/null || true

# Navigate to project directory
cd /home/xinhuiqian/Flow-Matching/flow_matching_mnist

# Run training
/home/xinhuiqian/.conda/envs/pnp-dm/bin/python main_1.py \
  --config=/home/xinhuiqian/Flow-Matching/flow_matching_mnist/configs/rectified_flow/afhq_cat_pytorch_rf_gaussian.py \
  --mode=train \
  --workdir=./workdir

