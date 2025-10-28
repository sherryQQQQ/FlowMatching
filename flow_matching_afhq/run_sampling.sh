#!/bin/bash
# Sample images from trained model

cd /home/xinhuiqian/Flow-Matching/flow_matching_mnist

# Default values
CHECKPOINT=${1:-checkpoint_10.pth}
NUM_SAMPLES=${2:-64}

echo "Sampling with checkpoint: $CHECKPOINT"
echo "Number of samples: $NUM_SAMPLES"

/home/xinhuiqian/.conda/envs/pnp-dm/bin/python sample_images.py \
  --config=/home/xinhuiqian/Flow-Matching/flow_matching_mnist/configs/rectified_flow/afhq_cat_pytorch_rf_gaussian.py \
  --workdir=./workdir \
  --checkpoint=$CHECKPOINT \
  --num_samples=$NUM_SAMPLES \
  --output_dir=samples

echo "Sampling complete! Check ./workdir/samples/ for results."


