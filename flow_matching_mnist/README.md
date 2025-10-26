# Flow Matching MNIST - Usage Guide

## Requirements

- **PyTorch 1.8+**: Required for loading trained models
- **Python 3.7+**
- **CUDA** (recommended for training)

⚠️ **PyTorch Version Compatibility**: If you encounter errors when loading checkpoints, you may need to update PyTorch:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

## Training

To train the Flow Matching model:

```bash
python main.py --workdir ./workdir --n_epochs 50
```

### Training Arguments:
- `--config`: Config name (default: 'mnist')
- `--workdir`: Working directory for outputs (default: './workdir')
- `--n_epochs`: Number of training epochs (overrides config)
- `--checkpoint`: Checkpoint path to resume training from

## Sampling

To generate samples from a trained model:

```bash
python sample.py --checkpoint ./workdir/checkpoints/final_model.pth --n_samples 64
```

### Sampling Arguments:
- `--checkpoint`: Path to trained model checkpoint (required)
- `--n_samples`: Number of samples to generate (default: 64)
- `--n_steps`: Number of sampling steps (default: 100)
- `--workdir`: Working directory for outputs (default: './workdir')
- `--output_name`: Output file name (default: 'generated_samples')
- `--trajectory`: Generate trajectory visualization
- `--save_every`: Save trajectory every N steps (default: 10)
- `--use_ema`: Use EMA model for sampling (default: True)

### Examples:

1. **Basic sampling:**
```bash
python sample.py --checkpoint ./workdir/checkpoints/final_model.pth
```

2. **High-quality sampling with more steps:**
```bash
python sample.py --checkpoint ./workdir/checkpoints/final_model.pth --n_steps 200 --n_samples 100
```

3. **Generate with trajectory visualization:**
```bash
python sample.py --checkpoint ./workdir/checkpoints/final_model.pth --trajectory --n_samples 16
```

4. **Custom output name:**
```bash
python sample.py --checkpoint ./workdir/checkpoints/final_model.pth --output_name my_samples
```

### Troubleshooting

If you encounter PyTorch version compatibility issues:

1. **Update PyTorch** (recommended):
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

2. **Test without trained model**:
```bash
python test_sample.py --n_samples 16
```

This will generate random samples to test the visualization pipeline.

## Output Files

### Training outputs:
- `workdir/loss_curves.png`: Training and validation loss curves
- `workdir/checkpoints/`: Model checkpoints
- `workdir/samples/`: Generated samples during training

### Sampling outputs:
- `workdir/samples/generated_samples.png`: Final generated samples
- `workdir/samples/generated_samples_trajectory.png`: Generation process (if --trajectory is used)
