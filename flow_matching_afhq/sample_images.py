"""
Sample images from trained Flow Matching model
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging

from models import ddpm, ncsnv2, ncsnpp  # Import to register models
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sampling
import sde_lib

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("checkpoint", "checkpoint_10.pth", "Checkpoint filename (e.g., checkpoint_10.pth)")
flags.DEFINE_integer("num_samples", 64, "Number of samples to generate")
flags.DEFINE_string("output_dir", "./samples", "Output directory for samples")
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
    # Create output directory
    output_dir = os.path.join(FLAGS.workdir, FLAGS.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = FLAGS.config
    
    # Setup device
    device = config.device
    
    # Create model
    score_model = mutils.create_model(config)
    score_model = score_model.to(device)
    score_model.eval()
    
    # Setup EMA
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    
    # Load checkpoint
    checkpoint_path = os.path.join(FLAGS.workdir, "checkpoints", FLAGS.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model' in checkpoint:
        score_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        score_model.load_state_dict(checkpoint, strict=False)
    
    # Load EMA state
    if 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
    
    # Copy EMA parameters to model
    ema.copy_to(score_model.parameters())
    
    logging.info(f"Checkpoint loaded. Generating {FLAGS.num_samples} samples...")
    
    # Setup SDE
    if config.training.sde.lower() == 'rectified_flow':
        sde = sde_lib.RectifiedFlow(
            init_type=config.sampling.init_type,
            noise_scale=config.sampling.init_noise_scale,
            use_ode_sampler=config.sampling.use_ode_sampler,
            sigma_var=config.sampling.sigma_variance,
            ode_tol=config.sampling.ode_tol,
            sample_N=config.sampling.sample_N
        )
    else:
        raise NotImplementedError(f"SDE {config.training.sde} not implemented")
    
    # Setup sampling function
    sampling_shape = (FLAGS.num_samples, config.data.num_channels, 
                     config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                          inverse_scaler=lambda x: (x + 1.) / 2., 
                                          eps=1e-5)
    
    # Generate samples
    with torch.no_grad():
        samples, n = sampling_fn(score_model)
    
    # Convert to numpy and denormalize
    samples = samples.cpu().numpy()
    samples = np.clip(samples * 255, 0, 255).astype(np.uint8)
    
    # Save individual images
    for i in range(FLAGS.num_samples):
        img = samples[i].transpose(1, 2, 0)  # CHW -> HWC
        img_path = os.path.join(output_dir, f"sample_{i:04d}.png")
        Image.fromarray(img).save(img_path)
    
    logging.info(f"Saved {FLAGS.num_samples} individual images to {output_dir}")
    
    # Create a grid visualization
    n_rows = int(np.sqrt(FLAGS.num_samples))
    n_cols = int(np.ceil(FLAGS.num_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if FLAGS.num_samples > 1 else [axes]
    
    for i in range(FLAGS.num_samples):
        img = samples[i].transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(FLAGS.num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"grid_{FLAGS.checkpoint.replace('.pth', '')}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved grid visualization to {grid_path}")
    logging.info("Sampling complete!")


if __name__ == "__main__":
    app.run(main)


