import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from trainers.mnist import FlowMatchingNCSNpp
from configs.flow_matching_mnist import get_config

def generate_samples(flow_model, config, n_samples=64, n_steps=100, use_ema=True):
    """Generate samples using the trained model"""
    print(f'Generating {n_samples} samples with {n_steps} steps...')
    
    samples = flow_model.sample(
        n_samples=n_samples,
        n_steps=n_steps,
        use_ema=use_ema
    )
    
    return samples

def save_samples(samples, save_path, title="Generated Samples"):
    """Save samples to file"""
    samples = samples.cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range for proper visualization
    samples = (samples + 1.0) / 2.0
    samples = np.clip(samples, 0, 1)
    
    # Calculate grid size
    n_samples = len(samples)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_samples:
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide empty subplots
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Samples saved to: {save_path}')

def generate_with_trajectory(flow_model, config, n_samples=16, n_steps=100, save_every=10, use_ema=True):
    """Generate samples and save intermediate steps"""
    print(f'Generating {n_samples} samples with trajectory tracking...')
    
    final_samples, trajectory = flow_model.sample_with_trajectory(
        n_samples=n_samples,
        n_steps=n_steps,
        save_every=save_every,
        use_ema=use_ema
    )
    
    return final_samples, trajectory

def save_trajectory(trajectory, save_path, title="Generation Process"):
    """Save generation trajectory"""
    n_steps = len(trajectory)
    sample_idx = 0
    
    n_cols = min(11, n_steps)
    fig, axes = plt.subplots(1, n_cols, figsize=(2*n_cols, 2))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < n_steps:
            img = trajectory[i][sample_idx, 0].cpu().numpy()
            
            # Convert from [-1, 1] to [0, 1]
            img = (img + 1.0) / 2.0
            img = np.clip(img, 0, 1)
            
            ax.imshow(img, cmap='gray')
            
            step = i * (100 // (n_steps - 1)) if n_steps > 1 else 0
            ax.set_title(f't={step/100:.1f}', fontsize=10)
            ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Trajectory saved to: {save_path}')

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained Flow Matching model')
    parser.add_argument('--config', type=str, default='mnist', 
                       help='Config name')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--workdir', type=str, default='./workdir',
                       help='Working directory for outputs')
    parser.add_argument('--n_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--n_steps', type=int, default=100,
                       help='Number of sampling steps')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save trajectory every N steps')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA model for sampling')
    parser.add_argument('--trajectory', action='store_true',
                       help='Generate trajectory visualization')
    parser.add_argument('--output_name', type=str, default='generated_samples',
                       help='Output file name (without extension)')
    
    args = parser.parse_args()
    
    config = get_config()
    
    # Override config sampling parameters if provided
    if args.n_steps != 100:
        config.sampling.n_steps = args.n_steps
    
    # Create output directory
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'samples'), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading trained Flow Matching model...')
    flow_model = FlowMatchingNCSNpp(config, device=device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except RuntimeError as e:
        if "version_" in str(e) and "kMaxSupportedFileFormatVersion" in str(e):
            print("⚠️  PyTorch version compatibility issue detected.")
            print("💡 Try using a checkpoint from an earlier epoch or update PyTorch.")
            print(f"   Error: {e}")
            return
        else:
            raise e
    
    flow_model.model.load_state_dict(checkpoint['model'])
    flow_model.ema.load_state_dict(checkpoint['ema'])
    print('Checkpoint loaded successfully!')
    
    if args.trajectory:
        # Generate samples with trajectory
        final_samples, trajectory = generate_with_trajectory(
            flow_model, config, 
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            save_every=args.save_every,
            use_ema=args.use_ema
        )
        
        # Save final samples
        samples_path = os.path.join(args.workdir, 'samples', f'{args.output_name}.png')
        save_samples(final_samples, samples_path, f'Generated Samples (Epoch {checkpoint.get("epoch", "Unknown")})')
        
        # Save trajectory
        trajectory_path = os.path.join(args.workdir, 'samples', f'{args.output_name}_trajectory.png')
        save_trajectory(trajectory, trajectory_path, f'Generation Process (Epoch {checkpoint.get("epoch", "Unknown")})')
        
    else:
        # Generate samples only
        samples = generate_samples(
            flow_model, config,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            use_ema=args.use_ema
        )
        
        # Save samples
        samples_path = os.path.join(args.workdir, 'samples', f'{args.output_name}.png')
        save_samples(samples, samples_path, f'Generated Samples (Epoch {checkpoint.get("epoch", "Unknown")})')
    
    print(f'\n🎉 Sampling completed!')
    print(f'📁 Outputs saved in: {os.path.join(args.workdir, "samples")}')

if __name__ == '__main__':
    main()
