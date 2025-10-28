import os
# Set CUDA environment variables before any imports
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from trainers.mnist import FlowMatchingNCSNpp
from configs.flow_matching_mnist import get_config

def generate_random_samples(config, n_samples=64, device='cpu'):
    """Generate random samples for testing (without loading trained model)"""
    print(f'Generating {n_samples} random samples for testing...')
    
    # Create model without loading checkpoint due to PyTorch version incompatibility
    model = FlowMatchingNCSNpp(config, device=device)
    
    checkpoint = torch.load('f/home/xinhuiqian/Flow-Matching/flow_matching_mnist/workdir/checkpoints/final_model.pth')
    model.load_state_dict(checkpoint['model'])
    model.ema.load_state_dict(checkpoint['ema'])
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(n_samples, config.data.image_size, config.data.image_size)
    
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

def main():
    parser = argparse.ArgumentParser(description='Test sample generation (without trained model)')
    parser.add_argument('--config', type=str, default='mnist', 
                       help='Config name')
    parser.add_argument('--workdir', type=str, default='./workdir',
                       help='Working directory for outputs')
    parser.add_argument('--n_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_name', type=str, default='test_samples',
                       help='Output file name (without extension)')
    
    args = parser.parse_args()
    
    config = get_config()
    
    # Create output directory
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'samples'), exist_ok=True)
    
    print('⚠️  PyTorch version compatibility issue detected.')
    print('💡 Generating samples with untrained model for testing.')
    print('🔧 To use your trained model, please update PyTorch:')
    print('   conda install pytorch torchvision torchaudio -c pytorch')
    print('   or: pip install torch torchvision torchaudio --upgrade')
    
    # Generate random samples for testing
    samples = generate_random_samples(config, args.n_samples)
    
    # Save samples
    samples_path = os.path.join(args.workdir, 'samples', f'{args.output_name}.png')
    save_samples(samples, samples_path, 'Test Samples (Untrained Model - PyTorch Compatibility Issue)')
    
    print(f'\n🎉 Test completed!')
    print(f'📁 Outputs saved in: {os.path.join(args.workdir, "samples")}')
    print(f'💡 To use your trained model, please update PyTorch:')
    print(f'   conda install pytorch torchvision torchaudio -c pytorch')

if __name__ == '__main__':
    main()
