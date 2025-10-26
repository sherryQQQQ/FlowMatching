import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ml_collections

from trainers.mnist import FlowMatchingNCSNpp
from configs.flow_matching_mnist import get_config

def get_dataloader(config, split='train'):
    transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=(split == 'train'),
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching model with NCSN++')
    parser.add_argument('--config', type=str, default='mnist', 
                       help='Config name')
    parser.add_argument('--workdir', type=str, default='./workdir',
                       help='Working directory')
    parser.add_argument('--n_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path to resume from')
    args = parser.parse_args()
    
    config = get_config()
    
    if args.n_epochs is not None:
        config.training.n_epochs = args.n_epochs
    
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.workdir, 'checkpoints'), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Initializing Flow Matching model with NCSN++...')
    flow_model = FlowMatchingNCSNpp(config, device=device)
    
    if args.checkpoint is not None:
        print(f'Loading checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        flow_model.model.load_state_dict(checkpoint['model'])
        flow_model.ema.load_state_dict(checkpoint['ema'])
        if 'optimizer' in checkpoint:
            flow_model.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint loaded successfully!')
    
    print('Loading MNIST dataset...')
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    print(f'Training dataset size: {len(train_loader.dataset)}')
    print(f'Validation dataset size: {len(val_loader.dataset)}')
    print(f'Batch size: {config.training.batch_size}')
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')
    
    print(f'\nStarting training for {config.training.n_epochs} epochs...')
    train_with_checkpoints(flow_model, train_loader, val_loader, config, args.workdir)

def train_with_checkpoints(flow_model, train_loader, val_loader, config, workdir):
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # Initialize loss tracking
    train_losses = []
    val_losses = []
    epochs = []
    
    flow_model.model.train()
    
    for epoch in range(config.training.n_epochs):
        epoch_train_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.n_epochs}')
        for x, _ in pbar:
            x = x.to(flow_model.device)
            x = x * 2.0 - 1.0
            
            loss = flow_model.train_step(x)
            epoch_train_loss += loss
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = epoch_train_loss / n_batches
        
        # Validation phase
        print('Validating...')
        val_loss = flow_model.validate(val_loader)
        
        # Record losses
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch + 1)
        
        print(f'Epoch {epoch+1}/{config.training.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Plot and save loss curves
        plot_loss_curves(epochs, train_losses, val_losses, workdir)
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                workdir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint(flow_model, epoch, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
        
        if (epoch + 1) % 5 == 0:
            print('Generating samples...')
            final_samples, trajectory = flow_model.sample_with_trajectory(
                n_samples=16,
                n_steps=config.sampling.n_steps,
                save_every=10,
                use_ema=True
            )
            
            save_dir = os.path.join(workdir, 'samples')
            save_samples(final_samples, epoch+1, save_dir)
            plot_generation_process(trajectory, epoch+1, save_dir)
    
    final_path = os.path.join(workdir, 'checkpoints', 'final_model.pth')
    save_checkpoint(flow_model, config.training.n_epochs, final_path)
    print(f'Final model saved to {final_path}')

def save_checkpoint(flow_model, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model': flow_model.model.state_dict(),
        'ema': flow_model.ema.state_dict(),
        'optimizer': flow_model.optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def save_samples(samples, epoch, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    
    samples = samples.cpu().numpy()
    
    # [-1,1] -> [0,1]
    samples = (samples + 1.0) / 2.0
    samples = np.clip(samples, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
    
    plt.suptitle(f'Generated Samples - Epoch {epoch}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Saved samples: {save_path}')

def plot_generation_process(trajectory, epoch, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_steps = len(trajectory)
    sample_idx = 0
    
    n_cols = min(11, n_steps)
    fig, axes = plt.subplots(1, n_cols, figsize=(2*n_cols, 2))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < n_steps:
            img = trajectory[i][sample_idx, 0].cpu().numpy()
            
            img = (img + 1.0) / 2.0
            img = np.clip(img, 0, 1)
            
            ax.imshow(img, cmap='gray')
            
            step = i * (100 // (n_steps - 1)) if n_steps > 1 else 0
            ax.set_title(f't={step/100:.1f}', fontsize=10)
            ax.axis('off')
    
    plt.suptitle(f'Generation Process - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'process_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Saved generation process: {save_path}')

def plot_loss_curves(epochs, train_losses, val_losses, workdir):
    """Plot training and validation loss curves"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    
    # Plot validation loss
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)
    
    # Add some padding
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(workdir, 'loss_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Loss curves saved to: {save_path}')

if __name__ == '__main__':
    main()