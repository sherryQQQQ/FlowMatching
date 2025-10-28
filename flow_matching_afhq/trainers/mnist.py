import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import ncsnpp
from models import utils as mutils
from models import ema as ema

class FlowMatchingNCSNpp:
    """Flow Matching model using NCSN++ architecture"""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = config
            
        model = mutils.create_model(config)
        if hasattr(model, 'module'):
            self.model = model.module.to(device)
        else:
            self.model = model.to(device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay
        )
        
        self.ema = ema.ExponentialMovingAverage(
            self.model.parameters(), 
            decay=config.model.ema_rate
        )
        
    def optimal_transport_flow(self, x0, x1, t):
        """
        Compute the optimal transport flow (CFM with linear interpolation)
        Args:
            x0: Source distribution samples (noise) - [0,1]
            x1: Target distribution samples (data) - [0,1]
            t: Time values in [0, 1]
        Returns:
            xt: Interpolated samples at time t
            ut: Target velocity field
        """
        t = t.view(-1, 1, 1, 1)
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0  # target velocity
        return xt, ut
    
    def train_step(self, x1):
        """Single training step for Flow Matching"""
        batch_size = x1.shape[0]
        
        x0 = torch.randn_like(x1, device=self.device)
        
        # 确保时间参数形状正确 [batch_size, 1, 1, 1] 用于广播
        t = torch.rand(batch_size, device=self.device).view(-1, 1, 1, 1)
        
        xt, ut = self.optimal_transport_flow(x0, x1, t)
        
        # 模型期望时间参数为标量值，不是广播形状
        t_scalar = t.view(-1)  # 重新整形为 [batch_size]
        
        vt = self.model(xt, t_scalar)
        
        loss = nn.MSELoss()(vt, ut)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # gradient clip
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.optim.grad_clip
        )
        
        self.optimizer.step()
        
        # update EMA
        self.ema.update(self.model.parameters())
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model on validation set"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for x, _ in dataloader:
            x = x.to(self.device)
            x = x * 2.0 - 1.0  # Convert to [-1, 1]
            
            batch_size = x.shape[0]
            x0 = torch.randn_like(x, device=self.device)
            t = torch.rand(batch_size, device=self.device).view(-1, 1, 1, 1)
            
            xt, ut = self.optimal_transport_flow(x0, x, t)
            t_scalar = t.view(-1)
            vt = self.model(xt, t_scalar)
            
            loss = nn.MSELoss()(vt, ut)
            total_loss += loss.item()
            n_batches += 1
        
        self.model.train()
        return total_loss / n_batches if n_batches > 0 else 0
    
    @torch.no_grad()
    def sample(self, n_samples=16, n_steps=100, use_ema=True):
        """Generate samples using Euler method (ODE solver)"""
        
        # use EMA model for samplingq
        if use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        
        # initial noise
        x = torch.randn(n_samples, self.config.data.num_channels, 
                       self.config.data.image_size, 
                       self.config.data.image_size, 
                       device=self.device)
        
        dt = 1.0 / n_steps
        
        # Euler integration
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=self.device)
            v = self.model(x, t)
            x = x + v * dt
        
        self.model.train()
        
        # restore training model parameters
        if use_ema:
            self.ema.restore(self.model.parameters())
        
        return x
    
    @torch.no_grad()
    def sample_with_trajectory(self, n_samples=1, n_steps=100, save_every=10, use_ema=True):
        """Generate samples and save intermediate steps"""
        
        if use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        
        x = torch.randn(n_samples, self.config.data.num_channels,
                       self.config.data.image_size,
                       self.config.data.image_size,
                       device=self.device)
        
        dt = 1.0 / n_steps
        trajectory = [x.clone()]
        
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=self.device)
            v = self.model(x, t)
            x = x + v * dt
            
            if (i + 1) % save_every == 0 or i == n_steps - 1:
                trajectory.append(x.clone())
        
        self.model.train()
        
        if use_ema:
            self.ema.restore(self.model.parameters())
        
        return x, trajectory
    
    def train(self, dataloader, n_epochs=50):
        """Train the flow matching model"""
        self.model.train()
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for x, _ in pbar:
                x = x.to(self.device)
                # normalize to [-1,1]
                x = x * 2.0 - 1.0
                
                loss = self.train_step(x)
                epoch_loss += loss
                n_batches += 1
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = epoch_loss / n_batches
            print(f'Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}')
            
            # sample periodically
            if (epoch + 1) % 5 == 0:
                final_samples, trajectory = self.sample_with_trajectory(
                    n_samples=16,
                    n_steps=100,
                    save_every=10
                )
                self.plot_samples(final_samples, epoch+1, './samples')
    
    def plot_samples(self, samples, epoch, save_dir='./samples'):
        """Plot generated samples"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        samples = samples.cpu().numpy()
        
        # de-normalize: [-1,1] -> [0,1]
        samples = (samples + 1.0) / 2.0
        samples = np.clip(samples, 0, 1)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                ax.imshow(samples[i, 0], cmap='gray')
                ax.axis('off')
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'✅ Saved samples: {save_path}')