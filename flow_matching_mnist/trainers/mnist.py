import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入NCSN++
from models import ncsnpp
from models import utils as mutils
from models import ema as ema

class FlowMatchingNCSNpp:
    """Flow Matching model using NCSN++ architecture"""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = config
        
        self.model = mutils.create_model(config).to(device)

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
            x0: Source distribution samples (noise) - 连续值
            x1: Target distribution samples (data) - 连续值 [0,1]
            t: Time values in [0, 1]
        Returns:
            xt: Interpolated samples at time t
            ut: Target velocity field
        """
        t = t.view(-1, 1, 1, 1)
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0  # 目标velocity
        return xt, ut
    
    def train_step(self, x1):
        """Single training step for Flow Matching"""
        batch_size = x1.shape[0]
        
        x0 = torch.randn_like(x1, device=self.device)
        
        t = torch.rand(batch_size, device=self.device)
        
        xt, ut = self.optimal_transport_flow(x0, x1, t)
        

        vt = self.model(xt, t)
        
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
                
                # 数据归一化到[-1, 1]或[0, 1] (根据你的config)
                # NCSN++通常期望数据在[-1, 1]
                x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
                
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
                self.plot_samples(final_samples, epoch+1)
    
    def plot_samples(self, samples, epoch):
        """Plot generated samples"""
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
        plt.savefig(f'samples/epoch_{epoch:03d}.png', dpi=150)
        plt.close()