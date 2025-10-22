
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom SiLU activation for older PyTorch versions
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings similar to diffusion models"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)).to(device) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalUNet(nn.Module):
    """Simple U-Net architecture with time conditioning for flow matching"""
    def __init__(self, in_channels=1, out_channels=1, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2)
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            SiLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            SiLU()
        )
        
        # Time conditioning projections
        self.time_proj1 = nn.Linear(time_dim * 2, 64)
        self.time_proj2 = nn.Linear(time_dim * 2, 128)
        self.time_proj3 = nn.Linear(time_dim * 2, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            SiLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            SiLU()
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            SiLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            SiLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            SiLU(),
            nn.Conv2d(64, out_channels, 1)
        )
        
    def forward(self, x, t):
        # Get time embeddings
        t_emb = self.time_mlp(t)
        
        # Encoder with time conditioning
        e1 = self.enc1(x)
        e1 = e1 + self.time_proj1(t_emb)[:, :, None, None]
        
        e2 = self.enc2(e1)
        e2 = e2 + self.time_proj2(t_emb)[:, :, None, None]
        
        e3 = self.enc3(e2)
        e3 = e3 + self.time_proj3(t_emb)[:, :, None, None]
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        out = self.dec1(d2)
        
        return out