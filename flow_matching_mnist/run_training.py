#!/usr/bin/env python3
## Trial 1: Using simple network for flow matching on MNISt
import torch
import numpy as np
from trainers.mnist import FlowMatching
from datasets.mnist import load_mnist_data
torch.manual_seed(42)
np.random.seed(42)


def main():
    print("Loading MNIST data...")
    train_loader = load_mnist_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    flow_model = FlowMatching(device=device)
    
    print('Starting training...')
    flow_model.train(train_loader, n_epochs=50)
    
    torch.save(flow_model.model.state_dict(), 'flow_matching_mnist_unet.pth')
    print('Model saved as flow_matching_mnist.pth')

if __name__ == "__main__":
    main()
