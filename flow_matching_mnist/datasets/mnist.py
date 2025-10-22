import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np
def get_mnist_loaders(root, batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),                     # [0,1]
        transforms.Normalize((0.5,), (0.5,)),      # [-1,1]
    ])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    return train_loader


def load_mnist_data():
    """Load MNIST data using the same approach as the notebook"""
    # Use the existing MNIST data from the notebook
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0  
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)  
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_train_tensor = (x_train_tensor - 0.5) / 0.5
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    return train_loader
