import torch
from torchvision import datasets, transforms
from .core import GMI_Dataset
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from pathlib import Path

class MNIST(GMI_Dataset):
    def __init__(self, 
                 train=True, 
                 transform=None, 
                 download=True, 
                 images_only=False,
                 root=None):
        
        # Set default root to GMI dataset path if not provided
        if root is None:
            # Use the current working directory (gmi_base) and create the proper path
            root = './gmi_data/datasets/MNIST'

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        self.dataset = datasets.MNIST(
                                root=root,
                                train=train,
                                transform=transform,
                                download=download)

        self.images_only = images_only

    def __getitem__(self, index):
        # Handle single index (most common case)
        if isinstance(index, int):
            result = self.dataset[index]
            print(f"[MNIST.__getitem__] index: {index}, result: {result}")
            data, target = result
            if self.images_only:
                # Ensure single sample has [channel, height, width] format
                if data.dim() == 2:
                    data = data.unsqueeze(0)  # Add channel dimension
                return data
            else:
                return data, target
        
        # Handle slice, convert it to a list of indices
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self.dataset))))
        elif isinstance(index, torch.Tensor):
            index = index.to(torch.int64).tolist()
            
        if isinstance(index, list):
            data_list = []
            target_list = []
            for i in index:
                data, target = self.dataset[i]
                data_list.append(data)
                target_list.append(target)
            data_batch = torch.stack(data_list)
            target_batch = torch.tensor(target_list)
            
            if self.images_only:
                return data_batch
            else:
                return data_batch, target_batch
    
    def __len__(self):
        return len(self.dataset)
    
    def visualize(self, save_path: Optional[str] = None, num_samples_per_class: int = 10) -> None:
        """
        Visualize MNIST dataset with a grid: each row is a digit (0-9), each column is a sample for that digit.
        
        Args:
            save_path: Optional path to save the visualization
            num_samples_per_class: Number of samples to show per digit (default 10)
        """
        # Collect the first num_samples_per_class samples for each digit
        digit_samples = {i: [] for i in range(10)}
        
        for i in range(len(self)):
            if all(len(samples) == num_samples_per_class for samples in digit_samples.values()):
                break
            data = self[i]
            if isinstance(data, tuple):
                sample, label = data
            else:
                sample = data
                label = None
            if label is not None and label in digit_samples and len(digit_samples[label]) < num_samples_per_class:
                digit_samples[label].append(sample)
        
        # Flatten to a list in row-major order (0s, then 1s, ..., 9s)
        samples = []
        labels = []
        for digit in range(10):
            samples.extend(digit_samples[digit])
            labels.extend([digit]*len(digit_samples[digit]))
        
        # Pad if not enough samples (shouldn't happen with MNIST)
        while len(samples) < 10 * num_samples_per_class:
            samples.append(torch.zeros_like(samples[0]))
            labels.append(-1)
        
        # Convert to tensor if needed
        if not isinstance(samples[0], torch.Tensor):
            samples = [torch.tensor(sample) for sample in samples]
        
        # Stack samples
        samples_tensor = torch.stack(samples)
        
        # Always maintain [batch, channel, height, width] convention
        # For visualization, we need to permute to [batch, height, width, channel] for matplotlib
        if samples_tensor.dim() == 4:  # [batch, channels, height, width]
            samples_tensor = samples_tensor.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        # Create grid with 10 rows and num_samples_per_class columns
        fig, axes = plt.subplots(10, num_samples_per_class, figsize=(2*num_samples_per_class, 20))
        axes = axes.flatten()
        
        for i, (sample, ax) in enumerate(zip(samples_tensor, axes)):
            # Handle different channel configurations for display
            if sample.shape[-1] == 1:  # Grayscale with channel dimension
                ax.imshow(sample.squeeze(-1).numpy(), cmap='gray')
            elif sample.shape[-1] == 3:  # RGB
                ax.imshow(sample.numpy())
            else:  # Other cases
                ax.imshow(sample.numpy())
            
            if labels[i] != -1:
                ax.set_title(f'Digit: {labels[i]}', fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f'MNIST: First {num_samples_per_class} samples of each digit (rows)', fontsize=16)
        plt.tight_layout()
        
        # Add extra headroom to prevent title overlap
        plt.subplots_adjust(top=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved MNIST visualization to {save_path}")
        
        plt.show()