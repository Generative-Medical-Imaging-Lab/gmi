import torch
from torchvision import datasets, transforms
import medmnist
from .core import GMI_Dataset
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from PIL import Image
import random

class MedMNIST(GMI_Dataset):
    def __init__(self,
                 dataset_name,
                 split,
                 transform=None,
                 target_transform=None,
                 download=True,
                 as_rgb=False,
                 root=None,
                 size=None,
                 mmap_mode=None,
                 images_only=False):
        # Map dataset_name to medmnist class
        medmnist_map = {
            'PathMNIST': medmnist.PathMNIST,
            'ChestMNIST': medmnist.ChestMNIST,
            'DermaMNIST': medmnist.DermaMNIST,
            'OCTMNIST': medmnist.OCTMNIST,
            'PneumoniaMNIST': medmnist.PneumoniaMNIST,
            'RetinaMNIST': medmnist.RetinaMNIST,
            'BreastMNIST': medmnist.BreastMNIST,
            'BloodMNIST': medmnist.BloodMNIST,
            'TissueMNIST': medmnist.TissueMNIST,
            'OrganAMNIST': medmnist.OrganAMNIST,
            'OrganCMNIST': medmnist.OrganCMNIST,
            'OrganSMNIST': medmnist.OrganSMNIST,
            'OrganMNIST3D': medmnist.OrganMNIST3D,
            'NoduleMNIST3D': medmnist.NoduleMNIST3D,
            'AdrenalMNIST3D': medmnist.AdrenalMNIST3D,
            'FractureMNIST3D': medmnist.FractureMNIST3D,
            'VesselMNIST3D': medmnist.VesselMNIST3D,
            'SynapseMNIST3D': medmnist.SynapseMNIST3D
        }
        if dataset_name not in medmnist_map:
            raise ValueError(f'MedMNIST dataset name not recognized: {dataset_name}')
        medmnist_dataset = medmnist_map[dataset_name]

        self.medmnist_dataset = medmnist_dataset(
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode
        )
        
        # Store metadata
        self.dataset_name = dataset_name
        self.split = split
        self.images_only = images_only
        
        # Detect shapes and number of classes
        self._detect_shapes_and_classes()
    
    def _detect_shapes_and_classes(self):
        """Detect image shape, label shape, and number of unique classes."""
        # Get a sample to detect shapes
        sample_data, sample_label = self.medmnist_dataset[0]
        
        # Convert PIL image to tensor and detect image shape
        if isinstance(sample_data, Image.Image):
            # Convert PIL to tensor
            to_tensor = transforms.ToTensor()
            sample_tensor = to_tensor(sample_data)
            self.image_shape = sample_tensor.shape
        else:
            self.image_shape = sample_data.shape if hasattr(sample_data, 'shape') else None
        
        # Detect label shape
        if hasattr(sample_label, 'shape'):
            self.label_shape = sample_label.shape
        else:
            self.label_shape = (1,) if isinstance(sample_label, (int, float)) else (len(sample_label),)
        
        # Handle multi-label vs categorical
        if isinstance(sample_label, np.ndarray) and len(sample_label.shape) == 1 and sample_label.shape[0] > 1:
            # Multi-label dataset
            self.num_classes = sample_label.shape[0]
            self.is_multi_label = True
        else:
            # Categorical dataset
            sample_size = min(1000, len(self.medmnist_dataset))
            labels = []
            
            # Sample more broadly across the dataset instead of just the first samples
            if len(self.medmnist_dataset) > sample_size:
                # Use random sampling to get better class distribution
                indices = random.sample(range(len(self.medmnist_dataset)), sample_size)
                for idx in indices:
                    _, label = self.medmnist_dataset[idx]
                    labels.append(label)
            else:
                # If dataset is small, use all samples
                for i in range(len(self.medmnist_dataset)):
                    _, label = self.medmnist_dataset[i]
                    labels.append(label)
            
            if isinstance(labels[0], np.ndarray):
                # Extract the actual class values from numpy arrays
                if labels[0].shape == (1,):
                    # Single class per sample: [class_idx]
                    labels_array = np.array([int(l[0]) for l in labels])
                else:
                    # Multiple classes per sample: [class1, class2, ...]
                    labels_array = np.array([int(np.argmax(l)) for l in labels])
            else:
                labels_array = np.array(labels)
            
            unique_labels = np.unique(labels_array)
            self.num_classes = len(unique_labels)
            self.unique_labels = unique_labels
            self.is_multi_label = False
        
        print(f"📊 {self.dataset_name} ({self.split}): {len(self.medmnist_dataset)} samples, "
              f"image shape: {self.image_shape}, label shape: {self.label_shape}, "
              f"classes: {self.num_classes}")

    def __getitem__(self, index):
        data, target = self.medmnist_dataset[index]
        
        # Convert PIL image to tensor if needed
        if isinstance(data, Image.Image):
            to_tensor = transforms.ToTensor()
            data = to_tensor(data)
        
        if self.images_only:
            return data
        else:
            return data, target
    
    def __len__(self):
        return len(self.medmnist_dataset)
    
    def visualize(self, save_path: Optional[str] = None, num_samples_per_class: int = 10) -> None:
        """
        Visualize MedMNIST dataset with samples for each class/condition.
        
        Args:
            save_path: Optional path to save the visualization
            num_samples_per_class: Number of samples to show per class (default 10)
        """
        print(f"🎨 Visualizing {self.dataset_name} with {num_samples_per_class} samples per class...")
        
        if self.is_multi_label:
            # For multi-label datasets, show one-hot samples for each condition
            self._visualize_multi_label(save_path, num_samples_per_class)
        else:
            # For categorical datasets, show samples for each class
            self._visualize_categorical(save_path, num_samples_per_class)
    
    def _visualize_multi_label(self, save_path: Optional[str] = None, num_samples_per_class: int = 10) -> None:
        """Visualize multi-label dataset showing one-hot samples for each condition."""
        print(f"🔍 Filtering for one-hot samples (exactly one condition active) and healthy samples (all zeros)...")
        
        # Collect one-hot samples for each condition + healthy samples
        # Condition 0: healthy (all zeros), Conditions 1-N: individual conditions
        condition_samples = {i: [] for i in range(self.num_classes + 1)}  # +1 for healthy
        
        # Sample through dataset to find one-hot cases and healthy cases
        max_iterations = len(self) * 3  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            # Check if we have enough samples for all conditions
            if all(len(samples) >= num_samples_per_class for samples in condition_samples.values()):
                break
            
            # Get a random sample
            idx = np.random.randint(0, len(self))
            try:
                data, label = self[idx]
                
                if isinstance(label, np.ndarray):
                    active_labels = np.where(label == 1)[0]
                    
                    if len(active_labels) == 0:  # Healthy case (all zeros)
                        if len(condition_samples[0]) < num_samples_per_class:
                            condition_samples[0].append(data)
                    
                    elif len(active_labels) == 1:  # Exactly one condition active
                        condition_idx = active_labels[0] + 1  # Shift by 1 to make room for healthy
                        if len(condition_samples[condition_idx]) < num_samples_per_class:
                            condition_samples[condition_idx].append(data)
                
            except Exception as e:
                print(f"Warning: Could not load sample {idx}: {e}")
            
            iteration += 1
        
        # Flatten samples for visualization
        samples = []
        labels = []
        condition_names = []
        
        # Add healthy samples first
        samples.extend(condition_samples[0][:num_samples_per_class])
        labels.extend([0] * len(condition_samples[0][:num_samples_per_class]))
        condition_names.extend(["Healthy"] * len(condition_samples[0][:num_samples_per_class]))
        
        # Add individual condition samples
        for condition_idx in range(1, self.num_classes + 1):
            samples.extend(condition_samples[condition_idx][:num_samples_per_class])
            labels.extend([condition_idx] * len(condition_samples[condition_idx][:num_samples_per_class]))
            condition_names.extend([f"Condition {condition_idx-1}"] * len(condition_samples[condition_idx][:num_samples_per_class]))
        
        if not samples:
            raise RuntimeError("No samples could be found for visualization")
        
        print(f"📊 Found samples for {len([s for s in condition_samples.values() if s])} out of {self.num_classes + 1} conditions (including healthy)")
        
        # Create visualization with updated class count
        self._create_visualization_grid(samples, labels, condition_names, save_path, num_samples_per_class, total_classes=self.num_classes + 1)
    
    def _visualize_categorical(self, save_path: Optional[str] = None, num_samples_per_class: int = 10) -> None:
        """Visualize categorical dataset showing samples for each class."""
        # Collect samples for each class
        class_samples = {i: [] for i in range(self.num_classes)}
        
        # Collect samples until we have enough for each class
        max_iterations = len(self) * 2  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            # Check if we have enough samples for all classes
            if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
                break
            
            # Get a random sample
            idx = np.random.randint(0, len(self))
            try:
                data, label = self[idx]
                
                # Find which class this belongs to
                if isinstance(label, np.ndarray):
                    if label.shape == (1,):
                        class_idx = int(label[0])  # Extract the actual class value
                    else:
                        class_idx = int(np.argmax(label))
                else:
                    class_idx = int(label)
                
                if class_idx in class_samples and len(class_samples[class_idx]) < num_samples_per_class:
                    class_samples[class_idx].append(data)
                        
            except Exception as e:
                print(f"Warning: Could not load sample {idx}: {e}")
            
            iteration += 1
        
        # Flatten samples for visualization
        samples = []
        labels = []
        class_names = []
        
        for class_idx, samples_list in class_samples.items():
            samples.extend(samples_list[:num_samples_per_class])
            labels.extend([class_idx] * len(samples_list[:num_samples_per_class]))
            class_names.extend([f"Class {class_idx}"] * len(samples_list[:num_samples_per_class]))
        
        if not samples:
            raise RuntimeError("No samples could be loaded for visualization")
        
        # Create visualization
        self._create_visualization_grid(samples, labels, class_names, save_path, num_samples_per_class)
    
    def _create_visualization_grid(self, samples, labels, names, save_path, num_samples_per_class, total_classes=None):
        """Create the visualization grid."""
        # Convert to tensor if needed
        if not isinstance(samples[0], torch.Tensor):
            samples = [torch.tensor(sample) for sample in samples]
        
        # Stack samples
        samples_tensor = torch.stack(samples)
        
        # Always maintain [batch, channel, height, width] convention
        # For visualization, we need to permute to [batch, height, width, channel] for matplotlib
        if samples_tensor.dim() == 4:  # [batch, channels, height, width]
            samples_tensor = samples_tensor.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        # Create visualization grid
        num_classes = total_classes if total_classes else len(set(labels))
        fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(2*num_samples_per_class, 2*num_classes))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        elif num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (sample, ax) in enumerate(zip(samples_tensor, axes.flatten())):
            # Handle different channel configurations for display
            if sample.shape[-1] == 1:  # Grayscale with channel dimension
                ax.imshow(sample.squeeze(-1).numpy(), cmap='gray')
            elif sample.shape[-1] == 3:  # RGB
                ax.imshow(sample.numpy())
            else:  # Other cases
                ax.imshow(sample.numpy())
            
            if i < len(names):
                ax.set_title(f'{names[i]}', fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(samples), len(axes.flatten())):
            axes.flatten()[i].axis('off')
        
        title_suffix = "conditions" if self.is_multi_label else "classes"
        plt.suptitle(f'{self.dataset_name} ({self.split}) - {num_samples_per_class} samples per {title_suffix} ({num_classes} {title_suffix})', fontsize=16)
        plt.tight_layout()
        
        # Add extra headroom to prevent title overlap
        plt.subplots_adjust(top=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {self.dataset_name} visualization to {save_path}")
        
        plt.show()