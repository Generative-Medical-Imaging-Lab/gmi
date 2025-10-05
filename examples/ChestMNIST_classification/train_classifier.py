import torch
import torch.nn as nn
import sys
import os
import numpy as np
import torchvision.transforms as transforms
from transformers import AutoModel
from typing import Tuple, List

import gmi

# ChestMNIST class labels codebook
CHEST_CODEBOOK = {
    0: 'atelectasis',
    1: 'cardiomegaly', 
    2: 'effusion',
    3: 'infiltration',
    4: 'mass',
    5: 'nodule',
    6: 'pneumonia',
    7: 'pneumothorax',
    8: 'consolidation',
    9: 'edema',
    10: 'emphysema',
    11: 'fibrosis',
    12: 'pleural',
    13: 'hernia'
}

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies augmentations to training data"""
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        
        if self.transform:
            # Convert to PIL Image format for transforms
            if isinstance(image, torch.Tensor):
                # For grayscale images, ensure proper format [1, H, W] with values in [0, 1]
                image = torch.clamp(image, 0, 1)
                # Convert single channel to 3 channels for PIL compatibility
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = self.transform(image)
                # Convert back to single channel
                if image.shape[0] == 3:
                    image = image.mean(dim=0, keepdim=True)
        
        return image, label

class ResNet50ChestClassifier(torch.nn.Module):
    """ResNet50-based classifier for ChestMNIST multi-label classification"""
    
    def __init__(self, num_labels=14, prior_logits=None):
        super().__init__()
        
        # Convert 1-channel to 3-channel for ResNet50 compatibility
        self.input_conv = torch.nn.Conv2d(1, 3, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.input_conv.weight)
        torch.nn.init.zeros_(self.input_conv.bias)
        
        # Load ResNet50 from Hugging Face (keeps original 3-channel input)
        self.resnet = AutoModel.from_pretrained("microsoft/resnet-50")
        
        # Get ResNet50 feature dimension (should be 2048 for ResNet50)
        resnet_features = 2048
        
        # Multi-label classification head (14 binary outputs)
        self.classifier = torch.nn.Linear(resnet_features, num_labels)
        
        # Prior logits (if provided)
        if prior_logits is not None:
            if isinstance(prior_logits, np.ndarray):
                prior_logits = torch.from_numpy(prior_logits).float()
            self.register_buffer('prior_logits', prior_logits)
        else:
            self.register_buffer('prior_logits', torch.zeros(num_labels))
    
    def forward_likelihood_only(self, x):
        """Forward pass returning only likelihood logits (no prior)"""
        # Convert 1-channel to 3-channel
        x_rgb = self.input_conv(x)
        
        # Get features from ResNet50
        outputs = self.resnet(x_rgb)
        features = outputs.pooler_output  # [batch_size, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, 2048]
        
        # Multi-label logits
        logits = self.classifier(features)  # [batch_size, 14]
        
        return logits
    
    def forward(self, x):
        """Forward pass returning posterior logits (likelihood + prior)"""
        likelihood_logits = self.forward_likelihood_only(x)
        posterior_logits = likelihood_logits + self.prior_logits.unsqueeze(0)
        return posterior_logits
    
    def predict_probabilities(self, x):
        """Get predicted probabilities using sigmoid activation"""
        with torch.no_grad():
            logits = self.forward_likelihood_only(x)
            return torch.sigmoid(logits)

def labels_to_multihot(labels_batch, device=None):
    """Convert batch of labels to multi-hot tensor format"""
    if isinstance(labels_batch, list):
        # Convert list of numpy arrays/tensors to single tensor
        labels_list = []
        for label in labels_batch:
            if isinstance(label, torch.Tensor):
                labels_list.append(label.float())
            else:
                labels_list.append(torch.from_numpy(np.array(label)).float())
        multihot_tensor = torch.stack(labels_list)
    else:
        # Already a tensor batch
        multihot_tensor = labels_batch.float()
    
    if device is not None:
        multihot_tensor = multihot_tensor.to(device)
    
    return multihot_tensor

def compute_base_rates(dataloader, num_labels=14):
    """Compute base rates for each class from training data"""
    print("Computing base rates from training dataset...")
    label_counts = np.zeros(num_labels)
    total_samples = 0
    
    for batch_idx, (_, labels) in enumerate(dataloader):
        if batch_idx >= 100:  # Limit for efficiency
            break
        
        labels_tensor = labels_to_multihot(labels)
        label_counts += labels_tensor.sum(dim=0).numpy()
        total_samples += len(labels_tensor)
    
    base_rates = label_counts / total_samples
    prior_logits = np.log(base_rates / (1 - base_rates + 1e-10))
    
    print(f"Base rates computed from {total_samples} samples:")
    for i, rate in enumerate(base_rates):
        print(f"  {CHEST_CODEBOOK[i]}: {rate:.4f}")
    
    return base_rates, prior_logits

def create_data_transforms(config):
    """Create data augmentation transforms based on config"""
    augmentation_config = config.augmentation
    
    train_transforms = transforms.Compose([
        transforms.RandomAffine(
            degrees=augmentation_config['rotation_degrees'],
            translate=augmentation_config['translate'],
            scale=augmentation_config['scale'],
            shear=augmentation_config['shear']
        ),
        transforms.ColorJitter(
            brightness=augmentation_config['brightness'],
            contrast=augmentation_config['contrast']
        )
    ])
    
    return train_transforms

def create_datasets_and_loaders(config):
    """Create datasets and data loaders"""
    data_config = config.data
    
    # Create datasets
    dataset_train = gmi.datasets.MedMNIST(
        data_config['dataset_name'], 
        split='train',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True
    )
    
    dataset_val = gmi.datasets.MedMNIST(
        data_config['dataset_name'],
        split='val',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True
    )
    
    dataset_test = gmi.datasets.MedMNIST(
        data_config['dataset_name'],
        split='test',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True
    )
    
    # Create augmented training dataset
    train_transforms = create_data_transforms(config)
    dataset_train_augmented = AugmentedDataset(dataset_train, train_transforms)
    
    # Create data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train_augmented, 
        batch_size=data_config['batch_size'], 
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=data_config['batch_size'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=data_config['batch_size'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    return (dataset_train, dataset_val, dataset_test, 
            dataloader_train, dataloader_val, dataloader_test)

def create_model_and_optimizer(config, prior_logits):
    """Create model and optimizer"""
    model_config = config.model
    training_config = config.training
    
    # Convert prior logits to tensor
    prior_logits_tensor = torch.from_numpy(prior_logits).float()
    
    # Create model
    model = ResNet50ChestClassifier(
        num_labels=model_config['num_labels'], 
        prior_logits=prior_logits_tensor
    ).to(config.device)
    
    # Create conditional multilabel using GMI
    conditional_multilabel = gmi.random_variable.ConditionalMultilabelBinaryRandomVariable(
        logit_function=model,
        codebook=CHEST_CODEBOOK
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        conditional_multilabel.parameters(), 
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay'])
    )
    
    return model, conditional_multilabel, optimizer