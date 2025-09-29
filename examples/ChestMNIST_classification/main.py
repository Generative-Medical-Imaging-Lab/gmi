from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from transformers import AutoModel, AutoConfig

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

medmnist_dataset_root = '/workspace/gmi/gmi_data/datasets/medmnist_dataset_root/'
if not os.path.exists(medmnist_dataset_root):
    os.makedirs(medmnist_dataset_root)

medmnist_name = 'ChestMNIST'
batch_size = 128

medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))

print("=== ChestMNIST Dataset Investigation ===")

# Download and investigate the ChestMNIST dataset
print("Downloading ChestMNIST dataset...")
dataset_train = gmi.datasets.MedMNIST(medmnist_name, 
                                      split='train',
                                      root=medmnist_dataset_root,
                                      size=64, 
                                      download=True)

dataset_val = gmi.datasets.MedMNIST(medmnist_name,
                                    split='val',
                                    root=medmnist_dataset_root,
                                    size=64, 
                                    download=True)

dataset_test = gmi.datasets.MedMNIST(medmnist_name,
                                     split='test',
                                     root=medmnist_dataset_root,
                                     size=64, 
                                     download=True)

print(f"Dataset sizes:")
print(f"  Train: {len(dataset_train)}")
print(f"  Val: {len(dataset_val)}")  
print(f"  Test: {len(dataset_test)}")

# Get sample data to understand structure
sample_img, sample_label = dataset_train[0]
print(f"\nSample data structure:")
print(f"  Image shape: {sample_img.shape}")
print(f"  Image dtype: {sample_img.dtype}")
print(f"  Image min/max: {sample_img.min():.3f} / {sample_img.max():.3f}")
print(f"  Label shape: {sample_label.shape}")
print(f"  Label dtype: {sample_label.dtype}")
print(f"  Sample label: {sample_label}")

# ChestMNIST class labels codebook (14 binary labels for multi-label classification)
chest_codebook = {
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

print(f"\nChestMNIST has {len(chest_codebook)} labels (multi-label binary classification):")
for idx, label_name in chest_codebook.items():
    print(f"  {idx}: {label_name}")

# Analyze label distribution
print(f"\nAnalyzing first 20 samples:")
label_counts = np.zeros(14)
for i in range(min(20, len(dataset_train))):
    img, label = dataset_train[i]
    active_labels = [j for j, val in enumerate(label) if val == 1]
    label_counts += label
    print(f"Sample {i:2d}: {label} -> active: {active_labels} (total: {label.sum()})")

print(f"\nLabel frequency in first 20 samples:")
for idx, count in enumerate(label_counts):
    print(f"  {chest_codebook[idx]}: {int(count)}")

# Check for samples with no labels and multiple labels
print(f"\nLabel statistics (first 100 samples):")
no_labels_count = 0
multi_labels_count = 0
single_label_count = 0

for i in range(min(100, len(dataset_train))):
    img, label = dataset_train[i]
    label_sum = label.sum()
    if label_sum == 0:
        no_labels_count += 1
    elif label_sum == 1:
        single_label_count += 1
    else:
        multi_labels_count += 1

print(f"  No labels (all zeros): {no_labels_count}")
print(f"  Single label: {single_label_count}")
print(f"  Multiple labels: {multi_labels_count}")

print(f"\nThis confirms ChestMNIST is a multi-label binary classification task")
print(f"Each sample can have 0, 1, or multiple conditions simultaneously")

# Function to convert numpy labels to tensor multi-hot
def labels_to_multihot(labels, device=None):
    """Convert multi-label array to tensor multi-hot encoding."""
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.float32)
    elif not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    # Ensure labels are on the correct device
    if device is not None:
        labels = labels.to(device)
    
    return labels

# Define augmentation transforms for training data (grayscale images)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # Random vertical flip  
    transforms.RandomRotation(degrees=90, expand=False),  # Random rotation by multiples of 90 degrees
    transforms.RandomChoice([  # Additional rotations to cover all 8 orientations
        transforms.RandomRotation(degrees=(0, 0)),      # 0 degrees
        transforms.RandomRotation(degrees=(90, 90)),    # 90 degrees
        transforms.RandomRotation(degrees=(180, 180)),  # 180 degrees
        transforms.RandomRotation(degrees=(270, 270)),  # 270 degrees
    ]),
    transforms.ToTensor(),  # Convert back to tensor
])

# Create augmented training dataset
class AugmentedDataset(torch.utils.data.Dataset):
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

# Create augmented training dataset (only for training, not val/test)
dataset_train_augmented = AugmentedDataset(dataset_train, train_transforms)

# Create data loaders
dataloader_train = torch.utils.data.DataLoader(dataset_train_augmented, 
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                num_workers=4)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                num_workers=4)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                num_workers=4)

# Compute base rates for each class from training dataloader (more efficient)
print("Computing base rates from training dataset using dataloader...")
label_counts = np.zeros(14)
total_samples = 0

# Use a temporary dataloader for base rate computation
temp_dataloader = torch.utils.data.DataLoader(dataset_train, 
                                              batch_size=batch_size,  # Larger batch for efficiency
                                              shuffle=False,  # No need to shuffle for counting
                                              num_workers=4)

from tqdm import tqdm
for batch_images, batch_labels in tqdm(temp_dataloader, desc="Counting labels"):
    # Convert batch labels to numpy
    if isinstance(batch_labels, torch.Tensor):
        batch_labels_np = batch_labels.numpy()
    else:
        batch_labels_np = np.array(batch_labels)
    
    # Accumulate counts
    label_counts += batch_labels_np.sum(axis=0)
    total_samples += len(batch_labels_np)

base_rates = label_counts / total_samples
# Handle edge case where base rate is 0 or 1 to avoid infinite logits
base_rates = np.clip(base_rates, 1e-7, 1-1e-7)
prior_logits = np.log(base_rates / (1 - base_rates))  # Convert to logits

print("Base rates for each condition:")
for i, rate in enumerate(base_rates):
    print(f"  {chest_codebook[i]}: {rate:.4f} ({int(label_counts[i])}/{total_samples})")

# Convert to tensor for use in model
prior_logits_tensor = torch.tensor(prior_logits, dtype=torch.float32, device=device)

# # Define CNN classifier for 14 binary labels (1-channel input)  [COMMENTED OUT]
# class ChestCNNClassifier(torch.nn.Module):
#     def __init__(self, num_labels=14, base_channels=64):
#         super().__init__()
#         self.features = torch.nn.Sequential(
#             # First block
#             torch.nn.Conv2d(1, base_channels, kernel_size=3, padding=1),  # 1 channel input
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.MaxPool2d(2),  # 64x64 -> 32x32
#             
#             # Second block  
#             torch.nn.Conv2d(base_channels, 2*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(2*base_channels, 2*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.MaxPool2d(2),  # 32x32 -> 16x16
#             
#             # Third block
#             torch.nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(4*base_channels, 4*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.MaxPool2d(2),  # 16x16 -> 8x8
#             
#             # Fourth block
#             torch.nn.Conv2d(4*base_channels, 8*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(8*base_channels, 8*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.MaxPool2d(2),  # 8x8 -> 4x4
#             
#             # Fifth block
#             torch.nn.Conv2d(8*base_channels, 16*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(16*base_channels, 16*base_channels, kernel_size=3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.AdaptiveAvgPool2d(1)  # Global average pooling -> 1x1
#         )
#         
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             # torch.nn.Dropout(0.5),
#             torch.nn.Linear(16*base_channels, 8*base_channels),
#             torch.nn.SiLU(),
#             # torch.nn.Dropout(0.5),
#             torch.nn.Linear(8*base_channels, num_labels)  # 14 binary outputs
#         )
#     
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
# 
# cnn_classifier = ChestCNNClassifier(num_labels=14).to(device)

# Define ResNet50-based classifier for 14 binary labels (1-channel input)
class ResNet50ChestClassifier(torch.nn.Module):
    def __init__(self, num_labels=14, prior_logits=None):
        super().__init__()
        
        # Convert 1-channel to 3-channel for ResNet50 compatibility
        self.channel_adapter = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialize the adapter to replicate the single channel to all 3 channels
        with torch.no_grad():
            self.channel_adapter.weight.fill_(1.0/3.0)  # Average the single channel across RGB
        
        # Load ResNet50 from Hugging Face (keeps original 3-channel input)
        self.resnet = AutoModel.from_pretrained("microsoft/resnet-50")
        
        # Store prior logits for Bayesian classification
        if prior_logits is not None:
            self.register_buffer('prior_logits', prior_logits)
        else:
            self.register_buffer('prior_logits', torch.zeros(num_labels))
        
        # We'll determine the feature size dynamically
        self.feature_size = None
        self.classifier = None
    
    def _build_classifier(self, feature_size, num_labels):
        """Build classifier once we know the feature size"""
        return torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_labels)  # 14 binary outputs
        )
    
    def forward(self, x):
        # Convert 1-channel to 3-channel
        x_rgb = self.channel_adapter(x)  # [batch_size, 1, H, W] -> [batch_size, 3, H, W]
        
        # Get features from ResNet50
        outputs = self.resnet(x_rgb)
        
        # Try pooler_output first, if not available use last_hidden_state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            # Use last hidden state and apply global average pooling
            last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            features = last_hidden.mean(dim=1)  # Global average pooling
        
        # Flatten features in case they have extra dimensions
        features = features.view(features.size(0), -1)
        
        # Build classifier on first forward pass
        if self.classifier is None:
            self.feature_size = features.size(1)
            self.classifier = self._build_classifier(self.feature_size, 14).to(features.device)
            print(f"Built classifier with input size: {self.feature_size}")
        
        # Pass through classifier to get likelihood logits
        likelihood_logits = self.classifier(features)
        
        # Add prior logits to get posterior logits (MAP)
        posterior_logits = likelihood_logits + self.prior_logits.unsqueeze(0)
        
        return posterior_logits
    
    def forward_likelihood_only(self, x):
        """Forward pass returning only likelihood logits (for display)"""
        # Convert 1-channel to 3-channel
        x_rgb = self.channel_adapter(x)
        
        # Get features from ResNet50
        outputs = self.resnet(x_rgb)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            features = last_hidden.mean(dim=1)
        
        features = features.view(features.size(0), -1)
        
        # Build classifier on first forward pass if needed
        if self.classifier is None:
            self.feature_size = features.size(1)
            self.classifier = self._build_classifier(self.feature_size, 14).to(features.device)
            print(f"Built classifier with input size: {self.feature_size}")
        
        # Return only likelihood logits (no prior)
        likelihood_logits = self.classifier(features)
        return likelihood_logits

# Create ResNet50 classifier with prior logits
print("Loading ResNet50 model from Hugging Face...")
cnn_classifier = ResNet50ChestClassifier(num_labels=14, prior_logits=prior_logits_tensor).to(device)
print(f"ResNet50 model loaded with {sum(p.numel() for p in cnn_classifier.parameters()):,} parameters")

# Create conditional multi-label binary random variable
conditional_multilabel = gmi.random_variable.ConditionalMultilabelBinaryRandomVariable(
    logit_function=cnn_classifier,
    codebook=chest_codebook
).to(device)

# Define optimizer
optimizer = torch.optim.Adam(conditional_multilabel.parameters(), lr=1e-3)

# Training parameters
num_epochs = 100
num_iterations = 100
num_iterations_val = 20

# Storage for losses and predictions
all_train_losses = []
all_val_losses = []
all_predictions = []

print(f"\n=== Starting ChestMNIST Multi-Label Training ===")
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in conditional_multilabel.parameters()):,}")
print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_val)}")
print(f"Test samples: {len(dataset_test)}")

# Set up visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7), dpi=300)

# Adjust layout to prevent label cutoff
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.9, wspace=0.3)

def animate(frame):
    """Animation function called by FuncAnimation"""
    global all_train_losses, all_val_losses, all_predictions
    
    print(f"Training epoch {frame + 1}/{num_epochs}...")
    
    # Prepare data with multi-hot encoding for this epoch
    def train_loss_closure(images, labels):
        conditional_multilabel.train()
        images = images.to(device)
        
        # Convert labels to multi-hot tensor
        labels_multihot = labels_to_multihot(labels, device=device)
        
        return -conditional_multilabel.log_prob(images, labels_multihot).mean()
    
    def val_loss_closure(images, labels):
        conditional_multilabel.eval()
        with torch.no_grad():
            images = images.to(device)
            
            # Convert labels to multi-hot tensor
            labels_multihot = labels_to_multihot(labels, device=device)
            
            return -conditional_multilabel.log_prob(images, labels_multihot).mean()
    
    # Train for one epoch
    train_losses, val_losses = gmi.train(
        train_data=dataloader_train, 
        val_data=dataloader_val,
        train_loss_closure=train_loss_closure,
        val_loss_closure=val_loss_closure,
        num_epochs=1, 
        num_iterations=num_iterations,
        num_iterations_val=num_iterations_val,
        optimizer=optimizer,
        device=device, 
        verbose=True
    )
    
    # Store losses
    all_train_losses.extend(train_losses)
    all_val_losses.extend(val_losses)
    
    with torch.no_grad():
        # Randomly sample a target focus: 15 options (0-13 for each class, 14 for "no findings")
        # Class 0-13: look for samples with that class positive
        # Class 14: look for samples with all labels = 0 (no findings)
        target_focus = np.random.randint(0, 15)
        
        # Convert test dataset to list and shuffle for random sampling
        test_samples = []
        for i in range(len(dataset_test)):
            test_samples.append((i, *dataset_test[i]))
        np.random.shuffle(test_samples)
        
        # Search for a sample matching our target focus
        single_image = None
        single_label = None
        
        for idx, image, label in test_samples:
            # Convert label to numpy if needed
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.array(label)
            
            # Check if this sample matches our target focus
            if target_focus < 14:
                # Looking for samples with target_focus class positive
                if label_np[target_focus] == 1:
                    single_image = image.unsqueeze(0).to(device)  # Add batch dim and move to device
                    single_label = label_np
                    break
            else:
                # Looking for "no findings" samples (all labels = 0)
                if np.sum(label_np) == 0:
                    single_image = image.unsqueeze(0).to(device)  # Add batch dim and move to device
                    single_label = label_np
                    break
        
        # Fallback: if we couldn't find target focus, use first test sample
        if single_image is None:
            image, label = dataset_test[0]
            single_image = image.unsqueeze(0).to(device)
            single_label = label.numpy() if isinstance(label, torch.Tensor) else np.array(label)
        
        # Get predicted probabilities for all labels (likelihood only, not posterior)
        with torch.no_grad():
            likelihood_logits = cnn_classifier.forward_likelihood_only(single_image)
            predicted_probs = torch.sigmoid(likelihood_logits).cpu().numpy()[0]  # [14]
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Plot 1: Training and validation loss curves (logarithmic scale)
        epochs_so_far = list(range(1, len(all_train_losses) + 1))
        if all_train_losses:
            ax1.plot(epochs_so_far, all_train_losses, 'b-', label='Train Loss', linewidth=2)
        if all_val_losses:
            ax1.plot(epochs_so_far, all_val_losses, 'r-', label='Val Loss', linewidth=2)
        
        ax1.set_xlim(0, num_epochs)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set logarithmic scale and y-limits based on current training loss
        ax1.set_yscale('log')
        if all_train_losses:
            current_train_loss = all_train_losses[-1]  # Most recent training loss
            
            # Set y-limits to [0.5x, 2x] the current training loss
            y_min = 0.5 * current_train_loss
            y_max = 2.0 * current_train_loss
            
            ax1.set_ylim(y_min, y_max)
        
        # Plot 2: Show single image (middle) - grayscale
        img_display = single_image[0, 0].cpu().numpy()  # Remove batch and channel dims [H, W]
        
        ax2.imshow(img_display, cmap='gray')
        active_conditions = [chest_codebook[i] for i, val in enumerate(single_label) if val == 1]
        if active_conditions:
            title_text = f'True: {", ".join(active_conditions[:3])}'
            if len(active_conditions) > 3:
                title_text += f" (+{len(active_conditions)-3} more)"
        else:
            title_text = 'True: No Findings'
        ax2.set_title(title_text)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot 3: Horizontal bar chart showing predicted probabilities vs ground truth
        y_positions = np.arange(14)
        
        # Plot predicted probabilities as horizontal bars
        bars = ax3.barh(y_positions, predicted_probs, alpha=0.7, color=(233/255, 213/255, 50/255))
        
        # Initialize all ground truth points off-view (reset defaults)
        # Create scatter plot with all points initially at x=100 (off-view)
        ground_truth_x = np.full(14, 100.0)  # All points start off-view
        ground_truth_y = np.arange(14)
        
        # For each true label, place point at predicted_probability + 0.1
        for i, is_present in enumerate(single_label):
            if is_present == 1:
                ground_truth_x[i] = predicted_probs[i] + 0.1
        
        # Plot all ground truth points (visible ones at correct positions, invisible ones off-view)
        scatter = ax3.scatter(ground_truth_x, ground_truth_y, color=(21/255, 96/255, 130/255), 
                             s=100, zorder=5, clip_on=False)
        
        # Add decision threshold line at x=0.5
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=4)
        
        ax3.set_title('Likelihood Probabilities vs Ground Truth Multi-Hot')
        ax3.set_xlabel('Probability (Likelihood Only)')
        ax3.set_xlim(0, 1.2)  # Extended to accommodate ground truth dots
        ax3.set_yticks(y_positions)
        ax3.set_yticklabels([chest_codebook[i] for i in range(14)], fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(233/255, 213/255, 50/255), alpha=0.7, label='Likelihood Prob'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(21/255, 96/255, 130/255), 
                      markersize=8, label='Ground Truth'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ]
        ax3.legend(handles=legend_elements, loc='lower right')
        
        # Update figure title with loss information
        train_loss = train_losses[0] if train_losses else 0
        val_loss = val_losses[0] if val_losses else 0
        fig.suptitle(f'ChestMNIST Multi-Label Classification - Epoch {frame + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    return ax1, ax2, ax3

# Create animation-safe dataloaders
print("Setting up animation-safe dataloaders...")
dataloader_train_anim = torch.utils.data.DataLoader(dataset_train_augmented, batch_size=batch_size, shuffle=True, num_workers=0)
dataloader_val_anim = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

# Store original dataloaders
dataloader_train_orig = dataloader_train
dataloader_val_orig = dataloader_val

# Use animation-safe dataloaders
dataloader_train = dataloader_train_anim
dataloader_val = dataloader_val_anim

# Create the animation
print("Creating training animation...")
ani = animation.FuncAnimation(
    fig, animate, frames=num_epochs, interval=1000, blit=False, repeat=False
)

# Save as MP4
animation_path = os.path.join(medmnist_example_dir, 'chestmnist_multilabel_animation.mp4')
print(f"Saving MP4 animation to: {animation_path}")

writer = animation.FFMpegWriter(fps=2, bitrate=1200)
ani.save(animation_path, writer=writer)
print(f"MP4 animation saved: {animation_path}")

# Restore original dataloaders
dataloader_train = dataloader_train_orig
dataloader_val = dataloader_val_orig

print(f"Final train loss: {all_train_losses[-1]:.4f}" if all_train_losses else "No training completed")
print(f"Final val loss: {all_val_losses[-1]:.4f}" if all_val_losses else "No validation completed")

# Final evaluation on test set
print("\n=== Final Evaluation on Test Set ===")
conditional_multilabel.eval()

# Multi-label metrics
total_samples = 0
total_exact_matches = 0
total_hamming_accuracy = 0
label_precisions = np.zeros(14)
label_recalls = np.zeros(14)
label_f1s = np.zeros(14)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader_test):
        if batch_idx >= 50:  # Limit evaluation for speed
            break
            
        images = images.to(device)
        
        # Convert labels to tensor
        true_labels = labels_to_multihot(labels, device=device)  # [batch_size, 14]
        
        # Get predictions (thresholded at 0.5)
        predicted_probs = conditional_multilabel.predict_probabilities(images)
        predicted_binary = (predicted_probs > 0.5).float()
        
        # Convert to numpy for evaluation
        true_np = true_labels.cpu().numpy()
        pred_np = predicted_binary.cpu().numpy()
        
        # Exact match accuracy (all labels must match)
        exact_matches = np.all(true_np == pred_np, axis=1)
        total_exact_matches += exact_matches.sum()
        
        # Hamming accuracy (average per-label accuracy)
        hamming_acc = np.mean(true_np == pred_np, axis=1)
        total_hamming_accuracy += hamming_acc.sum()
        
        total_samples += len(images)
        
        # Per-label metrics
        for label_idx in range(14):
            true_label = true_np[:, label_idx]
            pred_label = pred_np[:, label_idx]
            
            # Precision, Recall, F1 for this label
            tp = ((true_label == 1) & (pred_label == 1)).sum()
            fp = ((true_label == 0) & (pred_label == 1)).sum()
            fn = ((true_label == 1) & (pred_label == 0)).sum()
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            label_precisions[label_idx] += precision * len(images)
            label_recalls[label_idx] += recall * len(images)
            label_f1s[label_idx] += f1 * len(images)

# Calculate averages
exact_match_accuracy = total_exact_matches / total_samples if total_samples > 0 else 0
hamming_accuracy = total_hamming_accuracy / total_samples if total_samples > 0 else 0
label_precisions /= total_samples
label_recalls /= total_samples
label_f1s /= total_samples

print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({total_exact_matches}/{total_samples})")
print(f"Hamming Accuracy (avg per-label): {hamming_accuracy:.4f}")
print(f"Macro-averaged F1: {np.mean(label_f1s):.4f}")

print("\nPer-label metrics:")
for i in range(14):
    print(f"  {chest_codebook[i][:20]:20s}: P={label_precisions[i]:.3f} R={label_recalls[i]:.3f} F1={label_f1s[i]:.3f}")

plt.show()