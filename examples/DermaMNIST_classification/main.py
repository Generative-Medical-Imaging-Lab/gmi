from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

medmnist_dataset_root = '/workspace/gmi/gmi_data/datasets/medmnist_dataset_root/'
if not os.path.exists(medmnist_dataset_root):
    os.makedirs(medmnist_dataset_root)

medmnist_name = 'DermaMNIST'
batch_size = 64

medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))

# DermaMNIST class labels codebook
derma_codebook = {
    0: 'actinic keratoses and intraepithelial carcinoma',
    1: 'basal cell carcinoma', 
    2: 'benign keratosis-like lesions',
    3: 'dermatofibroma',
    4: 'melanoma',
    5: 'melanocytic nevi',
    6: 'vascular lesions'
}

# Function to convert class labels to one-hot
def labels_to_onehot(labels, num_classes=7, device=None):
    """Convert class labels to one-hot encoding."""
    if isinstance(labels, np.ndarray):
        if labels.shape == (1,):
            labels = labels[0]  # Extract scalar from array
        labels = torch.tensor(labels, dtype=torch.long)
    elif not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Handle batch dimension
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    elif labels.dim() == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    
    # Ensure labels are on the correct device
    if device is not None:
        labels = labels.to(device)
    
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Create the datasets with size=64 for 3x64x64 images
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

# Define augmentation transforms for training data
# This includes all possible flips and rotations
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
                # Ensure image is in proper format [C, H, W] with values in [0, 1]
                image = torch.clamp(image, 0, 1)
                image = self.transform(image)
        
        return image, label

# Create augmented training dataset (only for training, not val/test)
dataset_train_augmented = AugmentedDataset(dataset_train, train_transforms)

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

# Define CNN classifier for 7 classes
class DermaCNNClassifier(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = torch.nn.Sequential(
            # First block
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),  # 64x64 -> 32x32
            
            # Second block  
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            # Third block
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),  # 16x16 -> 8x8
            
            # Fourth block
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),  # 8x8 -> 4x4
            
            # Fifth block
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1)  # Global average pooling -> 1x1
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn_classifier = DermaCNNClassifier(num_classes=7).to(device)

# Create conditional categorical random variable
conditional_categorical = gmi.random_variable.ConditionalCategoricalRandomVariable(
    logit_function=cnn_classifier,
    codebook=derma_codebook
).to(device)

# Define optimizer
optimizer = torch.optim.Adam(conditional_categorical.parameters(), lr=1e-4)

# Training parameters
num_epochs = 200
num_iterations = 100
num_iterations_val = 20

# Storage for losses and predictions
all_train_losses = []
all_val_losses = []
all_predictions = []

# Set up visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7), dpi=300)

# Adjust layout to prevent label cutoff
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.9, wspace=0.3)

def animate(frame):
    """Animation function called by FuncAnimation"""
    global all_train_losses, all_val_losses, all_predictions
    
    print(f"Training epoch {frame + 1}/{num_epochs}...")
    
    # Prepare data with one-hot encoding for this epoch
    def train_loss_closure(images, labels):
        conditional_categorical.train()
        images = images.to(device)
        
        # Convert labels to one-hot
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
        labels_onehot = labels_to_onehot(labels, num_classes=7, device=device)
        
        return -conditional_categorical.log_prob(images, labels_onehot).mean()
    
    def val_loss_closure(images, labels):
        conditional_categorical.eval()
        with torch.no_grad():
            images = images.to(device)
            
            # Convert labels to one-hot
            if isinstance(labels, torch.Tensor):
                if labels.dim() == 2 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
            labels_onehot = labels_to_onehot(labels, num_classes=7, device=device)
            
            return -conditional_categorical.log_prob(images, labels_onehot).mean()
    
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
        # Randomly sample a class uniformly for this frame
        target_class = np.random.randint(0, 7)
        
        # Search through validation data to find an image of the target class
        val_iter = iter(dataloader_val)
        single_image = None
        single_label = None
        
        # Keep sampling batches until we find the target class
        max_attempts = 50  # Prevent infinite loop
        attempts = 0
        
        while single_image is None and attempts < max_attempts:
            try:
                images, labels = next(val_iter)
                images = images.to(device)
                
                # Convert labels to numpy for easy comparison
                if isinstance(labels, torch.Tensor):
                    if labels.dim() == 2 and labels.shape[1] == 1:
                        labels_np = labels.squeeze(1).numpy()
                    else:
                        labels_np = labels.numpy()
                else:
                    labels_np = np.array(labels)
                
                # Find indices where label matches target class
                matching_indices = np.where(labels_np == target_class)[0]
                
                if len(matching_indices) > 0:
                    # Randomly select one of the matching images
                    selected_idx = np.random.choice(matching_indices)
                    single_image = images[selected_idx:selected_idx+1]  # Keep batch dimension
                    single_label = labels_np[selected_idx]
                    break
                    
            except StopIteration:
                # Reset iterator if we reach the end
                val_iter = iter(dataloader_val)
            
            attempts += 1
        
        # Fallback: if we couldn't find the target class, use the first image from a batch
        if single_image is None:
            val_batch = next(iter(dataloader_val))
            images, labels = val_batch
            images = images.to(device)
            single_image = images[0:1]
            single_label = labels[0]
            
            # Convert single label for display
            if isinstance(single_label, torch.Tensor):
                if single_label.dim() == 1 and single_label.shape[0] == 1:
                    single_label = single_label.item()
                elif single_label.dim() == 0:
                    single_label = single_label.item()
        
        # Ensure single_label is an integer
        if isinstance(single_label, (torch.Tensor, np.ndarray)):
            single_label = int(single_label)
        
        # Get predicted probabilities using log_prob and exponential
        single_label_onehot = labels_to_onehot(torch.tensor([single_label]), num_classes=7, device=device)
        log_probs = conditional_categorical.log_prob(single_image, single_label_onehot)
        
        # Get probabilities for all classes by computing log_prob for each class
        all_class_probs = []
        for class_idx in range(7):
            class_onehot = torch.zeros(1, 7, device=device)
            class_onehot[0, class_idx] = 1.0
            class_log_prob = conditional_categorical.log_prob(single_image, class_onehot)
            all_class_probs.append(torch.exp(class_log_prob).item())
        
        # Normalize to get proper probabilities (they should already sum to 1)
        all_class_probs = np.array(all_class_probs)
        all_class_probs = all_class_probs / all_class_probs.sum()
        
        # Create true one-hot vector
        true_onehot = np.zeros(7)
        true_onehot[single_label] = 1.0
        
        # Store predictions for tracking
        predicted_class = np.argmax(all_class_probs)
        all_predictions.append(predicted_class)
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Plot 1: Training and validation loss curves
        epochs_so_far = list(range(1, len(all_train_losses) + 1))
        if all_train_losses:
            ax1.plot(epochs_so_far, all_train_losses, 'b-', label='Train Loss', linewidth=2)
        if all_val_losses:
            ax1.plot(epochs_so_far, all_val_losses, 'r-', label='Val Loss', linewidth=2)
        
        ax1.set_xlim(0, num_epochs)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Show single image (middle)
        img_display = single_image[0].cpu().permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)  # Ensure values are in [0,1] range
        
        ax2.imshow(img_display)
        ax2.set_title(f'True Label: {derma_codebook[single_label]}')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot 3: Horizontal bar chart showing predicted probabilities with true label marked
        y_positions = np.arange(7)
        
        # Plot predicted probabilities as horizontal bars
        bars = ax3.barh(y_positions, all_class_probs, alpha=0.7, color=(233/255, 213/255, 50/255))
        
        # Add blue circle on the true label
        true_prob = all_class_probs[single_label]
        circle_x = true_prob + 0.1  # Place circle 0.1 higher than predicted probability
        ax3.scatter(circle_x, single_label, color=(21/255, 96/255, 130/255), s=100, zorder=5, label='True Label')
        
        ax3.set_title('Predicted Probabilities')
        ax3.set_xlabel('Probability Mass')
        ax3.set_xlim(0, 1.2)  # Extended to 1.2 to accommodate the blue circle
        ax3.set_yticks(y_positions)
        ax3.set_yticklabels([derma_codebook[i][:12] + '...' for i in range(7)], fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Update figure title with loss information
        train_loss = train_losses[0] if train_losses else 0
        val_loss = val_losses[0] if val_losses else 0
        fig.suptitle(f'DermaMNIST Classification - Epoch {frame + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
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
animation_path = os.path.join(medmnist_example_dir, 'dermamnist_classification_animation.mp4')
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
conditional_categorical.eval()
test_correct = 0
test_total = 0
class_correct = [0] * 7
class_total = [0] * 7

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader_test):
        if batch_idx >= 50:  # Limit evaluation for speed
            break
            
        images = images.to(device)
        
        # Convert labels for comparison
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
        true_labels = labels.numpy()
        
        # Get predictions
        predictions = conditional_categorical.predict_classes(images).cpu().numpy()
        
        # Calculate accuracy
        correct = (predictions == true_labels)
        test_correct += correct.sum()
        test_total += len(predictions)
        
        # Per-class accuracy
        for i in range(len(predictions)):
            label = true_labels[i]
            class_correct[label] += correct[i]
            class_total[label] += 1

test_accuracy = test_correct / test_total if test_total > 0 else 0
print(f"Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_total})")

print("\nPer-class accuracy:")
for i in range(7):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"  {derma_codebook[i][:30]}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

plt.show()

