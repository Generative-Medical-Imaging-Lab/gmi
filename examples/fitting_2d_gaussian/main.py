import torch

# define a torch dataset
import torchvision

import gmi
import os
import pandas as pd
import numpy as np
import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_dataset = gmi.datasets.kaggle.UCIMLBreastCancerWisconsinDataset(split='train', download=True)
drop_columns = train_dataset.df.columns.tolist(); drop_columns.pop(10); drop_columns.pop(6) # only keep symmetry and smoothness

train_dataset = gmi.datasets.kaggle.UCIMLBreastCancerWisconsinDataset(split='train', download=True, drop_columns=drop_columns)
val_dataset = gmi.datasets.kaggle.UCIMLBreastCancerWisconsinDataset(split='val', download=True, drop_columns=drop_columns)
test_dataset = gmi.datasets.kaggle.UCIMLBreastCancerWisconsinDataset(split='test', download=True, drop_columns=drop_columns)

data_dim = train_dataset[0:1][0].size(1)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

myTrainableGaussian = gmi.random_variable.TrainableGaussian(dim=data_dim).to(device)

optimizer = torch.optim.Adam(myTrainableGaussian.parameters(), lr=2e-4)

num_epochs = 100

train_losses, val_losses = gmi.train(
        train_data=train_dataloader, 
        val_data=val_dataloader,
        train_loss_closure=myTrainableGaussian.train_loss_closure,
        val_loss_closure=myTrainableGaussian.eval_loss_closure,
        num_epochs=1, 
        num_iterations=1000,
        num_iterations_val=100,
        optimizer=optimizer,
        device=device, 
        verbose=True
    )





# Create animation showing Gaussian training progress
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
# fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Get data for plotting (first 2 dimensions only for visualization)
train_data_np = train_dataset[:][0][:, :2].cpu().numpy()
val_data_np = val_dataset[:][0][:, :2].cpu().numpy()
test_data_np = test_dataset[:][0][:, :2].cpu().numpy()

# Plot datasets and store references to preserve them during animation
train_scatter = ax.scatter(train_data_np[:, 0], train_data_np[:, 1], c='blue', label='Training Data', alpha=0.6, s=20)
val_scatter = ax.scatter(val_data_np[:, 0], val_data_np[:, 1], c='green', label='Validation Data', alpha=0.6, s=20)
test_scatter = ax.scatter(test_data_np[:, 0], test_data_np[:, 1], c='red', label='Test Data', alpha=0.6, s=20)

# Initialize empty plots for samples and contours
ln_independent_samples = ax.scatter([], [], c='orange', label='Model Samples', alpha=0.8, s=30)

# Keep track of data collections to preserve them
data_collections = {train_scatter, val_scatter, test_scatter, ln_independent_samples}
# ln_langevin_samples = ax.scatter([], [], c='purple', label='Langevin Samples', alpha=0.8, s=30)  # Commented out

# Set up grid for contour plotting with square aspect ratio
# Combine all data to find global min/max
all_data = np.vstack([train_data_np, val_data_np, test_data_np])
x_data_min, x_data_max = all_data[:, 0].min(), all_data[:, 0].max()
y_data_min, y_data_max = all_data[:, 1].min(), all_data[:, 1].max()

# Find the maximum extent in either direction
x_extent = x_data_max - x_data_min
y_extent = y_data_max - y_data_min
max_extent = max(x_extent, y_extent)

# Calculate centers
x_center = (x_data_min + x_data_max) / 2
y_center = (y_data_min + y_data_max) / 2

# Create square bounds with 110% of max extent
half_extent = (max_extent * 1.1) / 2
x_min, x_max = x_center - half_extent, x_center + half_extent
y_min, y_max = y_center - half_extent, y_center + half_extent

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(loc='upper right')
ax.set_xlabel('Feature 1: Smoothness')
ax.set_ylabel('Feature 2: Symmetry')

# Storage for losses across all epochs
all_train_losses = []
all_val_losses = []

def animate(frame):
    """Animation function called by FuncAnimation"""
    global all_train_losses, all_val_losses
    
    print(f"Training epoch {frame + 1}/{num_epochs}...")
    
    # Train for one epoch
    train_losses, val_losses = gmi.train(
        train_data=train_dataloader, 
        val_data=val_dataloader,
        train_loss_closure=myTrainableGaussian.train_loss_closure,
        val_loss_closure=myTrainableGaussian.eval_loss_closure,
        num_epochs=1, 
        num_iterations=1000,
        num_iterations_val=100,
        optimizer=optimizer,
        device=device, 
        verbose=True
    )
    
    # Store losses
    all_train_losses.extend(train_losses)
    all_val_losses.extend(val_losses)
    
    with torch.no_grad():
        # Sample from the trained Gaussian
        independent_samples = myTrainableGaussian.sample(100).cpu().numpy()
        ln_independent_samples.set_offsets(independent_samples[:, :2])
        
        # Calculate log probabilities for contour plot (only first 2 dimensions)
        if data_dim >= 2:
            # Create grid with zeros for higher dimensions if data_dim > 2
            if data_dim > 2:
                grid_full = torch.zeros(grid_points.shape[0], data_dim, device=device)
                grid_full[:, :2] = grid_points
            else:
                grid_full = grid_points
                
            log_probs = myTrainableGaussian.log_prob(grid_full).cpu().numpy()
            log_probs = log_probs.reshape(xx.shape)
            
            # Clear only contour collections, preserve data scatter plots
            collections_to_remove = []
            for collection in ax.collections:
                if collection not in data_collections:
                    collections_to_remove.append(collection)
            
            for collection in collections_to_remove:
                collection.remove()
            
            # Plot contours at 1, 2, and 3 standard deviations
            try:
                # Calculate contour levels for 1, 2, and 3 standard deviations
                log_prob_max = log_probs.max()
                sigma_1_level = log_prob_max - 0.5  # 1 standard deviation
                sigma_2_level = log_prob_max - 2.0  # 2 standard deviations  
                sigma_3_level = log_prob_max - 4.5  # 3 standard deviations
                
                levels = [sigma_3_level, sigma_2_level, sigma_1_level]
                contours = ax.contour(xx, yy, log_probs, levels=levels, colors='black', alpha=0.7, linewidths=1.5)
                
                # Handle different matplotlib versions for labeling contours
                try:
                    # Try newer matplotlib API first
                    if hasattr(contours, 'collections') and contours.collections:
                        labels = ['3σ', '2σ', '1σ']
                        for i, collection in enumerate(contours.collections):
                            if i < len(labels):
                                collection.set_label(labels[i])
                except AttributeError:
                    # Fallback for different matplotlib versions
                    pass
            except Exception as e:
                print(f"Warning: Could not create contours for frame {frame}: {e}")
        
        # Update title with loss information
        train_loss = train_losses[0] if train_losses else 0
        val_loss = val_losses[0] if val_losses else 0
        ax.set_title(f'Epoch {frame + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    return ln_independent_samples,

# Create the animation
print("Creating training animation...")

# Temporarily recreate dataloaders with num_workers=0 to avoid multiprocessing conflicts during animation
print("Setting up animation-safe dataloaders...")
train_dataloader_anim = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader_anim = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Store original dataloaders to restore later
train_dataloader_orig = train_dataloader
val_dataloader_orig = val_dataloader

# Use animation-safe dataloaders
train_dataloader = train_dataloader_anim
val_dataloader = val_dataloader_anim

ani = animation.FuncAnimation(
    fig, animate, frames=num_epochs, interval=500, blit=False, repeat=False
)

# Save as MP4 with optimized settings
animation_path = os.path.join(os.path.dirname(__file__), 'gaussian_training_animation.mp4')
print(f"Saving MP4 animation to: {animation_path}")


# Fallback with simpler settings
writer = animation.FFMpegWriter(fps=10, bitrate=800)
ani.save(animation_path, writer=writer)
print(f"MP4 animation saved with fallback settings: {animation_path}")


# Restore original DataLoaders
train_dataloader = train_dataloader_orig
val_dataloader = val_dataloader_orig

print(f"Final train loss: {all_train_losses[-1]:.4f}" if all_train_losses else "No training completed")
print(f"Final val loss: {all_val_losses[-1]:.4f}" if all_val_losses else "No validation completed")

plt.show()



    

