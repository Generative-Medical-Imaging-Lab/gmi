import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import gmi

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Olivetti Faces dataset (flattened for high-dimensional Gaussian fitting)
train_dataset = gmi.datasets.sklearn.OlivettiFacesDataset(download=True, flatten=True)
print(f"Dataset size: {len(train_dataset)}")
print(f"Data dimensionality: {train_dataset[0][0].shape}")

# Split dataset into train/val (80/20 split)
dataset_size = len(train_dataset)
# train_size = int(0.8 * dataset_size)
# val_size = dataset_size - train_size

# train_dataset, val_dataset = torch.utils.data.random_split(
#     dataset, [train_size, val_size], 
#     generator=torch.Generator().manual_seed(42)
# )

# Create data loaders
batch_size = 400  # Smaller batch size for high-dimensional data
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
)
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
# )

# print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Get data dimensionality
data_dim = train_dataset[0][0].numel()
print(f"Data dimension: {data_dim}")

# Create sparse Gaussian model
num_principal_components = 400  # Much smaller than 4096 for sparsity
eps = 1e-2

print(f"Creating sparse Gaussian with {num_principal_components} principal components")
sparse_gaussian = gmi.random_variable.TrainableSparseGaussian(
    dim=data_dim, 
    num_principal_components=num_principal_components,
    eps=eps
).to(device)


sample_mean = train_dataset[:][0].mean(dim=0).to(device)
# sparse_gaussian.mu is a parameter, we need to set it to this value without raising any errors or breaking the computational graph
with torch.no_grad():
    sparse_gaussian.mu.copy_(sample_mean)

data_matrix = train_dataset[:][0].to(device)
centered_data = data_matrix - sparse_gaussian.mu.unsqueeze(0)
cov_matrix = (centered_data.T @ centered_data) / (data_matrix.size(0))
# take the svd
U, S, Vt = torch.linalg.svd(cov_matrix)
# get the first num_principal_components components
U_reduced = U[:, :num_principal_components]
S_reduced = S[:num_principal_components]
# reconstruct the low-rank covariance matrix
low_rank_cov = U_reduced @ torch.diag(S_reduced) @ U_reduced.T
# now do it for sigma, where Sigma = sigma @ sigma^T + eps * I
sigma_matrix = U_reduced @ torch.diag(torch.sqrt(S_reduced)) 
with torch.no_grad():
    sparse_gaussian.sigma.copy_(sigma_matrix)

# Optimizer
optimizer = torch.optim.Adam(sparse_gaussian.parameters(), lr=1e-3)

print("Starting training...")

# Train the model
num_epochs = 1
iterations_per_epoch = len(train_dataloader)
# val_iterations_per_epoch = len(val_dataloader)

print(f"Training for {num_epochs} epochs with {iterations_per_epoch} iterations per epoch")

# train_losses, val_losses = gmi.train(
#     train_data=train_dataloader,
#     val_data=val_dataloader,
#     train_loss_closure=sparse_gaussian.train_loss_closure,
#     val_loss_closure=sparse_gaussian.eval_loss_closure,
#     num_epochs=num_epochs,
#     num_iterations=iterations_per_epoch,
#     num_iterations_val=val_iterations_per_epoch,
#     optimizer=optimizer,
#     device=device,
#     verbose=True
# )
train_losses, val_losses = gmi.train(
    train_data=train_dataloader,
    train_loss_closure=sparse_gaussian.train_loss_closure,
    num_epochs=num_epochs,
    num_iterations=iterations_per_epoch,
    optimizer=optimizer,
    device=device,
    verbose=True
)

print("Training completed!")

# Generate samples and visualize
print("Generating samples...")
with torch.no_grad():
    # Generate samples
    num_samples = 16
    samples = sparse_gaussian.sample(num_samples).cpu().numpy()
    
    # Get some real data for comparison
    real_data = []
    for i, (data,) in enumerate(train_dataloader):
        real_data.append(data)
        if i * batch_size >= num_samples:
            break
    real_data = torch.cat(real_data, dim=0)[:num_samples].cpu().numpy()
    
    # Reshape to images (64x64)
    sample_images = samples.reshape(num_samples, 64, 64)
    real_images = real_data.reshape(num_samples, 64, 64)
    
    # Create visualization
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Sparse Gaussian Face Generation: Real vs Generated', fontsize=16)
    
    for i in range(num_samples):
        # Real images (top two rows)
        row = i // 8
        col = i % 8
        if row < 2:
            axes[row, col].imshow(real_images[i], cmap='gray',vmin=0,vmax=1)
            axes[row, col].set_title(f'Real {i+1}' if row == 0 else '')
            axes[row, col].axis('off')
        
        # Generated images (bottom two rows)  
        if row < 2:
            # axes[row + 2, col].imshow(sample_images[i], cmap='gray', vmin=0,vmax=1)
            axes[row + 2, col].imshow(sparse_gaussian.mu.reshape(64,64).cpu().detach(), cmap='gray', vmin=0,vmax=1)
            axes[row + 2, col].set_title(f'Generated {i+1}' if row + 2 == 2 else '')
            axes[row + 2, col].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(__file__), 'face_generation_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also save loss plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Training Progress: Sparse Gaussian Face Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    loss_path = os.path.join(os.path.dirname(__file__), 'training_losses.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved to: {loss_path}")
    
    plt.show()

print(f"Final train loss: {train_losses[-1]:.4f}")
# print(f"Final validation loss: {val_losses[-1]:.4f}")

# Print model statistics
print(f"\nModel Statistics:")
print(f"Data dimension: {data_dim}")
print(f"Principal components: {num_principal_components}")  
print(f"Regularization eps: {eps}")
print(f"Total parameters: {sum(p.numel() for p in sparse_gaussian.parameters())}")
full_gaussian_params = data_dim + data_dim * (data_dim + 1) // 2
print(f"Full Gaussian would have: {full_gaussian_params} parameters")
print(f"Parameter reduction: {100 * (1 - sum(p.numel() for p in sparse_gaussian.parameters()) / full_gaussian_params):.1f}%")