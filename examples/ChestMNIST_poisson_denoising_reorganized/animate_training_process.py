import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm

def create_animation_safe_loaders(dataloader_train, dataloader_val, batch_size):
    """Create animation-safe data loaders with num_workers=0"""
    # Get original datasets
    dataset_train = dataloader_train.dataset
    dataset_val = dataloader_val.dataset
    
    # Create animation-safe loaders
    dataloader_train_anim = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataloader_val_anim = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return dataloader_train_anim, dataloader_val_anim

def animate_training_process(config, physics_simulator, conditional_denoiser, optimizer,
                           dataloader_train, dataloader_val, dataset_test):
    """Create animated visualization of the training process"""
    import gmi
    
    training_config = config.training
    output_config = config.output
    
    num_epochs = training_config['num_epochs']
    num_iterations = training_config['num_iterations']
    num_iterations_val = training_config['num_iterations_val']
    device = config.device
    
    # Storage for losses and training history
    all_train_losses = []
    all_val_losses = []
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'epochs': []
    }
    
    print(f"\n=== Starting ChestMNIST Poisson Denoising Training Animation ===")
    print(f"Device: {device}")
    print(f"Physics - μ: {config.physics['mu']}, I₀: {config.physics['I0']}")
    print(f"Model parameters: {sum(p.numel() for p in conditional_denoiser.parameters()):,}")
    print(f"Training samples: {len(dataloader_train.dataset)}")
    print(f"Validation samples: {len(dataloader_val.dataset)}")
    print(f"Test samples: {len(dataset_test)}")
    
    # Set up visualization with 5 subplots in a row (loss curve + 4 images)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5), dpi=300)
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.85, wspace=0.3)
    
    def animate(frame):
        """Animation function called by FuncAnimation"""
        nonlocal all_train_losses, all_val_losses, training_history
        
        print(f"Training epoch {frame + 1}/{num_epochs}...")
        
        def train_loss_closure(x_batch):
            conditional_denoiser.train()
            x_batch = x_batch.to(device)
            
            # Simulate noisy measurements using physics simulator
            y_batch = physics_simulator.forward_model(x_batch)
            
            # Compute negative log-likelihood (for minimization)
            return -conditional_denoiser.log_prob(y_batch, x_batch).mean()
        
        def val_loss_closure(x_batch):
            conditional_denoiser.eval()
            with torch.no_grad():
                x_batch = x_batch.to(device)
                
                # Simulate noisy measurements
                y_batch = physics_simulator.forward_model(x_batch)
                
                return -conditional_denoiser.log_prob(y_batch, x_batch).mean()
        
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
        training_history['train_losses'].extend(train_losses)
        training_history['val_losses'].extend(val_losses)
        training_history['epochs'].append(frame + 1)
        
        # Generate sample visualizations
        conditional_denoiser.eval()
        with torch.no_grad():
            # Get a random test sample
            test_idx = np.random.randint(0, len(dataset_test))
            x_sample = dataset_test[test_idx]  # Only image, no label since images_only=True
            x_sample = x_sample.unsqueeze(0).to(device)  # Add batch dimension
            
            # Simulate noisy measurement
            y_sample = physics_simulator.forward_model(x_sample)
            
            # Get log corrected measurement
            y_corr_sample = conditional_denoiser.get_log_corrected(y_sample)
            
            # Get denoised reconstruction
            x_pred_sample = conditional_denoiser.get_mean_estimate(y_sample)
            
            # Convert to numpy for plotting
            original = x_sample[0, 0].cpu().numpy()  # [H, W]
            noisy = y_sample[0, 0].cpu().numpy()     # [H, W]
            corrections = y_corr_sample[0, 0].cpu().numpy()  # [H, W]
            reconstructions = x_pred_sample[0, 0].cpu().numpy()  # [H, W]
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        
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
        
        # Plot 2: Original clean image [0,1]
        ax2.imshow(original, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Original\nClean Image', fontsize=10)
        ax2.axis('off')
        
        # Plot 3: Noisy measurements [0, I0*1.1]
        I0_val = config.physics["I0"]
        ax3.imshow(noisy, cmap='gray', vmin=0, vmax=I0_val*1.1)
        ax3.set_title(f'Noisy Measurements\n(Poisson, μ={config.physics["mu"]}, I₀={I0_val})', fontsize=10)
        ax3.axis('off')
        
        # Plot 4: Log corrected measurements (use natural range, not [0,1])
        ax4.imshow(corrections, cmap='gray')
        ax4.set_title('Log Corrected\n(-log(y/I₀))', fontsize=10)
        ax4.axis('off')
        
        # Plot 5: Denoised reconstruction [0,1]
        ax5.imshow(reconstructions, cmap='gray', vmin=0, vmax=1)
        mse = np.mean((original - reconstructions) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ax5.set_title(f'Denoised Result\n(PSNR: {psnr:.2f}dB)', fontsize=10)
        ax5.axis('off')
        
        # Update figure title with loss information
        train_loss = train_losses[0] if train_losses else 0
        val_loss = val_losses[0] if val_losses else 0
        fig.suptitle(f'X-ray Denoising - Epoch {frame + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}', 
                    fontsize=14)
        
        return ax1, ax2, ax3, ax4, ax5
    
    # Create animation-safe dataloaders
    print("Setting up animation-safe dataloaders...")
    dataloader_train_anim, dataloader_val_anim = create_animation_safe_loaders(
        dataloader_train, dataloader_val, config.data['batch_size']
    )
    
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
    animation_path = os.path.join(output_config['output_dir'], output_config['animation_path'])
    print(f"Saving MP4 animation to: {animation_path}")
    
    writer = animation.FFMpegWriter(
        fps=output_config['animation_fps'], 
        bitrate=output_config['animation_bitrate']
    )
    ani.save(animation_path, writer=writer)
    print(f"MP4 animation saved: {animation_path}")
    
    # Restore original dataloaders
    dataloader_train = dataloader_train_orig
    dataloader_val = dataloader_val_orig
    
    print(f"Final train loss: {all_train_losses[-1]:.4f}" if all_train_losses else "No training completed")
    print(f"Final val loss: {all_val_losses[-1]:.4f}" if all_val_losses else "No validation completed")
    
    return all_train_losses, all_val_losses, training_history