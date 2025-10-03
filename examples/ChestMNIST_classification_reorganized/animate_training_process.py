import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from train_classifier import labels_to_multihot, CHEST_CODEBOOK

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

def animate_training_process(config, model, conditional_multilabel, optimizer, 
                           dataloader_train, dataloader_val, dataset_test):
    """Create animated visualization of the training process"""
    import gmi
    
    training_config = config.training
    output_config = config.output
    
    num_epochs = training_config['num_epochs']
    num_iterations = training_config['num_iterations']
    num_iterations_val = training_config['num_iterations_val']
    device = config.device
    
    # Storage for losses and predictions
    all_train_losses = []
    all_val_losses = []
    
    print(f"\n=== Starting ChestMNIST Multi-Label Training Animation ===")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in conditional_multilabel.parameters()):,}")
    print(f"Training samples: {len(dataloader_train.dataset)}")
    print(f"Validation samples: {len(dataloader_val.dataset)}")
    print(f"Test samples: {len(dataset_test)}")
    
    # Set up visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7), dpi=300)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.9, wspace=0.3)
    
    def animate(frame):
        """Animation function called by FuncAnimation"""
        nonlocal all_train_losses, all_val_losses
        
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
                        single_image = image.unsqueeze(0).to(device)
                        single_label = label_np
                        break
                else:
                    # Looking for "no findings" samples (all labels = 0)
                    if np.sum(label_np) == 0:
                        single_image = image.unsqueeze(0).to(device)
                        single_label = label_np
                        break
            
            # Fallback: if we couldn't find target focus, use first test sample
            if single_image is None:
                image, label = dataset_test[0]
                single_image = image.unsqueeze(0).to(device)
                single_label = label.numpy() if isinstance(label, torch.Tensor) else np.array(label)
            
            # Get predicted probabilities for all labels (likelihood only, not posterior)
            with torch.no_grad():
                likelihood_logits = model.forward_likelihood_only(single_image)
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
                current_train_loss = all_train_losses[-1]
                y_min = 0.9 * current_train_loss
                y_max = 1.1 * current_train_loss
                ax1.set_ylim(y_min, y_max)
            
            # Plot 2: Show single image (middle) - grayscale
            img_display = single_image[0, 0].cpu().numpy()  # Remove batch and channel dims [H, W]
            
            ax2.imshow(img_display, cmap='gray')
            active_conditions = [CHEST_CODEBOOK[i] for i, val in enumerate(single_label) if val == 1]
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
            
            # Initialize all ground truth points off-view
            ground_truth_x = np.full(14, 100.0)  # All points start off-view
            ground_truth_y = np.arange(14)
            
            # For each true label, place point at predicted_probability + 0.1
            for i, is_present in enumerate(single_label):
                if is_present == 1:
                    ground_truth_x[i] = predicted_probs[i] + 0.1
            
            # Plot all ground truth points
            scatter = ax3.scatter(ground_truth_x, ground_truth_y, color=(21/255, 96/255, 130/255), 
                                 s=100, zorder=5, clip_on=False)
            
            # Add decision threshold line at x=0.5
            ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=4)
            
            ax3.set_title('Likelihood Probabilities vs Ground Truth Multi-Hot')
            ax3.set_xlabel('Probability (Likelihood Only)')
            ax3.set_xlim(0, 1.2)
            ax3.set_yticks(y_positions)
            ax3.set_yticklabels([CHEST_CODEBOOK[i] for i in range(14)], fontsize=8)
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
    
    return all_train_losses, all_val_losses