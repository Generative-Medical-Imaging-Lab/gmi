import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import wandb

def train(train_loader, loss_closure, num_epochs=10, num_iterations=100,
          optimizer=None, lr=1e-3, lr_scheduler=None,
          device='cuda' if torch.cuda.is_available() else 'cpu',fabric=None,
          validation_loader=None, num_iterations_val=10, verbose=True, very_verbose=False,
          early_stopping=False, patience=10, val_loss_smoothing=0.9, min_delta=1e-6, 
          use_ema=False, ema_decay=0.999, 
          wandb_project=None, wandb_config=None):
    """
    Trains any model with an Adam optimizer using a provided loss closure, with optional early stopping, EMA, and WandB logging.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        loss_closure (function): A function that calculates and returns the loss, taking a batch of data as input.
        num_epochs (int): The number of epochs to train for.
        num_iterations (int): The number of iterations per epoch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to use for training. If None, uses Adam.
        lr (float): The learning rate for the Adam optimizer.
        device (str): The device to train on, 'cuda' or 'cpu'.
        validation_loader (torch.utils.data.DataLoader, optional): DataLoader providing batches of validation data.
        verbose (bool): If True, prints training and validation losses at each epoch.
        early_stopping (bool): If True, enables early stopping based on validation loss.
        patience (int): Number of epochs with no improvement to wait before stopping when early stopping is enabled.
        min_delta (float): Minimum change to qualify as an improvement for early stopping.
        use_ema (bool): If True, applies Exponential Moving Average to model weights.
        ema_decay (float): Decay factor for EMA.
        val_loss_smoothing (float): Smoothing factor for exponential running average of validation loss.
        wandb_project (str): WandB project name.
        wandb_config (dict): Additional configuration for WandB.
    
    Returns:
        tuple: Two lists of average losses recorded at each epoch for training and validation (if provided).
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(loss_closure.parameters(), lr=lr)
    if num_iterations_val is None:
        num_iterations_val = num_iterations


    if wandb_project is not None:
        use_wandb = True
    else:
        use_wandb = False

    # Initialize WandB if enabled
    if use_wandb:
        wandb.init(project=wandb_project, config=wandb_config)
    
    # Initialize EMA if enabled
    ema = ExponentialMovingAverage(loss_closure.parameters(), decay=ema_decay) if use_ema else None
    
    # Early stopping variables
    smoothed_val_loss = None
    patience_counter = 0

    # Trackers for loss history
    train_losses, val_losses = [], []
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(validation_loader) if validation_loader else None

    for epoch in range(num_epochs):
        loss_closure.train()
        train_batch_losses = []

        for _ in tqdm(range(num_iterations), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            try:
                batch_data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_data = next(train_loader_iter)

            # Move data to device



            # batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data) \
            #     if isinstance(batch_data, (tuple, list)) else batch_data.to(device)
            
            if isinstance(batch_data, (tuple, list)):
                
                batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data) \
                    if isinstance(batch_data, (tuple, list)) else batch_data.to(device)
            elif isinstance(batch_data, torch.Tensor):
                batch_data = [batch_data.to(device)]

            optimizer.zero_grad()
            loss = loss_closure(*batch_data if isinstance(batch_data, (tuple, list)) else batch_data)
            
            if fabric is None:
                loss.backward()
            else:
                fabric.backward(loss)
            
            
            optimizer.step()

            # Apply EMA to weights
            if ema:
                ema.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # Log training loss to WandB
            if use_wandb:
                wandb.log({"train_loss": loss.item()})

            train_batch_losses.append(loss.item())
            if very_verbose:
                print(f"Training Batch Loss: {loss.item():.4f}")
        
        # Record average train loss for the epoch
        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_losses.append(train_epoch_loss)

        # Validation phase
        if validation_loader:
            loss_closure.eval()
            val_batch_losses = []
            with torch.no_grad():
                for _ in tqdm(range(num_iterations_val), desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    try:
                        batch_data = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(validation_loader)
                        batch_data = next(val_loader_iter)

                    # batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data) \
                    #     if isinstance(batch_data, (tuple, list)) else batch_data.to(device)
                    
                    if isinstance(batch_data, (tuple, list)):
                        batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data) \
                            if isinstance(batch_data, (tuple, list)) else batch_data.to(device)
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = [batch_data.to(device)]
                    
                    
                    
                    val_loss = loss_closure(*batch_data if isinstance(batch_data, (tuple, list)) else batch_data)
                    val_batch_losses.append(val_loss.item())

                    if use_wandb:
                        wandb.log({"val_loss": val_loss.item()})

            # Record average validation loss
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)

            # Smooth the validation loss
            if smoothed_val_loss is None:
                smoothed_val_loss = val_epoch_loss
            else:
                smoothed_val_loss = val_loss_smoothing * smoothed_val_loss + (1 - val_loss_smoothing) * val_epoch_loss

            # Print losses if verbose
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Smoothed Val Loss: {smoothed_val_loss:.4f}")
            
            # Early stopping logic based on smoothed validation loss
            
            if epoch == 0:
                best_val_loss = smoothed_val_loss*2

            if early_stopping:
                if smoothed_val_loss < best_val_loss - min_delta:
                    best_val_loss = smoothed_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1} due to no improvement in smoothed validation loss.")
                        break

            if smoothed_val_loss < best_val_loss:
                best_val_loss = smoothed_val_loss
                if use_wandb:
                    wandb.run.summary["best_smoothed_val_loss"] = best_val_loss
                    wandb.run.summary["smoothed_val_loss"] = smoothed_val_loss

        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f}")
    
    # Apply EMA weights before returning, if using EMA
    if ema:
        ema.store()
        ema.copy_to()
    
    return (train_losses, val_losses) if validation_loader else train_losses
