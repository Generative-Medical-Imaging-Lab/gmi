import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import wandb
from typing import Callable, Dict, Any, Optional
import time

def extract_model_from_loss_closure(loss_closure):
    """Extract the model from a loss closure, supporting different structures."""
    # Try different common patterns for accessing the model
    if hasattr(loss_closure, 'parent') and hasattr(loss_closure.parent, 'diffusion_backbone'):
        # Diffusion model pattern
        return loss_closure.parent.diffusion_backbone
    elif hasattr(loss_closure, 'reconstructor'):
        # Direct reconstructor pattern
        return loss_closure.reconstructor
    elif hasattr(loss_closure, 'model'):
        # Generic model pattern
        return loss_closure.model
    elif hasattr(loss_closure, 'parameters'):
        # If the loss closure itself is the model
        return loss_closure
    else:
        # Fallback: try to find any nn.Module in the loss closure
        for attr_name in dir(loss_closure):
            attr = getattr(loss_closure, attr_name)
            if isinstance(attr, nn.Module):
                return attr
        raise ValueError("Could not extract model from loss closure. Please ensure the loss closure has a model accessible via .parent.diffusion_backbone, .reconstructor, .model, or as a direct nn.Module attribute.")

class save_best_model():
    """A callback class that saves the model state when the validation loss improves.
    This class implements a simple callback mechanism that tracks the best validation loss
    and saves the model's state dictionary when a new best loss is achieved. It can also handle
    Exponential Moving Average (EMA) if provided.
    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        save_path (str): File path where the model state should be saved.
        ema (optional, ExponentialMovingAverage): EMA instance for model parameter averaging.
            Defaults to None.
    Attributes:
        model (torch.nn.Module): The model being monitored.
        ema (ExponentialMovingAverage): EMA instance for parameter averaging.
        save_path (str): Path where model state is saved.
        best_loss (float): The best validation loss observed so far.
    Example:
        >>> model = MyModel()
        >>> saver = save_best_model(model, 'best_model.pth')
        >>> # During training
        >>> val_loss = validate(model)
        >>> saver(val_loss)
    """
    def __init__(self, model, save_path, ema=None):
        self.model = model
        self.ema = ema
        self.save_path = save_path
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            if self.ema:
                self.ema.store()     
                self.ema.copy_to()   
            torch.save(self.model.state_dict(), self.save_path)
            if self.ema:
                self.ema.restore()
            print(f"This is the best val loss so far: {loss:.4f}. The model was saved!")

def train(
    train_data,
    num_epochs=10,
    num_iterations=100,
    optimizer=None,
    lr=1e-3,
    lr_scheduler=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    fabric=None,
    val_data=None,
    test_data=None,
    num_iterations_val=10,
    num_iterations_test=10,
    final_test_iterations=None,
    verbose=True,
    very_verbose=False,
    early_stopping=False,
    patience=10,
    val_loss_smoothing=0.9,
    min_delta=1e-6,
    use_ema=False,
    ema_decay=0.999,
    wandb_project=None,
    wandb_config=None,
    save_best_model_path=None,
    model_to_save=None,
    eval_fn: Optional[Callable[[nn.Module, str, dict, int], Dict[str, Any]]] = None,
    epochs_per_evaluation: Optional[int] = None,
    train_batch_size=None,
    val_batch_size=None,
    test_batch_size=None,
    train_num_workers=None,
    val_num_workers=None,
    test_num_workers=None,
    shuffle_train=None,
    shuffle_val=None,
    shuffle_test=None,
    train_loss_closure=None,
    val_loss_closure=None,
    test_closure=None,
    experiment_name=None
):
    """
    Trains any model with an Adam optimizer using a provided loss closure, with optional early stopping, EMA, WandB logging,
    and periodic evaluation.

    Args:
        train_data (Dataset or DataLoader): Training data. If DataLoader, train_batch_size, train_num_workers, shuffle_train must be None.
        val_data (Dataset or DataLoader, optional): Validation data. If DataLoader, val_batch_size, val_num_workers, shuffle_val must be None.
        test_data (Dataset or DataLoader, optional): Test data. If DataLoader, test_batch_size, test_num_workers, shuffle_test must be None.
        train_batch_size (int or None): Batch size for training DataLoader (required if train_data is a Dataset).
        val_batch_size (int or None): Batch size for validation DataLoader (required if val_data is a Dataset).
        test_batch_size (int or None): Batch size for test DataLoader (required if test_data is a Dataset).
        train_num_workers (int or None): Number of workers for training DataLoader (required if train_data is a Dataset).
        val_num_workers (int or None): Number of workers for validation DataLoader (required if val_data is a Dataset).
        test_num_workers (int or None): Number of workers for test DataLoader (required if test_data is a Dataset).
        shuffle_train (bool or None): Whether to shuffle training data (required if train_data is a Dataset).
        shuffle_val (bool or None): Whether to shuffle validation data (required if val_data is a Dataset).
        shuffle_test (bool or None): Whether to shuffle test data (required if test_data is a Dataset).
        train_loss_closure (nn.Module, optional): Module for training loss calculation. If None, must be provided.
        val_loss_closure (nn.Module, optional): Module for validation loss calculation. If None, uses train_loss_closure.
        test_closure (nn.Module, optional): Module for test evaluation that returns a dict of metrics and file paths.
        num_epochs (int): The number of epochs to train for.
        num_iterations (int): The number of iterations per epoch.
        num_iterations_val (int): The number of validation iterations per epoch.
        num_iterations_test (int): The number of test iterations per epoch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to use for training. If None, uses Adam.
        lr (float): The learning rate for the Adam optimizer.
        device (str): The device to train on, 'cuda' or 'cpu'.
        fabric (optional, fabric): Fabric instance for distributed training.
        val_loss_smoothing (float): Smoothing factor for exponential running average of validation loss.
        min_delta (float): Minimum change to qualify as an improvement for early stopping.
        use_ema (bool): If True, applies Exponential Moving Average to model weights.
        ema_decay (float): Decay factor for EMA.
        wandb_project (str): WandB project name.
        wandb_config (dict): Additional configuration for WandB.
        save_best_model_path (str, optional): Path to save the best model. If None, best model is not saved.
        model_to_save (nn.Module, optional): The model to save as best. If None, best model is not saved.
        eval_fn (Callable, optional): Function that takes (model, wandb_project, wandb_config, epoch) and returns evaluation metrics.
        epochs_per_evaluation (int, optional): Run evaluation every N epochs. If None, no evaluation is run.
    Returns:
        tuple: Lists of average losses and evaluation metrics recorded at each epoch.
    """
    import torch.utils.data
    import os
    
    # Set default closures
    if train_loss_closure is None:
        raise ValueError("train_loss_closure must be provided")
    if val_loss_closure is None:
        val_loss_closure = train_loss_closure
    
    # Assert test_closure is an nn.Module if provided
    if test_closure is not None:
        assert isinstance(test_closure, nn.Module), "test_closure must be an nn.Module"
    
    # Helper to wrap dataset in DataLoader if needed
    def make_loader(data, batch_size, shuffle, num_workers, split_name):
        if data is None:
            return None
        if isinstance(data, torch.utils.data.DataLoader):
            assert batch_size is None, f"{split_name}_batch_size must be None if {split_name}_data is a DataLoader"
            assert num_workers is None, f"{split_name}_num_workers must be None if {split_name}_data is a DataLoader"
            assert shuffle is None, f"shuffle_{split_name} must be None if {split_name}_data is a DataLoader"
            return data
        assert batch_size is not None, f"{split_name}_batch_size must be provided if {split_name}_data is a Dataset"
        assert num_workers is not None, f"{split_name}_num_workers must be provided if {split_name}_data is a Dataset"
        assert shuffle is not None, f"shuffle_{split_name} must be provided if {split_name}_data is a Dataset"
        return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    train_loader = make_loader(train_data, train_batch_size, shuffle_train, train_num_workers, 'train')
    validation_loader = make_loader(val_data, val_batch_size, shuffle_val, val_num_workers, 'val')
    test_loader = make_loader(test_data, test_batch_size, shuffle_test, test_num_workers, 'test')

    if optimizer is None:
        optimizer = torch.optim.Adam(train_loss_closure.parameters(), lr=lr)
    if num_iterations_val is None:
        num_iterations_val = num_iterations
    if num_iterations_test is None:
        num_iterations_test = num_iterations

    if wandb_project is not None:
        use_wandb = True
    else:
        use_wandb = False

    # Initialize WandB if enabled
    if use_wandb:
        try:
            # Get experiment name from config or parameter
            run_name = None
            if experiment_name:
                run_name = experiment_name
            elif wandb_config and isinstance(wandb_config, dict):
                run_name = wandb_config.get('experiment_name')
            
            wandb.init(project=wandb_project, config=wandb_config, name=run_name)
        except Exception as e:
            print(f"Warning: WandB initialization failed ({e}). Falling back to disabled mode.")
            wandb.init(project=wandb_project, config=wandb_config, mode='disabled')
            use_wandb = True  # Keep using WandB but in disabled mode
    
    # Check GPU usage (silent)
    if torch.cuda.is_available():
        pass  # GPU info available but not printed
    
    # Initialize EMA if enabled
    ema = ExponentialMovingAverage(train_loss_closure.parameters(), decay=ema_decay) if use_ema else None
    
    # Setup best model saving if requested
    if save_best_model_path is not None and model_to_save is not None:
        if use_ema:
            save_best = save_best_model(model_to_save, save_best_model_path, ema=ema)
        else:
            save_best = save_best_model(model_to_save, save_best_model_path)
    else:
        save_best = None
    
    # Early stopping variables
    smoothed_val_loss = None
    patience_counter = 0
    best_val_loss = float('inf')  # Initialize to infinity so any loss will be better

    # Trackers for loss history and evaluation metrics
    train_losses, val_losses = [], []
    eval_metrics = []
    test_metrics_history = []  # Track test metrics for each epoch
    final_test_metrics = None  # Track final test metrics
    final_test_summary = {}  # Track final test summary
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(validation_loader) if validation_loader else None
    test_loader_iter = iter(test_loader) if test_loader else None

    # Extract model for evaluation
    model = extract_model_from_loss_closure(train_loss_closure)

    for epoch in range(num_epochs):
        train_loss_closure.train()
        train_batch_losses = []

        epoch_start_time = time.time()

        for _ in tqdm(range(num_iterations), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            try:
                batch_data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_data = next(train_loader_iter)

            if isinstance(batch_data, (tuple, list)):
                batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data)
            elif isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device)

            optimizer.zero_grad()
            loss = train_loss_closure(batch_data)
            
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
                wandb.log({"train_loss": loss.item(), "epoch": epoch + 1})

            train_batch_losses.append(loss.item())
            if very_verbose:
                print(f"Training Batch Loss: {loss.item():.4f}")
        
        epoch_train_time = time.time() - epoch_start_time
        
        # Record average train loss for the epoch
        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_losses.append(train_epoch_loss)

        # Log epoch-level metrics to WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_train_loss": train_epoch_loss,
                "epoch_train_time_seconds": epoch_train_time
            })

        # Run evaluation if configured
        if eval_fn is not None and epochs_per_evaluation is not None and (epoch + 1) % epochs_per_evaluation == 0:
            # Store EMA weights if using EMA
            if ema:
                ema.store()
                ema.copy_to()
            
            # Put model in eval mode and run evaluation
            model.eval()
            with torch.no_grad():
                metrics = eval_fn(model, wandb_project, wandb_config, epoch+1)
            model.train()  # Return to train mode
            
            eval_metrics.append(metrics)
            
            # Print evaluation metrics if verbose
            if verbose:
                print(f"Evaluation metrics at epoch {epoch + 1}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
            
            # Restore EMA weights if using EMA
            if ema:
                ema.restore()

        # Validation phase
        if validation_loader:
            val_loss_closure.eval()
            val_batch_losses = []
            with torch.no_grad():
                for _ in tqdm(range(num_iterations_val), desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    try:
                        batch_data = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(validation_loader)
                        batch_data = next(val_loader_iter)

                    if isinstance(batch_data, (tuple, list)):
                        # If it's a tuple/list, extract just the images (first element) for validation
                        images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                        batch_data = images
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    
                    val_loss = val_loss_closure(batch_data)
                    val_batch_losses.append(val_loss.item())

                    if use_wandb:
                        wandb.log({"val_loss": val_loss.item(), "epoch": epoch + 1})

            # Record average validation loss
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)

            # Smooth the validation loss
            if smoothed_val_loss is None:
                smoothed_val_loss = val_epoch_loss
            else:
                smoothed_val_loss = val_loss_smoothing * smoothed_val_loss + (1 - val_loss_smoothing) * val_epoch_loss
            
            # Determine which validation loss to use for early stopping and patience
            # If val_loss_smoothing is 0, use raw validation loss; otherwise use smoothed
            val_loss_for_tracking = val_epoch_loss if val_loss_smoothing == 0 else smoothed_val_loss
            
            # Update best validation loss and patience counter
            if val_loss_for_tracking < best_val_loss - min_delta:
                best_val_loss = val_loss_for_tracking
                patience_counter = 0
                # Save best model
                if save_best is not None:
                    save_best(val_loss_for_tracking)
                # Update WandB summary
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
            else:
                patience_counter += 1

            # Early stopping check
            if early_stopping and patience_counter >= patience:
                loss_type = "raw validation loss" if val_loss_smoothing == 0 else "smoothed validation loss"
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in {loss_type}. The {loss_type} is {val_loss_for_tracking:.4f} and the best validation loss is {best_val_loss:.4f}.")
                break

            # Log validation metrics to WandB
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_val_loss": val_epoch_loss,
                    "epoch_smoothed_val_loss": smoothed_val_loss,
                    "val_loss_for_tracking": val_loss_for_tracking,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter
                })

            # Print losses if verbose
            if verbose:
                loss_type = "Raw Val Loss" if val_loss_smoothing == 0 else "Smoothed Val Loss"
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | {loss_type}: {val_loss_for_tracking:.4f} | Best {loss_type}: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f}")
        
        # Test phase
        if test_loader and test_closure is not None:
            test_start_time = time.time()
            test_closure.eval()
            test_metrics = {}
            test_files = []
            
            with torch.no_grad():
                for iteration in tqdm(range(num_iterations_test), desc=f"Test Epoch {epoch + 1}/{num_epochs}"):
                    try:
                        batch_data = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        batch_data = next(test_loader_iter)

                    if isinstance(batch_data, (tuple, list)):
                        # If it's a tuple/list, extract just the images (first element) for testing
                        images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                        batch_data = images
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    
                    # Test closure returns a dict
                    test_output = test_closure(batch_data, epoch=epoch + 1, iteration=iteration)
                    
                    # Process test output and log immediately to WandB
                    if isinstance(test_output, dict):
                        for key, value in test_output.items():
                            if key not in test_metrics:
                                test_metrics[key] = []
                            test_metrics[key].append(value)
                            
                            # Log numeric metrics immediately to WandB every iteration
                            if isinstance(value, (int, float)) and use_wandb:
                                wandb.log({f"test_{key}": value, "epoch": epoch + 1})
                            
                            # Log media files immediately to WandB with consistent key
                            if isinstance(value, str) and os.path.exists(value):
                                file_ext = os.path.splitext(value)[1].lower()
                                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov']:
                                    try:
                                        media_key = f"test_{key}"
                                        if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
                                            wandb.log({media_key: wandb.Image(value), "epoch": epoch + 1})
                                        else:
                                            wandb.log({media_key: wandb.Video(value), "epoch": epoch + 1})
                                    except Exception as e:
                                        print(f"Warning: Could not log file {value} to wandb: {e}")
                    else:
                        # If not a dict, treat as a single metric
                        if 'test_output' not in test_metrics:
                            test_metrics['test_output'] = []
                        test_metrics['test_output'].append(test_output)
            
            test_time = time.time() - test_start_time
            
            # Log averaged metrics for the epoch (in addition to per-iteration metrics)
            if use_wandb and test_metrics:
                for metric_name, values in test_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        # Numeric metrics - average them for epoch summary
                        avg_value = sum(values) / len(values)
                        wandb.log({f"test_{metric_name}": avg_value})
                    else:
                        # Non-numeric values - just log the last one
                        wandb.log({f"test_{metric_name}": values[-1]})
            
            # Store test metrics for this epoch
            epoch_test_summary = {}
            if test_metrics:
                for metric_name, values in test_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        avg_value = sum(values) / len(values)
                        epoch_test_summary[metric_name] = avg_value
                    else:
                        epoch_test_summary[metric_name] = values[-1]
            test_metrics_history.append(epoch_test_summary)
            
            # Print test metrics if verbose
            if verbose and test_metrics:
                print(f"Test metrics at epoch {epoch + 1}:")
                for metric_name, values in test_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        avg_value = sum(values) / len(values)
                        print(f"  test_{metric_name}: {avg_value:.4f}")
                    else:
                        print(f"  test_{metric_name}: {values[-1]}")
    
    # Apply EMA weights before returning, if using EMA
    if ema:
        ema.store()
        ema.copy_to()
    
    # Final test evaluation - always run if configured, regardless of early stopping
    if final_test_iterations is not None and test_loader and test_closure is not None:
        print(f"\nRunning final test evaluation with {final_test_iterations} iterations...")
        test_closure.eval()
        final_test_metrics = {}
        per_sample_metrics = []  # List of dicts for each sample
        import numpy as np
        import csv
        from pathlib import Path
        
        with torch.no_grad():
            if final_test_iterations == 'all':
                # Iterate through the test DataLoader once, no repeats
                for batch_data in tqdm(test_loader, desc="Final Test Evaluation (all)"):
                    if isinstance(batch_data, (tuple, list)):
                        images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                        batch_data = images
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    # Test closure returns a dict of batch metrics
                    # We'll compute metrics for each sample in the batch
                    images = batch_data
                    measurements = test_closure.parent.sample_measurements_given_images(1, images)
                    if measurements.dim() > images.dim():
                        measurements = measurements[0]
                    reconstructions = test_closure.parent.sample_reconstructions_given_measurements(1, measurements)
                    if reconstructions.dim() > images.dim():
                        reconstructions = reconstructions[0]
                    for i in range(images.shape[0]):
                        img = images[i:i+1]
                        rec = reconstructions[i:i+1]
                        rmse = test_closure._compute_rmse(rec, img)
                        psnr = test_closure._compute_psnr(rec, img)
                        ssim = test_closure._compute_ssim(rec, img)
                        lpips = test_closure._compute_lpips(rec, img)
                        per_sample_metrics.append({
                            'index': len(per_sample_metrics),
                            'rmse': rmse,
                            'psnr': psnr,
                            'ssim': ssim,
                            'lpips': lpips
                        })
                # Save per-sample metrics to CSV
                if save_best_model_path is not None:
                    per_sample_csv = Path(save_best_model_path).parent / "final_test_metrics_per_sample.csv"
                    with open(per_sample_csv, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['index', 'rmse', 'psnr', 'ssim', 'lpips'])
                        writer.writeheader()
                        for row in per_sample_metrics:
                            writer.writerow(row)
                    if verbose:
                        print(f"Per-sample final test metrics saved to: {per_sample_csv}")
                # Compute summary statistics
                if per_sample_metrics:
                    for metric in ['rmse', 'psnr', 'ssim', 'lpips']:
                        values = np.array([row[metric] for row in per_sample_metrics])
                        final_test_summary[f'{metric}_mean'] = float(np.mean(values))
                        final_test_summary[f'{metric}_std'] = float(np.std(values))
                        final_test_summary[f'{metric}_min'] = float(np.min(values))
                        final_test_summary[f'{metric}_max'] = float(np.max(values))
            else:
                for iteration in tqdm(range(final_test_iterations), desc="Final Test Evaluation"):
                    try:
                        batch_data = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        batch_data = next(test_loader_iter)

                    if isinstance(batch_data, (tuple, list)):
                        images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                        batch_data = images
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    # Test closure returns a dict
                    test_output = test_closure(batch_data, epoch=num_epochs, iteration=iteration)
                    if isinstance(test_output, dict):
                        for key, value in test_output.items():
                            if key not in final_test_metrics:
                                final_test_metrics[key] = []
                            final_test_metrics[key].append(value)
                    else:
                        if 'test_output' not in final_test_metrics:
                            final_test_metrics['test_output'] = []
                        final_test_metrics['test_output'].append(test_output)
                # Store final test metrics summary
                if final_test_metrics:
                    for metric_name, values in final_test_metrics.items():
                        if all(isinstance(v, (int, float)) for v in values):
                            avg_value = sum(values) / len(values)
                            final_test_summary[metric_name] = avg_value
                        else:
                            final_test_summary[metric_name] = values[-1]
        # Print final test metrics if verbose
        if verbose and (final_test_metrics or per_sample_metrics):
            print(f"\nFinal test evaluation metrics:")
            if per_sample_metrics:
                for metric in ['rmse', 'psnr', 'ssim', 'lpips']:
                    values = [row[metric] for row in per_sample_metrics]
                    print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")
            if final_test_metrics:
                for metric_name, values in final_test_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        avg_value = sum(values) / len(values)
                        print(f"  final_test_{metric_name}: {avg_value:.4f}")
                    else:
                        print(f"  final_test_{metric_name}: {values[-1]}")
    
    # Save training history locally if we have a save path
    if save_best_model_path is not None:
        import json
        import os
        from pathlib import Path
        
        # Create training history file
        history_path = Path(save_best_model_path).parent / "training_history.json"
        history_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "eval_metrics": eval_metrics,
            "test_metrics_history": test_metrics_history,  # Per-epoch test metrics
            "final_test_metrics": final_test_summary,  # Final test evaluation summary
            "num_epochs": len(train_losses),
            "actual_epochs_trained": len(train_losses),  # May be less than num_epochs due to early stopping
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_val_epoch": val_losses.index(min(val_losses)) + 1 if val_losses else None,
            "early_stopping_triggered": len(train_losses) < num_epochs,
            "training_config": {
                "num_epochs": num_epochs,
                "num_iterations": num_iterations,
                "num_iterations_val": num_iterations_val,
                "num_iterations_test": num_iterations_test,
                "final_test_iterations": final_test_iterations,
                "early_stopping": early_stopping,
                "patience": patience,
                "val_loss_smoothing": val_loss_smoothing,
                "min_delta": min_delta,
                "use_ema": use_ema,
                "ema_decay": ema_decay,
                "learning_rate": lr
            }
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        if verbose:
            print(f"Training history saved to: {history_path}")
        
        # Note: Detailed test metrics and summaries are now primarily logged to WandB
        # and downloaded at the end of training via the ImageReconstructionTask
    
    # Finish WandB run if it was initialized
    if use_wandb:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish WandB run: {e}")
    
    return train_losses, val_losses, eval_metrics
