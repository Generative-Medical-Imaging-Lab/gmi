import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import wandb
from typing import Callable, Dict, Any, Optional

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
    test_closure=None
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
            wandb.init(project=wandb_project, config=wandb_config)
        except Exception as e:
            print(f"Warning: WandB initialization failed ({e}). Falling back to disabled mode.")
            wandb.init(project=wandb_project, config=wandb_config, mode='disabled')
            use_wandb = True  # Keep using WandB but in disabled mode
    
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

    # Trackers for loss history and evaluation metrics
    train_losses, val_losses = [], []
    eval_metrics = []
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(validation_loader) if validation_loader else None
    test_loader_iter = iter(test_loader) if test_loader else None

    # Extract model for evaluation
    model = extract_model_from_loss_closure(train_loss_closure)

    for epoch in range(num_epochs):
        train_loss_closure.train()
        train_batch_losses = []

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
                wandb.log({"train_loss": loss.item()})

            train_batch_losses.append(loss.item())
            if very_verbose:
                print(f"Training Batch Loss: {loss.item():.4f}")
        
        # Record average train loss for the epoch
        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_losses.append(train_epoch_loss)

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
                        wandb.log({"val_loss": val_loss.item()})

            # Record average validation loss
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            
            if save_best is not None:
                save_best(val_epoch_loss)

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
        
        # Test phase
        if test_loader and test_closure is not None:
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
                                wandb.log({f"test_{key}": value})
                            
                            # Log media files immediately to WandB with consistent key
                            if isinstance(value, str) and os.path.exists(value):
                                file_ext = os.path.splitext(value)[1].lower()
                                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov']:
                                    try:
                                        media_key = f"test_{key}"
                                        if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
                                            wandb.log({media_key: wandb.Image(value)})
                                        else:
                                            wandb.log({media_key: wandb.Video(value)})
                                    except Exception as e:
                                        print(f"Warning: Could not log file {value} to wandb: {e}")
                    else:
                        # If not a dict, treat as a single metric
                        if 'test_output' not in test_metrics:
                            test_metrics['test_output'] = []
                        test_metrics['test_output'].append(test_output)
            
            # Log averaged metrics for the epoch (in addition to per-iteration metrics)
            if use_wandb and test_metrics:
                for metric_name, values in test_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        # Numeric metrics - average them for epoch summary
                        avg_value = sum(values) / len(values)
                        wandb.log({f"test_{metric_name}_epoch_avg": avg_value})
                    else:
                        # Non-numeric values - just log the last one
                        wandb.log({f"test_{metric_name}_epoch_avg": values[-1]})
            
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
    
    # Final test evaluation
    if final_test_iterations is not None and test_loader and test_closure is not None:
        print(f"\nRunning final test evaluation with {final_test_iterations} iterations...")
        test_closure.eval()
        final_test_metrics = {}
        
        with torch.no_grad():
            for iteration in tqdm(range(final_test_iterations), desc="Final Test Evaluation"):
                try:
                    batch_data = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    batch_data = next(test_loader_iter)

                if isinstance(batch_data, (tuple, list)):
                    # If it's a tuple/list, extract just the images (first element) for final testing
                    images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                    batch_data = images
                elif isinstance(batch_data, torch.Tensor):
                    batch_data = batch_data.to(device)
                
                # Test closure returns a dict
                test_output = test_closure(batch_data, epoch=num_epochs, iteration=iteration)
                
                # Process test output and log immediately to WandB
                if isinstance(test_output, dict):
                    for key, value in test_output.items():
                        if key not in final_test_metrics:
                            final_test_metrics[key] = []
                        final_test_metrics[key].append(value)
                        
                        # Log numeric metrics immediately to WandB every iteration
                        if isinstance(value, (int, float)) and use_wandb:
                            wandb.log({f"final_test_{key}": value})
                        
                        # Log media files immediately to WandB with consistent key
                        if isinstance(value, str) and os.path.exists(value):
                            file_ext = os.path.splitext(value)[1].lower()
                            if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov']:
                                try:
                                    media_key = f"final_test_{key}"
                                    if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
                                        wandb.log({media_key: wandb.Image(value)})
                                    else:
                                        wandb.log({media_key: wandb.Video(value)})
                                except Exception as e:
                                    print(f"Warning: Could not log file {value} to wandb: {e}")
                else:
                    # If not a dict, treat as a single metric
                    if 'test_output' not in final_test_metrics:
                        final_test_metrics['test_output'] = []
                    final_test_metrics['test_output'].append(test_output)
        
        # Log averaged metrics for the final test evaluation
        if use_wandb and final_test_metrics:
            for metric_name, values in final_test_metrics.items():
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric metrics - average them for final summary
                    avg_value = sum(values) / len(values)
                    wandb.log({f"final_test_{metric_name}_avg": avg_value})
                else:
                    # Non-numeric values - just log the last one
                    wandb.log({f"final_test_{metric_name}_avg": values[-1]})
        
        # Print final test metrics if verbose
        if verbose and final_test_metrics:
            print(f"\nFinal test evaluation metrics:")
            for metric_name, values in final_test_metrics.items():
                if all(isinstance(v, (int, float)) for v in values):
                    avg_value = sum(values) / len(values)
                    print(f"  final_test_{metric_name}: {avg_value:.4f}")
                else:
                    print(f"  final_test_{metric_name}: {values[-1]}")
    
    return train_losses, val_losses, eval_metrics
