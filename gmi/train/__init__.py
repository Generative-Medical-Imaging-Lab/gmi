import torch
import torch.nn as nn

def train(  train_loader, 
            loss_closure, 
            num_epochs=10, 
            num_iterations=100,
            optimizer=None,
            lr=1e-3, 
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            validation_loader=None, 
            num_iterations_val=10,
            verbose=True):
    """
    Trains any model with an Adam optimizer using a provided loss closure.
    
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
    
    Returns:
        tuple: Two lists of average losses recorded at each epoch for training and validation (if provided).
    """
    # Initialize the optimizer (assumes parameters are accessible within the closure)
    if optimizer is None:
        if lr is None:
            lr = 1e-3
        optimizer = torch.optim.Adam(loss_closure.parameters(), lr=lr)

    if num_iterations_val is None:
        num_iterations_val = num_iterations

    train_loader_iter = iter(train_loader)
    
    if validation_loader is not None:
        val_loader_iter = iter(validation_loader)


    # Store average loss values for each epoch
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):

        # Initialize lists to store batch losses
        train_batch_losses = []
        
        # Set model to training mode
        loss_closure.train()  

        # Iterate over each batch of training data
        for i in range(num_iterations):

            # Get the next batch of data
            try:
                batch_data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_data = next(train_loader_iter)

            # Move batch data to the specified device
            if isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device)
            else:
                assert isinstance(batch_data, tuple) or isinstance(batch_data, list), "Batch data must be a tensor or tuple or list"
                batch_data_list = []
                for batch_data_element in batch_data:
                    if isinstance(batch_data_element, torch.Tensor):
                        batch_data_element = batch_data_element.to(device)
                    batch_data_list.append(batch_data_element)
                batch_data = tuple(batch_data_list)
            # Zero gradients before each step
            optimizer.zero_grad()
            
            # Calculate loss using closure, passing in the current batch
            
            if isinstance(batch_data, torch.Tensor):
                loss = loss_closure(batch_data)
            else:
                loss = loss_closure(*batch_data)
            
            # Perform backpropagation
            loss.backward()
            
            # Take optimization step
            optimizer.step()
            
            # Record batch loss
            train_batch_losses.append(loss.item())
        
        # Record average training loss for the epoch
        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_losses.append(train_epoch_loss)
        
        # Validation (if validation_loader is provided)
        if validation_loader:
            # Set model to evaluation mode
            loss_closure.eval()  

            # Initialize lists to store batch losses
            val_batch_losses = []
            
            # Iterate over each batch of validation data
            with torch.no_grad():  # Disable gradient computation for validation
                for i in range(num_iterations_val):

                    # Get the next batch of data
                    try:
                        batch_data = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(validation_loader)
                        batch_data = next(val_loader_iter)

                    # Move batch data to the specified device
                        
                    if isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    else:
                        assert isinstance(batch_data, tuple) or isinstance(batch_data, list), "Batch data must be a tensor or tuple or list"
                        batch_data_list = []
                        for batch_data_element in batch_data:
                            if isinstance(batch_data_element, torch.Tensor):
                                batch_data_element = batch_data_element.to(device)
                            batch_data_list.append(batch_data_element)
                        batch_data = tuple(batch_data_list)

                    # Calculate validation loss using closure
                    
                    if isinstance(batch_data, torch.Tensor):
                        val_loss = loss_closure(batch_data)
                    else:
                        val_loss = loss_closure(*batch_data)


                    val_batch_losses.append(val_loss.item())
            
            # Record average validation loss for the epoch
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}")
    
    return (train_losses, val_losses) if validation_loader else train_losses


class LossClosure(nn.Module):
    def __init__(self, model, loss_fn):
        super(LossClosure, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch_data):

        # Forward pass through the model
        output = self.model(batch_data)
        
        # Calculate loss
        loss = self.loss_fn(output, batch_data)
        
        return loss