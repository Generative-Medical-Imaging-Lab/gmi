import torch

# define a torch dataset
import torchvision

import gmi
import os
import pandas as pd
import numpy as np


example_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(example_dir, "health_data.csv")

class GeneralUsupervisedProbabilisticModel(torch.nn.Module):
    def __init__(self, 
                model: torch.nn.Module | gmi.random_variable.RandomVariable,
                train_data: torch.utils.data.Dataset | torch.utils.data.DataLoader | torch.Tensor =None,
                val_data: torch.utils.data.Dataset | torch.utils.data.DataLoader | torch.Tensor=None,
                test_data: torch.utils.data.Dataset | torch.utils.data.DataLoader | torch.Tensor=None,
                optimizer: torch.optim.Optimizer = None,
                device: torch.device = None):

        super(GeneralUsupervisedProbabilisticModel, self).__init__()

        if isinstance(train_data, torch.utils.data.Dataset) or isinstance(train_data, torch.Tensor):
            self.train_data = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)
        else:
            self.train_data = train_data

        if isinstance(val_data, torch.utils.data.Dataset) or isinstance(val_data, torch.Tensor):
            self.val_data = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)
        else:
            self.val_data = val_data

        if isinstance(test_data, torch.utils.data.Dataset) or isinstance(test_data, torch.Tensor):
            self.test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
        else:
            self.test_data = test_data
            
        if isinstance(model, torch.nn.Module):
            self.model = gmi.random_variable.from_log_prob.RandomVariableFromLogProb(model)
        else:
            self.model = model

        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train_step(self, batch):
        self.model.train()
        loss = -self.model.log_prob(batch).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            loss = -self.model.log_prob(batch).mean()
        return loss.item()

    def test_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            loss = -self.model.log_prob(batch).mean()
        return loss.item()
    
    def sample(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples)
        return samples
    
    def log_prob(self, samples):
        self.model.eval()
        with torch.no_grad():
            log_probs = self.model.log_prob(samples)
        return log_probs
    
    def score(self, samples):
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score(samples)
        return scores
    

def main():

    data = pd.read_csv(data_path)
    print(data.head())
    print(data.columns)
    print(f"Data types:\n{data.dtypes}")
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")

    train_data_iter = iter(self.train_data)
    val_data_iter = iter(self.val_data)

    train_losses_all_epochs = []
    val_losses_all_epochs = []

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        for iter in range(num_iterations_train):
            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(self.train_data)
                batch = next(train_data_iter)
            batch = batch.to(self.device)
            loss = self.train_step(batch)
            train_losses.append(loss)
        if self.val_data is not None:
            for iter in range(num_iterations_val):
                try:
                    batch = next(val_data_iter)
                except StopIteration:
                    val_data_iter = iter(self.val_data)
                    batch = next(val_data_iter)
                batch = batch.to(self.device)
                loss = self.validate_step(batch)
                val_losses.append(loss)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
        train_losses_all_epochs.append(np.mean(train_losses))
        val_losses_all_epochs.append(np.mean(val_losses))
