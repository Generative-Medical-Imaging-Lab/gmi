
import torch
import sklearn
import numpy as np

class OlivettiFacesDataset(torch.utils.data.Dataset):
    def __init__(self, download=False, flatten=True):
        self.data = None
        self.flatten = flatten
        if download:
            self._download()
        self._load_data()

    def _download(self):
        dataset = sklearn.datasets.fetch_olivetti_faces()
        self.images = dataset.images
        self.data = dataset.data

    def _load_data(self):
        if self.data is None:
            raise RuntimeError("Data not downloaded. Call _download() first.")
        if self.flatten:
            # Return flattened 4096-dim vectors for training
            self.processed_data = self.data
        else:
            # Return 64x64 images for visualization
            self.processed_data = self.data.reshape(-1, 64, 64)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return (torch.tensor(self.processed_data[idx], dtype=torch.float32),)
    
    def get_image_shape(self):
        """Return the shape for reshaping flattened data back to image"""
        return (64, 64)