import torch
import kagglehub


class KaggleHubDataset(torch.utils.data.Dataset):
    def __init__(self, name, download=False):
        self.name = name
        if download:
            self.path = self._download()
    def _download(self):
        return kagglehub.dataset_download(self.name)
    def _load_data(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        raise NotImplementedError