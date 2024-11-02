
import torch
from torchvision import datasets, transforms

class MNIST( torch.utils.data.Dataset):
    def __init__(self, 
                 train=True, 
                 transform=None, 
                 download=True, 
                 images_only=False):

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        self.dataset = datasets.MNIST(
                                root='./data',
                                train=train,
                                transform=transform,
                                download=download)

        self.images_only = images_only

    def __getitem__(self, index):
        # if its a slice, convert it to a list of indices
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self.dataset))))
        if isinstance(index, int):
            index = [index]
        if isinstance(index, torch.Tensor):
            index = index.to(torch.int64).tolist()
        if isinstance(index, list):
            data_list = []
            target_list = []
            for i in index:
                data, target = self.dataset[i]
                data_list.append(data)
                target_list.append(target)
            data_batch = torch.stack(data_list)
            target_batch = torch.tensor(target_list)
            
            if self.images_only:
                return data_batch
            else:
                return data_batch, target_batch
    
    def __len__(self):
        return len(self.dataset)