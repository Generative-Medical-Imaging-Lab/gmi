import torch
from torchvision import datasets, transforms
import medmnist

class MedMNIST( torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_name, 
                 split='train', 
                 transform=transforms.Compose([transforms.ToTensor()]),
                 target_transform=None,
                 download=True, 
                 images_only=False):

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        if dataset_name == 'PathMNIST':
            medmnist_dataset = medmnist.PathMNIST
        elif dataset_name == 'ChestMNIST':
            medmnist_dataset = medmnist.ChestMNIST
        elif dataset_name == 'DermaMNIST':
            medmnist_dataset = medmnist.DermaMNIST
        elif dataset_name == 'OCTMNIST':
            medmnist_dataset = medmnist.OCTMNIST
        elif dataset_name == 'PneumoniaMNIST':
            medmnist_dataset = medmnist.PneumoniaMNIST
        elif dataset_name == 'RetinaMNIST':
            medmnist_dataset = medmnist.RetinaMNIST
        elif dataset_name == 'BreastMNIST':
            medmnist_dataset = medmnist.BreastMNIST
        elif dataset_name == 'BloodMNIST':
            medmnist_dataset = medmnist.BloodMNIST
        elif dataset_name == 'TissueMNIST':
            medmnist_dataset = medmnist.TissueMNIST
        elif dataset_name == 'OrganAMNIST':
            medmnist_dataset = medmnist.OrganAMNIST
        elif dataset_name == 'OrganCMNIST':
            medmnist_dataset = medmnist.OrganCMNIST
        elif dataset_name == 'OrganSMNIST':
            medmnist_dataset = medmnist.OrganSMNIST
        elif dataset_name == 'OrganMNIST3D':
            medmnist_dataset = medmnist.OrganMNIST3D
        elif dataset_name == 'NoduleMNIST3D':
            medmnist_dataset = medmnist.NoduleMNIST3D
        elif dataset_name == 'AdrenalMNIST3D':
            medmnist_dataset = medmnist.AdrenalMNIST3D
        elif dataset_name == 'FractureMNIST3D':
            medmnist_dataset = medmnist.FractureMNIST3D
        elif dataset_name == 'VesselMNIST3D':
            medmnist_dataset = medmnist.VesselMNIST3D
        elif dataset_name == 'SynapseMNIST3D':
            medmnist_dataset = medmnist.SynapseMNIST3D
        else:
            raise ValueError('MedMNIST dataset name not recognized. Please choose from: PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D')

        self.medmnist_dataset = medmnist_dataset(
                                split=split,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)

        self.images_only = images_only

    def __getitem__(self, index):
        # # if its a slice, convert it to a list of indices
        # if isinstance(index, slice):
        #     index = list(range(*index.indices(len(self.medmnist_dataset))))
        # if isinstance(index, int):
        #     index = [index]
        # if isinstance(index, torch.Tensor):
        #     index = index.to(torch.int64).tolist()
        # if isinstance(index, list):
        #     data_list = []
        #     target_list = []
        #     for i in index:
        #         data, target = self.medmnist_dataset[i]
        #         data_list.append(data)
        #         target_list.append(target)
        #     data_batch = torch.stack(data_list)
        #     target_batch = torch.tensor(target_list)
            
            data, target = self.medmnist_dataset[index]

            if self.images_only:
                return data
            else:
                return data, target
    
    def __len__(self):
        return len(self.medmnist_dataset)