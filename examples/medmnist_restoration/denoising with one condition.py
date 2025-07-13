import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from diffusers import UNet2DModel

import gmi
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
medmnist_name = 'OrganAMNIST'
batch_size = 64
images_only = True
medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(medmnist_example_dir, 'models', 'conditional_diffusion_model.pth')
print(f"MedMNIST example directory: {medmnist_example_dir}")


#---------------------- Preparing the data ----------------------
class DuplicateDataset(Dataset):
    """A PyTorch Dataset class that creates duplicates of images with optional transformations.
    This dataset wrapper takes a base dataset and creates pairs of images where the second image
    is a transformed copy of the original, optionally with added noise (simulating some kind of meassurment)
    Args:
        base_dataset (Dataset): The original dataset to duplicate images from.
        noise_simulation (callable, optional): A function/transform to add noise to the copied
            images after copy_transform is applied. Default is None.
    Returns:
        tuple: A tuple containing:
            - image (Tensor): The original image from the base dataset
            - cond (Tensor): Concatenated tensor of transformed copy (simulated meassurment) and expanded class label
    Shape:
        - image: Original image shape from base dataset
        - cond: (C+1, H, W) where C is the number of channels in the image, 
          and H, W are height and width
    Note:
        The label is expanded to match the spatial dimensions (H,W) of the image and
        concatenated with the transformed copy along the channel dimension.
    """
    def __init__(self, base_dataset, noise_simulation=None):
        self.base_dataset = base_dataset
        self.noise_simulation = noise_simulation
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset[idx]
        image = torch.as_tensor(image)

        if hasattr(image, "clone"):
            image_copy = image.clone()
            
        else:
            image_copy = image.copy()

        if self.noise_simulation is not None:
            image_copy = self.noise_simulation(image_copy)

        
        return [[image.to(device), image_copy.to(device)]] #the list within a list is necessary because loss closure always applies
    # * to any input but down the line the code expects the image and the condition to come as a list or tuple.

dataset_train = gmi.datasets.MedMNIST(medmnist_name, 
                                        split='train', 
                                        download=True, 
                                        images_only=images_only)

dataset_val = gmi.datasets.MedMNIST(medmnist_name,
                                    split='val',
                                    download=True,
                                    images_only=images_only)

dataset_test = gmi.datasets.MedMNIST(medmnist_name,
                                     split='test',
                                     download=True,
                                     images_only=images_only)

white_noise_adder = gmi.distribution.AdditiveWhiteGaussianNoise(
                                                    noise_standard_deviation=0.2)
def add_noise(y):
    """"
    This function contains the meassurment simulation. At the moment it just adds white noise to the image.
    """
    y = white_noise_adder.sample(y)
    return y

dataset_train = DuplicateDataset(dataset_train, noise_simulation=add_noise)
dataset_val = DuplicateDataset(dataset_val, noise_simulation=add_noise)
dataset_test = DuplicateDataset(dataset_test, noise_simulation=add_noise)
dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

#---------------------- Defining the model ----------------------
print("Defining the model...")
forward_SDE = gmi.sde.VarianceExplodingSDE(noise_variance=lambda t: t,
                                               noise_variance_prime=lambda t: t * 0.0 + 1.0)

class ConditionalUNet(UNet2DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, sample, timestep):
        return super().forward(sample, timestep)
    
class DiffusionBackbone(nn.Module):
    def __init__(self):
        super(DiffusionBackbone, self).__init__()

        self.unet = ConditionalUNet(
            sample_size=32,
            in_channels=2,   # Changed from 1 to 2 to get the info about the labels (our condition) in. Se below
            out_channels=1,
            layers_per_block=2,
            norm_num_groups=1,
            block_out_channels=(32, 64, 128),
            down_block_types=("AttnDownBlock2D", 
                              "AttnDownBlock2D", 
                              "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", 
                            "AttnUpBlock2D", 
                            "AttnUpBlock2D"),
        )
    
    def forward(self, x_t, t, y):
        # x_t expected to have shap [B, 1, H, W]!
        x_t_cat = torch.cat([x_t, y], dim=1)# Concatenate along the chan dimension to [B, 2, H, W]
        
        x_t_cat = torch.nn.functional.pad(x_t_cat, (2, 2, 2, 2), mode='constant', value=0)
        
        x_0_pred = self.unet(x_t_cat, t.squeeze())[0]#We take the first element. This is the "denoised" image

        x_0_pred = x_0_pred[:, :, 2:30, 2:30] #Removing padding of output
        return x_0_pred


diffusion_backbone = DiffusionBackbone().to(device)
conditional_diffusion_model = gmi.diffusion.DiffusionModel(forward_SDE, diffusion_backbone)

if not os.path.isdir(os.path.join(medmnist_example_dir, 'models')):
    os.makedirs(os.path.join(medmnist_example_dir, 'models'))
if os.path.isfile(model_path):
    conditional_diffusion_model.diffusion_backbone.load_state_dict(torch.load(model_path))
else:
    print("No state dict found. Proceeding without loading.")

#---------------------- Training the model ----------------------
class TimeSampler(gmi.samplers.Sampler):
    def __init__(self):
        super(TimeSampler, self).__init__()
    def sample(self, batch_size):
        log10_t_mu = -1.0
        log10_t_sigma = 1.0 
        log10_t = torch.randn(batch_size, 1) * log10_t_sigma + log10_t_mu
        t = 10**log10_t
        return t
    
time_sampler = TimeSampler()
loss_closure = conditional_diffusion_model.loss_closure(time_sampler=time_sampler)
optimizer = torch.optim.Adam(conditional_diffusion_model.diffusion_backbone.parameters(), lr=1e-3)

if not os.path.isfile(model_path):
    gmi.train(
        dataloader_train, 
        loss_closure, 
        num_epochs=12, 
        num_iterations=1000,
        optimizer=optimizer,
        lr=1e-3, 
        device=device, 
        validation_loader=dataloader_val, 
        num_iterations_val=100,
        verbose=True,
        patience=1
    )
    torch.save(conditional_diffusion_model.diffusion_backbone.state_dict(), model_path)
else:
    print("Model already trained. Skipping training.")

#---------------------- Plotting some examples ----------------------
print("Plotting some examples now, depending on the number of examples this might take a while...")
n_examples = 5
T = 1000
t0 = 0.0001
nSteps = 512
fig, ax = plt.subplots(n_examples, 3, figsize=(3*2, n_examples*2))

ax[0,0].set_title(f"Simulated measurement")
ax[0,1].set_title(f"Reconstruction")
ax[0,2].set_title(f"Ground truth")
for i in tqdm(range(n_examples)):
    sample = dataset_test[i][0]
    x_0, conds = sample
    conds = conds.unsqueeze(0)
    x_0 = x_0.unsqueeze(0)
    
    x_t = conditional_diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, t=torch.tensor([T]).to(device))
    timesteps = (torch.linspace(1.0, 0.0, nSteps).to(device) ** 5.0) * (T - t0) + t0
    conditional_diffusion_model.eval()
    x_t_all = conditional_diffusion_model.sample_reverse_process(
        x_t.to(device),
        timesteps,
        sampler='euler',
        return_all=True,
        verbose=False,
        y=conds.to(device)
    )
    
    x_reconstructed = x_t_all[-1][0, 0, :, :].cpu().detach().numpy()
    x_0 = x_0[0, 0, :, :].cpu().detach().numpy()
    y = conds[0, 0, :, :].cpu().detach().numpy()

    ax[i, 0].imshow(y, cmap='gray', vmin=0, vmax=1)
    ax[i, 1].imshow(x_reconstructed, cmap='gray', vmin=0, vmax=1)
    ax[i, 2].imshow(x_0, cmap='gray', vmin=0, vmax=1)

ax = ax.flatten()
for a in ax:
    a.set_axis_off()
plt.show()
    



