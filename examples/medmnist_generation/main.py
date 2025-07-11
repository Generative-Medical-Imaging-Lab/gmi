from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms
from typing import cast
import numpy as np


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

medmnist_name = 'BloodMNIST'
batch_size = 32

medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))

# Path to where MedMNIST data should live (relative to repository root: gmi_base/gmi_data/datasets/MedMNIST)
gmi_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # one level up from examples/
medmnist_root = os.path.join(gmi_base_dir, 'gmi_data', 'datasets', 'MedMNIST', medmnist_name)
os.makedirs(medmnist_root, exist_ok=True)

# Create the dataset with the target_transform set to discard the label

dataset_train = gmi.datasets.MedMNIST(
    medmnist_name,
    split='train',
    download=True,
    images_only=True,
    root=medmnist_root,
)

dataset_val = gmi.datasets.MedMNIST(
    medmnist_name,
    split='val',
    download=True,
    images_only=True,
    root=medmnist_root,
)

dataset_test = gmi.datasets.MedMNIST(
    medmnist_name,
    split='test',
    download=True,
    images_only=True,
    root=medmnist_root,
)

dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4)

forward_SDE = gmi.sde.SongVarianceExplodingSDE(noise_variance = lambda t: t, noise_variance_prime = lambda t: t*0.0 + 1.0)

# Use the smaller DiffusersUnet2D_Size28 network specifically designed for 28x28 images
# This matches the simple_unet_1ch.yaml configuration from modular_configs
diffusion_backbone = gmi.network.DiffusersUnet2D_Size28(
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),  # Much smaller than the original (32, 64, 128)
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),  # No attention blocks for simplicity
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")  # No attention blocks for simplicity
).to(device)

class TimeSampler(gmi.samplers.Sampler):
    def __init__(self):
        super(TimeSampler, self).__init__()

    def sample(self, batch_size):
        
        log10_t_mu = -1.0
        log10_t_sigma = 1.0 
        log10_t = torch.randn(batch_size) * log10_t_sigma + log10_t_mu
        t = 10**log10_t
        return t
    
time_sampler = TimeSampler()


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, x_0, x_t, t):
        loss_weights = 1/t
        loss_weights = loss_weights.reshape(-1, 1, 1, 1)
        return torch.mean(loss_weights*(x_0 - x_t)**2)

weighted_mse_loss_fn = WeightedMSELoss()


unconditional_diffusion_model = gmi.diffusion.DiffusionModel(diffusion_backbone, forward_SDE=forward_SDE, training_loss_fn=weighted_mse_loss_fn, training_time_sampler=time_sampler)
ve_weights_path = os.path.join(medmnist_example_dir, 'unconditional_diffusion_model_VE.pth')
if os.path.exists(ve_weights_path):
    # Cast to clarify the type for the linter
    backbone = cast(gmi.network.DiffusersUnet2D_Size28, unconditional_diffusion_model.diffusion_backbone)
    backbone.load_state_dict(torch.load(ve_weights_path))
    print(f"Loaded VE pre-trained weights from {ve_weights_path}")
else:
    print(f"VE pre-trained weights not found at {ve_weights_path}. Using random initialization and/or will train from scratch.")


if not os.path.exists(medmnist_example_dir + '/unconditional_diffusion_model.pth'):
    # train the denoiser using the new train_diffusion_model method
    # Parameters aligned with modular_configs/diffusion_training_config.yaml
    train_losses, val_losses, eval_metrics = unconditional_diffusion_model.train_diffusion_model(
        dataset=dataset_train,
        val_data=dataset_val,
        test_data=dataset_test,
        train_batch_size=64,  # Reduced from batch_size for memory efficiency (modular configs uses 64)
        val_batch_size=64,
        test_batch_size=64,
        train_num_workers=2,  # Reduced from 4 to match modular configs
        val_num_workers=2,
        test_num_workers=2,
        shuffle_train=True,
        shuffle_val=True,  # Changed to True to match modular configs
        shuffle_test=False,
        num_epochs=1000,  # Increased from 10 to 500 for better convergence (modular configs uses 500)
        num_iterations_train=100,
        num_iterations_val=10,
        num_iterations_test=5,  # Increased from 1 to 5 for better test evaluation
        learning_rate=1e-4,  # Reduced from 1e-3 to 1e-4 for diffusion models (modular configs uses 1e-4)
        use_ema=True,
        ema_decay=0.999,
        early_stopping=True,
        patience=100,  # Increased from 10 to 50 for convergence-based early stopping
        val_loss_smoothing=0.9,
        min_delta=1e-6,
        verbose=True,
        very_verbose=False,
        save_checkpoints=True,
        experiment_name="medmnist_diffusion_experiment",
        output_dir=medmnist_example_dir,
        epochs_per_evaluation=None,  # Added to match modular configs structure
        test_save_plots=True,
        test_plot_vmin=0,
        test_plot_vmax=1,
        final_test_iterations='20',  # Added to match modular configs (runs final test on subset)
        # Reverse process sampling parameters from modular configs
        reverse_t_start=1.0,
        reverse_t_end=0.0,
        reverse_spacing='linear',
        reverse_sampler='euler',
        reverse_timesteps=50,
        # WandB logging from modular configs
        wandb_project="gmi-medmnist-diffusion",  # Enable WandB logging
        wandb_config=None
    )

    # save weights
    backbone = cast(gmi.network.DiffusersUnet2D_Size28, unconditional_diffusion_model.diffusion_backbone)
    torch.save(backbone.state_dict(), medmnist_example_dir + '/unconditional_diffusion_model.pth')

else:
    # load weights
    backbone = cast(gmi.network.DiffusersUnet2D_Size28, unconditional_diffusion_model.diffusion_backbone)
    backbone.load_state_dict(torch.load(medmnist_example_dir + '/unconditional_diffusion_model.pth'))



T = 1000.0
t0 = 0.0001
nSteps=512

x_0 = dataset_test[0].view(1,3,28,28).to(device)

x_t = unconditional_diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, t=torch.tensor([T]).to(device))

# timesteps = torch.linspace(1.0, 0.0, nSteps).to(device)**5.0 *(T-t0) + t0
# go logarithmically
timesteps = torch.logspace(np.log10(T), np.log10(t0), nSteps).to(device)

unconditional_diffusion_model.eval()
x_t_all = unconditional_diffusion_model.sample_reverse_process(x_t, timesteps, sampler='euler', return_all=True, verbose=True)

from matplotlib import animation
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(xlim=(0, 28), ylim=(0, 28))

# Handle RGB images properly - transpose from (C, H, W) to (H, W, C) for matplotlib
def get_rgb_image(tensor):
    """Convert tensor from (C, H, W) to (H, W, C) format for RGB display"""
    # tensor shape: (1, 3, 28, 28) -> take first sample and transpose
    img = tensor[0].cpu().detach().numpy()  # (3, 28, 28)
    img = np.transpose(img, (1, 2, 0))     # (28, 28, 3)
    return img

# Calculate dynamic scaling based on timesteps
def get_dynamic_scale(t_idx):
    """Calculate dynamic vmin and vmax based on timestep index"""
    # Get the current timestep value
    t = timesteps[t_idx].item()
    # Calculate noise scale: sqrt(t) represents the noise level
    noise_scale = np.sqrt(t)
    # Dynamic range: vmin = 0 - 3*noise_scale, vmax = 1 + 3*noise_scale
    vmin = 0 - 3 * noise_scale
    vmax = 1 + 3 * noise_scale
    return vmin, vmax

# Initialize with first frame
first_img = get_rgb_image(x_t_all[0])
vmin, vmax = get_dynamic_scale(0)
im = ax.imshow(first_img, vmin=vmin, vmax=vmax)

# Add colorbar to show the dynamic scaling
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pixel Value')

def update(iFrame):
    print('Animating frame ', iFrame)
    jFrame = (nSteps//128)*iFrame
    # Get image for current frame
    img = get_rgb_image(x_t_all[jFrame])
    # Get dynamic scaling for current timestep
    vmin, vmax = get_dynamic_scale(jFrame)
    current_t = timesteps[jFrame].item()
    # Update both the image data and the color scale
    im.set_array(img)
    im.set_clim(vmin, vmax)
    # Update colorbar label to show current timestep and range
    cbar.set_label(f'Pixel Value (t={current_t:.3f}, range=[{vmin:.2f}, {vmax:.2f}])')
    return im,

ani = animation.FuncAnimation(fig, update, frames=128, interval=100, blit=True)
writer = animation.writers['ffmpeg'](fps=15)
ani.save(medmnist_example_dir + '/reverse_diffusion.mp4', writer=writer)

print()



# images, measurements, reconstructions = mnist_denoising_task.sample_images_measurements_reconstructions(
#                                     image_batch_size=9,
#                                     measurement_batch_size=9,
#                                     reconstruction_batch_size=9)



# from matplotlib import pyplot as plt

# images = images.cpu().detach().numpy()
# measurements = measurements.cpu().detach().numpy()
# reconstructions = reconstructions.cpu().detach().numpy()

# fig = plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1)
#     # Handle RGB images - transpose from (C, H, W) to (H, W, C)
#     img = images[0,0,i].cpu().detach().numpy()  # (3, 28, 28)
#     img = np.transpose(img, (1, 2, 0))         # (28, 28, 3)
#     img = np.clip(img, 0, 1)                   # Clip to [0, 1] range
#     ax.imshow(img, vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/images.png')

# fig = plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1)
#     # Handle RGB images - transpose from (C, H, W) to (H, W, C)
#     img = measurements[0,0,i].cpu().detach().numpy()  # (3, 28, 28)
#     img = np.transpose(img, (1, 2, 0))               # (28, 28, 3)
#     img = np.clip(img, 0, 1)                         # Clip to [0, 1] range
#     ax.imshow(img, vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/measurements.png')

# fig = plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1)
#     # Handle RGB images - transpose from (C, H, W) to (H, W, C)
#     img = reconstructions[0,0,i].cpu().detach().numpy()  # (3, 28, 28)
#     img = np.transpose(img, (1, 2, 0))                  # (28, 28, 3)
#     img = np.clip(img, 0, 1)                            # Clip to [0, 1] range
#     ax.imshow(img, vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/reconstructions.png')




