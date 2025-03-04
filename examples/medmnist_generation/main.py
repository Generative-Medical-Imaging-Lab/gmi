from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms
from diffusers import UNet2DModel


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

medmnist_name = 'OrganAMNIST'
batch_size = 32

medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))

# Create the dataset with the target_transform set to discard the label

dataset_train = gmi.datasets.MedMNIST(medmnist_name, 
                                      split='train', 
                                      download=True, 
                                      images_only=True)

dataset_val = gmi.datasets.MedMNIST(medmnist_name,
                                    split='val',
                                    download=True,
                                    images_only=True)

dataset_test = gmi.datasets.MedMNIST(medmnist_name,
                                        split='test',
                                        download=True,
                                        images_only=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                num_workers=16,
                                                drop_last=True)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=16)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)

forward_SDE = gmi.sde.SongVarianceExplodingSDE(noise_variance = lambda t: t, noise_variance_prime = lambda t: t*0.0 + 1.0)

class DiffusionBackbone(torch.nn.Module):
    def __init__(self):
        super(DiffusionBackbone, self).__init__()
        
        # Create a UNet2DModel for noise prediction given x_t and t
        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            norm_num_groups=1,
            block_out_channels=(32, 64, 128),
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
        )

    def forward(self, x_t, t):
        x_t = torch.nn.functional.pad(x_t, (2, 2, 2, 2), mode='constant', value=0)
        x_0_pred = self.unet(x_t, t.squeeze())[0]
        x_0_pred = x_0_pred[:,:,2:30,2:30]
        return x_0_pred


diffusion_backbone = DiffusionBackbone().to(device)

unconditional_diffusion_model = gmi.diffusion.DiffusionModel(forward_SDE, diffusion_backbone)
unconditional_diffusion_model.diffusion_backbone.unet.load_state_dict(torch.load(medmnist_example_dir + '/unconditional_diffusion_model_VE.pth'))



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


loss_closure = unconditional_diffusion_model.loss_closure(time_sampler=time_sampler)


if not os.path.exists(medmnist_example_dir + '/unconditional_diffusion_model.pth'):
    # train the denoiser
    gmi.train(  dataloader_train, 
                loss_closure, 
                num_epochs=100, 
                num_iterations=1000,
                optimizer=None,
                lr=1e-3, 
                device=device, 
                validation_loader=dataloader_val, 
                num_iterations_val=10,
                verbose=True)


    # save weights
    torch.save(unconditional_diffusion_model.diffusion_backbone.unet.state_dict(), medmnist_example_dir + '/unconditional_diffusion_model.pth')

else:
    # load weights
    unconditional_diffusion_model.diffusion_backbone.unet.load_state_dict(torch.load(medmnist_example_dir + '/unconditional_diffusion_model.pth'))

T = 1000.0
t0 = 0.0001
nSteps=512

x_0 = dataset_test[0][0].view(1,1,28,28).to(device)

x_t = unconditional_diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, t=torch.tensor([T]).to(device))

timesteps = torch.linspace(1.0, 0.0, nSteps).to(device)**5.0 *(T-t0) + t0

unconditional_diffusion_model.eval()
x_t_all = unconditional_diffusion_model.sample_reverse_process(x_t, timesteps, sampler='euler', return_all=True, verbose=True)

from matplotlib import animation
from matplotlib import pyplot as plt

fig = plt.figure()
ax = plt.axes(xlim=(0, 28), ylim=(0, 28))
im = ax.imshow(x_t_all[0][0,0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)

def update(iFrame):
    print('Animating frame ', iFrame)
    jFrame = (nSteps//128)*iFrame
    im.set_array(x_t_all[jFrame][0,0].cpu().detach().numpy())
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
#     ax.imshow(images[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/images.png')

# fig = plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1)
#     ax.imshow(measurements[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/measurements.png')

# fig = plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1)
#     ax.imshow(reconstructions[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
#     ax.axis('off')
# plt.savefig(medmnist_example_dir + '/reconstructions.png')




