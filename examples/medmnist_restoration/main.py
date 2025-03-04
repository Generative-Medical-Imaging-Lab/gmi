from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

medmnist_name = 'OrganAMNIST'
batch_size = 2048

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
                                                num_workers=1)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)


# define the measurement simulator
white_noise_adder = gmi.distribution.AdditiveWhiteGaussianNoise(
                                                    noise_standard_deviation=0.1)

denoiser = gmi.network.SimpleCNN(input_channels=1,
                                    output_channels=1,
                                    hidden_channels_list=[16, 32, 64, 128, 256, 128, 64, 32, 16],
                                    activation=torch.nn.SiLU(),
                                    dim=2).to(device)

# define the denoising task
mnist_denoising_task = gmi.tasks.ImageReconstructionTask(
                                        image_dataset = dataset_train,
                                        measurement_simulator = white_noise_adder,
                                        image_reconstructor = denoiser,
                                        device=device)


loss_closure = mnist_denoising_task.loss_closure(torch.nn.MSELoss())

# train the denoiser
gmi.train(  dataloader_train, 
            loss_closure, 
            num_epochs=100, 
            num_iterations=100,
            optimizer=None,
            lr=1e-3, 
            device=device, 
            validation_loader=dataloader_val, 
            num_iterations_val=10,
            verbose=True)




images, measurements, reconstructions = mnist_denoising_task.sample_images_measurements_reconstructions(
                                    image_batch_size=9,
                                    measurement_batch_size=9,
                                    reconstruction_batch_size=9)



from matplotlib import pyplot as plt

images = images.cpu().detach().numpy()
measurements = measurements.cpu().detach().numpy()
reconstructions = reconstructions.cpu().detach().numpy()

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(images[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(medmnist_example_dir + '/images.png')

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(measurements[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(medmnist_example_dir + '/measurements.png')

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(reconstructions[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(medmnist_example_dir + '/reconstructions.png')




