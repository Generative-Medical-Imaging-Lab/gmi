

import torch
import gmi


import os
mnist_denoising_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define the dataset
MNIST_Dataset_train = gmi.datasets.MNIST(
                                    train=True, 
                                    download=True,
                                    images_only=True)

MNIST_Dataset_test = gmi.datasets.MNIST(
                                    train=False, 
                                    download=True,
                                    images_only=True)

# define the measurement simulator
white_noise_adder = gmi.distributions.AdditiveWhiteGaussianNoise(
                                                    noise_variance=0.1)

# define the image reconstructor
densenet_denoiser = gmi.networks.DenseNet(
                                    input_shape=(1, 28, 28), 
                                    output_shape=(1, 28, 28), 
                                    hidden_channels_list=[1024, 1024, 1024], 
                                    activation=torch.nn.SiLU()).to(device)

# define the denoising task
mnist_denoising_task = gmi.tasks.ImageReconstructionTask(
                                        image_dataset = MNIST_Dataset_train,
                                        measurement_simulator = white_noise_adder,
                                        image_reconstructor = densenet_denoiser,
                                        task_evaluator = 'rmse',
                                        device=device)

# train the denoiser
batch_size = 128
mnist_denoising_task.train_reconstructor(
                                    batch_size, 
                                    num_epochs=100, 
                                    num_iterations=100, 
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
    ax.imshow(images[0,0,i][0], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(mnist_denoising_dir + '/images.png')

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(measurements[0,0,i][0], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(mnist_denoising_dir + '/measurements.png')

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(reconstructions[0,0,i][0], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.savefig(mnist_denoising_dir + '/reconstructions.png')



