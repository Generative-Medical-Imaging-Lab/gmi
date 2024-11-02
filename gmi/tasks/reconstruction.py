
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage

from ..samplers import Sampler, DatasetSampler, DataLoaderSampler, ModuleSampler

class ImageReconstructionTask(nn.Module):
    def __init__(self, 
                 image_dataset,
                 measurement_simulator,
                 image_reconstructor,
                 task_evaluator='rmse',
                 device=None):
        
        # initialize the parent class
        super(ImageReconstructionTask, self).__init__()
        
        # if image_dataset is a torch.utils.data.Dataset, convert it to a DatasetSampler
        if isinstance(image_dataset, torch.utils.data.Dataset):
            image_dataset = DatasetSampler(image_dataset)

        # if image_dataset is a torch.utils.data.DataLoader, convert it to a DataLoaderSampler
        if isinstance(image_dataset, torch.utils.data.DataLoader):
            image_dataset = DataLoaderSampler(image_dataset)

        # if measurement_simulator is a torch.nn.Module, convert it to a ModuleSampler
        if isinstance(measurement_simulator, torch.nn.Module):
            measurement_simulator = ModuleSampler(measurement_simulator)

        # if image_reconstructor is a torch.nn.Module, convert it to a ModuleSampler
        if isinstance(image_reconstructor, torch.nn.Module):
            image_reconstructor = ModuleSampler(image_reconstructor)

        # assert that image_dataset, measurement_simulator, and image_reconstructor are instances of Sampler
        assert isinstance(image_dataset, Sampler), 'image_dataset must be an instance of torch.utils.data.Dataset, torch.utils.data.DataLoader, or gmi.Sampler'
        assert isinstance(measurement_simulator, Sampler), 'measurement_simulator must be an instance of torch.nn.Module or gmi.Sampler'
        assert isinstance(image_reconstructor, Sampler), 'image_reconstructor must be an instance of torch.nn.Module or gmi.Sampler'

        # set the image_dataset, measurement_simulator, and image_reconstructor
        self.image_dataset = image_dataset
        self.measurement_simulator = measurement_simulator
        self.image_reconstructor = image_reconstructor
        
        # set the task_evaluator
        if isinstance(task_evaluator, str):
            if task_evaluator == 'mse':
                self.task_evaluator = nn.MSELoss()
            elif task_evaluator == 'rmse':
                self.task_evaluator = lambda x, y: torch.sqrt(nn.MSELoss()(x, y))
            else:
                opts_lst = ['mse', 'rmse']
                raise ValueError(f"task_metric must be one of {opts_lst}")
            
        if isinstance(task_evaluator, nn.Module):
            self.task_evaluator = task_evaluator

        self.device=device

    def sample_images(self, image_batch_size):
        # call the image_dataset sampler
        images = self.image_dataset.sample(image_batch_size)
        assert isinstance(images, torch.Tensor), 'image_dataset.sample() must return a torch.Tensor'
        if self.device is not None:
            images = images.to(self.device)
        return images
    
    def sample_measurements_given_images(self, measurement_batch_size, images):
        # call the measurement_simulator conditional sampler
        measurements = self.measurement_simulator.sample(measurement_batch_size, images)
        assert isinstance(measurements, torch.Tensor), 'measurement_simulator.sample() must return a torch.Tensor'
        if self.device is not None:
            measurements = measurements.to(self.device)
        return measurements
    
    def sample_reconstructions_given_measurements(self, reconstruction_batch_size, measurements):
        # call the image_reconstructor conditional sampler
        reconstructions = self.image_reconstructor(reconstruction_batch_size, measurements)
        assert isinstance(reconstructions, torch.Tensor), 'image_reconstructor.sample() must return a torch.Tensor'
        if self.device is not None:
            reconstructions = reconstructions.to(self.device)
        return reconstructions
    
    def sample_images_measurements(self, image_batch_size, measurement_batch_size):
        # call the image_dataset sampler
        images = self.sample_images(image_batch_size)
        # call the measurement_simulator conditional sampler
        measurements = self.sample_measurements_given_images(measurement_batch_size, images)
        assert isinstance(measurements, torch.Tensor), 'sample_measurements_given_images() must return a torch.Tensor'
        # add a dimension for the measurement batches
        images.unsqueeze(0) 
        return images, measurements
    
    def sample_images_measurements_reconstructions(self, image_batch_size, measurement_batch_size, reconstruction_batch_size):
        # call the image_dataset sampler and the measurement_simulator conditional sampler
        images, measurements = self.sample_images_measurements(image_batch_size, measurement_batch_size)
        # combine the image and measurement batch dimensions into one batch dimension
        measurements = measurements.view(image_batch_size*measurement_batch_size, *measurements.shape[2:])
        # call the image_reconstructor conditional sampler
        reconstructions = self.sample_reconstructions_given_measurements(reconstruction_batch_size, measurements)
        assert isinstance(reconstructions, torch.Tensor), 'sample_reconstructions_given_measurements() must return a torch.Tensor'
        # reshape the images to have reconstruction, measurement, and image batch dimensions
        images = images.view(1,1,image_batch_size, *images.shape[1:])
        # reshape the measurements to have reconstruction, measurement, and image batch dimensions
        measurements = measurements.view(1, measurement_batch_size, image_batch_size, *measurements.shape[1:])
        # reshape the reconstructions to have reconstruction, measurement, and image batch dimensions
        reconstructions = reconstructions.view(reconstruction_batch_size, measurement_batch_size, image_batch_size, *reconstructions.shape[2:])
        return images, measurements, reconstructions

    def forward(self, image_batch_size, measurement_batch_size, reconstruction_batch_size):
        return self.sample_images_measurements_reconstructions(image_batch_size, measurement_batch_size, reconstruction_batch_size)

    def train_reconstructor(self, batch_size, num_epochs=100, num_iterations=100, lr=None, optimizer=None, verbose=True, **kwargs):

        if optimizer is None:
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.Adam(self.image_reconstructor.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                true_images, _, reconstructions = self.sample_images_measurements_reconstructions(image_batch_size=batch_size, measurement_batch_size=1, reconstruction_batch_size=1)
                assert isinstance(true_images, torch.Tensor)
                assert isinstance(reconstructions, torch.Tensor)
                reconstructions = reconstructions.reshape(true_images.shape)
                loss = self.task_evaluator(true_images, reconstructions)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch}: Loss: {loss.item()}")
        
        return
    








from ..diffusion import UnconditionalDiffusionModel

class DiffusionBridgeImageReconstructor(nn.Module):
            def __init__(self, initial_reconstructor, diffusion_model, final_reconstructor):
                super(DiffusionBridgeImageReconstructor, self).__init__()
                assert isinstance(initial_reconstructor, nn.Module)
                assert isinstance(diffusion_model, UnconditionalDiffusionModel)
                assert isinstance(final_reconstructor, nn.Module)
                self.initial_reconstructor = initial_reconstructor
                self.diffusion_model = diffusion_model
                self.final_reconstructor = final_reconstructor
                
            def forward(self,measurements,timesteps=None, num_timesteps=None, sampler='euler', verbose=False):

                assert isinstance(self.initial_reconstructor, nn.Module)
                assert isinstance(self.diffusion_model, UnconditionalDiffusionModel)
                assert isinstance(self.final_reconstructor, nn.Module)

                x_1 = self.initial_reconstructor(measurements)

                if timesteps is None:
                    if num_timesteps is None:
                        num_timesteps = 32
                    timesteps = torch.linspace(1, 0, num_timesteps+1).to(x_1.device)
              
                assert isinstance(timesteps, torch.Tensor)
                assert timesteps.ndim == 1
                assert timesteps.shape[0] > 1
                assert timesteps[0] == 1.0
                assert timesteps[-1] == 0.0
                for i in range(1, timesteps.shape[0]):
                    assert timesteps[i] < timesteps[i-1]

                x_0 = self.diffusion_model.reverse_sample(x_1, timesteps, sampler=sampler, return_all=False, verbose=verbose)
                
                reconstructions = self.final_reconstructor(x_0)
                return reconstructions

class DiffusionBridgeModel(ImageReconstructionTask):
    def __init__(self, 
                 image_dataset,
                 measurement_simulator,
                 image_reconstructor,                 
                 task_evaluator='rmse'):
        
        assert isinstance(image_reconstructor, DiffusionBridgeImageReconstructor)
            
        super(DiffusionBridgeModel, self).__init__(image_dataset, 
                                                    measurement_simulator, 
                                                    image_reconstructor, 
                                                    task_evaluator=task_evaluator)
        
    def train_diffusion_backbone(self, 
                                 *args, 
                                 num_epochs=100, 
                                 num_iterations_per_epoch=100, 
                                 num_epochs_per_save=None, 
                                 weights_filename=None,
                                 optimizer=None, 
                                 verbose=True, 
                                 time_sampler=None,
                                 ema=False,
                                 **kwargs):
                
        assert isinstance(self.image_reconstructor, DiffusionBridgeImageReconstructor)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.image_reconstructor.diffusion_model.diffusion_backbone.parameters(), lr=1e-3)

        if ema:
            ema =  ExponentialMovingAverage(self.image_reconstructor.diffusion_model.diffusion_backbone.parameters(), decay=0.995)
        
        if time_sampler is None:
            assert isinstance(self.image_reconstructor.diffusion_model.diffusion_backbone, torch.nn.Module)
            time_sampler = lambda x_shape: torch.rand(x_shape)

        if num_epochs_per_save is not None:
            assert weights_filename is not None
            assert isinstance(weights_filename, str)

        train_loss = torch.zeros(num_epochs, dtype=torch.float32)

        for epoch in range(num_epochs):
            loss_sum = 0
            for iteration in range(num_iterations_per_epoch):
                optimizer.zero_grad()
                x_0 = self.sample_images(*args, **kwargs)
                batch_size = x_0.shape[0]
                t = time_sampler((batch_size, 1)).to(x_0.device)
                noise = torch.randn_like(x_0)
                x_t = self.image_reconstructor.diffusion_model.sample_x_t_given_x_0_and_noise(x_0, noise, t) # forward process
                
                
                # x_0_pred = self.image_reconstructor.diffusion_model.predict_x_0_given_x_t(x_t, t) # reverse prediction
                # loss = torch.mean((x_0_pred - x_0)**2.0)

                noise_pred = self.image_reconstructor.diffusion_model.predict_noise_given_x_t(x_t, t) # reverse prediction
                loss = torch.mean((noise_pred - noise)**2.0)

                soft_tissue_mask = x_0 < 1.5
                loss += 9.0*torch.mean((noise_pred[soft_tissue_mask] - noise[soft_tissue_mask])**2.0)


                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

                if ema:
                    ema.update()

            if num_epochs_per_save is not None and epoch % num_epochs_per_save == 0:
                torch.save(self.image_reconstructor.diffusion_model.diffusion_backbone.state_dict(), weights_filename)

            
            train_loss[epoch] = loss_sum/num_iterations_per_epoch
            
            if verbose:
                print(f"Epoch {epoch}: Loss: {train_loss[epoch].item()}")

        return train_loss
    




