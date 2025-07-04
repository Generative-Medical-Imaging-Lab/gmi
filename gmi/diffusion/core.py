import torch
from torch import nn
from ..sde import LinearSDE, StandardWienerSDE
from ..distribution import UniformDistribution
from ..samplers import Sampler
from ..linalg import InvertibleLinearOperator


class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 diffusion_backbone,
                 forward_SDE=None,
                 training_loss_fn=None,
                 training_time_sampler=None,
                 training_time_uncertainty_sampler=None):
        """
        This is an abstract base class for diffusion models.
        """

        assert isinstance(diffusion_backbone, torch.nn.Module)

        super(DiffusionModel, self).__init__()

        if forward_SDE is None:
            forward_SDE = StandardWienerSDE()

        if training_loss_fn is None:
            training_loss_fn = torch.nn.MSELoss()

        if training_time_sampler is None:
            training_time_sampler = UniformDistribution(0.0, 1.0)

        if training_time_uncertainty_sampler is None:
            class IdentitySampler(Sampler):
                def sample(self, t):
                    return t
            training_time_uncertainty_sampler = IdentitySampler()

        assert isinstance(forward_SDE, LinearSDE)
        assert isinstance(diffusion_backbone, torch.nn.Module)
        assert isinstance(training_time_sampler, Sampler)
        assert isinstance(training_loss_fn, torch.nn.Module)

        self.diffusion_backbone = diffusion_backbone
        self.forward_SDE = forward_SDE
        self.training_loss_fn = training_loss_fn
        self.training_time_sampler = training_time_sampler
        self.training_time_uncertainty_sampler = training_time_uncertainty_sampler


    # use union of two data types for typing
    def forward(self, batch_data: torch.Tensor | list):
        """
        This method implements the training loss closure of the diffusion model.
        It computes the loss between the predicted x_0 and the true x_0.
        parameters:
            batch_data: torch.Tensor or list
                The input tensor to the linear operator.
                If it is a tensor, it is assumed to be the true x_0.
                If it is a list, it is assumed to be a tuple of (x_0, y).
        returns:
            loss: torch.Tensor
                The loss between the predicted x_0 and the true x_0.
        """

        if isinstance(batch_data, torch.Tensor):
            x_0 = batch_data
            y=None
        elif isinstance(batch_data, list):
            assert len(batch_data) == 2, "batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)"
            x_0 = batch_data[0]
            y = batch_data[1]
        else:
            raise ValueError("batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)")

        batch_size = x_0.shape[0]

        t = self.training_time_sampler.sample(batch_size).to(x_0.device)
        tau = self.training_time_uncertainty_sampler.sample(t).to(x_0.device)
        x_t = self.forward_SDE.sample_x_t_given_x_0(x_0, tau)
        x_0_pred = self.predict_x_0(x_t, t, y)
        loss = self.training_loss_fn(x_0, x_0_pred, t)
        return loss
        
    def sample_reverse_process(self, x_t, timesteps, sampler='euler', return_all=False, y=None, verbose=False):
        """
        This method samples from the reverse SDE.

        parameters:
            x_t: torch.Tensor
                The initial condition.
            timesteps: int
                The number of timesteps to sample.
            sampler: str
                The method used to compute the forward update. Currently, only 'euler' and 'heun' are supported.
            return_all: bool
                If True, the method returns all intermediate samples.
            y: torch.Tensor
                The conditional input to the reverse SDE.
        returns:
            x: torch.Tensor
                The output tensor.
        """
        # we assume the diffusion_backbone estimates the posterior mean of x_0 given x_t, t, and y
        def mean_estimator(x_t, t):
            return self.predict_x_0(x_t, t, y)

        # define the reverse SDE, based on the mean estimator,
        # Tweedies's formula to get the score function, 
        # Anderson's formula to get the reverse SDE
        reverse_SDE = self.forward_SDE.reverse_SDE_given_mean_estimator(mean_estimator)

        return reverse_SDE.sample(x_t, timesteps, sampler, return_all, verbose)
    
    def sample_reverse_process_DPS(self,    
                                x_t, 
                                log_likelihood_fn, 
                                timesteps, 
                                likelihood_weight=1.0,
                                jacobian_method='Backpropagation', # Backpropagation or Identity
                                sampler='euler', 
                                return_all=False, 
                                y=None, 
                                verbose=False):
        """
        Samples from the reverse SDE using diffusion posterior sampling (DPS).

        Parameters:
            x_t: torch.Tensor
                The initial condition.
            log_likelihood_fn: Callable
                Function that returns scalar log-likelihood given x_0.
            timesteps: int
                Number of timesteps to sample.
            likelihood_weight: float
                Weight on the likelihood score.
            jacobian_method: str
                One of ['Backpropagation', 'Identity']
            sampler: str
                Sampler method: 'euler' or 'heun'.
            return_all: bool
                If True, return all intermediate samples.
            y: torch.Tensor
                Optional conditional input.
        Returns:
            x: torch.Tensor
                Final output tensor.
        """

        def prior_score_estimator(x_t, t, x_0):
            Sigma_t = self.forward_SDE.Sigma(t)
            Sigma_t_inv = Sigma_t.inverse_LinearOperator()
            return Sigma_t_inv @ (x_0 - x_t)

        def posterior_score_estimator(x_t, t):
            x_t = x_t.detach().requires_grad_(True)
            x_0_pred = self.predict_x_0(x_t, t, y)
            x_0_pred.retain_grad()

            prior_score = prior_score_estimator(x_t, t, x_0_pred)

            # Zero any old gradients
            if x_t.grad is not None:
                x_t.grad.zero_()
            if x_0_pred.grad is not None:
                x_0_pred.grad.zero_()

            if jacobian_method == 'Backpropagation':
                log_likelihood = log_likelihood_fn(x_0_pred)
                log_likelihood.backward()
                if x_t.grad is None:
                    raise RuntimeError("x_t.grad is None. Did predict_x_0 disconnect x_t from the graph?")
                likelihood_score = x_t.grad.clone()
            elif jacobian_method == 'Identity':
                # Detach x_0_pred so backward does not go through predict_x_0
                x_0_pred_detached = x_0_pred.detach().requires_grad_(True)
                x_0_pred_detached.retain_grad()
                log_likelihood = log_likelihood_fn(x_0_pred_detached)
                log_likelihood.backward()
                if x_0_pred_detached.grad is None:
                    raise RuntimeError("x_0_pred_detached.grad is None. Check that x_0_pred_detached.requires_grad=True.")
                likelihood_score = x_0_pred_detached.grad.clone()
            else:
                raise ValueError(f"Unsupported jacobian_method: {jacobian_method}")

            posterior_score = prior_score + likelihood_weight * likelihood_score
            return posterior_score

        reverse_SDE = self.forward_SDE.reverse_SDE_given_score_estimator(posterior_score_estimator)
        return reverse_SDE.sample(x_t, timesteps, sampler, return_all, verbose)

            
    def predict_x_0(self, x_t: torch.Tensor, t: torch.Tensor, y=None):
        """
        This method predicts x_0 given x_t.

        parameters:
            x_t: torch.Tensor
                The sample at time t.
            t: float
                The time step.
        returns:
            x_0: torch.Tensor
                The predicted initial condition.
        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)

        if y is None:
            x_0_pred =  self.diffusion_backbone(x_t, t)
        else:
            x_0_pred =  self.diffusion_backbone(x_t, t, y)

        return x_0_pred

class DiffusionBackbone(torch.nn.Module):
    def __init__(self,
                 x_t_encoder,
                 t_encoder,
                 x_0_predictor,
                 y_encoder=None,
                 pass_t_to_x_0_predictor=False,
                 c_skip=None,
                 c_out=None,
                 c_in=None,
                 c_noise=None):
        
        """
        
        This is designed to implement a diffusion backbone. It predicts x_0 given x_t, t, and y embeddings.

        x_t is a sample from the forward or reverse diffusion process at time t, it is a tensor of shape [batch_size, *x_t.shape]
        t is the time step. We assume it is a tensor of shape [batch_size, 1]
        y is an optional conditional input to the diffusion backbone, it is a tensor of shape [batch_size, *y.shape]


        parameters:
            x_t_encoder: torch.nn.Module
                The neural network that encodes information from x_t.
            t_encoder: torch.nn.Module
                The neural network that encodes information from t.
            x_0_predictor: torch.nn.Module
                The neural network that predicts x_0 given x_t, t, and y embeddings.
            y_encoder: torch.nn.Module
                The optional neural network that encodes information from y.
        """
        
        assert isinstance(x_t_encoder, torch.nn.Module)
        assert isinstance(t_encoder, torch.nn.Module)
        assert isinstance(x_0_predictor, torch.nn.Module)

        if y_encoder is not None:
            assert isinstance(y_encoder, torch.nn.Module)

        super(DiffusionBackbone, self).__init__()

        self.x_t_encoder = x_t_encoder
        self.t_encoder = t_encoder
        self.x_0_predictor = x_0_predictor
        self.y_encoder = y_encoder

        self.pass_t_to_x_0_predictor = pass_t_to_x_0_predictor

        def expand_t_to_x_shape(t, x_shape):
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == x_shape[0], f"t.shape[0] = {t.shape[0]}, x_shape[0] = {x_shape[0]}"
            t_shape = [t.shape[0]] + [1]*len(x_shape[1:])
            t = t.reshape(t_shape)
            return t

        if c_skip is None:
            c_skip = lambda t,x_shape: 0.0
        
        if c_out is None:
            c_out = lambda t,x_shape: 1.0

        if c_in is None:
            c_in = lambda t,x_shape: 1.0
            
        if c_noise is None:
            c_noise = lambda t,x_shape: 1.0

        self.c_skip = c_skip
        self.c_out = c_out
        self.c_in = c_in
        self.c_noise = c_noise

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y=None):

        """
        This method implements the forward pass of the diffusion backbone.

        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)
        
        x_t_embedding = self.x_t_encoder(x_t)
        t_embedding = self.t_encoder(t)

        if y is not None:
                y_embedding = self.y_encoder(y)

        if self.pass_t_to_x_0_predictor:
            if y is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding, (self.c_noise(t,t.shape)*t).squeeze())
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, (self.c_noise(t,t.shape)*t).squeeze())
        else:
            if y is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding)
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding)


        x_0_pred = self.c_skip(t,x_t.shape)*x_t + self.c_out(t,x_out.shape)*x_out

        return x_0_pred



