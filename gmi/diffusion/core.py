import torch
from ..sde import LinearSDE


class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 forward_SDE,
                 diffusion_backbone
                 ):
        """
        This is an abstract base class for diffusion models.
        """

        assert isinstance(forward_SDE, LinearSDE)
        assert isinstance(diffusion_backbone, torch.nn.Module)

        super(DiffusionModel, self).__init__()

        self.diffusion_backbone = diffusion_backbone
        self.forward_SDE = forward_SDE

    def forward(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return self.forward_SDE.sample_x_t_given_x_0(x_0, t)
    
    def sample_forward_process(self, x_t, timesteps, sampler='euler', return_all=False):
        """
        This method samples from the forward SDE.

        parameters:
            x_t: torch.Tensor
                The initial condition.
            timesteps: int
                The number of timesteps to sample.
            sampler: str
                The method used to compute the forward update. Currently, only 'euler' and 'heun' are supported.
        returns:
            x: torch.Tensor
                The output tensor.
        """

        return self.forward_SDE.sample(x_t, timesteps, sampler, return_all)

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




        # denoising preconditioners

        # if c_skip is None:
        #     c_skip = lambda t,x_shape: 0.0
        
        # if c_out is None:
        #     c_out = lambda t,x_shape: 1.0

        # if c_in is None:
        #     c_in = lambda t,x_shape: 1.0

        # if c_noise is None:
        #     c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))


        
        # SGM preconditioners

        # if c_skip is None:
        #     c_skip = lambda t,x_shape: 1.0
        
        # if c_out is None:
        #     c_out = lambda t,x_shape: torch.sqrt(expand_t_to_x_shape(t, x_shape))

        # if c_in is None:
        #     c_in = lambda t,x_shape: 1.0

        # if c_noise is None:
        #     c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))




        # EDM preconditioners

        # var_data = 0.25

        # if c_skip is None:
        #     c_skip = lambda t,x_shape: var_data/(var_data + expand_t_to_x_shape(t,x_shape))
        
        # if c_out is None:
        #     c_out = lambda t,x_shape: torch.sqrt(expand_t_to_x_shape(t, x_shape)) * torch.sqrt(var_data/(var_data + expand_t_to_x_shape(t,x_shape)))

        # if c_in is None:
        #     c_in = lambda t,x_shape: 1.0 / torch.sqrt(var_data + expand_t_to_x_shape(t,x_shape))

        # if c_noise is None:
        #     c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))



        # custom preconditioners
        # testing if time dependence of previous case is the issue

        # var_data = 0.25

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



        # if self.pass_t_to_x_0_predictor:
        #     if y is not None:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, y_embedding, t.squeeze())
        #     else:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, t.squeeze())
        # else:
        #     if y is not None:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, y_embedding)
        #     else:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding)


        # x_0_pred = x_out
        

        return x_0_pred



