import torch
from torch import nn
from .core import Distribution
from ..linear_operator import LinearOperator, ScalarLinearOperator


class LogNormalDistribution(Distribution):
    def __init__(self, mu, Sigma):
        super(LogNormalDistribution, self).__init__()

        if isinstance(mu, int):
            mu = float(mu)

        if isinstance(Sigma, int):
            Sigma = float(Sigma)

        if isinstance(mu, float):
            mu = torch.tensor(mu)

        if isinstance(Sigma, float):
            Sigma = ScalarLinearOperator(Sigma)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(Sigma, LinearOperator)

        self.mu = mu


        self.Sigma = Sigma

    def sample(self, batch_size):
        total_shape = [batch_size] + list(self.mu.shape)
        white_noise = torch.randn(total_shape, device=self.mu.device)
        sqrt_Sigma = self.Sigma.sqrt_LinearOperator()
        correlated_noise =  sqrt_Sigma @ white_noise
        return torch.exp(self.mu + correlated_noise)
    def log_prob(self, x):
        raise NotImplementedError("LogNormalDistribution does not support log_prob yet")