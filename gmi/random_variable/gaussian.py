import torch
from torch import nn

import math

from .core import RandomVariable

from ..linear_system import LinearSystem, Scalar

class GaussianRandomVariable(RandomVariable):
    def __init__(self, mu, Sigma):
        super(GaussianRandomVariable, self).__init__()

        assert isinstance(mu, torch.Tensor)
        assert isinstance(Sigma, LinearSystem)

        self.mu = mu
        self.Sigma = Sigma

    def sample(self):
        white_noise = torch.randn(self.mu.shape, device=self.mu.device)
        sqrt_Sigma = self.Sigma.sqrt_LinearSystem()
        correlated_noise =  sqrt_Sigma(white_noise)
        return self.mu + correlated_noise
    
    def mahalanobis_distance(self, x):
        res = (x - self.mu)
        weighted_res = self.Sigma.inv_LinearSystem() @ res
        return torch.sum(res * weighted_res)
    
    def log_prob(self, x):
        d  = torch.prod(torch.tensor(self.mu.shape)).float()
        constant_term = - d * torch.log(2 * torch.tensor([3.141592653589793]))
        log_det = self.Sigma.logdet()
        mahalanobis_distance = self.mahalanobis_distance(x)
        return 0.5 * constant_term - 0.5 * log_det - 0.5 * mahalanobis_distance
    
    def log_prob_plus_constant(self, x):
        mahalanobis_distance = self.mahalanobis_distance(x)
        return  - 0.5 * mahalanobis_distance
    
    def score(self, x):
        return self.Sigma.inv_LinearSystem() @ (self.mu - x)
        


class ConditionalGaussianRandomVariable(RandomVariable):
    def __init__(self, mu_fn, Sigma_fn):
        super(ConditionalGaussianRandomVariable, self).__init__()

        self.mu_fn = mu_fn
        self.Sigma_fn = Sigma_fn
    
    def evaluate(self, y):
        mu = self.mu_fn(y)
        Sigma = self.Sigma_fn(y)
        return GaussianRandomVariable(mu, Sigma)
    
    def sample(self, y, *args, **kwargs):
        return self.evaluate(y).sample(*args, **kwargs)
    
    def mahalanobis_distance(self, y, x):
        return self.evaluate(y).mahalanobis_distance(x)
    
    def log_prob(self, y, x):
        return self.evaluate(y).log_prob(x)
    
    def log_prob_plus_constant(self, y, x):
        return self.evaluate(y).log_prob_plus_constant(x)
    
    def score(self, y, x):
        return self.evaluate(y).score(x)
    


class LinearSystemGaussianNoise(ConditionalGaussianRandomVariable):
    def __init__(self, linear_system, noise_covariance):
        mu_fn = lambda y: linear_system(y)
        Sigma_fn = lambda y: noise_covariance
        super(LinearSystemGaussianNoise, self).__init__(mu_fn, Sigma_fn)
        self.linear_system = linear_system
        self.noise_covariance = noise_covariance



class AdditiveWhiteGaussianNoise(ConditionalGaussianRandomVariable):
    def __init__(self, noise_standard_deviation):
        noise_variance = noise_standard_deviation ** 2
        noise_covariance_linear_system = Scalar(noise_variance)
        mu_fn = lambda y: y
        Sigma_fn = lambda y: noise_covariance_linear_system
        super(AdditiveWhiteGaussianNoise, self).__init__(mu_fn, Sigma_fn)
        self.noise_variance = noise_variance





class TrainableGaussian(RandomVariable):
    def __init__(self, dim):
        super(TrainableGaussian, self).__init__()
        self.dim = dim
        self.mu = torch.nn.Parameter(torch.zeros(dim))
        self.sigma = torch.nn.Parameter(torch.eye(dim))
    def get_Sigma(self):
        return self.sigma @ self.sigma.T
    def set_Sigma(self, value):
        self.sigma.data = torch.linalg.cholesky(value).data
    Sigma = property(get_Sigma, set_Sigma)
    def get_invSigma(self):
        try:
            inv_Sigma = torch.linalg.inv(self.Sigma)
        except Exception as e:
            print(f"Warning, Error in computing invSigma, adding small identity for stability")
            inv_Sigma = torch.linalg.inv(self.Sigma + 1e-3 * torch.eye(self.dim, device=self.Sigma.device))
        return inv_Sigma

    def set_invSigma(self, value):
        self.Sigma = torch.linalg.inv(value)
    invSigma = property(get_invSigma, set_invSigma)
    def sample(self, batch_size):
        white_noise = torch.randn((batch_size, len(self.mu)), device=self.mu.device)
        correlated_noise = white_noise @ self.sigma.T
        return self.mu.unsqueeze(0) + correlated_noise
    def log_prob(self, x):
       assert x.dim() == 2 # (batch_size, dim)
       assert x.size(1) == self.dim
       batch_size = x.size(0)
       constant_term = torch.ones([], device=x.device) * self.dim * 2* math.log(2 * math.pi)
       log_det = torch.logdet(self.Sigma)
       res = x - self.mu.unsqueeze(0)
       weighted_res = torch.einsum('bi,ij->bj', res, self.invSigma)
       mahalanobis_distance = torch.einsum('bi,bi->b', res, weighted_res).view(batch_size,1,1)
       return -0.5 * (constant_term + log_det + mahalanobis_distance).view(batch_size)
    def score(self, x):
        # score is -invSigma @ (x - mu)
        res = x - self.mu.unsqueeze(0)
        return torch.einsum('bi,ij->bj', res, -self.invSigma)