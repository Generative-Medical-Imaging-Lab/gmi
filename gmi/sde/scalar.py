# random_tensor_laboratory/diffusion/sde.py

import torch
import torch.nn as nn
import torch
from torch.autograd.functional import jacobian
from .core import LinearSDE
from ..linalg import ScalarLinearOperator
import numpy as np


        
class ScalarSDE(LinearSDE):
    def __init__(self, signal_scale, noise_variance, signal_scale_prime=None, noise_variance_prime=None):
        """
        This class implements a scalar stochastic differential equation (SDE).

        dx = f(t) x dt + g(t) dw

        Parameters:
            signal_scale: callable
                Function that returns a scalar representing the system response.
            noise_variance: callable
                Function that returns a scalar representing the covariance.
            signal_scale_prime: callable, optional
                Function that returns the time derivative of signal_scale. If not provided, it will be computed automatically.
            noise_variance_prime: callable, optional
                Function that returns the time derivative of noise_variance. If not provided, it will be computed automatically.
        """

        # handle the default case where signal_scale_prime is not provided, use autograd
        if signal_scale_prime is None:
            signal_scale_prime = lambda t: torch.autograd.grad(signal_scale(t), t, create_graph=True)[0]

        # handle the default case where noise_variance_prime is not provided, use autograd
        if noise_variance_prime is None:
            noise_variance_prime = lambda t: torch.autograd.grad(noise_variance(t), t, create_graph=True)[0]

        H = lambda t: ScalarLinearOperator(signal_scale(t))
        Sigma = lambda t: ScalarLinearOperator(noise_variance(t))

        H_prime = lambda t: ScalarLinearOperator(signal_scale_prime(t))
        Sigma_prime = lambda t: ScalarLinearOperator(noise_variance_prime(t))

        super(ScalarSDE, self).__init__(H, Sigma, H_prime, Sigma_prime)



class SongVarianceExplodingSDE(ScalarSDE):
    def __init__(self, noise_variance, noise_variance_prime=None):
        """
        This class implements a Song variance-exploding process, which is a mean-reverting process with a variance-exploding term.
        
        parameters:
            sigma_1: float
                The standard deviation at t=1 (the variance at t=1 is G*G^T = sigma_1^2)
        """
        signal_scale = lambda t: 1.0
        signal_scale_prime = lambda t: 0.0

        super(SongVarianceExplodingSDE, self).__init__(signal_scale=signal_scale, noise_variance=noise_variance, signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)

class SongVariancePreservingSDE(ScalarSDE):
    def __init__(self, beta=5.0):
        """
        This class implements a Song variance-preserving process, which is a mean-reverting process with a variance-preserving term.
        
        parameters:
            beta: float
                The variance-preserving coefficient.
        """

        if isinstance(beta, float):
            beta = torch.tensor(beta)

        signal_scale = lambda t: torch.exp(-0.5*beta*t)
        signal_scale_prime = lambda t: -0.5*beta*torch.exp(-0.5*beta*t)
        noise_variance = lambda t: beta*t
        noise_variance_prime = lambda t: beta

        super(SongVariancePreservingSDE, self).__init__(signal_scale=signal_scale, noise_variance=noise_variance, signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)


class StandardWienerSDE(SongVarianceExplodingSDE):
    def __init__(self):
        """
        This class implements a Wiener process, which is a Song variance-exploding process with sigma_1 = 1.
        """

        noise_variance = lambda t: t
        noise_variance_prime = lambda t: 0*t + 1.0
        super(StandardWienerSDE, self).__init__(noise_variance=noise_variance,
                                            noise_variance_prime=noise_variance_prime)
