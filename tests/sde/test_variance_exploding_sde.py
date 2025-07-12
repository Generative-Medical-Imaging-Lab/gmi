import torch
import pytest
from gmi.sde import VarianceExplodingSDE

def test_variance_exploding_sde_instantiation():
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    assert isinstance(sde, VarianceExplodingSDE)

def test_sample_increases_variance(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    x0 = sample_tensor_4d
    timesteps = torch.linspace(0, 1, 10)
    xT = sde.sample(x0, timesteps)
    var0 = x0.var().item()
    varT = xT.var().item()
    # The variance should increase due to added noise
    assert varT > var0

def test_mean_response_x_t_given_x_0(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    x0 = sample_tensor_4d
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    # For variance exploding, mean is just x0 (signal_scale = 1.0)
    assert torch.allclose(mean, x0)

def test_sample_x_t_given_x_0(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    x0 = sample_tensor_4d
    t = torch.tensor(1.0)
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape

def test_sample_x_t_given_x_0_and_noise(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    x0 = sample_tensor_4d
    noise = torch.randn_like(x0)
    t = torch.tensor(1.0)
    xt = sde.sample_x_t_given_x_0_and_noise(x0, noise, t)
    assert xt.shape == x0.shape

def test_reverse_SDE_given_score_estimator_runs(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    def dummy_score(x, t):
        return torch.zeros_like(x)
    rev = sde.reverse_SDE_given_score_estimator(dummy_score)
    x = sample_tensor_4d
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor)

def test_reverse_SDE_given_mean_estimator_runs(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    def dummy_mean(x, t):
        return x
    rev = sde.reverse_SDE_given_mean_estimator(dummy_mean)
    x = sample_tensor_4d
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor)

def test_reverse_SDE_given_noise_estimator_runs(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    def dummy_noise(x, t):
        return torch.zeros_like(x)
    rev = sde.reverse_SDE_given_noise_estimator(dummy_noise)
    x = sample_tensor_4d
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor)

def test_reverse_SDE_given_posterior_mean_estimator_runs(sample_tensor_4d):
    sde = VarianceExplodingSDE(noise_variance=lambda t: t, noise_variance_prime=lambda t: 1.0)
    def dummy_posterior_mean(x, t):
        return x
    rev = sde.reverse_SDE_given_posterior_mean_estimator(dummy_posterior_mean)
    x = sample_tensor_4d
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor) 