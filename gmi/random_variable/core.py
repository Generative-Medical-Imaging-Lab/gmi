
import torch
from torch import nn

from ..samplers import Sampler

class RandomVariable(Sampler, torch.distributions.Distribution):
    def __init__(self):
        super(RandomVariable, self).__init__()
    
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError

    
