
import torch
from torch import nn

from ..samplers import Sampler

class Distribution(Sampler, torch.distributions.Distribution):
    def __init__(self):
        super(Distribution, self).__init__()
    
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError

    
