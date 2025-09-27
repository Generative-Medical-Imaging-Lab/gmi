
from .core import *
# from .dataset import *
from .gaussian import *
from .lognormal import *
from .uniform import *
from .from_log_prob import *

from . import core
# from . import dataset
from . import gaussian
from . import lognormal
from . import uniform
from . import from_log_prob

# Backward compatibility aliases
Distribution = RandomVariable
GaussianDistribution = GaussianRandomVariable
ConditionalGaussianDistribution = ConditionalGaussianRandomVariable
UniformDistribution = UniformRandomVariable
LogNormalDistribution = LogNormalRandomVariable
DistributionFromLogProb = RandomVariableFromLogProb