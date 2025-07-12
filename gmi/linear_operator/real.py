import torch
from abc import ABC, abstractmethod
from .base import LinearOperator

class RealLinearOperator(LinearOperator, ABC):
    def __init__(self):
        super().__init__()

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        # For real operators, conjugate is just forward
        return self.forward(x)

    def conjugate_LinearOperator(self):
        return self

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.transpose(y)

    def conjugate_transpose_LinearOperator(self):
        return self 