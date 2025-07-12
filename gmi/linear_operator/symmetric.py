import torch
from abc import ABC, abstractmethod
from .square import SquareLinearOperator

class SymmetricLinearOperator(SquareLinearOperator, ABC):
    def __init__(self):
        super().__init__()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For symmetric operators, transpose = forward
        return self.forward(y)

    def transpose_LinearOperator(self):
        return self
    
 