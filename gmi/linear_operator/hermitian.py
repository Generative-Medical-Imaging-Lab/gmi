import torch
from abc import ABC, abstractmethod
from .square import SquareLinearOperator

class HermitianLinearOperator(SquareLinearOperator, ABC):
    def __init__(self):
        super().__init__()

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For hermitian operators, conjugate_transpose = forward
        return self.forward(y)

    def conjugate_transpose_LinearOperator(self):
        return self 