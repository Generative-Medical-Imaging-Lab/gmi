import torch
from .invertible import InvertibleLinearOperator

class UnitaryLinearOperator(InvertibleLinearOperator):
    def __init__(self):
        super().__init__()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # For unitary operators, inverse = conjugate_transpose
        return self.conjugate_transpose(y)

    def inverse_LinearOperator(self):
        from .conjugate_transpose import ConjugateTransposeLinearOperator
        return ConjugateTransposeLinearOperator(self) 