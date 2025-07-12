import torch
from .base import LinearOperator

class ConjugateLinearOperator(LinearOperator):
    def __init__(self, base_matrix_operator):
        """
        This class represents the conjugate of another linear operator.
        
        parameters:
            base_matrix_operator: LinearOperator
                The linear operator to which the conjugate should be applied.
        """
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(y)

    def base_operator(self):
        return ConjugateLinearOperator(self.base_matrix_operator)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self):
        return f"ConjugateLinearOperator({self.base_matrix_operator})" 