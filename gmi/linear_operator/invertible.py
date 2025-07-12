import torch
from abc import ABC, abstractmethod
from .square import SquareLinearOperator

class InvertibleLinearOperator(SquareLinearOperator, ABC):
    def __init__(self):
        super().__init__()

    @property
    def is_invertible(self) -> bool:
        """
        Check if this linear operator is invertible.
        
        Default implementation returns True for invertible operators.
        Subclasses can override this to provide specific logic.
        
        returns:
            bool: True by default, can be overridden by subclasses.
        """
        return True

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the linear operator.
        returns:
            x: torch.Tensor
                The result of applying the inverse of the linear operator to the input tensor.
        """
        pass

    def inverse_LinearOperator(self):
        from .inverse import InverseLinearOperator
        return InverseLinearOperator(self) 