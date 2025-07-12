"""
Inverse Linear Operator

This module provides the InverseLinearOperator class, which represents the inverse
of an invertible linear operator.
"""

import torch
from typing import Optional

from .invertible import InvertibleLinearOperator
from .base import LinearOperator


class InverseLinearOperator(InvertibleLinearOperator):
    """
    Linear operator that represents the inverse of another invertible linear operator.
    
    This class wraps an invertible linear operator and provides its inverse
    as the forward operation. The inverse of this operator is the original operator.
    
    Parameters:
        base_matrix_operator: InvertibleLinearOperator
            The invertible linear operator to invert.
    """
    
    def __init__(self, base_matrix_operator: InvertibleLinearOperator):
        """
        Initialize the inverse linear operator.
        
        Parameters:
            base_matrix_operator: InvertibleLinearOperator
                The invertible linear operator to invert.
        """
        assert isinstance(base_matrix_operator, InvertibleLinearOperator), \
            "The input linear operator should be an InvertibleLinearOperator object."
        
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse of the base operator to the input.
        
        Parameters:
            x: torch.Tensor
                Input tensor.
                
        Returns:
            torch.Tensor: Result of applying the inverse operator.
        """
        return self.base_matrix_operator.inverse(x)
    
    def inverse_LinearOperator(self) -> LinearOperator:
        """
        Return the original base operator as the inverse of this operator.
        
        Returns:
            LinearOperator: The original base operator.
        """
        return self.base_matrix_operator.forward_LinearOperator()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the original base operator to the input.
        
        Parameters:
            y: torch.Tensor
                Input tensor.
                
        Returns:
            torch.Tensor: Result of applying the original operator.
        """
        return self.base_matrix_operator.forward(y) 