import torch
from .base import LinearOperator

class CompositeLinearOperator(LinearOperator):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.
        
        It inherits from LinearOperator.
        
        parameters:
            matrix_operators: list of LinearOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """
        super().__init__()
        
        # Convert to list if it's a ListConfig from Hydra
        if hasattr(matrix_operators, '__iter__') and not isinstance(matrix_operators, list):
            matrix_operators = list(matrix_operators)
        
        assert isinstance(matrix_operators, list), "The operators should be provided as a list of LinearOperator objects."
        
        # Allow empty list - will act as identity operator
        if len(matrix_operators) == 0:
            from .identity import IdentityLinearOperator
            self.matrix_operators = [IdentityLinearOperator()]
        else:
            # Instantiate operators if they're configs
            instantiated_operators = []
            for operator in matrix_operators:
                if hasattr(operator, '_target_'):
                    # This is a config, need to instantiate it
                    from hydra.utils import instantiate
                    instantiated_operator = instantiate(operator)
                    instantiated_operators.append(instantiated_operator)
                else:
                    # This is already an operator
                    assert isinstance(operator, LinearOperator), "All operators should be LinearOperator objects."
                    instantiated_operators.append(operator)
            
            self.matrix_operators = instantiated_operators
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the composite linear operator.
        
        parameters:
            x: torch.Tensor
                The input tensor to the composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the composite linear operator to the input tensor.
        """
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.forward(result)
        return result
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the composite linear operator.
        
        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the transpose of the composite linear operator to the input tensor.
        """
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.transpose(result)
        return result
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate of the composite linear operator.
        
        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate of the composite linear operator to the input tensor.
        """
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.conjugate(result)
        return result
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the composite linear operator.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the composite linear operator to the input tensor.
        """
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.conjugate_transpose(result)
        return result

# Removed inline tests and __main__ execution block. 