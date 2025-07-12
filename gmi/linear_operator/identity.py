import torch
from .scalar import ScalarLinearOperator
from .real import RealLinearOperator
from .hermitian import HermitianLinearOperator
from .unitary import UnitaryLinearOperator

class IdentityLinearOperator(ScalarLinearOperator, RealLinearOperator, HermitianLinearOperator, UnitaryLinearOperator):
    def __init__(self):
        """
        Identity linear operator (the identity matrix).
        """
        super().__init__(scalar=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear operator.
        
        parameters:
            x: torch.Tensor
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the linear operator to the input tensor.
        """
        return x
    

    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear operator.
        
        parameters:
            other: IdentityLinearOperator or ScalarLinearOperator or DiagonalLinearOperator
                The other linear operator to add.
        returns:
            result: ScalarLinearOperator or DiagonalLinearOperator
                The sum of the two linear operators.
        """
        from .scalar import ScalarLinearOperator
        from .diagonal import DiagonalLinearOperator
        
        if isinstance(other, IdentityLinearOperator):
            return ScalarLinearOperator(2.0)
        elif isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(1.0 + other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Identity + Diagonal = Diagonal with 1 added to each element
            return DiagonalLinearOperator(1.0 + other.diagonal_vector)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for IdentityLinearOperator")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear operator.
        
        parameters:
            other: IdentityLinearOperator or ScalarLinearOperator or DiagonalLinearOperator
                The other linear operator to subtract.
        returns:
            result: ScalarLinearOperator or DiagonalLinearOperator
                The difference of the two linear operators.
        """
        from .scalar import ScalarLinearOperator
        from .diagonal import DiagonalLinearOperator
        
        if isinstance(other, IdentityLinearOperator):
            return ScalarLinearOperator(0.0)
        elif isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(1.0 - other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Identity - Diagonal = Diagonal with 1 subtracted from each element
            return DiagonalLinearOperator(1.0 - other.diagonal_vector)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for IdentityLinearOperator")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear operator.
        
        parameters:
            other: IdentityLinearOperator or ScalarLinearOperator or DiagonalLinearOperator or torch.Tensor
                The other linear operator to multiply, or a tensor.
        returns:
            result: IdentityLinearOperator or ScalarLinearOperator or DiagonalLinearOperator or torch.Tensor
                The product of the two linear operators, or the result of applying to tensor.
        """
        from .scalar import ScalarLinearOperator
        from .diagonal import DiagonalLinearOperator
        
        if isinstance(other, torch.Tensor):
            return other
        elif isinstance(other, IdentityLinearOperator):
            return IdentityLinearOperator()
        elif isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Identity * Diagonal = Diagonal (unchanged)
            return DiagonalLinearOperator(other.diagonal_vector)
        else:
            # For other types, multiplication by identity returns the other operator
            return other
    
    def logdet(self):
        """
        This method returns the log determinant of the linear operator.
        
        returns:
            result: torch.Tensor
                The log determinant (which is 0 for identity).
        """
        return torch.tensor(0.0)

    def inverse_LinearOperator(self):
        """
        This method returns the inverse linear operator.
        
        returns:
            result: IdentityLinearOperator
                The inverse linear operator (which is the identity itself).
        """
        return IdentityLinearOperator()

    def sqrt_LinearOperator(self):
        """
        This method returns the square root linear operator.
        
        returns:
            result: IdentityLinearOperator
                The square root linear operator (which is the identity itself).
        """
        return IdentityLinearOperator()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear operator.
        
        For identity operators, transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the transpose of the linear operator to the input tensor.
        """
        return self.forward(y)

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the linear operator.
        
        For real identity operators, conjugate transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear operator to the input tensor.
        """
        return self.forward(y) 