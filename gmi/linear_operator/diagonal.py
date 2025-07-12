import torch
from .symmetric import SymmetricLinearOperator
from .invertible import InvertibleLinearOperator

class DiagonalLinearOperator(SymmetricLinearOperator, InvertibleLinearOperator):
    def __init__(self, diagonal_vector):
        """
        This class implements a diagonal linear operator.
        
        parameters:
            diagonal_vector: torch.Tensor or list or omegaconf.ListConfig
                The diagonal elements of the linear operator.
        """
        super().__init__()
        
        # Convert to torch.Tensor if needed
        if isinstance(diagonal_vector, (int, float)):
            diagonal_vector = torch.tensor(diagonal_vector)
        elif hasattr(diagonal_vector, '__iter__') and not isinstance(diagonal_vector, torch.Tensor):
            # Handle list/tuple/omegaconf.ListConfig from config
            diagonal_vector = torch.tensor(list(diagonal_vector))
        
        self.diagonal_vector = diagonal_vector
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this diagonal operator is invertible.
        
        returns:
            bool: True if diagonal vector contains no zeros, False otherwise.
        """
        return not torch.any(self.diagonal_vector == 0)
    
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
        return self.diagonal_vector * x
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate of the linear operator.
        
        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate of the linear operator to the input tensor.
        """
        return torch.conj(self.diagonal_vector) * x
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the linear operator to the input tensor.
        raises:
            ValueError: If the diagonal vector contains zeros.
        """
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return y / self.diagonal_vector
    
    def inverse_LinearOperator(self):
        """
        This method returns the inverse linear operator.
        
        returns:
            result: DiagonalLinearOperator
                The inverse linear operator.
        raises:
            ValueError: If the diagonal vector contains zeros.
        """
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return DiagonalLinearOperator(1 / self.diagonal_vector)
    
    def sqrt_LinearOperator(self):
        """
        This method returns the square root linear operator.
        
        returns:
            result: DiagonalLinearOperator
                The square root linear operator.
        """
        return DiagonalLinearOperator(torch.sqrt(self.diagonal_vector))
    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear operator.
        
        parameters:
            other: DiagonalLinearOperator or ScalarLinearOperator or IdentityLinearOperator
                The other linear operator to add.
        returns:
            result: DiagonalLinearOperator
                The sum of the two linear operators.
        """
        from .scalar import ScalarLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, DiagonalLinearOperator):
            return DiagonalLinearOperator(self.diagonal_vector + other.diagonal_vector)
        elif isinstance(other, ScalarLinearOperator):
            # Scalar + Diagonal = Diagonal with scalar added to each element
            return DiagonalLinearOperator(self.diagonal_vector + other.scalar)
        elif isinstance(other, IdentityLinearOperator):
            # Identity + Diagonal = Diagonal with 1 added to each element
            return DiagonalLinearOperator(self.diagonal_vector + 1.0)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for DiagonalLinearOperator")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear operator.
        
        parameters:
            other: DiagonalLinearOperator or ScalarLinearOperator or IdentityLinearOperator
                The other linear operator to subtract.
        returns:
            result: DiagonalLinearOperator
                The difference of the two linear operators.
        """
        from .scalar import ScalarLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, DiagonalLinearOperator):
            return DiagonalLinearOperator(self.diagonal_vector - other.diagonal_vector)
        elif isinstance(other, ScalarLinearOperator):
            # Diagonal - Scalar = Diagonal with scalar subtracted from each element
            return DiagonalLinearOperator(self.diagonal_vector - other.scalar)
        elif isinstance(other, IdentityLinearOperator):
            # Diagonal - Identity = Diagonal with 1 subtracted from each element
            return DiagonalLinearOperator(self.diagonal_vector - 1.0)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for DiagonalLinearOperator")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear operator.
        
        parameters:
            other: DiagonalLinearOperator or ScalarLinearOperator or IdentityLinearOperator
                The other linear operator to multiply.
        returns:
            result: DiagonalLinearOperator
                The product of the two linear operators.
        """
        from .scalar import ScalarLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, DiagonalLinearOperator):
            return DiagonalLinearOperator(self.diagonal_vector * other.diagonal_vector)
        elif isinstance(other, ScalarLinearOperator):
            # Diagonal * Scalar = Diagonal with each element multiplied by scalar
            return DiagonalLinearOperator(self.diagonal_vector * other.scalar)
        elif isinstance(other, IdentityLinearOperator):
            # Diagonal * Identity = Diagonal (unchanged)
            return DiagonalLinearOperator(self.diagonal_vector)
        else:
            raise ValueError(f"Multiplication with {type(other)} not supported for DiagonalLinearOperator")

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the linear operator.
        
        For diagonal operators, conjugate transpose equals conjugate.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear operator to the input tensor.
        """
        return self.conjugate(y)
