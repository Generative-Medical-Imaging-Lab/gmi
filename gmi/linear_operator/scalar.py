import torch
from .diagonal import DiagonalLinearOperator
from .invertible import InvertibleLinearOperator

class ScalarLinearOperator(DiagonalLinearOperator, InvertibleLinearOperator):
    def __init__(self, scalar):
        """
        This class implements a scalar linear operator (scalar multiple of identity).
        
        parameters:
            scalar: float or torch.Tensor or list or omegaconf.ListConfig
                The scalar to multiply the input tensor with.
        """
        if isinstance(scalar, (int, float, complex)):
            scalar = torch.tensor(scalar)
        elif hasattr(scalar, '__iter__') and not isinstance(scalar, torch.Tensor):
            # Handle list/tuple/omegaconf.ListConfig from config
            scalar = torch.tensor(list(scalar))
        
        # For scalar operator, we need to handle broadcasting properly
        # The diagonal vector will be the scalar value
        self.scalar = scalar
        super().__init__(scalar)
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this scalar operator is invertible.
        
        returns:
            bool: True if scalar is non-zero, False otherwise.
        """
        return not torch.any(self.scalar == 0)
    
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
        # Handle broadcasting for scalar multiplication
        if isinstance(self.scalar, torch.Tensor) and self.scalar.dim() > 0:
            for i, shape in enumerate(self.scalar.shape):
                assert x.shape[i] == shape or self.scalar.shape[i] == 1, f"Shape mismatch at dimension {i}: x.shape[{i}]={x.shape[i]}, scalar.shape[{i}]={self.scalar.shape[i]}"
            
            target_shape = list(self.scalar.shape)
            for i in range(len(x.shape) - len(self.scalar.shape)):
                target_shape.append(1)
            
            return self.scalar.reshape(target_shape).to(x.device) * x
        else:  # If scalar is a 0-dimensional tensor (scalar)
            return self.scalar.to(x.device) * x
    
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
        # Handle broadcasting for scalar multiplication
        if isinstance(self.scalar, torch.Tensor) and self.scalar.dim() > 0:
            for i, shape in enumerate(self.scalar.shape):
                assert x.shape[i] == shape or self.scalar.shape[i] == 1, f"Shape mismatch at dimension {i}: x.shape[{i}]={x.shape[i]}, scalar.shape[{i}]={self.scalar.shape[i]}"
            
            target_shape = list(self.scalar.shape)
            for i in range(len(x.shape) - len(self.scalar.shape)):
                target_shape.append(1)
            
            return torch.conj(self.scalar).reshape(target_shape).to(x.device) * x
        else:  # If scalar is a 0-dimensional tensor (scalar)
            return torch.conj(self.scalar).to(x.device) * x
    
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
            ValueError: If the scalar is zero.
        """
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return y / self.scalar
    
    def inverse_LinearOperator(self):
        """
        This method returns the inverse linear operator.
        
        returns:
            result: ScalarLinearOperator
                The inverse linear operator.
        raises:
            ValueError: If the scalar is zero.
        """
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return ScalarLinearOperator(1 / self.scalar)
    
    def sqrt_LinearOperator(self):
        """
        This method returns the square root linear operator.
        
        returns:
            result: ScalarLinearOperator
                The square root linear operator.
        """
        return ScalarLinearOperator(torch.sqrt(self.scalar))
    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear operator.
        
        parameters:
            other: ScalarLinearOperator or DiagonalLinearOperator or IdentityLinearOperator
                The other linear operator to add.
        returns:
            result: ScalarLinearOperator or DiagonalLinearOperator
                The sum of the two linear operators.
        """
        from .diagonal import DiagonalLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(self.scalar + other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Scalar + Diagonal = Diagonal with scalar added to each element
            return DiagonalLinearOperator(self.scalar + other.diagonal_vector)
        elif isinstance(other, IdentityLinearOperator):
            # Scalar + Identity = Scalar with 1 added
            return ScalarLinearOperator(self.scalar + 1.0)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for ScalarLinearOperator")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear operator.
        
        parameters:
            other: ScalarLinearOperator or DiagonalLinearOperator or IdentityLinearOperator
                The other linear operator to subtract.
        returns:
            result: ScalarLinearOperator or DiagonalLinearOperator
                The difference of the two linear operators.
        """
        from .diagonal import DiagonalLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(self.scalar - other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Scalar - Diagonal = Diagonal with scalar subtracted from each element
            return DiagonalLinearOperator(self.scalar - other.diagonal_vector)
        elif isinstance(other, IdentityLinearOperator):
            # Scalar - Identity = Scalar with 1 subtracted
            return ScalarLinearOperator(self.scalar - 1.0)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for ScalarLinearOperator")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear operator.
        
        parameters:
            other: ScalarLinearOperator or DiagonalLinearOperator or IdentityLinearOperator or torch.Tensor
                The other linear operator to multiply, or a tensor.
        returns:
            result: ScalarLinearOperator or DiagonalLinearOperator or torch.Tensor
                The product of the two linear operators, or the result of applying to tensor.
        """
        from .diagonal import DiagonalLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(other, torch.Tensor):
            return self.forward(other)
        elif isinstance(other, ScalarLinearOperator):
            return ScalarLinearOperator(self.scalar * other.scalar)
        elif isinstance(other, DiagonalLinearOperator):
            # Scalar * Diagonal = Diagonal with each element multiplied by scalar
            return DiagonalLinearOperator(self.scalar * other.diagonal_vector)
        elif isinstance(other, IdentityLinearOperator):
            # Scalar * Identity = Scalar (unchanged)
            return ScalarLinearOperator(self.scalar)
        else:
            raise ValueError(f"Multiplication with {type(other)} not supported for ScalarLinearOperator")
    
    def __matmul__(self, other):
        """
        This method implements the @ operator for matrix multiplication.
        
        parameters:
            other: LinearOperator or torch.Tensor
                The other linear operator to multiply, or a tensor.
        returns:
            result: LinearOperator or torch.Tensor
                The product of the two linear operators, or the result of applying to tensor.
        """
        return self.mat_mul(other)
    
    def logdet(self):
        """
        This method returns the log determinant of the linear operator.
        
        returns:
            result: torch.Tensor
                The log determinant.
        """
        return torch.log(torch.abs(self.scalar))

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear operator.
        
        For scalar operators, transpose equals forward.
        
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
        
        For real scalar operators, conjugate transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear operator.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear operator to the input tensor.
        """
        return self.forward(y)

    def transpose_LinearOperator(self):
        """
        This method returns the transpose linear operator.
        
        For scalar operators, transpose equals the operator itself.
        
        returns:
            result: ScalarLinearOperator
                The transpose linear operator (same as self).
        """
        return ScalarLinearOperator(self.scalar)

    def conjugate_LinearOperator(self):
        """
        This method returns the conjugate linear operator.
        
        For scalar operators, conjugate is the conjugate of the scalar.
        
        returns:
            result: ScalarLinearOperator
                The conjugate linear operator.
        """
        return ScalarLinearOperator(torch.conj(self.scalar))

    def conjugate_transpose_LinearOperator(self):
        """
        This method returns the conjugate transpose linear operator.
        
        For scalar operators, conjugate transpose equals conjugate.
        
        returns:
            result: ScalarLinearOperator
                The conjugate transpose linear operator.
        """
        return self.conjugate_LinearOperator()

# Removed inline test definitions and __main__ execution block. 