import torch
from .eigen_decomposition import EigenDecompositionLinearOperator
from .fourier_transform import FourierTransform
from .diagonal import DiagonalLinearOperator
from .symmetric import SymmetricLinearOperator

class FourierLinearOperator(EigenDecompositionLinearOperator, SymmetricLinearOperator):
    """
    Implements a linear operator that applies a filter in the Fourier domain.
    """
    def __init__(self, filter: torch.Tensor, dim):
        """
        Parameters:
            filter: torch.Tensor
                The filter to apply in the Fourier domain (should match the shape of the transformed dimensions).
            dim: int or tuple of ints
                The dimensions along which to perform the Fourier transform.
        """
        eigenvalue_matrix = DiagonalLinearOperator(filter)
        eigenvector_matrix = FourierTransform(dim=dim)
        super().__init__(eigenvalue_matrix, eigenvector_matrix)
        self.dim = dim
        self.filter = filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Fourier filter operation.
        
        For Fourier convolution, we need: Q^(-1) * Λ * Q * x
        where Q is the Fourier transform and Λ is the diagonal filter.
        """
        # Step 1: Transform to Fourier domain: Q * x
        x_fourier = self.eigenvector_matrix.forward(x)
        
        # Step 2: Apply the filter: Λ * (Q * x)
        filtered_fourier = self.eigenvalue_matrix.forward(x_fourier)
        
        # Step 3: Transform back: Q^(-1) * (Λ * (Q * x))
        result = self.eigenvector_matrix.inverse(filtered_fourier)
        
        return result

    def mat_add(self, other):
        assert isinstance(other, FourierLinearOperator), "Addition only supported for FourierLinearOperator."
        assert self.dim == other.dim, "Dimension mismatch."
        return FourierLinearOperator(self.filter + other.filter, dim=self.dim)

    def mat_sub(self, other):
        assert isinstance(other, FourierLinearOperator), "Subtraction only supported for FourierLinearOperator."
        assert self.dim == other.dim, "Dimension mismatch."
        return FourierLinearOperator(self.filter - other.filter, dim=self.dim)

    def mat_mul(self, other):
        assert isinstance(other, FourierLinearOperator), "Multiplication only supported for FourierLinearOperator."
        assert self.dim == other.dim, "Dimension mismatch."
        return FourierLinearOperator(self.filter * other.filter, dim=self.dim)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Use the parent class's inverse
        return super().inverse(y)

    def sqrt_LinearOperator(self):
        """
        Returns a FourierLinearOperator with the square root of the filter.
        """
        return FourierLinearOperator(torch.sqrt(self.filter), dim=self.dim) 