import torch
from .composite import CompositeLinearOperator
from .diagonal import DiagonalLinearOperator
from .invertible import InvertibleLinearOperator


class EigenDecompositionLinearOperator(CompositeLinearOperator, InvertibleLinearOperator):
    """
    This class represents a linear operator that is given by its eigenvalue decomposition.

    It inherits from CompositeLinearOperator.

    The operator is constructed as: A = Q * Λ * Q^(-1)
    where Q is the eigenvector matrix and Λ is the diagonal eigenvalue matrix.

    parameters:
        eigenvalue_matrix: DiagonalLinearOperator object
            The diagonal matrix of eigenvalues.
        eigenvector_matrix: InvertibleLinearOperator object
            The invertible matrix of eigenvectors.
    """
    def __init__(self, eigenvalue_matrix: DiagonalLinearOperator, eigenvector_matrix: InvertibleLinearOperator):
        assert isinstance(eigenvalue_matrix, DiagonalLinearOperator), "The eigenvalues should be a DiagonalLinearOperator object."
        assert isinstance(eigenvector_matrix, InvertibleLinearOperator), "The eigenvectors should be a InvertibleLinearOperator object."
        
        # Create the composite structure: Q * Λ * Q^(-1)
        operators = [eigenvector_matrix, eigenvalue_matrix, eigenvector_matrix.inverse_LinearOperator()]
        
        # Initialize CompositeLinearOperator with the operators
        CompositeLinearOperator.__init__(self, operators)
        
        # Set attributes
        self.eigenvalue_matrix = eigenvalue_matrix
        self.eigenvector_matrix = eigenvector_matrix

    @property
    def is_invertible(self) -> bool:
        """
        Check if this eigen decomposition operator is invertible.
        
        returns:
            bool: True if all eigenvalues are non-zero, False otherwise.
        """
        return self.eigenvalue_matrix.is_invertible

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eigenvector_matrix.forward(self.eigenvalue_matrix.forward(self.eigenvector_matrix.inverse(x)))


    def mat_add(self, M):
        """
        Matrix addition with another linear operator.
        
        parameters:
            M: LinearOperator
                The other linear operator to add.
        returns:
            result: LinearOperator
                The sum of the two linear operators.
        """
        # For eigen decomposition, we can only add with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(M, EigenDecompositionLinearOperator):
            # If both are eigen decompositions, we can add their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_add(M.eigenvalue_matrix)
                return EigenDecompositionLinearOperator(new_eigenvalues, self.eigenvector_matrix)
        
        # For other cases, use the composite approach
        return CompositeLinearOperator([self, IdentityLinearOperator()]).mat_add(M)

    def mat_sub(self, M):
        """
        Matrix subtraction with another linear operator.
        
        parameters:
            M: LinearOperator
                The other linear operator to subtract.
        returns:
            result: LinearOperator
                The difference of the two linear operators.
        """
        # For eigen decomposition, we can only subtract with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearOperator
        from .identity import IdentityLinearOperator
        
        if isinstance(M, EigenDecompositionLinearOperator):
            # If both are eigen decompositions, we can subtract their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_sub(M.eigenvalue_matrix)
                return EigenDecompositionLinearOperator(new_eigenvalues, self.eigenvector_matrix)
        
        # For other cases, use the composite approach
        return CompositeLinearOperator([self, IdentityLinearOperator()]).mat_sub(M)

    def mat_mul(self, M):
        """
        Matrix multiplication with another linear operator.
        
        parameters:
            M: LinearOperator or torch.Tensor
                The other linear operator to multiply, or a tensor.
        returns:
            result: LinearOperator or torch.Tensor
                The product of the two linear operators, or the result of applying to tensor.
        """
        if isinstance(M, torch.Tensor):
            return self.forward(M)
        elif isinstance(M, EigenDecompositionLinearOperator):
            # If both are eigen decompositions, we can multiply their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_mul(M.eigenvalue_matrix)
                return EigenDecompositionLinearOperator(new_eigenvalues, self.eigenvector_matrix)
        
        # For other cases, use the composite approach
        from .composite import CompositeLinearOperator
        return CompositeLinearOperator([self, M])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # For a diagonalizable operator, the inverse is Q * Λ^{-1} * Q^{-1}
        inv_eigenvalue_matrix = DiagonalLinearOperator(1.0 / self.eigenvalue_matrix.diagonal_vector)
        return self.eigenvector_matrix.forward(inv_eigenvalue_matrix.forward(self.eigenvector_matrix.inverse(y)))

    def transpose_LinearOperator(self):
        """
        This method returns the transpose linear operator.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the transpose is Q^(-T) * Λ * Q^T.
        Since Λ is diagonal (and thus symmetric), this simplifies to Q^(-T) * Λ * Q^T.
        
        returns:
            result: EigenDecompositionLinearOperator
                The transpose linear operator.
        """
        # For eigen decomposition, transpose is Q^(-T) * Λ * Q^T
        # Since Λ is diagonal and symmetric, we can use the same eigenvalue matrix
        # but with transposed eigenvector matrices
        Q_T = self.eigenvector_matrix.transpose_LinearOperator()
        Q_inv_T = self.eigenvector_matrix.inverse_LinearOperator().transpose_LinearOperator()
        
        # Create new EigenDecompositionLinearOperator with transposed structure
        # Note: This creates Q^(-T) * Λ * Q^T which is equivalent to the transpose
        return EigenDecompositionLinearOperator(self.eigenvalue_matrix, Q_T)

    def inverse_LinearOperator(self):
        """
        This method returns the inverse linear operator.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the inverse is Q * Λ^(-1) * Q^(-1).
        
        returns:
            result: EigenDecompositionLinearOperator
                The inverse linear operator.
        raises:
            ValueError: If any eigenvalue is zero.
        """
        if not self.is_invertible:
            raise ValueError("The operator is not invertible (has zero eigenvalues).")
        
        # Create inverse eigenvalue matrix
        inv_eigenvalue_matrix = DiagonalLinearOperator(1.0 / self.eigenvalue_matrix.diagonal_vector)
        
        # Create new EigenDecompositionLinearOperator with inverse eigenvalues
        return EigenDecompositionLinearOperator(inv_eigenvalue_matrix, self.eigenvector_matrix)

    def conjugate_LinearOperator(self):
        """
        This method returns the conjugate linear operator.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the conjugate is conj(Q) * conj(Λ) * conj(Q)^(-1).
        
        returns:
            result: EigenDecompositionLinearOperator
                The conjugate linear operator.
        """
        # Create conjugate eigenvalue matrix
        conj_eigenvalue_matrix = DiagonalLinearOperator(torch.conj(self.eigenvalue_matrix.diagonal_vector))
        
        # Create conjugate eigenvector matrix
        Q_conj = self.eigenvector_matrix.conjugate_LinearOperator()
        
        # Create new EigenDecompositionLinearOperator with conjugate structure
        return EigenDecompositionLinearOperator(conj_eigenvalue_matrix, Q_conj)

    def conjugate_transpose_LinearOperator(self):
        """
        This method returns the conjugate transpose linear operator.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the conjugate transpose is Q^(-H) * conj(Λ) * Q^H.
        
        returns:
            result: EigenDecompositionLinearOperator
                The conjugate transpose linear operator.
        """
        # Create conjugate eigenvalue matrix
        conj_eigenvalue_matrix = DiagonalLinearOperator(torch.conj(self.eigenvalue_matrix.diagonal_vector))
        
        # Create conjugate transpose eigenvector matrix
        Q_H = self.eigenvector_matrix.conjugate_transpose_LinearOperator()
        
        # Create new EigenDecompositionLinearOperator with conjugate transpose structure
        return EigenDecompositionLinearOperator(conj_eigenvalue_matrix, Q_H) 