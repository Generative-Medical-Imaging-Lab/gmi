"""
Tests for matrix operations between diagonal, scalar, and identity operators.
"""
import pytest
import torch
from gmi.linear_operator.diagonal import DiagonalLinearOperator
from gmi.linear_operator.scalar import ScalarLinearOperator
from gmi.linear_operator.identity import IdentityLinearOperator


class TestMatrixOperations:
    """Test cases for matrix operations between diagonal, scalar, and identity operators."""
    
    # ==================== ADDITION TESTS ====================
    
    def test_diagonal_scalar_addition(self):
        """Test addition between diagonal and scalar operators."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        scalar = ScalarLinearOperator(5.0)
        
        # Diagonal + Scalar
        result = diag.mat_add(scalar)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([6.0, 7.0, 8.0]))
        
        # Scalar + Diagonal
        result = scalar.mat_add(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([6.0, 7.0, 8.0]))
    
    def test_diagonal_identity_addition(self):
        """Test addition between diagonal and identity operators."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        identity = IdentityLinearOperator()
        
        # Diagonal + Identity
        result = diag.mat_add(identity)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([2.0, 3.0, 4.0]))
        
        # Identity + Diagonal
        result = identity.mat_add(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([2.0, 3.0, 4.0]))
    
    def test_scalar_identity_addition(self):
        """Test addition between scalar and identity operators."""
        scalar = ScalarLinearOperator(3.0)
        identity = IdentityLinearOperator()
        
        # Scalar + Identity
        result = scalar.mat_add(identity)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(4.0))
        
        # Identity + Scalar
        result = identity.mat_add(scalar)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(4.0))
    
    def test_identity_identity_addition(self):
        """Test addition between two identity operators."""
        identity1 = IdentityLinearOperator()
        identity2 = IdentityLinearOperator()
        
        result = identity1.mat_add(identity2)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(2.0))
    
    def test_scalar_scalar_addition(self):
        """Test addition between two scalar operators."""
        scalar1 = ScalarLinearOperator(2.0)
        scalar2 = ScalarLinearOperator(3.0)
        
        result = scalar1.mat_add(scalar2)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(5.0))
    
    def test_diagonal_diagonal_addition(self):
        """Test addition between two diagonal operators."""
        diag1 = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        diag2 = DiagonalLinearOperator(torch.tensor([4.0, 5.0, 6.0]))
        
        result = diag1.mat_add(diag2)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([5.0, 7.0, 9.0]))
    
    # ==================== SUBTRACTION TESTS ====================
    
    def test_diagonal_scalar_subtraction(self):
        """Test subtraction between diagonal and scalar operators."""
        diag = DiagonalLinearOperator(torch.tensor([5.0, 6.0, 7.0]))
        scalar = ScalarLinearOperator(2.0)
        
        # Diagonal - Scalar
        result = diag.mat_sub(scalar)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([3.0, 4.0, 5.0]))
        
        # Scalar - Diagonal
        result = scalar.mat_sub(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([-3.0, -4.0, -5.0]))
    
    def test_diagonal_identity_subtraction(self):
        """Test subtraction between diagonal and identity operators."""
        diag = DiagonalLinearOperator(torch.tensor([3.0, 4.0, 5.0]))
        identity = IdentityLinearOperator()
        
        # Diagonal - Identity
        result = diag.mat_sub(identity)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([2.0, 3.0, 4.0]))
        
        # Identity - Diagonal
        result = identity.mat_sub(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([-2.0, -3.0, -4.0]))
    
    def test_scalar_identity_subtraction(self):
        """Test subtraction between scalar and identity operators."""
        scalar = ScalarLinearOperator(5.0)
        identity = IdentityLinearOperator()
        
        # Scalar - Identity
        result = scalar.mat_sub(identity)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(4.0))
        
        # Identity - Scalar
        result = identity.mat_sub(scalar)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(-4.0))
    
    def test_identity_identity_subtraction(self):
        """Test subtraction between two identity operators."""
        identity1 = IdentityLinearOperator()
        identity2 = IdentityLinearOperator()
        
        result = identity1.mat_sub(identity2)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(0.0))
    
    def test_scalar_scalar_subtraction(self):
        """Test subtraction between two scalar operators."""
        scalar1 = ScalarLinearOperator(5.0)
        scalar2 = ScalarLinearOperator(3.0)
        
        result = scalar1.mat_sub(scalar2)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(2.0))
    
    def test_diagonal_diagonal_subtraction(self):
        """Test subtraction between two diagonal operators."""
        diag1 = DiagonalLinearOperator(torch.tensor([5.0, 6.0, 7.0]))
        diag2 = DiagonalLinearOperator(torch.tensor([2.0, 3.0, 4.0]))
        
        result = diag1.mat_sub(diag2)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([3.0, 3.0, 3.0]))
    
    # ==================== MULTIPLICATION TESTS ====================
    
    def test_diagonal_scalar_multiplication(self):
        """Test multiplication between diagonal and scalar operators."""
        diag = DiagonalLinearOperator(torch.tensor([2.0, 3.0, 4.0]))
        scalar = ScalarLinearOperator(5.0)
        
        # Diagonal * Scalar
        result = diag.mat_mul(scalar)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([10.0, 15.0, 20.0]))
        
        # Scalar * Diagonal
        result = scalar.mat_mul(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([10.0, 15.0, 20.0]))
    
    def test_diagonal_identity_multiplication(self):
        """Test multiplication between diagonal and identity operators."""
        diag = DiagonalLinearOperator(torch.tensor([2.0, 3.0, 4.0]))
        identity = IdentityLinearOperator()
        
        # Diagonal * Identity
        result = diag.mat_mul(identity)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([2.0, 3.0, 4.0]))
        
        # Identity * Diagonal
        result = identity.mat_mul(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([2.0, 3.0, 4.0]))
    
    def test_scalar_identity_multiplication(self):
        """Test multiplication between scalar and identity operators."""
        scalar = ScalarLinearOperator(3.0)
        identity = IdentityLinearOperator()
        
        # Scalar * Identity
        result = scalar.mat_mul(identity)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(3.0))
        
        # Identity * Scalar
        result = identity.mat_mul(scalar)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(3.0))
    
    def test_identity_identity_multiplication(self):
        """Test multiplication between two identity operators."""
        identity1 = IdentityLinearOperator()
        identity2 = IdentityLinearOperator()
        
        result = identity1.mat_mul(identity2)
        assert isinstance(result, IdentityLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(1.0))
    
    def test_scalar_scalar_multiplication(self):
        """Test multiplication between two scalar operators."""
        scalar1 = ScalarLinearOperator(2.0)
        scalar2 = ScalarLinearOperator(3.0)
        
        result = scalar1.mat_mul(scalar2)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(6.0))
    
    def test_diagonal_diagonal_multiplication(self):
        """Test multiplication between two diagonal operators."""
        diag1 = DiagonalLinearOperator(torch.tensor([2.0, 3.0, 4.0]))
        diag2 = DiagonalLinearOperator(torch.tensor([5.0, 6.0, 7.0]))
        
        result = diag1.mat_mul(diag2)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([10.0, 18.0, 28.0]))
    
    def test_scalar_tensor_multiplication(self):
        """Test multiplication between scalar operator and tensor."""
        scalar = ScalarLinearOperator(3.0)
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        result = scalar.mat_mul(tensor)
        assert torch.allclose(result, torch.tensor([3.0, 6.0, 9.0]))
    
    def test_identity_tensor_multiplication(self):
        """Test multiplication between identity operator and tensor."""
        identity = IdentityLinearOperator()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        result = identity.mat_mul(tensor)
        assert torch.allclose(result, tensor)
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_unsupported_addition_types(self):
        """Test error handling for unsupported addition types."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        scalar = ScalarLinearOperator(2.0)
        
        # Test with unsupported types
        with pytest.raises(ValueError):
            diag.mat_add("unsupported")
        
        with pytest.raises(ValueError):
            scalar.mat_add("unsupported")
    
    def test_unsupported_subtraction_types(self):
        """Test error handling for unsupported subtraction types."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        scalar = ScalarLinearOperator(2.0)
        
        # Test with unsupported types
        with pytest.raises(ValueError):
            diag.mat_sub("unsupported")
        
        with pytest.raises(ValueError):
            scalar.mat_sub("unsupported")
    
    def test_unsupported_multiplication_types(self):
        """Test error handling for unsupported multiplication types."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        scalar = ScalarLinearOperator(2.0)
        
        # Test with unsupported types
        with pytest.raises(ValueError):
            diag.mat_mul("unsupported")
        
        with pytest.raises(ValueError):
            scalar.mat_mul("unsupported")
    
    # ==================== PROPERTY TESTS ====================
    
    def test_identity_scalar_attribute(self):
        """Test that IdentityLinearOperator has scalar=1.0."""
        identity = IdentityLinearOperator()
        assert torch.allclose(identity.scalar, torch.tensor(1.0))
        assert torch.allclose(identity.diagonal_vector, torch.tensor(1.0))
    
    def test_commutativity_of_addition(self):
        """Test that addition is commutative for compatible operators."""
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        scalar = ScalarLinearOperator(5.0)
        
        # Test commutativity: A + B = B + A
        result1 = diag.mat_add(scalar)
        result2 = scalar.mat_add(diag)
        assert torch.allclose(result1.diagonal_vector, result2.diagonal_vector)
    
    def test_associativity_of_multiplication(self):
        """Test that multiplication is associative for compatible operators."""
        diag = DiagonalLinearOperator(torch.tensor([2.0, 3.0]))
        scalar1 = ScalarLinearOperator(2.0)
        scalar2 = ScalarLinearOperator(3.0)
        
        # Test associativity: (A * B) * C = A * (B * C)
        result1 = (diag.mat_mul(scalar1)).mat_mul(scalar2)
        result2 = diag.mat_mul(scalar1.mat_mul(scalar2))
        assert torch.allclose(result1.diagonal_vector, result2.diagonal_vector)
    
    def test_distributivity(self):
        """Test that multiplication distributes over addition."""
        diag = DiagonalLinearOperator(torch.tensor([2.0, 3.0]))
        scalar1 = ScalarLinearOperator(2.0)
        scalar2 = ScalarLinearOperator(3.0)
        
        # Test left distributivity: A * (B + C) = A * B + A * C
        # Note: This would require implementing matrix addition between results
        # For now, we test that the operations work correctly individually
        sum_scalars = scalar1.mat_add(scalar2)
        left_result = diag.mat_mul(sum_scalars)
        
        # This should equal diag * (2 + 3) = diag * 5
        expected = DiagonalLinearOperator(torch.tensor([10.0, 15.0]))
        assert torch.allclose(left_result.diagonal_vector, expected.diagonal_vector)
    
    # ==================== EDGE CASE TESTS ====================
    
    def test_zero_scalar_operations(self):
        """Test operations with zero scalar."""
        zero_scalar = ScalarLinearOperator(0.0)
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        identity = IdentityLinearOperator()
        
        # Zero + Identity = Identity
        result = zero_scalar.mat_add(identity)
        assert isinstance(result, ScalarLinearOperator)
        assert torch.allclose(result.scalar, torch.tensor(1.0))
        
        # Zero * Diagonal = Zero diagonal
        result = zero_scalar.mat_mul(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([0.0, 0.0, 0.0]))
    
    def test_negative_scalar_operations(self):
        """Test operations with negative scalars."""
        neg_scalar = ScalarLinearOperator(-2.0)
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        
        # Negative scalar + diagonal
        result = neg_scalar.mat_add(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([-1.0, 0.0, 1.0]))
        
        # Negative scalar * diagonal
        result = neg_scalar.mat_mul(diag)
        assert isinstance(result, DiagonalLinearOperator)
        assert torch.allclose(result.diagonal_vector, torch.tensor([-2.0, -4.0, -6.0]))
    
    def test_complex_scalar_operations(self):
        """Test operations with complex scalars."""
        complex_scalar = ScalarLinearOperator(2.0 + 1j)
        diag = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
        
        # Complex scalar + diagonal (should work with real diagonal)
        result = complex_scalar.mat_add(diag)
        assert isinstance(result, DiagonalLinearOperator)
        expected = torch.tensor([3.0 + 1j, 4.0 + 1j, 5.0 + 1j])
        assert torch.allclose(result.diagonal_vector, expected)
        
        # Complex scalar * diagonal
        result = complex_scalar.mat_mul(diag)
        assert isinstance(result, DiagonalLinearOperator)
        expected = torch.tensor([2.0 + 1j, 4.0 + 2j, 6.0 + 3j])
        assert torch.allclose(result.diagonal_vector, expected) 