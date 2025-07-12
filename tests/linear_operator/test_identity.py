"""
Tests for the IdentityLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.identity import IdentityLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestIdentityLinearOperator:
    """Test cases for the IdentityLinearOperator class."""
    
    def test_instantiation(self):
        """Test that IdentityLinearOperator can be instantiated directly."""
        op = IdentityLinearOperator()
        assert op is not None
    
    def test_forward_operation(self, sample_vector_2, sample_tensor_4d):
        """Test that identity operator preserves input."""
        op = IdentityLinearOperator()
        
        # Test with vector
        x = sample_vector_2
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
        
        # Test with tensor
        x = sample_tensor_4d
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
    
    def test_inheritance(self):
        """Test that IdentityLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        from gmi.linear_operator.square import SquareLinearOperator
        from gmi.linear_operator.scalar import ScalarLinearOperator
        from gmi.linear_operator.real import RealLinearOperator
        from gmi.linear_operator.hermitian import HermitianLinearOperator
        from gmi.linear_operator.unitary import UnitaryLinearOperator
        from gmi.linear_operator.invertible import InvertibleLinearOperator
        
        op = IdentityLinearOperator()
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        assert isinstance(op, SquareLinearOperator), "Should inherit from SquareLinearOperator"
        assert isinstance(op, ScalarLinearOperator), "Should inherit from ScalarLinearOperator"
        assert isinstance(op, RealLinearOperator), "Should inherit from RealLinearOperator"
        assert isinstance(op, HermitianLinearOperator), "Should inherit from HermitianLinearOperator"
        assert isinstance(op, UnitaryLinearOperator), "Should inherit from UnitaryLinearOperator"
        assert isinstance(op, InvertibleLinearOperator), "Should inherit from InvertibleLinearOperator"
    
    def test_is_invertible_property(self):
        """Test the is_invertible property inherited from ScalarLinearOperator."""
        op = IdentityLinearOperator()
        assert op.is_invertible, "Identity operator should be invertible (scalar = 1.0)"
    
    def test_config_instantiation(self):
        """Test that IdentityLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/identity.yaml")
        # Instantiate the linear_operator part directly to avoid wrapping
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, IdentityLinearOperator), "Should instantiate IdentityLinearOperator from config"
    
    def test_identity_properties(self, sample_vector_2):
        """Test that identity operator has all expected properties."""
        op = IdentityLinearOperator()
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, x)
        
        # Test transpose (should equal forward for identity)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, x)
        
        # Test conjugate (should equal forward for real identity)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, x)
        
        # Test conjugate_transpose (should equal forward for real identity)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, x)
        
        # Test inverse (should equal forward for identity)
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, x)
    
    def test_scalar_property(self):
        """Test that identity operator has scalar property equal to 1.0."""
        op = IdentityLinearOperator()
        assert torch.allclose(op.scalar, torch.tensor(1.0)), "Identity operator should have scalar value 1.0"
    
    def test_all_operations_preserve_input(self, sample_vector_2):
        """Test that all operations preserve the input (identity behavior)."""
        op = IdentityLinearOperator()
        x = sample_vector_2
        
        # Test all operations return the input unchanged
        operations = [
            op.forward,
            op.transpose,
            op.conjugate,
            op.conjugate_transpose,
            op.inverse
        ]
        
        for operation in operations:
            result = operation(x)
            assert torch.allclose(result, x), f"Operation {operation.__name__} should preserve input"
    
    def test_complex_input_handling(self):
        """Test that identity operator handles complex inputs correctly."""
        op = IdentityLinearOperator()
        
        # Test with complex tensor
        x_complex = torch.tensor([1.0 + 2j, 3.0 - 4j])
        y = op.forward(x_complex)
        assert torch.allclose(y, x_complex)
        
        # Test conjugate with complex input
        y_conj = op.conjugate(x_complex)
        assert torch.allclose(y_conj, x_complex)  # Identity preserves complex values
    
    def test_tensor_input_handling(self, sample_tensor_4d):
        """Test that identity operator handles multi-dimensional tensors correctly."""
        op = IdentityLinearOperator()
        x = sample_tensor_4d
        
        y = op.forward(x)
        assert torch.allclose(y, x)
        assert y.shape == x.shape
    
    def test_matrix_operations(self, sample_vector_2):
        """Test matrix operations with other operators."""
        from gmi.linear_operator.scalar import ScalarLinearOperator
        from gmi.linear_operator.diagonal import DiagonalLinearOperator
        
        op = IdentityLinearOperator()
        x = sample_vector_2
        
        # Test addition with IdentityLinearOperator
        other_identity = IdentityLinearOperator()
        sum_op = op.mat_add(other_identity)
        assert isinstance(sum_op, ScalarLinearOperator), "Identity + Identity should return ScalarLinearOperator"
        assert torch.allclose(sum_op.scalar, torch.tensor(2.0)), "Identity + Identity should equal 2.0"
        
        # Test addition with ScalarLinearOperator
        scalar_op = ScalarLinearOperator(scalar=3.0)
        sum_op = op.mat_add(scalar_op)
        assert isinstance(sum_op, ScalarLinearOperator), "Identity + Scalar should return ScalarLinearOperator"
        assert torch.allclose(sum_op.scalar, torch.tensor(4.0)), "Identity + 3.0 should equal 4.0"
        
        # Test addition with DiagonalLinearOperator
        diagonal_op = DiagonalLinearOperator(diagonal_vector=torch.tensor([1.0, 2.0]))
        sum_op = op.mat_add(diagonal_op)
        assert isinstance(sum_op, DiagonalLinearOperator), "Identity + Diagonal should return DiagonalLinearOperator"
        expected_diagonal = torch.tensor([2.0, 3.0])  # 1.0 + [1.0, 2.0]
        assert torch.allclose(sum_op.diagonal_vector, expected_diagonal)
        
        # Test subtraction with IdentityLinearOperator
        diff_op = op.mat_sub(other_identity)
        assert isinstance(diff_op, ScalarLinearOperator), "Identity - Identity should return ScalarLinearOperator"
        assert torch.allclose(diff_op.scalar, torch.tensor(0.0)), "Identity - Identity should equal 0.0"
        
        # Test subtraction with ScalarLinearOperator
        diff_op = op.mat_sub(scalar_op)
        assert isinstance(diff_op, ScalarLinearOperator), "Identity - Scalar should return ScalarLinearOperator"
        assert torch.allclose(diff_op.scalar, torch.tensor(-2.0)), "Identity - 3.0 should equal -2.0"
        
        # Test subtraction with DiagonalLinearOperator
        diff_op = op.mat_sub(diagonal_op)
        assert isinstance(diff_op, DiagonalLinearOperator), "Identity - Diagonal should return DiagonalLinearOperator"
        expected_diagonal = torch.tensor([0.0, -1.0])  # 1.0 - [1.0, 2.0]
        assert torch.allclose(diff_op.diagonal_vector, expected_diagonal)
        
        # Test multiplication with IdentityLinearOperator
        prod_op = op.mat_mul(other_identity)
        assert isinstance(prod_op, IdentityLinearOperator), "Identity * Identity should return IdentityLinearOperator"
        
        # Test multiplication with ScalarLinearOperator
        prod_op = op.mat_mul(scalar_op)
        assert isinstance(prod_op, ScalarLinearOperator), "Identity * Scalar should return ScalarLinearOperator"
        assert torch.allclose(prod_op.scalar, torch.tensor(3.0)), "Identity * 3.0 should equal 3.0"
        
        # Test multiplication with DiagonalLinearOperator
        prod_op = op.mat_mul(diagonal_op)
        assert isinstance(prod_op, DiagonalLinearOperator), "Identity * Diagonal should return DiagonalLinearOperator"
        assert torch.allclose(prod_op.diagonal_vector, diagonal_op.diagonal_vector), "Identity * Diagonal should preserve diagonal"
        
        # Test multiplication with tensor
        y = op.mat_mul(x)
        assert torch.allclose(y, x), "Identity * tensor should preserve tensor"
    
    def test_matrix_operations_error_handling(self):
        """Test that matrix operations raise appropriate errors for unsupported types."""
        from gmi.linear_operator.conjugate import ConjugateLinearOperator
        
        op = IdentityLinearOperator()
        unsupported_op = ConjugateLinearOperator(base_matrix_operator=op)
        
        # Test addition with unsupported type
        with pytest.raises(ValueError, match="Addition with.*not supported"):
            op.mat_add(unsupported_op)
        
        # Test subtraction with unsupported type
        with pytest.raises(ValueError, match="Subtraction with.*not supported"):
            op.mat_sub(unsupported_op)
        
        # Test multiplication with unsupported type
        # Note: mat_mul returns the other operator for unsupported types, so no error
        result = op.mat_mul(unsupported_op)
        assert result == unsupported_op, "Multiplication with unsupported type should return the other operator"
    
    def test_logdet(self):
        """Test log determinant calculation."""
        op = IdentityLinearOperator()
        logdet = op.logdet()
        
        expected = torch.tensor(0.0)
        assert torch.allclose(logdet, expected), "Log determinant of identity should be 0.0"
    
    def test_inverse_LinearOperator(self):
        """Test inverse linear operator method."""
        op = IdentityLinearOperator()
        inv_op = op.inverse_LinearOperator()
        
        assert isinstance(inv_op, IdentityLinearOperator), "Inverse of identity should be identity"
        assert inv_op is not op, "Should return a new instance"
    
    def test_sqrt_LinearOperator(self):
        """Test square root linear operator method."""
        op = IdentityLinearOperator()
        sqrt_op = op.sqrt_LinearOperator()
        
        assert isinstance(sqrt_op, IdentityLinearOperator), "Square root of identity should be identity"
        assert sqrt_op is not op, "Should return a new instance"
    
    def test_inverse_operations(self, sample_vector_2):
        """Test that inverse operations work correctly."""
        op = IdentityLinearOperator()
        x = sample_vector_2
        
        # Test inverse method
        y = op.inverse(x)
        assert torch.allclose(y, x), "Inverse should preserve input"
        
        # Test inverse_LinearOperator method
        inv_op = op.inverse_LinearOperator()
        y_inv = inv_op.forward(x)
        assert torch.allclose(y_inv, x), "Inverse operator should preserve input"
        
        # Test that inverse operator is identity
        assert isinstance(inv_op, IdentityLinearOperator), "Inverse operator should be IdentityLinearOperator" 