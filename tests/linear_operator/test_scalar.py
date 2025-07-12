"""
Tests for the ScalarLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.scalar import ScalarLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestScalarLinearOperator:
    """Test cases for the ScalarLinearOperator class."""
    
    def test_instantiation(self):
        """Test that ScalarLinearOperator can be instantiated directly."""
        op = ScalarLinearOperator(scalar=2.0)
        assert op is not None
        assert op.scalar == 2.0
    
    def test_forward_operation(self, sample_vector_2, sample_tensor_4d):
        """Test that scalar operator multiplies input by scalar."""
        op = ScalarLinearOperator(scalar=3.0)
        
        # Test with vector
        x = sample_vector_2
        y = op.forward(x)
        expected = 3.0 * x
        assert torch.allclose(y, expected), "Scalar operator should multiply by scalar"
        
        # Test with tensor
        x = sample_tensor_4d
        y = op.forward(x)
        expected = 3.0 * x
        assert torch.allclose(y, expected), "Scalar operator should multiply by scalar"
    
    def test_inheritance(self):
        """Test that ScalarLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        from gmi.linear_operator.square import SquareLinearOperator
        # from gmi.linear_operator.real import RealLinearOperator

        op = ScalarLinearOperator(scalar=2.0)
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        assert isinstance(op, SquareLinearOperator), "Should inherit from SquareLinearOperator"
        # ScalarLinearOperator is not always real, so we do not assert RealLinearOperator inheritance
    
    def test_invertible_inheritance(self):
        """Test that ScalarLinearOperator inherits from InvertibleLinearOperator."""
        from gmi.linear_operator.invertible import InvertibleLinearOperator
        
        op = ScalarLinearOperator(scalar=2.0)
        assert isinstance(op, InvertibleLinearOperator), "Should inherit from InvertibleLinearOperator"
    
    def test_is_invertible_property(self):
        """Test the is_invertible property."""
        # Test invertible scalar
        op = ScalarLinearOperator(scalar=2.0)
        assert op.is_invertible, "Non-zero scalar should be invertible"
        
        # Test non-invertible scalar
        op_zero = ScalarLinearOperator(scalar=0.0)
        assert not op_zero.is_invertible, "Zero scalar should not be invertible"
        
        # Test complex scalar
        op_complex = ScalarLinearOperator(scalar=1.0 + 2.0j)
        assert op_complex.is_invertible, "Non-zero complex scalar should be invertible"
        
        # Test tensor scalar
        op_tensor = ScalarLinearOperator(scalar=torch.tensor([1.0, 2.0, 3.0]))
        assert op_tensor.is_invertible, "Non-zero tensor scalar should be invertible"
        
        # Test tensor scalar with zeros
        op_tensor_zero = ScalarLinearOperator(scalar=torch.tensor([1.0, 0.0, 3.0]))
        assert not op_tensor_zero.is_invertible, "Tensor scalar with zeros should not be invertible"
    
    def test_config_instantiation(self):
        """Test that ScalarLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/scalar.yaml")
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, ScalarLinearOperator), "Should instantiate ScalarLinearOperator from config"
        assert torch.allclose(op.scalar, torch.tensor(3.0)), "Should load scalar from config"
    
    def test_scalar_properties(self, sample_vector_2):
        """Test that scalar operator has all expected properties."""
        op = ScalarLinearOperator(scalar=2.0)
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 2.0 * x)
        
        # Test transpose (should equal forward for scalar)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, 2.0 * x)
        
        # Test conjugate (should equal forward for real scalar)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, 2.0 * x)
        
        # Test conjugate_transpose (should equal forward for real scalar)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, 2.0 * x)
    
    def test_inverse_operations(self, sample_vector_2):
        """Test inverse operations for invertible scalar operators."""
        op = ScalarLinearOperator(scalar=2.0)
        x = sample_vector_2
        
        # Test inverse method
        y = op.inverse(x)
        expected = x / 2.0
        assert torch.allclose(y, expected), "Inverse should divide by scalar"
        
        # Test inverse_LinearOperator method
        inv_op = op.inverse_LinearOperator()
        assert isinstance(inv_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(inv_op.scalar, torch.tensor(0.5)), "Inverse operator should have reciprocal scalar"
        
        # Test that inverse operator works correctly
        y_inv = inv_op.forward(x)
        assert torch.allclose(y_inv, expected), "Inverse operator should give same result as inverse method"
    
    def test_non_invertible_operations(self, sample_vector_2):
        """Test that non-invertible scalar operators raise appropriate errors."""
        op_zero = ScalarLinearOperator(scalar=0.0)
        x = sample_vector_2
        
        # Test inverse method raises error
        with pytest.raises(ValueError, match="The scalar is zero"):
            op_zero.inverse(x)
        
        # Test inverse_LinearOperator method raises error
        with pytest.raises(ValueError, match="The scalar is zero"):
            op_zero.inverse_LinearOperator()
    
    def test_matrix_operations(self, sample_vector_2):
        """Test matrix operations with other operators."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        x = sample_vector_2
        
        # Test addition
        sum_op = op1.mat_add(op2)
        assert isinstance(sum_op, ScalarLinearOperator), "Addition should return ScalarLinearOperator"
        assert torch.allclose(sum_op.scalar, torch.tensor(5.0)), "Addition should sum scalars"
        
        # Test subtraction
        diff_op = op1.mat_sub(op2)
        assert isinstance(diff_op, ScalarLinearOperator), "Subtraction should return ScalarLinearOperator"
        assert torch.allclose(diff_op.scalar, torch.tensor(-1.0)), "Subtraction should subtract scalars"
        
        # Test multiplication
        prod_op = op1.mat_mul(op2)
        assert isinstance(prod_op, ScalarLinearOperator), "Multiplication should return ScalarLinearOperator"
        assert torch.allclose(prod_op.scalar, torch.tensor(6.0)), "Multiplication should multiply scalars"
        
        # Test multiplication with tensor
        y = op1.mat_mul(x)
        expected = 2.0 * x
        assert torch.allclose(y, expected), "Multiplication with tensor should apply forward"
    
    def test_sqrt_operation(self):
        """Test square root operation."""
        op = ScalarLinearOperator(scalar=4.0)
        sqrt_op = op.sqrt_LinearOperator()
        
        assert isinstance(sqrt_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(sqrt_op.scalar, torch.tensor(2.0)), "Square root should be 2.0"
    
    def test_logdet(self):
        """Test log determinant calculation."""
        op = ScalarLinearOperator(scalar=3.0)
        logdet = op.logdet()
        
        expected = torch.log(torch.tensor(3.0))
        assert torch.allclose(logdet, expected), "Log determinant should be log of absolute scalar"

    def test_different_scalars(self, sample_vector_2):
        """Test scalar operator with different scalar values."""
        x = sample_vector_2
        
        # Test with positive scalar
        op = ScalarLinearOperator(scalar=5.0)
        y = op.forward(x)
        assert torch.allclose(y, 5.0 * x)
        
        # Test with negative scalar
        op = ScalarLinearOperator(scalar=-2.0)
        y = op.forward(x)
        assert torch.allclose(y, -2.0 * x)
        
        # Test with zero scalar
        op = ScalarLinearOperator(scalar=0.0)
        y = op.forward(x)
        assert torch.allclose(y, torch.zeros_like(x))

    def test_transpose_LinearOperator(self):
        """Test the transpose_LinearOperator method."""
        # Test with real scalar
        op = ScalarLinearOperator(scalar=2.0)
        transpose_op = op.transpose_LinearOperator()
        
        assert isinstance(transpose_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(transpose_op.scalar, torch.tensor(2.0)), "Transpose should preserve scalar value"
        
        # Test with complex scalar
        op_complex = ScalarLinearOperator(scalar=1.0 + 2.0j)
        transpose_op_complex = op_complex.transpose_LinearOperator()
        
        assert isinstance(transpose_op_complex, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(transpose_op_complex.scalar, torch.tensor(1.0 + 2.0j)), "Transpose should preserve complex scalar value"
        
        # Test that transpose operator works correctly
        x = torch.tensor([1.0, 2.0, 3.0])
        y_original = op.forward(x)
        y_transpose = transpose_op.forward(x)
        assert torch.allclose(y_original, y_transpose), "Transpose operator should give same result as original"

    def test_conjugate_LinearOperator(self):
        """Test the conjugate_LinearOperator method."""
        # Test with real scalar
        op = ScalarLinearOperator(scalar=2.0)
        conjugate_op = op.conjugate_LinearOperator()
        
        assert isinstance(conjugate_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(conjugate_op.scalar, torch.tensor(2.0)), "Conjugate of real scalar should be unchanged"
        
        # Test with complex scalar
        op_complex = ScalarLinearOperator(scalar=1.0 + 2.0j)
        conjugate_op_complex = op_complex.conjugate_LinearOperator()
        
        assert isinstance(conjugate_op_complex, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(conjugate_op_complex.scalar, torch.tensor(1.0 - 2.0j)), "Conjugate should conjugate complex scalar"
        
        # Test that conjugate operator works correctly
        x = torch.tensor([1.0, 2.0, 3.0])
        y_original = op_complex.forward(x)
        y_conjugate = conjugate_op_complex.forward(x)
        expected_conjugate = torch.conj(y_original)
        assert torch.allclose(y_conjugate, expected_conjugate), "Conjugate operator should conjugate the result"

    def test_conjugate_transpose_LinearOperator(self):
        """Test the conjugate_transpose_LinearOperator method."""
        # Test with real scalar
        op = ScalarLinearOperator(scalar=2.0)
        conj_transpose_op = op.conjugate_transpose_LinearOperator()
        
        assert isinstance(conj_transpose_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(conj_transpose_op.scalar, torch.tensor(2.0)), "Conjugate transpose of real scalar should be unchanged"
        
        # Test with complex scalar
        op_complex = ScalarLinearOperator(scalar=1.0 + 2.0j)
        conj_transpose_op_complex = op_complex.conjugate_transpose_LinearOperator()
        
        assert isinstance(conj_transpose_op_complex, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(conj_transpose_op_complex.scalar, torch.tensor(1.0 - 2.0j)), "Conjugate transpose should conjugate complex scalar"
        
        # Test that conjugate transpose operator works correctly
        x = torch.tensor([1.0, 2.0, 3.0])
        y_original = op_complex.forward(x)
        y_conj_transpose = conj_transpose_op_complex.forward(x)
        expected_conj_transpose = torch.conj(y_original)
        assert torch.allclose(y_conj_transpose, expected_conj_transpose), "Conjugate transpose operator should conjugate the result"

    def test_operator_chain_operations(self, sample_vector_2):
        """Test chaining of the new LinearOperator methods."""
        op = ScalarLinearOperator(scalar=2.0)
        x = sample_vector_2
        
        # Test transpose -> inverse
        transpose_op = op.transpose_LinearOperator()
        inv_transpose_op = transpose_op.inverse_LinearOperator()
        assert isinstance(inv_transpose_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(inv_transpose_op.scalar, torch.tensor(0.5)), "Inverse of transpose should be 0.5"
        
        # Test inverse -> transpose
        inv_op = op.inverse_LinearOperator()
        transpose_inv_op = inv_op.transpose_LinearOperator()
        assert isinstance(transpose_inv_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(transpose_inv_op.scalar, torch.tensor(0.5)), "Transpose of inverse should be 0.5"
        
        # Test conjugate -> inverse
        conjugate_op = op.conjugate_LinearOperator()
        inv_conjugate_op = conjugate_op.inverse_LinearOperator()
        assert isinstance(inv_conjugate_op, ScalarLinearOperator), "Should return ScalarLinearOperator"
        assert torch.allclose(inv_conjugate_op.scalar, torch.tensor(0.5)), "Inverse of conjugate should be 0.5"
        
        # Test that operations commute correctly
        y1 = inv_transpose_op.forward(x)
        y2 = transpose_inv_op.forward(x)
        assert torch.allclose(y1, y2), "Inverse of transpose should equal transpose of inverse" 