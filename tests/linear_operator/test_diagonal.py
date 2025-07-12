"""
Tests for the DiagonalLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.diagonal import DiagonalLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestDiagonalLinearOperator:
    """Test cases for the DiagonalLinearOperator class."""
    
    def test_instantiation(self):
        """Test that DiagonalLinearOperator can be instantiated directly."""
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        assert op is not None
        assert torch.allclose(op.diagonal_vector, diagonal)
    
    def test_forward_operation(self, sample_vector_2):
        """Test that diagonal operator multiplies input by diagonal elements."""
        diagonal = torch.tensor([2.0, 3.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        
        x = sample_vector_2
        y = op.forward(x)
        expected = diagonal * x
        assert torch.allclose(y, expected), "Diagonal operator should multiply by diagonal elements"
    
    def test_inheritance(self):
        """Test that DiagonalLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        from gmi.linear_operator.square import SquareLinearOperator
        from gmi.linear_operator.symmetric import SymmetricLinearOperator
        from gmi.linear_operator.invertible import InvertibleLinearOperator

        op = DiagonalLinearOperator(diagonal_vector=torch.tensor([1.0, 2.0]))
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        assert isinstance(op, SquareLinearOperator), "Should inherit from SquareLinearOperator"
        assert isinstance(op, SymmetricLinearOperator), "Should inherit from SymmetricLinearOperator"
        assert isinstance(op, InvertibleLinearOperator), "Should inherit from InvertibleLinearOperator"
    
    def test_config_instantiation(self):
        """Test that DiagonalLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/diagonal.yaml")
        # Instantiate the linear_operator part directly to avoid wrapping
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, DiagonalLinearOperator), "Should instantiate DiagonalLinearOperator from config"
        expected_diagonal = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(op.diagonal_vector, expected_diagonal), "Should load diagonal vector from config"
    
    def test_diagonal_properties(self, sample_vector_2):
        """Test that diagonal operator has all expected properties."""
        diagonal = torch.tensor([2.0, 3.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, diagonal * x)
        
        # Test transpose (should equal forward for diagonal)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, diagonal * x)
        
        # Test conjugate (should equal forward for real diagonal)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, diagonal * x)
        
        # Test conjugate_transpose (should equal forward for real diagonal)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, diagonal * x)
        
        # Test inverse (should divide by diagonal elements)
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, x / diagonal)
    
    def test_different_diagonals(self, sample_vector_2):
        """Test diagonal operator with different diagonal values."""
        x = sample_vector_2
        
        # Test with positive diagonal
        diagonal = torch.tensor([1.0, 2.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        y = op.forward(x)
        assert torch.allclose(y, diagonal * x)
        
        # Test with negative diagonal
        diagonal = torch.tensor([-1.0, -2.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        y = op.forward(x)
        assert torch.allclose(y, diagonal * x)
        
        # Test with zero diagonal
        diagonal = torch.tensor([0.0, 1.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        y = op.forward(x)
        expected = torch.tensor([0.0, x[1]])
        assert torch.allclose(y, expected)
    
    def test_shape_validation(self):
        """Test that diagonal operator validates input shapes."""
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalLinearOperator(diagonal_vector=diagonal)
        
        # Should work with matching size
        x = torch.tensor([1.0, 2.0, 3.0])
        y = op.forward(x)
        assert y.shape == x.shape
        
        # Should raise error with mismatched size
        x = torch.tensor([1.0, 2.0])  # Wrong size
        with pytest.raises(Exception):
            op.forward(x)
    
    def test_is_invertible_property(self):
        """Test the is_invertible property."""
        # Test invertible diagonal
        op = DiagonalLinearOperator(diagonal_vector=torch.tensor([1.0, 2.0, 3.0]))
        assert op.is_invertible, "Non-zero diagonal should be invertible"
        
        # Test non-invertible diagonal (with zero)
        op_zero = DiagonalLinearOperator(diagonal_vector=torch.tensor([1.0, 0.0, 3.0]))
        assert not op_zero.is_invertible, "Diagonal with zero should not be invertible"
        
        # Test all zeros
        op_all_zero = DiagonalLinearOperator(diagonal_vector=torch.tensor([0.0, 0.0, 0.0]))
        assert not op_all_zero.is_invertible, "All-zero diagonal should not be invertible"
        
        # Test single element
        op_single = DiagonalLinearOperator(diagonal_vector=torch.tensor([2.0]))
        assert op_single.is_invertible, "Single non-zero element should be invertible"
        
        op_single_zero = DiagonalLinearOperator(diagonal_vector=torch.tensor([0.0]))
        assert not op_single_zero.is_invertible, "Single zero element should not be invertible" 