"""
Tests for the CompositeLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.composite import CompositeLinearOperator
from gmi.linear_operator.scalar import ScalarLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestCompositeLinearOperator:
    """Test cases for the CompositeLinearOperator class."""
    
    def test_instantiation(self):
        """Test that CompositeLinearOperator can be instantiated directly."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        op = CompositeLinearOperator(matrix_operators=[op1, op2])
        assert op is not None
        assert len(op.matrix_operators) == 2
    
    def test_forward_operation(self, sample_vector_2):
        """Test that composite operator applies operators in sequence."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        op = CompositeLinearOperator(matrix_operators=[op1, op2])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 6.0 * x  # 2.0 * 3.0 * x
        assert torch.allclose(y, expected), "Composite operator should apply operators in sequence"
    
    def test_inheritance(self):
        """Test that CompositeLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        # from gmi.linear_operator.real import RealLinearOperator

        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        op = CompositeLinearOperator(matrix_operators=[op1, op2])
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        # CompositeLinearOperator should NOT inherit from RealLinearOperator
    
    def test_config_instantiation(self):
        """Test that CompositeLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/composite.yaml")
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, CompositeLinearOperator), "Should instantiate CompositeLinearOperator from config"
    
    def test_composite_properties(self, sample_vector_2):
        """Test that composite operator has all expected properties."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        op = CompositeLinearOperator(matrix_operators=[op1, op2])
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 6.0 * x)
        
        # Test transpose (should reverse order and transpose each)
        y_transpose = op.transpose(x)
        # For scalar operators, transpose equals forward, so result should be same
        assert torch.allclose(y_transpose, 6.0 * x)
        
        # Test conjugate (should conjugate each operator)
        y_conjugate = op.conjugate(x)
        # For real scalar operators, conjugate equals forward
        assert torch.allclose(y_conjugate, 6.0 * x)
        
        # Test conjugate_transpose (should reverse order and conjugate_transpose each)
        y_conj_transpose = op.conjugate_transpose(x)
        # For real scalar operators, conjugate_transpose equals forward
        assert torch.allclose(y_conj_transpose, 6.0 * x)
    
    def test_empty_composite(self, sample_vector_2):
        """Test composite operator with empty list of operators."""
        op = CompositeLinearOperator(matrix_operators=[])
        x = sample_vector_2
        
        # Should act as identity operator
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_single_operator(self, sample_vector_2):
        """Test composite operator with single operator."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op = CompositeLinearOperator(matrix_operators=[op1])
        
        x = sample_vector_2
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
    
    def test_multiple_operators(self, sample_vector_2):
        """Test composite operator with multiple operators."""
        op1 = ScalarLinearOperator(scalar=2.0)
        op2 = ScalarLinearOperator(scalar=3.0)
        op3 = ScalarLinearOperator(scalar=4.0)
        op = CompositeLinearOperator(matrix_operators=[op1, op2, op3])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 24.0 * x  # 2.0 * 3.0 * 4.0 * x
        assert torch.allclose(y, expected) 