"""
Tests for the TransposeLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.transpose import TransposeLinearOperator
from gmi.linear_operator.scalar import ScalarLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestTransposeLinearOperator:
    """Test cases for the TransposeLinearOperator class."""
    
    def test_instantiation(self):
        """Test that TransposeLinearOperator can be instantiated directly."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        assert op is not None
        assert op.base_matrix_operator == base_op
    
    def test_forward_operation(self, sample_vector_2):
        """Test that transpose operator applies transpose of base operator."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        
        x = sample_vector_2
        y = op.forward(x)
        # For scalar operator, transpose equals forward
        expected = 2.0 * x
        assert torch.allclose(y, expected), "Transpose operator should apply transpose of base operator"
    
    def test_inheritance(self):
        """Test that TransposeLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        # from gmi.linear_operator.real import RealLinearOperator

        base_op = ScalarLinearOperator(scalar=2.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        # TransposeLinearOperator should NOT inherit from RealLinearOperator
    
    def test_config_instantiation(self):
        """Test that TransposeLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/transpose.yaml")
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, TransposeLinearOperator), "Should instantiate TransposeLinearOperator from config"
    
    def test_transpose_properties(self, sample_vector_2):
        """Test that transpose operator has all expected properties."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test forward (applies transpose of base)
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 2.0 * x)
        
        # Test transpose (should apply transpose of transpose = original)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, 2.0 * x)
        
        # Test conjugate (should conjugate the transpose)
        y_conjugate = op.conjugate(x)
        # For real scalar operator, conjugate equals forward
        assert torch.allclose(y_conjugate, 2.0 * x)
        
        # Test conjugate_transpose (should apply conjugate of transpose)
        y_conj_transpose = op.conjugate_transpose(x)
        # For real scalar operator, conjugate_transpose equals forward
        assert torch.allclose(y_conj_transpose, 2.0 * x)
    
    def test_transpose_of_transpose(self, sample_vector_2):
        """Test that transpose of transpose equals original."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Apply transpose operator
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
        
        # Apply transpose of transpose (should equal original base operator)
        op_transpose = TransposeLinearOperator(base_matrix_operator=op)
        y_transpose_of_transpose = op_transpose.forward(x)
        assert torch.allclose(y_transpose_of_transpose, 2.0 * x)
    
    def test_with_different_base_operator(self, sample_vector_2):
        """Test transpose operator with different base operators."""
        x = sample_vector_2
        
        # Test with different scalar
        base_op = ScalarLinearOperator(scalar=3.0)
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, 3.0 * x)
        
        # Test with identity operator
        from gmi.linear_operator.identity import IdentityLinearOperator
        base_op = IdentityLinearOperator()
        op = TransposeLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x) 