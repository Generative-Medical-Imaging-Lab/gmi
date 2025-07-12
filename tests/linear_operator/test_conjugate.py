"""
Tests for the ConjugateLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.conjugate import ConjugateLinearOperator
from gmi.linear_operator.scalar import ScalarLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestConjugateLinearOperator:
    """Test cases for the ConjugateLinearOperator class."""
    
    def test_instantiation(self):
        """Test that ConjugateLinearOperator can be instantiated directly."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        assert op is not None
        assert op.base_matrix_operator == base_op
    
    def test_forward_operation(self, sample_vector_2):
        """Test that conjugate operator applies conjugate of base operator."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        
        x = sample_vector_2
        y = op.forward(x)
        # For real scalar operator, conjugate equals forward
        expected = 2.0 * x
        assert torch.allclose(y, expected), "Conjugate operator should apply conjugate of base operator"
    
    def test_inheritance(self):
        """Test that ConjugateLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        # from gmi.linear_operator.real import RealLinearOperator

        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        # ConjugateLinearOperator should NOT inherit from RealLinearOperator
    
    def test_config_instantiation(self):
        """Test that ConjugateLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/conjugate.yaml")
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, ConjugateLinearOperator), "Should instantiate ConjugateLinearOperator from config"
    
    def test_conjugate_properties(self, sample_vector_2):
        """Test that conjugate operator has all expected properties."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test forward (applies conjugate of base)
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 2.0 * x)
        
        # Test conjugate (should apply conjugate of conjugate = original)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, 2.0 * x)
        
        # Test transpose (should transpose the conjugate)
        y_transpose = op.transpose(x)
        # For real scalar operator, transpose equals forward
        assert torch.allclose(y_transpose, 2.0 * x)
        
        # Test conjugate_transpose (should apply conjugate of transpose)
        y_conj_transpose = op.conjugate_transpose(x)
        # For real scalar operator, conjugate_transpose equals forward
        assert torch.allclose(y_conj_transpose, 2.0 * x)
    
    def test_conjugate_of_conjugate(self, sample_vector_2):
        """Test that conjugate of conjugate equals original."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Apply conjugate operator
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
        
        # Apply conjugate of conjugate (should equal original base operator)
        op_conjugate = ConjugateLinearOperator(base_matrix_operator=op)
        y_conjugate_of_conjugate = op_conjugate.forward(x)
        assert torch.allclose(y_conjugate_of_conjugate, 2.0 * x)
    
    def test_with_different_base_operator(self, sample_vector_2):
        """Test conjugate operator with different base operators."""
        x = sample_vector_2
        
        # Test with different scalar
        base_op = ScalarLinearOperator(scalar=3.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, 3.0 * x)
        
        # Test with identity operator
        from gmi.linear_operator.identity import IdentityLinearOperator
        base_op = IdentityLinearOperator()
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_with_complex_input(self, sample_vector_2):
        """Test conjugate operator with complex input."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateLinearOperator(base_matrix_operator=base_op)
        
        # Create complex input
        x_complex = sample_vector_2 + 1j * sample_vector_2
        y = op.forward(x_complex)
        
        # For real scalar operator, conjugate should preserve complex input
        expected = 2.0 * x_complex
        assert torch.allclose(y, expected) 