"""
Tests for the ConjugateTransposeLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.conjugate_transpose import ConjugateTransposeLinearOperator
from gmi.linear_operator.scalar import ScalarLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestConjugateTransposeLinearOperator:
    """Test cases for the ConjugateTransposeLinearOperator class."""
    
    def test_instantiation(self):
        """Test that ConjugateTransposeLinearOperator can be instantiated directly."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        assert op is not None
        assert op.base_matrix_operator == base_op
    
    def test_forward_operation(self, sample_vector_2):
        """Test that conjugate transpose operator applies conjugate transpose of base operator."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        
        x = sample_vector_2
        y = op.forward(x)
        # For real scalar operator, conjugate transpose equals forward
        expected = 2.0 * x
        assert torch.allclose(y, expected), "Conjugate transpose operator should apply conjugate transpose of base operator"
    
    def test_inheritance(self):
        """Test that ConjugateTransposeLinearOperator inherits correctly."""
        from gmi.linear_operator.base import LinearOperator
        # from gmi.linear_operator.real import RealLinearOperator

        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        assert isinstance(op, LinearOperator), "Should inherit from LinearOperator"
        # ConjugateTransposeLinearOperator should NOT inherit from RealLinearOperator
    
    def test_config_instantiation(self):
        """Test that ConjugateTransposeLinearOperator can be instantiated from config."""
        cfg = compose(config_name="linear_operator/conjugate_transpose.yaml")
        op = instantiate(cfg.linear_operator)
        assert isinstance(op, ConjugateTransposeLinearOperator), "Should instantiate ConjugateTransposeLinearOperator from config"
        assert isinstance(op.base_matrix_operator, ScalarLinearOperator), "Should load base operator from config"
        assert op.base_matrix_operator.scalar == 2.0, "Should load base operator parameters from config"
    
    def test_conjugate_transpose_properties(self, sample_vector_2):
        """Test that conjugate transpose operator has all expected properties."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test forward (applies conjugate transpose of base)
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 2.0 * x)
        
        # Test conjugate_transpose (should apply conjugate_transpose of conjugate_transpose = original)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, 2.0 * x)
        
        # Test transpose (should transpose the conjugate transpose)
        y_transpose = op.transpose(x)
        # For real scalar operator, transpose equals forward
        assert torch.allclose(y_transpose, 2.0 * x)
        
        # Test conjugate (should conjugate the conjugate transpose)
        y_conjugate = op.conjugate(x)
        # For real scalar operator, conjugate equals forward
        assert torch.allclose(y_conjugate, 2.0 * x)
    
    def test_conjugate_transpose_of_conjugate_transpose(self, sample_vector_2):
        """Test that conjugate transpose of conjugate transpose equals original."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Apply conjugate transpose operator
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
        
        # Apply conjugate transpose of conjugate transpose (should equal original base operator)
        op_conj_transpose = ConjugateTransposeLinearOperator(base_matrix_operator=op)
        y_conj_transpose_of_conj_transpose = op_conj_transpose.forward(x)
        assert torch.allclose(y_conj_transpose_of_conj_transpose, 2.0 * x)
    
    def test_with_different_base_operator(self, sample_vector_2):
        """Test conjugate transpose operator with different base operators."""
        x = sample_vector_2
        
        # Test with different scalar
        base_op = ScalarLinearOperator(scalar=3.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, 3.0 * x)
        
        # Test with identity operator
        from gmi.linear_operator.identity import IdentityLinearOperator
        base_op = IdentityLinearOperator()
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_with_complex_input(self, sample_vector_2):
        """Test conjugate transpose operator with complex input."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        
        # Create complex input
        x_complex = sample_vector_2 + 1j * sample_vector_2
        y = op.forward(x_complex)
        
        # For real scalar operator, conjugate transpose should preserve complex input
        expected = 2.0 * x_complex
        assert torch.allclose(y, expected)
    
    def test_relationship_with_transpose_and_conjugate(self, sample_vector_2):
        """Test that conjugate transpose equals conjugate of transpose and transpose of conjugate."""
        base_op = ScalarLinearOperator(scalar=2.0)
        op = ConjugateTransposeLinearOperator(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test that conjugate_transpose equals conjugate of transpose
        from gmi.linear_operator.transpose import TransposeLinearOperator
        from gmi.linear_operator.conjugate import ConjugateLinearOperator
        
        transpose_op = TransposeLinearOperator(base_matrix_operator=base_op)
        conjugate_of_transpose = ConjugateLinearOperator(base_matrix_operator=transpose_op)
        
        y1 = op.forward(x)
        y2 = conjugate_of_transpose.forward(x)
        assert torch.allclose(y1, y2)
        
        # Test that conjugate_transpose equals transpose of conjugate
        conjugate_op = ConjugateLinearOperator(base_matrix_operator=base_op)
        transpose_of_conjugate = TransposeLinearOperator(base_matrix_operator=conjugate_op)
        
        y3 = transpose_of_conjugate.forward(x)
        assert torch.allclose(y1, y3) 