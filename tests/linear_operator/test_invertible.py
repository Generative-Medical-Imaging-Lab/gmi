"""
Tests for the InvertibleLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.invertible import InvertibleLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")


class TestInvertibleLinearOperator:
    """Test cases for the InvertibleLinearOperator class."""
    
    def test_abstract_class_instantiation(self):
        """Test that InvertibleLinearOperator cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            InvertibleLinearOperator()
    
    def test_concrete_subclass_implementation(self, sample_tensor_2d):
        """Test that a concrete subclass works correctly."""
        class SimpleInvertibleOperator(InvertibleLinearOperator):
            def __init__(self, factor=2.0):
                super().__init__()
                self.factor = factor

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.factor * x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y / self.factor
        
        op = SimpleInvertibleOperator(factor=3.0)
        x = sample_tensor_2d
        
        # Test forward
        y = op.forward(x)
        expected_forward = 3.0 * x
        assert torch.allclose(y, expected_forward), f"Forward mismatch: {y} vs {expected_forward}"
        
        # Test inverse
        x_recovered = op.inverse(y)
        assert torch.allclose(x_recovered, x), f"Inverse mismatch: {x_recovered} vs {x}"
        
        # Test that forward and inverse are truly inverses
        x_double_recovered = op.inverse(op.forward(x))
        assert torch.allclose(x_double_recovered, x), "Double inverse should recover original"
    
    def test_square_property_inheritance(self, sample_tensor_3d):
        """Test that InvertibleLinearOperator inherits square property correctly."""
        class SquareInvertibleOperator(InvertibleLinearOperator):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x  # Identity operator
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y  # Identity operator
        
        op = SquareInvertibleOperator()
        
        # Test square property: input_shape = output_shape
        output_shape = (2, 3, 4)
        input_shape = op.input_shape_given_output_shape(output_shape)
        assert input_shape == output_shape, f"Expected {output_shape}, got {input_shape}"
        
        # Test that it's a square operator
        from gmi.linear_operator.square import SquareLinearOperator
        assert isinstance(op, SquareLinearOperator)
    
    def test_inverse_linear_operator_method(self, sample_tensor_2d):
        """Test the inverse_LinearOperator method."""
        class ScaleInvertibleOperator(InvertibleLinearOperator):
            def __init__(self, scale=2.0):
                super().__init__()
                self.scale = scale

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scale * x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y / self.scale
        
        op = ScaleInvertibleOperator(scale=2.0)
        x = sample_tensor_2d
        
        # Test that inverse_LinearOperator returns an InverseLinearOperator
        inverse_op = op.inverse_LinearOperator()
        from gmi.linear_operator.inverse import InverseLinearOperator
        assert isinstance(inverse_op, InverseLinearOperator)
        
        # Test that the inverse operator works correctly
        y = op.forward(x)
        x_recovered = inverse_op.forward(y)
        assert torch.allclose(x_recovered, x), "InverseLinearOperator should recover original input"
    
    def test_invertible_property_verification(self, sample_vector_2):
        """Test that invertible operators satisfy the invertible property."""
        # Create a simple invertible operator (scalar)
        from gmi.linear_operator.scalar import ScalarLinearOperator
        op = ScalarLinearOperator(scalar=2.0)
        
        # Use the sample vector directly, don't reshape
        x = sample_vector_2
        
        # Apply forward then inverse
        y = op.forward(x)
        x_recovered = op.inverse(y)
        
        # Should recover original input
        assert torch.allclose(x_recovered, x, atol=1e-6), "Invertible operator should recover original input"
        
        # Apply inverse then forward
        y_inverse = op.inverse(x)
        x_recovered_2 = op.forward(y_inverse)
        
        # Should also recover original input
        assert torch.allclose(x_recovered_2, x, atol=1e-6), "Invertible operator should recover original input in both directions"
    
    def test_inheritance_from_square_operator(self, sample_tensor_2d):
        """Test that InvertibleLinearOperator inherits correctly from SquareLinearOperator."""
        class IdentityInvertibleOperator(InvertibleLinearOperator):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y
        
        op = IdentityInvertibleOperator()
        x = sample_tensor_2d
        
        # Test that it's a SquareLinearOperator
        from gmi.linear_operator.square import SquareLinearOperator
        assert isinstance(op, SquareLinearOperator)
        
        # Test that it's a LinearOperator
        from gmi.linear_operator.base import LinearOperator
        assert isinstance(op, LinearOperator)
        
        # Test basic operations
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
        
        x_recovered = op.inverse(y)
        assert torch.allclose(x_recovered, x), "Identity inverse should preserve input"
    
    def test_config_instantiation(self):
        """Test that InvertibleLinearOperator cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_operator/invertible.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg.linear_operator) 