"""
Tests for the SquareLinearOperator class.
"""
import pytest
import torch
from gmi.linear_operator.square import SquareLinearOperator

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../../configs")

class TestSquareLinearOperator:
    def test_abstract_class_instantiation(self):
        """Test that SquareLinearOperator cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SquareLinearOperator()

    def test_concrete_subclass_implementation(self, sample_vector_2):
        """Test a concrete subclass of SquareLinearOperator works correctly."""
        from gmi.linear_operator.real import RealLinearOperator
        class DoubleOperator(SquareLinearOperator, RealLinearOperator):
            def __init__(self):
                super().__init__()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        double_op = DoubleOperator()
        x = sample_vector_2
        y = double_op.forward(x)
        assert torch.allclose(y, 2.0 * x), f"Expected {2.0 * x}, got {y}"

    def test_input_shape_equals_output_shape(self):
        """Test that input_shape_given_output_shape returns the same shape for square operators."""
        from gmi.linear_operator.real import RealLinearOperator
        class DummySquareOperator(SquareLinearOperator, RealLinearOperator):
            def forward(self, x):
                return x
        op = DummySquareOperator()
        output_shape = (2, 3, 4)
        input_shape = op.input_shape_given_output_shape(output_shape)
        assert input_shape == output_shape, f"Expected {output_shape}, got {input_shape}"
    
    def test_config_instantiation(self):
        """Test that SquareLinearOperator cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_operator/square.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg.linear_operator) 