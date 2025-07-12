import torch
from .composite import CompositeLinearOperator
from .invertible import InvertibleLinearOperator

class InvertibleCompositeLinearOperator(CompositeLinearOperator, InvertibleLinearOperator):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple invertible linear operators.
        
        It inherits from CompositeLinearOperator and InvertibleLinearOperator.
        
        parameters:
            matrix_operators: list of InvertibleLinearOperator objects
                The list of invertible linear operators to be composed. The product is taken in the order they are provided.
        """
        # Convert to list if it's a ListConfig from Hydra
        if hasattr(matrix_operators, '__iter__') and not isinstance(matrix_operators, list):
            matrix_operators = list(matrix_operators)
        
        assert isinstance(matrix_operators, list), "The operators should be provided as a list of InvertibleLinearOperator objects."
        assert len(matrix_operators) > 0, "At least one operator should be provided."
        
        # Instantiate operators if they're configs and validate they're invertible
        instantiated_operators = []
        for operator in matrix_operators:
            if hasattr(operator, '_target_'):
                # This is a config, need to instantiate it
                from hydra.utils import instantiate
                instantiated_operator = instantiate(operator)
                instantiated_operators.append(instantiated_operator)
            else:
                # This is already an operator
                instantiated_operators.append(operator)
            
            # Validate that all operators are invertible
            assert isinstance(instantiated_operators[-1], InvertibleLinearOperator), "All operators should be InvertibleLinearOperator objects."
        
        # Initialize the composite operator
        super().__init__(instantiated_operators)
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this composite linear operator is invertible.
        
        A composite operator is invertible if and only if all its component operators are invertible.
        
        returns:
            bool: True if all component operators are invertible, False otherwise.
        """
        return all(operator.is_invertible for operator in self.matrix_operators)
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the invertible composite linear operator.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the invertible composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the invertible composite linear operator to the input tensor.
        """
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    
    def inverse_LinearOperator(self):
        """
        This method returns the inverse linear operator.
        
        returns:
            result: InvertibleCompositeLinearOperator
                The inverse linear operator.
        """
        return InvertibleCompositeLinearOperator([operator.inverse_LinearOperator() for operator in reversed(self.matrix_operators)])


if __name__ == "__main__":
    def test_from_python():
        print("Testing InvertibleCompositeLinearOperator from Python...")
        try:
            from .scalar import ScalarLinearOperator
            from .diagonal import DiagonalLinearOperator
            
            # Test with two invertible scalar operators
            op1 = ScalarLinearOperator(2.0)
            op2 = ScalarLinearOperator(3.0)
            composite = InvertibleCompositeLinearOperator([op1, op2])
            
            x = torch.tensor([1.0, 2.0, 3.0])
            
            # Test forward (should be 2.0 * 3.0 * x = 6.0 * x)
            y = composite.forward(x)
            expected = torch.tensor([6.0, 12.0, 18.0])
            assert torch.allclose(y, expected), f"Forward mismatch: {y} vs {expected}"
            print("SUCCESS: InvertibleCompositeLinearOperator forward works correctly")
            
            # Test inverse (should be (1/3.0) * (1/2.0) * x = (1/6.0) * x)
            x_recovered = composite.inverse(y)
            assert torch.allclose(x_recovered, x), f"Inverse mismatch: {x_recovered} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearOperator inverse works correctly")
            
            # Test inverse_LinearOperator
            inverse_op = composite.inverse_LinearOperator()
            x_recovered2 = inverse_op.forward(y)
            assert torch.allclose(x_recovered2, x), f"Inverse operator mismatch: {x_recovered2} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearOperator inverse_LinearOperator works correctly")
            
            # Test with diagonal and scalar operators
            diag_op = DiagonalLinearOperator(torch.tensor([1.0, 2.0, 3.0]))
            scalar_op = ScalarLinearOperator(2.0)
            composite2 = InvertibleCompositeLinearOperator([diag_op, scalar_op])
            
            y2 = composite2.forward(x)
            expected2 = torch.tensor([2.0, 8.0, 18.0])  # diag([1,2,3]) * 2 * [1,2,3]
            assert torch.allclose(y2, expected2), f"Complex forward mismatch: {y2} vs {expected2}"
            print("SUCCESS: InvertibleCompositeLinearOperator with diagonal and scalar works correctly")
            
            # Test inverse of complex operator
            x_recovered3 = composite2.inverse(y2)
            assert torch.allclose(x_recovered3, x), f"Complex inverse mismatch: {x_recovered3} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearOperator complex inverse works correctly")
            
            return True
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_from_config():
        print("Testing InvertibleCompositeLinearOperator from config...")
        try:
            from hydra import compose, initialize
            from hydra.utils import instantiate
            with initialize(version_base=None, config_path="../../configs"):
                cfg = compose(config_name="linear_operator/invertible_composite.yaml")
                op = instantiate(cfg.linear_operator)
                
                # Test the instantiated operator
                x = torch.tensor([1.0, 2.0, 3.0])
                y = op.forward(x)
                expected = torch.tensor([6.0, 12.0, 18.0])  # 2.0 * 3.0 * x
                assert torch.allclose(y, expected), f"Config forward mismatch: {y} vs {expected}"
                
                # Test inverse
                x_recovered = op.inverse(y)
                assert torch.allclose(x_recovered, x), f"Config inverse mismatch: {x_recovered} vs {x}"
                print("SUCCESS: InvertibleCompositeLinearOperator from config works correctly")
                return True
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success_python = test_from_python()
    success_config = test_from_config()
    if success_python and success_config:
        print("All tests passed!")
    else:
        print("Some tests failed!") 