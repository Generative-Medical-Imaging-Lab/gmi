# GMI Testing Framework

This directory contains the comprehensive test suite for the GMI (Generative Medical Imaging) package.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and common fixtures
├── linear_operator/         # Tests for linear operator modules
│   ├── test_base.py         # Base LinearOperator tests
│   ├── test_real.py         # RealLinearOperator tests
│   ├── test_symmetric.py    # SymmetricLinearOperator tests
│   ├── test_unitary.py      # UnitaryLinearOperator tests
│   └── test_invertible.py   # InvertibleLinearOperator tests
├── diffusion/               # Tests for diffusion modules (future)
├── network/                 # Tests for network modules (future)
├── datasets/                # Tests for dataset modules (future)
├── tasks/                   # Tests for task modules (future)
└── loss_function/           # Tests for loss function modules (future)
```

## Running Tests

### Using the Test Runner Script

```bash
# Run all tests
./run_tests.py

# Run only linear operator tests
./run_tests.py tests/linear_operator/

# Run with verbose output
./run_tests.py -v

# Run specific test markers
./run_tests.py -m "unit"

# List available test categories
./run_tests.py --list-tests

# Run without coverage reporting
./run_tests.py --no-coverage
```

### Using pytest Directly

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/linear_operator/test_base.py

# Run specific test class
python -m pytest tests/linear_operator/test_base.py::TestLinearOperator

# Run specific test method
python -m pytest tests/linear_operator/test_base.py::TestLinearOperator::test_abstract_class_instantiation

# Run with coverage
python -m pytest --cov=gmi --cov-report=html

# Run with markers
python -m pytest -m "unit"
```

## Test Categories

### Linear Operators

The linear operator tests cover the mathematical properties and implementations of various linear operator types:

- **Base LinearOperator**: Abstract base class and common functionality
- **RealLinearOperator**: Real-valued linear operators
- **SymmetricLinearOperator**: Symmetric linear operators
- **UnitaryLinearOperator**: Unitary linear operators
- **InvertibleLinearOperator**: Invertible linear operators

Each test verifies:
- Abstract class instantiation prevention
- Concrete subclass implementations
- Mathematical properties (symmetry, unitarity, etc.)
- Inheritance relationships
- Operator overloads (multiplication, matrix multiplication)

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `device`: Returns CPU or CUDA device
- `cpu_device`: Forces CPU device
- `temp_dir`: Creates temporary directory
- `sample_tensor_*`: Various tensor fixtures for testing
- `mock_wandb`: Mocks wandb logging
- `mock_hydra`: Mocks hydra configuration
- `sample_config_dict`: Sample configuration for testing

## Adding New Tests

### For New Modules

1. Create a new directory in `tests/` for your module
2. Add an `__init__.py` file
3. Create test files following the naming convention `test_*.py`
4. Use the existing fixtures from `conftest.py`

### Test File Structure

```python
"""
Tests for the ModuleName class.
"""
import pytest
import torch
from gmi.module_name import ModuleName


class TestModuleName:
    """Test cases for the ModuleName class."""
    
    def test_basic_functionality(self, sample_tensor_2d):
        """Test basic functionality."""
        # Your test code here
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Your test code here
        pass
```

### Test Guidelines

1. **Use descriptive test names**: Test method names should clearly describe what is being tested
2. **Test one thing at a time**: Each test should verify a single behavior
3. **Use appropriate assertions**: Use specific assertions like `torch.allclose()` for tensors
4. **Test both success and failure cases**: Include tests for error conditions
5. **Use fixtures**: Leverage the common fixtures for consistent test data
6. **Add docstrings**: Document what each test verifies

## Coverage

The test suite includes coverage reporting to ensure comprehensive testing:

- **Line coverage**: Percentage of code lines executed
- **Branch coverage**: Percentage of code branches executed
- **HTML reports**: Detailed coverage reports in `htmlcov/`

To view coverage reports:
```bash
# Generate HTML coverage report
python -m pytest --cov=gmi --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

- Tests run on both CPU and GPU (when available)
- Coverage reports are generated automatically
- Test results are reported in standard formats
- Integration with GitHub Actions or similar CI systems

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the GMI package is installed in development mode
   ```bash
   pip install -e .
   ```

2. **CUDA errors**: Tests automatically fall back to CPU if CUDA is not available

3. **Memory issues**: Use smaller tensor fixtures for memory-constrained environments

4. **Test discovery**: Ensure test files follow the naming convention `test_*.py`

### Debugging Tests

```bash
# Run with maximum verbosity
python -m pytest -vvv

# Run with print statements visible
python -m pytest -s

# Run specific failing test
python -m pytest tests/linear_operator/test_base.py::TestLinearOperator::test_specific_method -vvv -s
``` 