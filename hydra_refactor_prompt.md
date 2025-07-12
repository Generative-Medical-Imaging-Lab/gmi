# Hydra Refactor Plan for GMI

**Last updated:** 2024-12-19

## 1. Goals

1. Adopt Hydra for all configuration management across the GMI code-base.
2. Replace custom `load_*_from_config` helpers with Hydra‐native `_target_` object instantiation.
3. Provide a complete, granular configuration hierarchy covering **every** object that can be constructed inside `gmi` (datasets, models, SDEs, linear operators, distributions, samplers, losses, trainers, etc.).
4. Harmonise command-line entry points so they are *thin* wrappers around a single Hydra-driven `run(cfg: DictConfig)` function (no heavy lifting inside command classes).
5. Eliminate ad-hoc outputs under `gmi_data/outputs/`.  Examples should write inside their own directory; library code remains pure.
6. Enforce clean package structure:
   • **One class / function per Python file.**
   • `__init__.py` files must **only** perform import re-exports (no logic).
7. Every public object *must* ship with two tests:  
   • **`test_from_python`** – direct instantiation in Python.  
   • **`test_from_config`** – instantiation via the matching Hydra YAML.
8. Create an explicit, single source-of-truth for important paths (``GMI_BASE`` env-var or helper) so commands know where they run.
9. Introduce a common **RandomVariableModel** ABC – `DiffusionModel` and `ImageReconstructor` will subclass this (final stage).

> **Coding style:** *Do not use emojis in any code or config files.*

---

## 2. Configuration Layout (`configs/`)

```
configs/
│
├── dataset/
│   ├── mnist.yaml
│   ├── medmnist_blood.yaml
│   └── ...
│
├── model/
│   ├── diffusion/
│   │   └── unet28.yaml
│   └── reconstruction/
│       └── linear_conv.yaml
│
├── sde/
│   └── scalar.yaml
├── linalg/
│   └── fourier.yaml
├── distribution/
│   └── gaussian_awgn.yaml
├── sampler/
│   └── dataloader.yaml
├── loss_function/
│   └── mse.yaml
├── lr_scheduler/
│   └── linear_warmup.yaml
└── training/
    └── default.yaml
```
Each YAML contains a single object with its `_target_` and arguments.

### Example: `configs/dataset/mnist.yaml`
```yaml
_target_: gmi.datasets.mnist.MNIST
train: true
images_only: true
```

---

## 3. Command Refactor

1. Replace `gmi.commands.train_diffusion_model` & `train_image_reconstructor` with:
   ```python
   # gmi/commands/train_diffusion.py
   import hydra
   from omegaconf import DictConfig

   @hydra.main(config_path="../../configs", config_name="train/diffusion_default")
   def run(cfg: DictConfig):
       model = hydra.utils.instantiate(cfg.model)
       trainer = hydra.utils.instantiate(cfg.trainer)
       trainer.fit(model)
   ```
2. Similar structure for reconstruction.
3. CLI entry (`main.py`) just forwards to these `run` functions.

---

## 4. Tests

*Use pytest.*  For each python file `<module>.py`:
```
tests/
└── <module>/
    ├── test_from_python.py
    └── test_from_config.py
```

`test_from_config.py` example:
```python
from hydra import compose, initialize
from hydra.utils import instantiate

def test_mnist_from_cfg():
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="dataset/mnist.yaml")
        ds = instantiate(cfg)
    assert len(ds) > 0
```

---

## 5. Current Progress (Linear Operator Refactor)

### ✅ Completed
- **Inline Test Removal**: Removed all `__main__` blocks and inline test functions from `gmi/linear_operator/*.py` files
- **Inheritance Fixes**: 
  - `DiagonalLinearOperator`, `ScalarLinearOperator`, `IdentityLinearOperator` now inherit from `RealLinearOperator`
  - Wrapper operators (`TransposeLinearOperator`, `ConjugateLinearOperator`, `ConjugateTransposeLinearOperator`) inherit from `LinearOperator` (not `SquareLinearOperator` since transpose of non-square matrix is not square)
  - `CompositeLinearOperator` and `InvertibleCompositeLinearOperator` inherit from `LinearOperator` (composites may not be square)
- **Method Implementations**: Added missing `transpose()` and `conjugate_transpose()` methods to real operators
- **Parameter Naming**: Standardized all `transpose` and `conjugate_transpose` methods to use `y` as input parameter
- **Test Suite**: Comprehensive pytest files exist for all linear operator classes

### 🔄 In Progress
- **Config Instantiation Tests**: Fixing test expectations to match Hydra YAML structure
- **Method Implementations**: Adding missing methods to ensure no `NotImplementedError` calls

### 📋 Remaining Tasks
1. **Fix Config Tests**: Update remaining test files to expect correct Hydra instantiation structure
2. **Test Suite Validation**: Run full test suite and fix any remaining failures
3. **Extend to Other Modules**: Apply same refactor pattern to other GMI modules (datasets, models, etc.)
4. **Command Refactor**: Update CLI commands to use Hydra-driven approach
5. **Path Management**: Implement `GMI_BASE` environment variable or helper

### 🎯 Current Strategy
1. **Linear Operators First**: Complete the linear operator refactor as a template
2. **Incremental Testing**: Run tests after each change to catch issues early
3. **Method Completeness**: Ensure all abstract methods are implemented before moving to next module
4. **Config Alignment**: Make sure YAML configs and test expectations match

---

## 6. Path Management

Introduce `GMI_BASE` environment variable or helper function to provide single source of truth for project paths.

---

## 7. Testing Strategy

### Current Test Structure
- Each linear operator has comprehensive pytest file
- Tests cover: instantiation, inheritance, properties, config instantiation
- Config tests use `hydra.utils.instantiate()` with proper initialization

### Test Categories
1. **Direct Instantiation**: `test_instantiation()` - basic object creation
2. **Inheritance Verification**: `test_inheritance()` - check class hierarchy
3. **Property Testing**: `test_*_properties()` - verify mathematical properties
4. **Config Instantiation**: `test_config_instantiation()` - Hydra YAML loading
5. **Error Handling**: Test edge cases and invalid inputs

### Running Tests
```bash
# Run all linear operator tests
pytest tests/linear_operator/

# Run specific test file
pytest tests/linear_operator/test_scalar.py

# Run with verbose output
pytest tests/linear_operator/ -v

# Run with stop on first failure
pytest tests/linear_operator/ -x
```

---

## 8. Next Steps

1. **Complete Linear Operator Refactor**: Fix remaining test failures
2. **Validate Test Suite**: Ensure all tests pass
3. **Document Patterns**: Document successful refactor patterns for other modules
4. **Extend to Datasets**: Apply same pattern to `gmi/datasets/`
5. **Extend to Models**: Apply same pattern to `gmi/network/`
6. **Command Refactor**: Update CLI commands to use Hydra
7. **Integration Testing**: Test full pipeline with refactored components

---

## 9. Lessons Learned

- **Inheritance Matters**: Be careful about inheritance hierarchies - not all transposes are square
- **Method Completeness**: Real operators need explicit `transpose`/`conjugate_transpose` implementations
- **Config Structure**: YAML configs should return objects directly, not nested structures
- **Test Alignment**: Test expectations must match actual Hydra instantiation behavior
- **Incremental Approach**: Small changes with immediate testing prevents cascading failures