# Tests

Unit tests for TorchTitanTurbo NPU optimizations.

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit_tests/test_npu_rmsnorm.py

# Run specific test class
python -m pytest tests/unit_tests/test_converters.py::TestConverterInterface

# Run with verbose output
python -m pytest tests/ -v
```

## Test Categories

### Converter Tests (`test_converters.py`)
- Interface consistency tests
- Naming convention tests
- Initialization tests

### Module Tests
- `test_npu_rmsnorm.py` - NpuRMSNorm and NpuRMSNormConverter
- `test_npu_attention.py` - NpuScaledDotProductAttention and NpuSDPAConverter
- `test_npu_gmm.py` - (TODO) NpuGroupedExperts and NpuGroupedExpertsConverter

## Test Coverage

Tests verify:
1. ✅ Config build() creates correct module type
2. ✅ Converter replaces correct Config types
3. ✅ Mock NPU operators are called correctly
4. ✅ Interface consistency across all converters
5. ✅ Naming conventions

## Mocking NPU Operators

Tests use `unittest.mock.patch` to mock `torch_npu` operators:
- `torch_npu.npu_rms_norm`
- `torch_npu.npu_fusion_attention`

This allows testing without actual NPU hardware.

## Test Requirements

```bash
pip install pytest
```