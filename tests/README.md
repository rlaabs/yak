# Yak Framework Test Suite

This directory contains a comprehensive test suite for the Yak LLM chat framework. The tests are designed to catch regressions and ensure all components work correctly together.

## Test Structure

### Core Test Files

- **`test_core.py`** - Tests the main Yak class functionality including:
  - Initialization with different providers
  - Tool management (add, remove, schema generation)
  - History management (conversation history, system prompts)
  - Chat functionality (sync/async)
  - Save/load functionality

- **`test_utils.py`** - Tests utility functions including:
  - Tool call extraction from various response formats
  - Pydantic model to JSON schema conversion
  - Type hint to JSON schema conversion
  - Docstring parsing for parameter descriptions

- **`test_providers.py`** - Tests provider interface and functionality:
  - Provider interface compliance
  - Provider creation through factory methods
  - Provider-specific tool schema formats
  - Error handling in provider operations

- **`test_integration.py`** - End-to-end integration tests:
  - Complete conversation flows
  - Tool calling workflows
  - Multi-round conversations
  - History persistence
  - Error recovery scenarios
  - Concurrent usage patterns

- **`test_openrouter_tool_calling.py`** - Existing OpenRouter integration test

### Configuration Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`pytest.ini`** - Pytest settings and markers
- **`README.md`** - This documentation file

## Running Tests

### Prerequisites

Install test dependencies:
```bash
# Using uv (recommended)
uv sync --group dev

# Or using pip
pip install -e ".[test]"
```

### Quick Start

Use the test runner script for convenient test execution:

```bash
# Run all tests
python run_tests.py all

# Run specific test suites
python run_tests.py core
python run_tests.py utils
python run_tests.py providers
python run_tests.py integration

# Run fast tests only (excludes slow/API tests)
python run_tests.py fast

# Run with coverage report
python run_tests.py coverage
```

### Direct Pytest Usage

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestYakInitialization -v

# Run specific test method
pytest tests/test_core.py::TestYakInitialization::test_init_with_string_provider -v

# Run tests with markers
pytest tests/ -m "not slow and not requires_api"
pytest tests/ -m "integration"
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks to avoid external dependencies
- Fast execution (< 1 second per test)
- Located in: `test_core.py`, `test_utils.py`, `test_providers.py`

### Integration Tests
- Test multiple components working together
- Use mock providers to simulate real scenarios
- Test complete workflows end-to-end
- Located in: `test_integration.py`

### API Tests
- Test real provider integrations (require API keys)
- Marked with `@pytest.mark.requires_api`
- Located in: `test_openrouter_tool_calling.py`

## Test Markers

- **`integration`** - Integration tests that test multiple components
- **`slow`** - Tests that take longer to run
- **`requires_api`** - Tests that require API keys

## Mock Strategy

The test suite uses extensive mocking to:
- Avoid requiring API keys for most tests
- Ensure tests run quickly and reliably
- Test error conditions and edge cases
- Isolate components for unit testing

### Key Mock Classes

- **`MockProvider`** - Basic mock LLM provider for unit tests
- **`MockIntegrationProvider`** - Advanced mock with call history tracking
- Various provider-specific mocks for testing provider creation

## Coverage Goals

The test suite aims for high coverage of:
- **Core functionality** - All major Yak class methods
- **Provider interface** - All provider implementations
- **Tool calling** - Complete tool calling workflows
- **Error handling** - Error conditions and recovery
- **Edge cases** - Boundary conditions and unusual inputs

## Adding New Tests

### For New Features

1. Add unit tests to the appropriate test file
2. Add integration tests if the feature involves multiple components
3. Use appropriate fixtures from `conftest.py`
4. Follow existing naming conventions

### For Bug Fixes

1. Add a test that reproduces the bug
2. Verify the test fails before the fix
3. Verify the test passes after the fix
4. Consider adding related edge case tests

### Test Naming Convention

```python
def test_[component]_[action]_[condition]():
    """Test that [component] [action] when [condition]."""
```

Examples:
- `test_yak_initialization_with_string_provider()`
- `test_tool_call_extraction_with_invalid_json()`
- `test_provider_error_handling_during_generation()`

## Continuous Integration

The test suite is designed to run in CI environments:
- No external dependencies required for core tests
- Fast execution (< 30 seconds for full suite)
- Clear pass/fail indicators
- Detailed error reporting

## Troubleshooting

### Common Issues

1. **Import errors** - Ensure `src/` is in Python path (handled by `conftest.py`)
2. **Missing dependencies** - Install test dependencies with `pip install -e ".[test]"`
3. **Pydantic tests skipped** - Install pydantic for structured response tests
4. **API tests failing** - Set appropriate API keys or skip with `-m "not requires_api"`

### Debug Mode

Run tests with more verbose output:
```bash
pytest tests/ -v -s --tb=long
```

### Running Single Test with Debug

```bash
pytest tests/test_core.py::TestYakInitialization::test_init_with_string_provider -v -s
```

## Contributing

When contributing to the test suite:
1. Ensure all tests pass before submitting
2. Add tests for new functionality
3. Maintain or improve test coverage
4. Follow existing patterns and conventions
5. Update this README if adding new test categories or patterns
