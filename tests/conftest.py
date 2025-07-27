"""
Pytest configuration and shared fixtures for Yak framework tests.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_tools():
    """Fixture providing sample tools for testing."""
    
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    def echo_message(message: str) -> str:
        """Echo a message back."""
        return f"Echo: {message}"
    
    return [add_numbers, multiply_numbers, echo_message]


@pytest.fixture
def mock_api_responses():
    """Fixture providing common mock API responses."""
    return {
        "simple_response": "Hello! How can I help you?",
        "tool_call_response": "I'll calculate that for you.",
        "final_response": "The calculation is complete.",
        "error_response": "I encountered an error."
    }


@pytest.fixture
def sample_tool_calls():
    """Fixture providing sample tool call data."""
    return [
        {
            "id": "call_123",
            "name": "add_numbers", 
            "arguments": {"a": 5, "b": 3}
        },
        {
            "id": "call_124",
            "name": "multiply_numbers",
            "arguments": {"a": 2.5, "b": 4.0}
        }
    ]


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API keys"
    )
