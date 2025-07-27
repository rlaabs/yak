"""
Core framework tests for the Yak LLM chat framework.
Tests the main Yak class functionality including initialization, tool management,
history management, and provider integration.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from yak import Yak
from yak.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing without external dependencies."""
    
    def __init__(self, **kwargs):
        self.responses = kwargs.get('responses', ["Mock response"])
        self.response_index = 0
        self.tool_calls_responses = kwargs.get('tool_calls_responses', [])
        self.tool_calls_index = 0
    
    async def generate(self, messages: List[Dict[str, str]], tools: List[Dict] = None, **kwargs) -> str:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: List[Dict] = None, **kwargs) -> str:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        if self.tool_calls_index < len(self.tool_calls_responses):
            calls = self.tool_calls_responses[self.tool_calls_index]
            self.tool_calls_index += 1
            return calls
        return []


def sample_tool(x: int, y: int = 5) -> int:
    """
    Add two numbers together.
    
    Args:
        x: First number to add
        y: Second number to add (default 5)
    
    Returns:
        Sum of x and y
    """
    return x + y


def another_tool(message: str) -> str:
    """
    Echo a message.
    
    Args:
        message: The message to echo
    """
    return f"Echo: {message}"


class TestYakInitialization:
    """Test Yak initialization with different configurations."""
    
    def test_init_with_string_provider(self):
        """Test initialization with provider name string."""
        with patch('yak.core.OpenAIProvider') as mock_openai:
            mock_openai.return_value = MockProvider()
            yak = Yak(provider="openai", api_key="test-key")
            assert yak.provider is not None
            mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_init_with_provider_object(self):
        """Test initialization with provider instance."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        assert yak.provider is provider
    
    def test_init_with_invalid_provider(self):
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError, match="Provider must be a string or LLMProvider instance"):
            Yak(provider=123)
    
    def test_init_with_unknown_provider_string(self):
        """Test initialization with unknown provider string."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            Yak(provider="unknown")
    
    def test_init_with_tools(self):
        """Test initialization with tools."""
        provider = MockProvider()
        yak = Yak(provider=provider, tools=[sample_tool, another_tool])
        assert len(yak.tools) == 2
        assert "sample_tool" in yak.tools
        assert "another_tool" in yak.tools
    
    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        provider = MockProvider()
        system_prompt = "You are a helpful assistant."
        yak = Yak(provider=provider, system_prompt=system_prompt)
        
        assert yak.system_prompt == system_prompt
        assert len(yak.history) == 1
        assert yak.history[0]["role"] == "system"
        assert yak.history[0]["content"] == system_prompt
        assert len(yak.llm_history) == 1
        assert yak.llm_history[0]["role"] == "system"


class TestToolManagement:
    """Test tool management functionality."""
    
    def test_add_tool(self):
        """Test adding a tool."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        yak.add_tool(sample_tool)
        assert "sample_tool" in yak.tools
        assert yak.tools["sample_tool"] is sample_tool
    
    def test_remove_tool(self):
        """Test removing a tool."""
        provider = MockProvider()
        yak = Yak(provider=provider, tools=[sample_tool])
        
        assert "sample_tool" in yak.tools
        yak.remove_tool("sample_tool")
        assert "sample_tool" not in yak.tools
    
    def test_remove_nonexistent_tool(self):
        """Test removing a tool that doesn't exist."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        # Should not raise an error
        yak.remove_tool("nonexistent_tool")
    
    def test_get_tool_schemas_openai_format(self):
        """Test getting tool schemas in OpenAI format."""
        provider = MockProvider()
        yak = Yak(provider=provider, tools=[sample_tool])
        
        schemas = yak.get_tool_schemas(format_type="openai")
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample_tool"
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]
        assert "x" in schema["function"]["parameters"]["properties"]
        assert "y" in schema["function"]["parameters"]["properties"]
    
    def test_get_tool_schemas_anthropic_format(self):
        """Test getting tool schemas in Anthropic format."""
        provider = MockProvider()
        yak = Yak(provider=provider, tools=[sample_tool])
        
        schemas = yak.get_tool_schemas(format_type="anthropic")
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["name"] == "sample_tool"
        assert "input_schema" in schema
        assert "properties" in schema["input_schema"]


class TestHistoryManagement:
    """Test conversation history management."""
    
    def test_set_system_prompt(self):
        """Test setting system prompt."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        system_prompt = "You are a helpful assistant."
        yak.set_system_prompt(system_prompt)
        
        assert yak.system_prompt == system_prompt
        assert yak.history[0]["role"] == "system"
        assert yak.history[0]["content"] == system_prompt
        assert yak.llm_history[0]["role"] == "system"
        assert yak.llm_history[0]["content"] == system_prompt
    
    def test_update_system_prompt(self):
        """Test updating existing system prompt."""
        provider = MockProvider()
        initial_prompt = "Initial prompt"
        yak = Yak(provider=provider, system_prompt=initial_prompt)
        
        new_prompt = "Updated prompt"
        yak.set_system_prompt(new_prompt)
        
        assert yak.system_prompt == new_prompt
        assert len(yak.history) == 1  # Should replace, not add
        assert yak.history[0]["content"] == new_prompt
    
    def test_reset_llm_history_keep_system(self):
        """Test resetting LLM history while keeping system prompt."""
        provider = MockProvider()
        system_prompt = "You are a helpful assistant."
        yak = Yak(provider=provider, system_prompt=system_prompt)
        
        # Add some conversation
        yak.llm_history.append({"role": "user", "content": "Hello"})
        yak.llm_history.append({"role": "assistant", "content": "Hi there"})
        
        yak.reset_llm_history(keep_system_prompt=True)
        
        assert len(yak.llm_history) == 1
        assert yak.llm_history[0]["role"] == "system"
        assert yak.llm_history[0]["content"] == system_prompt
    
    def test_reset_llm_history_remove_system(self):
        """Test resetting LLM history and removing system prompt."""
        provider = MockProvider()
        system_prompt = "You are a helpful assistant."
        yak = Yak(provider=provider, system_prompt=system_prompt)
        
        # Add some conversation
        yak.llm_history.append({"role": "user", "content": "Hello"})
        yak.llm_history.append({"role": "assistant", "content": "Hi there"})
        
        yak.reset_llm_history(keep_system_prompt=False)
        
        assert len(yak.llm_history) == 0
    
    def test_save_and_load_history(self):
        """Test saving and loading conversation history."""
        provider = MockProvider()
        system_prompt = "You are a helpful assistant."
        yak = Yak(provider=provider, system_prompt=system_prompt)
        
        # Add some conversation
        yak.history.append({"role": "user", "content": "Hello"})
        yak.history.append({"role": "assistant", "content": "Hi there"})
        yak.llm_history.append({"role": "user", "content": "Hello"})
        yak.llm_history.append({"role": "assistant", "content": "Hi there"})
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            yak.save_history(temp_path)
            
            # Create new Yak instance and load history
            new_yak = Yak(provider=MockProvider())
            new_yak.load_history(temp_path)
            
            assert new_yak.system_prompt == system_prompt
            assert len(new_yak.history) == 3  # system + user + assistant
            assert len(new_yak.llm_history) == 3
            assert new_yak.history[1]["content"] == "Hello"
            assert new_yak.history[2]["content"] == "Hi there"
        finally:
            Path(temp_path).unlink()


class TestChatFunctionality:
    """Test chat functionality with mocked providers."""
    
    def test_chat_sync_simple(self):
        """Test simple synchronous chat without tools."""
        provider = MockProvider(responses=["Hello! How can I help you?"])
        yak = Yak(provider=provider)
        
        response = yak.chat("Hello")
        
        assert response == "Hello! How can I help you?"
        assert len(yak.history) == 2  # user + assistant
        assert yak.history[0]["role"] == "user"
        assert yak.history[0]["content"] == "Hello"
        assert yak.history[1]["role"] == "assistant"
        assert yak.history[1]["content"] == "Hello! How can I help you?"
    
    @pytest.mark.asyncio
    async def test_chat_async_simple(self):
        """Test simple asynchronous chat without tools."""
        provider = MockProvider(responses=["Hello! How can I help you?"])
        yak = Yak(provider=provider)
        
        response = await yak.chat_async("Hello")
        
        assert response == "Hello! How can I help you?"
        assert len(yak.history) == 2  # user + assistant
    
    def test_chat_with_tool_calling(self):
        """Test chat with tool calling."""
        # Mock provider that returns a tool call, then a final response
        tool_calls = [{"id": "call_123", "name": "sample_tool", "arguments": {"x": 10, "y": 5}}]
        provider = MockProvider(
            responses=["I'll calculate that for you.", "The result is 15."],
            tool_calls_responses=[tool_calls, []]  # First response has tool calls, second doesn't
        )
        
        yak = Yak(provider=provider, tools=[sample_tool])
        
        response = yak.chat("What is 10 + 5?")
        
        # Should have: user message, assistant message, tool message, assistant final message
        assert len(yak.history) >= 3
        assert response == "The result is 15."
    
    def test_chat_with_nonexistent_tool(self):
        """Test chat when LLM tries to call a nonexistent tool."""
        tool_calls = [{"id": "call_123", "name": "nonexistent_tool", "arguments": {"x": 10}}]
        provider = MockProvider(
            responses=["I'll use a tool.", "I apologize for the error."],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider)
        
        response = yak.chat("Do something")
        
        # Should handle the error gracefully
        assert "I apologize for the error." in response
    
    def test_chat_with_tool_execution_error(self):
        """Test chat when tool execution fails."""
        def failing_tool(x: int) -> int:
            """A tool that always fails."""
            raise ValueError("This tool always fails")
        
        tool_calls = [{"id": "call_123", "name": "failing_tool", "arguments": {"x": 10}}]
        provider = MockProvider(
            responses=["I'll use a tool.", "I encountered an error."],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider, tools=[failing_tool])
        
        response = yak.chat("Use the failing tool")
        
        # Should handle the error gracefully
        assert "I encountered an error." in response


class TestSchemaGeneration:
    """Test tool schema generation from function signatures."""
    
    def test_generate_tool_schema_basic(self):
        """Test basic tool schema generation."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        schema = yak.generate_tool_schema(sample_tool, format_type="openai")
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample_tool"
        assert "Add two numbers together." in schema["function"]["description"]
        
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "x" in params["properties"]
        assert "y" in params["properties"]
        assert params["properties"]["x"]["type"] == "integer"
        assert params["properties"]["y"]["type"] == "integer"
        assert "x" in params["required"]
        assert "y" not in params["required"]  # Has default value
    
    def test_generate_tool_schema_with_custom_name(self):
        """Test tool schema generation with custom name."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        schema = yak.generate_tool_schema(sample_tool, name="custom_name", format_type="openai")
        
        assert schema["function"]["name"] == "custom_name"
    
    def test_generate_tool_schema_anthropic_format(self):
        """Test tool schema generation in Anthropic format."""
        provider = MockProvider()
        yak = Yak(provider=provider)
        
        schema = yak.generate_tool_schema(sample_tool, format_type="anthropic")
        
        assert "type" not in schema  # Anthropic format doesn't have top-level type
        assert schema["name"] == "sample_tool"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


if __name__ == "__main__":
    pytest.main([__file__])
