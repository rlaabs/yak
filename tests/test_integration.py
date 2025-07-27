"""
Integration tests for the Yak framework.
Tests end-to-end functionality with mocked providers to ensure all components work together.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Optional

from yak import Yak
from yak.providers.base import LLMProvider

# Test pydantic functionality if available
try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
    
    class TaskResponse(BaseModel):
        task: str
        completed: bool
        result: Optional[str] = None
        
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    TaskResponse = None


class MockIntegrationProvider(LLMProvider):
    """Mock provider for integration testing with configurable responses."""
    
    def __init__(self, responses=None, tool_calls_responses=None, **kwargs):
        self.responses = responses or ["Default response"]
        self.response_index = 0
        self.tool_calls_responses = tool_calls_responses or []
        self.tool_calls_index = 0
        self.call_history = []
    
    async def generate(self, messages: List[Dict[str, str]], tools: List[Dict] = None, **kwargs) -> str:
        self.call_history.append({
            'method': 'generate_async',
            'messages': messages,
            'tools': tools,
            'kwargs': kwargs
        })
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: List[Dict] = None, **kwargs) -> str:
        self.call_history.append({
            'method': 'generate_sync',
            'messages': messages,
            'tools': tools,
            'kwargs': kwargs
        })
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        if self.tool_calls_index < len(self.tool_calls_responses):
            calls = self.tool_calls_responses[self.tool_calls_index]
            self.tool_calls_index += 1
            return calls
        return []
    
    def build_tool_result_message(self, tool_call_id: str, tool_name: str, result: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        }


def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform mathematical operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_weather(city: str) -> str:
    """
    Get weather information for a city.
    
    Args:
        city: The city to get weather for
    """
    return f"The weather in {city} is sunny and 72°F"


def failing_tool(message: str) -> str:
    """
    A tool that always fails for testing error handling.
    
    Args:
        message: A message (ignored)
    """
    raise RuntimeError("This tool always fails")


class TestEndToEndConversations:
    """Test complete conversation flows."""
    
    def test_simple_conversation_flow(self):
        """Test a simple back-and-forth conversation."""
        provider = MockIntegrationProvider(responses=[
            "Hello! How can I help you today?",
            "I'd be happy to help you with math problems.",
            "Goodbye! Have a great day!"
        ])
        
        yak = Yak(provider=provider, system_prompt="You are a helpful assistant.")
        
        # First exchange
        response1 = yak.chat("Hello")
        assert response1 == "Hello! How can I help you today?"
        
        # Second exchange
        response2 = yak.chat("Can you help me with math?")
        assert response2 == "I'd be happy to help you with math problems."
        
        # Third exchange
        response3 = yak.chat("Thanks, goodbye!")
        assert response3 == "Goodbye! Have a great day!"
        
        # Check conversation history
        assert len(yak.history) == 7  # system + 3 user + 3 assistant
        assert yak.history[0]["role"] == "system"
        assert yak.history[1]["role"] == "user"
        assert yak.history[1]["content"] == "Hello"
        assert yak.history[2]["role"] == "assistant"
        assert yak.history[2]["content"] == "Hello! How can I help you today?"
    
    @pytest.mark.asyncio
    async def test_async_conversation_flow(self):
        """Test asynchronous conversation flow."""
        provider = MockIntegrationProvider(responses=[
            "Hello! I'm ready to help.",
            "Sure, I can assist with that."
        ])
        
        yak = Yak(provider=provider)
        
        response1 = await yak.chat_async("Hello")
        assert response1 == "Hello! I'm ready to help."
        
        response2 = await yak.chat_async("Can you help me?")
        assert response2 == "Sure, I can assist with that."
        
        # Verify async methods were called
        assert len(provider.call_history) == 2
        assert all(call['method'] == 'generate_async' for call in provider.call_history)


class TestToolCallingIntegration:
    """Test complete tool calling workflows."""
    
    def test_single_tool_call_flow(self):
        """Test a complete single tool call workflow."""
        tool_calls = [{"id": "call_123", "name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 3}}]
        provider = MockIntegrationProvider(
            responses=[
                "I'll calculate that for you.",
                "The result of 5 + 3 is 8."
            ],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider, tools=[calculator])
        
        response = yak.chat("What is 5 + 3?")
        
        assert response == "The result of 5 + 3 is 8."
        
        # Check that tool was executed and result was added to history
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "8" in tool_messages[0]["content"] or "8.0" in tool_messages[0]["content"]
    
    def test_multiple_tool_calls_flow(self):
        """Test workflow with multiple tool calls."""
        tool_calls_round1 = [
            {"id": "call_123", "name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 3}},
            {"id": "call_124", "name": "calculator", "arguments": {"operation": "multiply", "a": 8, "b": 2}}
        ]
        
        provider = MockIntegrationProvider(
            responses=[
                "I'll do both calculations.",
                "The results are 8 and 16 respectively."
            ],
            tool_calls_responses=[tool_calls_round1, []]
        )
        
        yak = Yak(provider=provider, tools=[calculator])
        
        response = yak.chat("Calculate 5+3 and 8*2")
        
        assert response == "The results are 8 and 16 respectively."
        
        # Should have two tool result messages
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 2
    
    def test_tool_call_with_error_handling(self):
        """Test tool call error handling integration."""
        tool_calls = [{"id": "call_123", "name": "calculator", "arguments": {"operation": "divide", "a": 10, "b": 0}}]
        provider = MockIntegrationProvider(
            responses=[
                "I'll calculate that division.",
                "I encountered an error: division by zero is not allowed."
            ],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider, tools=[calculator])
        
        response = yak.chat("What is 10 divided by 0?")
        
        assert "error" in response.lower()
        
        # Check that error was recorded in tool message
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "error" in tool_messages[0]["content"].lower()
    
    def test_nonexistent_tool_handling(self):
        """Test handling of calls to nonexistent tools."""
        tool_calls = [{"id": "call_123", "name": "nonexistent_tool", "arguments": {"param": "value"}}]
        provider = MockIntegrationProvider(
            responses=[
                "I'll use a tool to help.",
                "I apologize, that tool is not available."
            ],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider, tools=[calculator])
        
        response = yak.chat("Use the special tool")
        
        assert "apologize" in response.lower() or "not available" in response.lower()
        
        # Check error was recorded
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "No such tool" in tool_messages[0]["content"]


class TestMultiRoundToolCalling:
    """Test multi-round conversations with tool calling."""
    
    def test_multi_round_tool_conversation(self):
        """Test conversation with multiple rounds of tool calling."""
        # Round 1: Calculate 5+3
        tool_calls_1 = [{"id": "call_1", "name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 3}}]
        # Round 2: Get weather
        tool_calls_2 = [{"id": "call_2", "name": "get_weather", "arguments": {"city": "New York"}}]
        
        provider = MockIntegrationProvider(
            responses=[
                "I'll calculate that.",
                "The result is 8. What else can I help with?",
                "I'll check the weather.",
                "The weather in New York is sunny and 72°F. Anything else?"
            ],
            tool_calls_responses=[tool_calls_1, [], tool_calls_2, []]
        )
        
        yak = Yak(provider=provider, tools=[calculator, get_weather])
        
        # First conversation with tool
        response1 = yak.chat("What is 5 + 3?")
        assert "8" in response1
        
        # Second conversation with different tool
        response2 = yak.chat("What's the weather in New York?")
        assert "sunny" in response2 and "72" in response2
        
        # Check history has all the expected messages
        assert len(yak.history) >= 6  # 2 user + 2 assistant + 2 tool messages
        
        # Verify both tools were called
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 2
        tool_names = [msg.get("name") for msg in tool_messages]
        assert "calculator" in tool_names
        assert "get_weather" in tool_names


class TestHistoryPersistence:
    """Test conversation history persistence and loading."""
    
    def test_save_and_load_conversation_with_tools(self):
        """Test saving and loading conversation history with tool calls."""
        provider = MockIntegrationProvider(responses=["I'll help with that calculation."])
        yak = Yak(provider=provider, tools=[calculator], system_prompt="You are a math assistant.")
        
        # Add some conversation history manually for testing
        yak.history.extend([
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "assistant", "content": "I'll calculate that for you."},
            {"role": "tool", "name": "calculator", "content": "4"},
            {"role": "assistant", "content": "The result is 4."}
        ])
        
        yak.llm_history.extend([
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "assistant", "content": "I'll calculate that for you."},
            {"role": "tool", "name": "calculator", "content": "4"},
            {"role": "assistant", "content": "The result is 4."}
        ])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            yak.save_history(temp_path)
            
            # Create new Yak instance and load
            new_provider = MockIntegrationProvider(responses=["Continuing conversation..."])
            new_yak = Yak(provider=new_provider, tools=[calculator])
            new_yak.load_history(temp_path)
            
            # Verify loaded state
            assert new_yak.system_prompt == "You are a math assistant."
            assert len(new_yak.history) == 5  # system + 4 conversation messages
            assert len(new_yak.tools) == 1
            assert "calculator" in new_yak.tools
            
            # Verify conversation can continue
            response = new_yak.chat("What was the last result?")
            assert response == "Continuing conversation..."
            
        finally:
            Path(temp_path).unlink()


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestStructuredResponseIntegration:
    """Test structured response format integration."""
    
    def test_structured_response_with_openai_provider(self):
        """Test structured response format with OpenAI-style provider."""
        with patch('yak.core.OpenAIProvider') as mock_openai:
            mock_provider = MockIntegrationProvider(responses=['{"task": "test", "completed": true, "result": "success"}'])
            # Make the mock provider look like an OpenAI provider
            mock_provider.__class__.__name__ = "OpenAIProvider"
            mock_openai.return_value = mock_provider
            
            yak = Yak(provider="openai", api_key="test")
            
            response = yak.chat("Complete this task", response_format=TaskResponse)
            
            # Verify response_format was passed to provider
            assert len(mock_provider.call_history) == 1
            call = mock_provider.call_history[0]
            assert 'response_format' in call['kwargs']
    
    def test_structured_response_without_pydantic_fallback(self):
        """Test graceful fallback when pydantic is not available for response format."""
        with patch('yak.utils.HAS_PYDANTIC', False):
            provider = MockIntegrationProvider(responses=["Regular response"])
            yak = Yak(provider=provider)
            
            # Should not raise error, just ignore response_format
            response = yak.chat("Test", response_format="dummy")
            assert response == "Regular response"


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_provider_error_recovery(self):
        """Test recovery from provider errors."""
        class FailingThenWorkingProvider(LLMProvider):
            def __init__(self):
                self.call_count = 0
            
            async def generate(self, messages, tools=None, **kwargs):
                return self.generate_sync(messages, tools, **kwargs)
            
            def generate_sync(self, messages, tools=None, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Provider temporarily failed")
                return "I'm working now!"
            
            def extract_tool_calls(self, response):
                return []
        
        provider = FailingThenWorkingProvider()
        yak = Yak(provider=provider)
        
        # First call should fail
        with pytest.raises(Exception, match="Provider temporarily failed"):
            yak.chat("Hello")
        
        # Second call should work
        response = yak.chat("Hello again")
        assert response == "I'm working now!"
    
    def test_tool_execution_error_recovery(self):
        """Test recovery from tool execution errors."""
        tool_calls = [{"id": "call_123", "name": "failing_tool", "arguments": {"message": "test"}}]
        provider = MockIntegrationProvider(
            responses=[
                "I'll use the tool.",
                "The tool failed, but I can still help you in other ways."
            ],
            tool_calls_responses=[tool_calls, []]
        )
        
        yak = Yak(provider=provider, tools=[failing_tool])
        
        response = yak.chat("Use the failing tool")
        
        # Should handle the error gracefully
        assert "other ways" in response
        
        # Error should be recorded in tool message
        tool_messages = [msg for msg in yak.history if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "error" in tool_messages[0]["content"].lower()


class TestMaxRoundsLimiting:
    """Test max_rounds parameter functionality."""
    
    def test_max_rounds_limiting(self):
        """Test that max_rounds limits the number of tool calling rounds."""
        # Create a scenario where tools keep calling each other
        tool_calls = [{"id": "call_123", "name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 1}}]
        provider = MockIntegrationProvider(
            responses=["I'll calculate."] * 10,  # Always wants to use tools
            tool_calls_responses=[tool_calls] * 10  # Always returns tool calls
        )
        
        yak = Yak(provider=provider, tools=[calculator])
        
        # Limit to 2 rounds
        response = yak.chat("Calculate something", max_rounds=2)
        
        # Should stop after 2 rounds even though provider keeps returning tool calls
        assert len(provider.call_history) == 2
        
        # Should have limited number of messages in history
        assistant_messages = [msg for msg in yak.history if msg.get("role") == "assistant"]
        assert len(assistant_messages) <= 2


class TestConcurrentUsage:
    """Test concurrent usage patterns."""
    
    @pytest.mark.asyncio
    async def test_multiple_async_conversations(self):
        """Test multiple async conversations can run concurrently."""
        import asyncio
        
        provider1 = MockIntegrationProvider(responses=["Response from provider 1"])
        provider2 = MockIntegrationProvider(responses=["Response from provider 2"])
        
        yak1 = Yak(provider=provider1)
        yak2 = Yak(provider=provider2)
        
        # Run conversations concurrently
        task1 = yak1.chat_async("Hello from conversation 1")
        task2 = yak2.chat_async("Hello from conversation 2")
        
        response1, response2 = await asyncio.gather(task1, task2)
        
        assert response1 == "Response from provider 1"
        assert response2 == "Response from provider 2"
        
        # Each should have their own history
        assert len(yak1.history) == 2
        assert len(yak2.history) == 2
        assert yak1.history[0]["content"] == "Hello from conversation 1"
        assert yak2.history[0]["content"] == "Hello from conversation 2"


if __name__ == "__main__":
    pytest.main([__file__])
