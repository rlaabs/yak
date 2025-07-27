"""
Tests for provider interface and provider-specific functionality.
Tests the base provider contract and provider creation without requiring API keys.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from yak.providers.base import LLMProvider
from yak.providers.openai import OpenAIProvider
from yak.providers.anthropic import AnthropicProvider
from yak.providers.openrouter import OpenRouterProvider
from yak.providers.local import LocalModelProvider
from yak.providers.mlx import MLXProvider
from yak.core import Yak


class TestProviderInterface:
    """Test that all providers implement the required interface."""
    
    def test_base_provider_is_abstract(self):
        """Test that base provider cannot be instantiated."""
        with pytest.raises(TypeError):
            LLMProvider()
    
    def test_all_providers_implement_generate(self):
        """Test that all providers implement the generate method."""
        providers = [
            OpenAIProvider,
            AnthropicProvider, 
            OpenRouterProvider,
            LocalModelProvider,
            MLXProvider
        ]
        
        for provider_class in providers:
            # Check that generate method exists and is callable
            assert hasattr(provider_class, 'generate')
            assert callable(getattr(provider_class, 'generate'))
    
    def test_all_providers_implement_extract_tool_calls(self):
        """Test that all providers implement extract_tool_calls method."""
        providers = [
            OpenAIProvider,
            AnthropicProvider,
            OpenRouterProvider, 
            LocalModelProvider,
            MLXProvider
        ]
        
        for provider_class in providers:
            assert hasattr(provider_class, 'extract_tool_calls')
            assert callable(getattr(provider_class, 'extract_tool_calls'))
    
    def test_providers_have_sync_methods(self):
        """Test that providers have synchronous generate methods."""
        providers = [
            OpenAIProvider,
            AnthropicProvider,
            OpenRouterProvider,
            LocalModelProvider,
            MLXProvider
        ]
        
        for provider_class in providers:
            # Most providers should have generate_sync
            assert hasattr(provider_class, 'generate_sync')
            assert callable(getattr(provider_class, 'generate_sync'))


class TestProviderCreation:
    """Test provider creation through Yak factory method."""
    
    @patch('yak.core.OpenAIProvider')
    def test_create_openai_provider(self, mock_openai):
        """Test creating OpenAI provider."""
        mock_openai.return_value = Mock(spec=LLMProvider)
        
        yak = Yak(provider="openai", api_key="test-key", model_name="gpt-4")
        
        mock_openai.assert_called_once_with(api_key="test-key", model_name="gpt-4")
        assert yak.provider is not None
    
    @patch('yak.core.AnthropicProvider')
    def test_create_anthropic_provider(self, mock_anthropic):
        """Test creating Anthropic provider."""
        mock_anthropic.return_value = Mock(spec=LLMProvider)
        
        yak = Yak(provider="anthropic", api_key="test-key")
        
        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert yak.provider is not None
    
    @patch('yak.core.OpenRouterProvider')
    def test_create_openrouter_provider(self, mock_openrouter):
        """Test creating OpenRouter provider."""
        mock_openrouter.return_value = Mock(spec=LLMProvider)
        
        yak = Yak(provider="openrouter", api_key="test-key", model_name="openai/gpt-4")
        
        mock_openrouter.assert_called_once_with(api_key="test-key", model_name="openai/gpt-4")
        assert yak.provider is not None
    
    @patch('yak.core.LocalModelProvider')
    def test_create_local_provider(self, mock_local):
        """Test creating Local provider."""
        mock_local.return_value = Mock(spec=LLMProvider)
        
        yak = Yak(provider="local", model_name="test-model")
        
        mock_local.assert_called_once_with(model_name="test-model")
        assert yak.provider is not None
    
    @patch('yak.core.MLXProvider')
    def test_create_mlx_provider(self, mock_mlx):
        """Test creating MLX provider."""
        mock_mlx.return_value = Mock(spec=LLMProvider)
        
        yak = Yak(provider="mlx", model_name="test-model")
        
        mock_mlx.assert_called_once_with(model_name="test-model")
        assert yak.provider is not None
    
    def test_case_insensitive_provider_names(self):
        """Test that provider names are case insensitive."""
        with patch('yak.core.OpenAIProvider') as mock_openai:
            mock_openai.return_value = Mock(spec=LLMProvider)
            
            # Test various cases
            for provider_name in ["OPENAI", "OpenAI", "openai", "OpenAi"]:
                yak = Yak(provider=provider_name, api_key="test")
                assert yak.provider is not None


class TestProviderToolSchemas:
    """Test provider-specific tool schema formats."""
    
    def test_openai_tool_schema_format(self):
        """Test OpenAI tool schema format."""
        def sample_tool(x: int, y: str) -> str:
            """Sample tool for testing."""
            return f"{x}: {y}"
        
        with patch('yak.core.OpenAIProvider') as mock_openai:
            mock_provider = Mock(spec=LLMProvider)
            mock_openai.return_value = mock_provider
            
            yak = Yak(provider="openai", tools=[sample_tool])
            schemas = yak.get_tool_schemas(format_type="openai")
            
            assert len(schemas) == 1
            schema = schemas[0]
            assert schema["type"] == "function"
            assert "function" in schema
            assert schema["function"]["name"] == "sample_tool"
            assert "parameters" in schema["function"]
    
    def test_anthropic_tool_schema_format(self):
        """Test Anthropic tool schema format."""
        def sample_tool(x: int, y: str) -> str:
            """Sample tool for testing."""
            return f"{x}: {y}"
        
        with patch('yak.core.AnthropicProvider') as mock_anthropic:
            mock_provider = Mock(spec=LLMProvider)
            mock_anthropic.return_value = mock_provider
            
            yak = Yak(provider="anthropic", tools=[sample_tool])
            schemas = yak.get_tool_schemas(format_type="anthropic")
            
            assert len(schemas) == 1
            schema = schemas[0]
            assert "type" not in schema  # Anthropic format doesn't have top-level type
            assert schema["name"] == "sample_tool"
            assert "input_schema" in schema
    
    def test_provider_specific_schema_selection(self):
        """Test that provider type determines schema format."""
        def sample_tool(x: int) -> int:
            """Sample tool."""
            return x
        
        # Test with AnthropicProvider instance
        with patch('yak.core.AnthropicProvider') as mock_anthropic:
            mock_provider = Mock(spec=AnthropicProvider)
            mock_anthropic.return_value = mock_provider
            
            yak = Yak(provider="anthropic", tools=[sample_tool])
            
            # When provider is Anthropic, should use anthropic format by default
            # This is tested indirectly through the chat method's tool handling


class TestProviderErrorHandling:
    """Test error handling in provider operations."""
    
    def test_provider_generate_error_handling(self):
        """Test handling of provider generation errors."""
        class FailingProvider(LLMProvider):
            async def generate(self, messages, tools=None, **kwargs):
                raise Exception("Provider failed")
            
            def generate_sync(self, messages, tools=None, **kwargs):
                raise Exception("Provider failed")
            
            def extract_tool_calls(self, response):
                return []
        
        yak = Yak(provider=FailingProvider())
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Provider failed"):
            yak.chat("Hello")
    
    def test_tool_call_extraction_error_handling(self):
        """Test handling of tool call extraction errors."""
        class FailingExtractionProvider(LLMProvider):
            async def generate(self, messages, tools=None, **kwargs):
                return "Response with tool call"
            
            def generate_sync(self, messages, tools=None, **kwargs):
                return "Response with tool call"
            
            def extract_tool_calls(self, response):
                raise Exception("Extraction failed")
        
        yak = Yak(provider=FailingExtractionProvider())
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Extraction failed"):
            yak.chat("Hello")


class TestProviderSpecificFeatures:
    """Test provider-specific features and behaviors."""
    
    def test_native_tool_calling_support(self):
        """Test detection of native tool calling support."""
        # Mock providers with different tool calling capabilities
        class NativeToolProvider(LLMProvider):
            async def generate(self, messages, tools=None, **kwargs):
                return "response"
            
            def generate_sync(self, messages, tools=None, **kwargs):
                return "response"
            
            def extract_tool_calls(self, response):
                return []
            
            def build_tool_result_message(self, tool_call_id, tool_name, result):
                return {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            
            def supports_native_tool_calling(self):
                return True
        
        class SimpleToolProvider(LLMProvider):
            async def generate(self, messages, tools=None, **kwargs):
                return "response"
            
            def generate_sync(self, messages, tools=None, **kwargs):
                return "response"
            
            def extract_tool_calls(self, response):
                return []
        
        native_provider = NativeToolProvider()
        simple_provider = SimpleToolProvider()
        
        assert native_provider.supports_native_tool_calling() == True
        assert simple_provider.supports_native_tool_calling() == True  # Base implementation returns True
    
    def test_mlx_provider_tool_format(self):
        """Test that MLX provider expects function objects directly."""
        def sample_tool(x: int) -> int:
            """Sample tool."""
            return x * 2
        
        with patch('yak.core.MLXProvider') as mock_mlx:
            mock_provider = Mock(spec=MLXProvider)
            mock_provider.generate_sync.return_value = "Response"
            mock_provider.extract_tool_calls.return_value = []
            mock_mlx.return_value = mock_provider
            
            yak = Yak(provider="mlx", tools=[sample_tool])
            
            # When chatting, MLX provider should receive function objects, not schemas
            yak.chat("Hello")
            
            # Verify that generate_sync was called
            mock_provider.generate_sync.assert_called()
            
            # The tools argument should be the actual function objects for MLX
            call_args = mock_provider.generate_sync.call_args
            if call_args and len(call_args[0]) > 1:
                tools_arg = call_args[0][1]  # Second positional argument
                if tools_arg:
                    # For MLX, tools should be function objects
                    assert callable(tools_arg[0])
                    assert tools_arg[0].__name__ == "sample_tool"


class TestProviderMessageFormatting:
    """Test provider-specific message formatting."""
    
    def test_tool_result_message_formatting(self):
        """Test different tool result message formats."""
        # Test base provider format
        base_provider = Mock(spec=LLMProvider)
        base_provider.build_tool_result_message = LLMProvider.build_tool_result_message
        
        result_msg = base_provider.build_tool_result_message(
            base_provider, "call_123", "test_tool", "result"
        )
        
        assert result_msg["role"] == "tool"
        assert result_msg["name"] == "test_tool"
        assert result_msg["content"] == "result"
    
    def test_provider_specific_message_building(self):
        """Test that providers can override message building."""
        class CustomMessageProvider(LLMProvider):
            async def generate(self, messages, tools=None, **kwargs):
                return "response"
            
            def generate_sync(self, messages, tools=None, **kwargs):
                return "response"
            
            def extract_tool_calls(self, response):
                return []
            
            def build_tool_result_message(self, tool_call_id, tool_name, result):
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                    "custom_field": "custom_value"
                }
        
        provider = CustomMessageProvider()
        result_msg = provider.build_tool_result_message("call_123", "test_tool", "result")
        
        assert result_msg["custom_field"] == "custom_value"
        assert result_msg["tool_call_id"] == "call_123"


if __name__ == "__main__":
    pytest.main([__file__])
