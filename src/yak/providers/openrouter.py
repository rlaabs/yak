"""
OpenRouter provider implementation.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..utils import extract_tool_calls
import requests

logger = logging.getLogger(__name__)

# Import with fallback
try:
    from openai import AsyncOpenAI, OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = OpenAI = None


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider - access to multiple LLMs through unified API."""
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        if not HAS_OPENAI:
            raise ImportError("OpenAI package not installed")
        
        self.model_name = model_name
        
        # Handle API key with environment variable fallback
        self.api_key = self._get_api_key(api_key)
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        self.sync_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=self.api_key
        )
        self.generation_kwargs = kwargs
        self._last_response = None
    
    def _get_api_key(self, provided_key: Optional[str]) -> Optional[str]:
        """Get API key from parameter or environment variable."""
        # First priority: explicitly provided key
        if provided_key:
            logger.info("Using provided OpenRouter API key")
            return provided_key
        
        # Second priority: environment variable
        env_key = os.getenv("OPENROUTER_API_KEY")
        if env_key:
            logger.info("Using OpenRouter API key from OPENROUTER_API_KEY environment variable")
            return env_key
        
        # No key found - warn user
        logger.warning(
            "⚠️  No OpenRouter API key found! Please either:\n"
            "   1. Set the OPENROUTER_API_KEY environment variable:\n"
            "      export OPENROUTER_API_KEY='your-api-key-here'\n"
            "   2. Pass api_key parameter:\n"
            "      Yak(provider='openrouter', api_key='your-api-key-here')\n"
            "   3. Get your API key from: https://openrouter.ai/keys"
        )
        return None
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Generate response using OpenRouter API."""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs
            }
            
            # Note: Tool calling support varies by model on OpenRouter
            if tools and self._supports_tools():
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**kwargs)
            self._last_response = response
            
            # Handle tool calls
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_text = ""
                for tool_call in message.tool_calls:
                    tool_calls_text += f"<tool_call>\n"
                    tool_calls_text += f"<n>{tool_call.function.name}</n>\n"
                    tool_calls_text += f"<arguments>{tool_call.function.arguments}</arguments>\n"
                    tool_calls_text += f"</tool_call>\n"
                
                content = message.content or ""
                return content + "\n" + tool_calls_text
            
            return message.content or ""
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error: {str(e)}"
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Synchronous generation for compatibility."""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs
            }
            
            if tools and self._supports_tools():
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.sync_client.chat.completions.create(**kwargs)
            self._last_response = response
            
            # Handle tool calls
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_text = ""
                for tool_call in message.tool_calls:
                    tool_calls_text += f"<tool_call>\n"
                    tool_calls_text += f"<n>{tool_call.function.name}</n>\n"
                    tool_calls_text += f"<arguments>{tool_call.function.arguments}</arguments>\n"
                    tool_calls_text += f"</tool_call>\n"
                
                content = message.content or ""
                return content + "\n" + tool_calls_text
            
            return message.content or ""
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error: {str(e)}"
    
    def _supports_tools(self) -> bool:
        """Check if the current model supports tool calling."""
        # Models that support function calling on OpenRouter
        def get_models_with_tool(data):
            """
            Extract model IDs where 'tool' is present in supported_parameters.

            :param data: A dictionary with a 'data' key containing a list of model dicts.
            :return: A list of model IDs where 'tool' is present in supported_parameters.
            """
            model_ids = []
            for model in data.get('data', []):
                supported = model.get('supported_parameters', [])
                if 'tools' in supported:
                    model_ids.append(model['id'])
            return model_ids
        
        response = requests.get("https://openrouter.ai/api/v1/models")
        models = response.json()
        tool_supported_models = get_models_with_tool(models)
        
        # Check if current model or model family supports tools
        return any(supported in self.model_name.lower() for supported in tool_supported_models)
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenRouter-formatted response."""
        return extract_tool_calls(response)

    def build_tool_result_message(self, tool_call_id: str, tool_name: str, result: str) -> Dict[str, Any]:
        """Build a properly formatted tool result message for OpenRouter."""
        # OpenRouter follows the OpenAI format which requires tool messages
        # to reference a specific tool_call_id from a previous message with tool_calls
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
