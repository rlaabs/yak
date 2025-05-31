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

# Import httpx for direct API calls
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None

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
    
    async def _direct_api_call(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
        """Make a direct API call to OpenRouter with json_schema support."""
        if not HAS_HTTPX:
            raise ImportError("httpx package is required for JSON schema support with OpenRouter. Install with: pip install httpx")
        
        try:
            # Create the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                # Simplified json response format for OpenRouter
                "response_format": {"type": "json_object"}
            }
            
            # Add instruction to return JSON in the format specified
            # Modify the last user message to include JSON format instructions
            if messages and messages[-1]["role"] == "user":
                original_content = messages[-1]["content"]
                schema_str = json.dumps(schema, indent=2)
                if "json" not in original_content.lower():
                    messages[-1]["content"] = (
                        f"{original_content}\n\n"
                        f"Please respond with JSON that follows this schema:\n"
                        f"```json\n{schema_str}\n```\n"
                        f"Ensure your response is valid JSON that matches this schema exactly."
                    )
            
            logger.info(f"OpenRouter direct API call payload: {json.dumps(payload, indent=2)}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                logger.info(f"OpenRouter response: {json.dumps(data, indent=2)}")
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"OpenRouter direct API call error: {e}")
            return f"Error with direct API call: {str(e)}"
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response using OpenRouter API."""
        try:
            # Check if we need to use the direct API fallback for response_format with schema
            if "response_format" in kwargs and isinstance(kwargs["response_format"], dict):
                # Check for schema directly or in json_schema field
                if "schema" in kwargs["response_format"]:
                    logger.info("Using direct API fallback for OpenRouter with JSON schema")
                    return await self._direct_api_call(messages, kwargs["response_format"]["schema"])
                elif "json_schema" in kwargs["response_format"] and "schema" in kwargs["response_format"]["json_schema"]:
                    logger.info("Using direct API fallback for OpenRouter with JSON schema (from json_schema field)")
                    return await self._direct_api_call(messages, kwargs["response_format"]["json_schema"]["schema"])
            
            # Combine default generation kwargs with any additional kwargs
            generation_kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs,
                **kwargs
            }
            
            # For regular response_format without schema, convert to simple json_object
            if "response_format" in generation_kwargs:
                logger.warning("Adjusting response_format for OpenRouter compatibility. Using simple JSON format.")
                
                # Tell the model to return JSON in the last message
                # Add JSON instruction to the last user message
                if messages and messages[-1]["role"] == "user":
                    original_content = messages[-1]["content"]
                    if "json" not in original_content.lower():
                        messages[-1]["content"] = original_content + "\n\nPlease format your response as JSON."
                
                # Use simpler response_format with just the type
                generation_kwargs["response_format"] = {"type": "json_object"}
            
            # Note: Tool calling support varies by model on OpenRouter
            if tools and self._supports_tools():
                generation_kwargs["tools"] = tools
                generation_kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**generation_kwargs)
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
    
    def _direct_api_call_sync(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
        """Make a direct synchronous API call to OpenRouter with json_schema support."""
        if not HAS_HTTPX:
            raise ImportError("httpx package is required for JSON schema support with OpenRouter. Install with: pip install httpx")
        
        try:
            # Create the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                # Simplified json response format for OpenRouter
                "response_format": {"type": "json_object"}
            }
            
            # Add instruction to return JSON in the format specified
            # Modify the last user message to include JSON format instructions
            if messages and messages[-1]["role"] == "user":
                original_content = messages[-1]["content"]
                schema_str = json.dumps(schema, indent=2)
                if "json" not in original_content.lower():
                    messages[-1]["content"] = (
                        f"{original_content}\n\n"
                        f"Please respond with JSON that follows this schema:\n"
                        f"```json\n{schema_str}\n```\n"
                        f"Ensure your response is valid JSON that matches this schema exactly."
                    )
            
            logger.info(f"OpenRouter direct API call payload (sync): {json.dumps(payload, indent=2)}")
            
            with httpx.Client() as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                logger.info(f"OpenRouter response (sync): {json.dumps(data, indent=2)}")
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"OpenRouter direct API call error: {e}")
            return f"Error with direct API call: {str(e)}"

    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Synchronous generation for compatibility."""
        try:
            # Check if we need to use the direct API fallback for response_format with schema
            if "response_format" in kwargs and isinstance(kwargs["response_format"], dict):
                # Check for schema directly or in json_schema field
                if "schema" in kwargs["response_format"]:
                    logger.info("Using direct API fallback for OpenRouter with JSON schema (sync)")
                    return self._direct_api_call_sync(messages, kwargs["response_format"]["schema"])
                elif "json_schema" in kwargs["response_format"] and "schema" in kwargs["response_format"]["json_schema"]:
                    logger.info("Using direct API fallback for OpenRouter with JSON schema (from json_schema field, sync)")
                    return self._direct_api_call_sync(messages, kwargs["response_format"]["json_schema"]["schema"])
            
            # Combine default generation kwargs with any additional kwargs
            generation_kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs,
                **kwargs
            }
            
            # For regular response_format without schema, convert to simple json_object
            if "response_format" in generation_kwargs:
                logger.warning("Adjusting response_format for OpenRouter compatibility. Using simple JSON format.")
                
                # Tell the model to return JSON in the last message
                # Add JSON instruction to the last user message
                if messages and messages[-1]["role"] == "user":
                    original_content = messages[-1]["content"]
                    if "json" not in original_content.lower():
                        messages[-1]["content"] = original_content + "\n\nPlease format your response as JSON."
                
                # Use simpler response_format with just the type
                generation_kwargs["response_format"] = {"type": "json_object"}
            
            if tools and self._supports_tools():
                generation_kwargs["tools"] = tools
                generation_kwargs["tool_choice"] = "auto"
            
            response = self.sync_client.chat.completions.create(**generation_kwargs)
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
