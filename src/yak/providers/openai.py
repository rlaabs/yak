"""
OpenAI provider implementation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..utils import extract_tool_calls

logger = logging.getLogger(__name__)

# Import with fallback
try:
    from openai import AsyncOpenAI, OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = OpenAI = None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        if not HAS_OPENAI:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.model_name = model_name
        
        # Handle API key with environment variable fallback
        self.api_key = self._get_api_key(api_key)
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
        self.generation_kwargs = kwargs
    
    def _get_api_key(self, provided_key: Optional[str]) -> Optional[str]:
        """Get API key from parameter or environment variable."""
        # First priority: explicitly provided key
        if provided_key:
            logger.info("Using provided OpenAI API key")
            return provided_key
        
        # Second priority: environment variable
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            logger.info("Using OpenAI API key from OPENAI_API_KEY environment variable")
            return env_key
        
        # No key found - warn user
        logger.warning(
            "⚠️  No OpenAI API key found! Please either:\n"
            "   1. Set the OPENAI_API_KEY environment variable:\n"
            "      export OPENAI_API_KEY='your-api-key-here'\n"
            "   2. Pass api_key parameter:\n"
            "      Yak(provider='openai', api_key='your-api-key-here')\n"
            "   3. Get your API key from: https://platform.openai.com/api-keys"
        )
        return None
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Generate response using OpenAI API."""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**kwargs)
            
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
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Synchronous generation for compatibility."""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self.generation_kwargs
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.sync_client.chat.completions.create(**kwargs)
            
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
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI-formatted response."""
        return extract_tool_calls(response)