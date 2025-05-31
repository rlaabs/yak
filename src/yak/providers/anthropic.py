"""
Anthropic provider implementation.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..utils import extract_tool_calls

logger = logging.getLogger(__name__)

# Import with fallback
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs):
        if not HAS_ANTHROPIC:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        self.model_name = model_name
        
        # Handle API key with environment variable fallback
        self.api_key = self._get_api_key(api_key)
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        self.generation_kwargs = kwargs
    
    def _get_api_key(self, provided_key: Optional[str]) -> Optional[str]:
        """Get API key from parameter or environment variable."""
        # First priority: explicitly provided key
        if provided_key:
            logger.info("Using provided Anthropic API key")
            return provided_key
        
        # Second priority: environment variable
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            logger.info("Using Anthropic API key from ANTHROPIC_API_KEY environment variable")
            return env_key
        
        # No key found - warn user
        logger.warning(
            "⚠️  No Anthropic API key found! Please either:\n"
            "   1. Set the ANTHROPIC_API_KEY environment variable:\n"
            "      export ANTHROPIC_API_KEY='your-api-key-here'\n"
            "   2. Pass api_key parameter:\n"
            "      Yak(provider='anthropic', api_key='your-api-key-here')\n"
            "   3. Get your API key from: https://console.anthropic.com/"
        )
        return None
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            # Separate system message from conversation
            system_message = ""
            conversation = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation.append(msg)
            
            # Combine default generation kwargs with any additional kwargs
            generation_kwargs = {
                "model": self.model_name,
                "messages": conversation,
                "max_tokens": self.generation_kwargs.get("max_tokens", 1000),
                **{k: v for k, v in self.generation_kwargs.items() if k != "max_tokens"},
                **kwargs
            }
            
            if system_message:
                generation_kwargs["system"] = system_message
            
            if tools:
                generation_kwargs["tools"] = tools
            
            response = await self.client.messages.create(**generation_kwargs)
            
            # Handle tool use
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                elif hasattr(block, 'name'):  # Tool use block
                    content += f"<tool_call>\n<n>{block.name}</n>\n<arguments>{json.dumps(block.input)}</arguments>\n</tool_call>\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Synchronous generation for compatibility."""
        try:
            # Separate system message from conversation
            system_message = ""
            conversation = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation.append(msg)
            
            # Combine default generation kwargs with any additional kwargs
            generation_kwargs = {
                "model": self.model_name,
                "messages": conversation,
                "max_tokens": self.generation_kwargs.get("max_tokens", 1000),
                **{k: v for k, v in self.generation_kwargs.items() if k != "max_tokens"},
                **kwargs
            }
            
            if system_message:
                generation_kwargs["system"] = system_message
            
            if tools:
                generation_kwargs["tools"] = tools
            
            response = self.sync_client.messages.create(**generation_kwargs)
            
            # Handle tool use
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                elif hasattr(block, 'name'):  # Tool use block
                    content += f"<tool_call>\n<n>{block.name}</n>\n<arguments>{json.dumps(block.input)}</arguments>\n</tool_call>\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic-formatted response."""
        return extract_tool_calls(response)
