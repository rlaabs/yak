"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Generate a response from the LLM. May return string or structured response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool schemas
            **kwargs: Additional provider-specific generation parameters
        """
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        pass
    
    def build_tool_result_message(self, tool_call_id: str, tool_name: str, result: str) -> Dict[str, Any]:
        """Build a tool result message. Override for provider-specific formats."""
        return {
            "role": "tool",
            "name": tool_name,
            "content": result
        }
    
    def supports_native_tool_calling(self) -> bool:
        """Returns True if this provider supports native tool calling format."""
        return hasattr(self, 'build_tool_result_message') and callable(getattr(self, 'build_tool_result_message'))
    
    def get_sync_method(self):
        """Get the synchronous generate method if available."""
        return getattr(self, 'generate_sync', None)