import json
import re
import asyncio
import logging
import inspect
from typing import List, Optional, Dict, Any, Union, Tuple, get_origin, get_args, Callable, Type
from pathlib import Path

from .providers import LLMProvider
from .utils import pydantic_model_to_json_schema
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.openrouter import OpenRouterProvider
from .providers.local import LocalModelProvider
from .providers.mlx import MLXProvider

logger = logging.getLogger(__name__)


class Yak:
    """LLM conversation manager with tool calling support."""
    
    def __init__(
        self,
        provider: Union[str, LLMProvider],
        tools: List[Callable] = None,
        system_prompt: Optional[str] = None,
        **provider_kwargs
    ):
        """
        Initialize Yak with an LLM provider and tools.
        
        Args:
            provider: Either a provider name ("openai", "anthropic", "openrouter", "local", "mlx") or LLMProvider instance
            tools: List of callable functions to use as tools
            system_prompt: Optional system prompt
            **provider_kwargs: Additional arguments for the provider
        """
        self.tools = {}
        if tools:
            self.tools = {tool.__name__: tool for tool in tools}
        
        self.system_prompt = system_prompt
        
        # Initialize provider
        if isinstance(provider, str):
            self.provider = self._create_provider(provider, **provider_kwargs)
        elif isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            raise ValueError("Provider must be a string or LLMProvider instance")
        
        # Conversation histories
        self.history: List[Dict[str, str]] = []
        self.llm_history: List[Dict[str, str]] = []
        
        # Add system prompt to history if provided
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
            self.llm_history.append({"role": "system", "content": self.system_prompt})
    
    def _create_provider(self, provider_name: str, **kwargs) -> LLMProvider:
        """Create a provider instance from name."""
        if provider_name.lower() == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name.lower() == "anthropic":
            return AnthropicProvider(**kwargs)
        elif provider_name.lower() == "openrouter":
            return OpenRouterProvider(**kwargs)
        elif provider_name.lower() == "local":
            return LocalModelProvider(**kwargs)
        elif provider_name.lower() == "mlx":
            return MLXProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}. Supported: openai, anthropic, openrouter, local, mlx")
    
    def _convert_type_to_json_schema(self, python_type) -> Dict[str, Any]:
        """Convert Python type hints to JSON schema format."""
        # Handle basic types
        if python_type is str:
            return {"type": "string"}
        elif python_type is int:
            return {"type": "integer"}
        elif python_type is float:
            return {"type": "number"}
        elif python_type is bool:
            return {"type": "boolean"}
        elif python_type in (list, List):
            return {"type": "array"}
        elif python_type is dict:
            return {"type": "object"}
        
        # Handle typing module types
        origin = get_origin(python_type)
        args = get_args(python_type)
        
        if origin is list or origin is List:
            if args:
                item_schema = self._convert_type_to_json_schema(args[0])
                return {"type": "array", "items": item_schema}
            return {"type": "array"}
        
        elif origin is dict or origin is Dict:
            return {"type": "object"}
        
        elif origin is Union:
            # Handle Optional[T] (Union[T, None])
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return self._convert_type_to_json_schema(non_none_args[0])
            # For other unions, default to string
            return {"type": "string"}
        
        # Default fallback
        return {"type": "string"}
    
    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring to extract description and parameter info."""
        if not docstring:
            return {"description": "", "params": {}}
        
        lines = docstring.strip().split('\n')
        description = lines[0].strip()
        
        # Look for Args/Parameters section
        params = {}
        in_params_section = False
        
        for line in lines[1:]:
            line = line.strip()
            if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'params:')):
                in_params_section = True
                continue
            elif line.lower().startswith(('returns:', 'return:', 'raises:', 'examples:')):
                in_params_section = False
                continue
            
            if in_params_section and ':' in line:
                # Match "param_name: description" or "param_name (type): description"
                match = re.match(r'\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+)', line)
                if match:
                    param_name, param_desc = match.groups()
                    params[param_name] = param_desc.strip()
        
        return {"description": description, "params": params}
    
    def generate_tool_schema(self, func, name: str = None, format_type: str = "openai") -> Dict[str, Any]:
        """
        Generate JSON schema for LLM tool use from a function's type hints and docstring.
        
        Args:
            func: The function to generate schema for
            name: Optional name override (defaults to function name)
            format_type: Schema format - "openai", "anthropic", or "standard"
        
        Returns:
            JSON schema dictionary for LLM tool use
        """
        tool_name = name or func.__name__
        signature = inspect.signature(func)
        
        # Parse docstring for description and parameter info
        doc_info = self._parse_docstring(func.__doc__ or "")
        description = doc_info["description"] or f"Execute {tool_name}"
        
        # Build properties from function parameters
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Get type information
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            type_info = self._convert_type_to_json_schema(param_type)
            
            # Add description from docstring
            param_desc = doc_info["params"].get(param_name, f"Parameter {param_name}")
            type_info["description"] = param_desc
            
            properties[param_name] = type_info
            
            # Required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Build schema based on format
        parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        if format_type == "openai":
            return {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters_schema
                }
            }
        elif format_type == "anthropic":
            return {
                "name": tool_name,
                "description": description,
                "input_schema": parameters_schema
            }
        else:  # standard
            return {
                "name": tool_name,
                "description": description,
                "parameters": parameters_schema
            }
    
    def add_tool(self, func: Callable) -> None:
        """Add a tool function."""
        self.tools[func.__name__] = func
    
    def remove_tool(self, name: str) -> None:
        """Remove a tool by name."""
        if name in self.tools:
            del self.tools[name]
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
        
        # Update both histories
        for history in [self.history, self.llm_history]:
            # Remove existing system message if present
            if history and history[0]["role"] == "system":
                history.pop(0)
            
            # Add new system message at the beginning
            history.insert(0, {"role": "system", "content": system_prompt})
    
    def reset_llm_history(self, keep_system_prompt: bool = True) -> None:
        """Clear the LLM conversation history, optionally keeping the system prompt."""
        if keep_system_prompt and self.system_prompt:
            self.llm_history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.llm_history = []
    
    def get_tool_schemas(self, format_type: str = "openai") -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        schemas = []
        for tool in self.tools.values():
            schema = self.generate_tool_schema(tool, format_type=format_type)
            schemas.append(schema)
        return schemas
    
    async def chat_async(self, user_input: str, max_rounds: int = 3, response_format: Optional[Type] = None) -> str:
        """
        Async version of chat method.
        Continue the conversation by adding the user_input to history,
        generating a reply, and running any tool calls.
        
        Args:
            user_input: The user's message to process
            max_rounds: Maximum number of LLM response/tool calling rounds to perform
            response_format: Optional Pydantic model class to specify the response format
        
        Returns:
            The LLM's response
        """
        # Append user message to both histories
        user_msg = {"role": "user", "content": user_input}
        self.history.append(user_msg)
        self.llm_history.append(user_msg)
        
        for round_num in range(max_rounds):
            logger.info(f"Chat round {round_num + 1}/{max_rounds}")
            
            # Get tool schemas for the provider
            tools = None
            if self.tools:
                format_type = "anthropic" if isinstance(self.provider, AnthropicProvider) else "openai"
                tools = self.get_tool_schemas(format_type=format_type)
            
            # Convert response_format to appropriate JSON schema if provided
            generation_kwargs = {}
            if response_format:
                try:
                    if isinstance(self.provider, OpenAIProvider):
                        # OpenAI only supports {"type": "json_object"}
                        response_format_schema = pydantic_model_to_json_schema(response_format, provider_type="openai")
                        generation_kwargs["response_format"] = response_format_schema
                        logger.info(f"Using response format schema for OpenAI: {response_format_schema}")
                    elif isinstance(self.provider, OpenRouterProvider):
                        # OpenRouter supports full JSON schema
                        response_format_schema = pydantic_model_to_json_schema(response_format, provider_type="openrouter")
                        generation_kwargs["response_format"] = response_format_schema
                        logger.info(f"Using response format schema for OpenRouter: {response_format_schema}")
                except ImportError as e:
                    logger.warning(f"Could not use response_format: {e}")
                except ValueError as e:
                    logger.warning(f"Invalid response_format: {e}")
            
            # Generate response
            reply = await self.provider.generate(self.llm_history, tools, **generation_kwargs)
            
            # Append assistant reply to both histories
            assistant_msg = {"role": "assistant", "content": reply}
            self.history.append(assistant_msg)
            self.llm_history.append(assistant_msg)
            
            # Check for tool calls in assistant reply
            calls = self.provider.extract_tool_calls(reply)
            if not calls:
                break
            
            logger.info(f"Executing {len(calls)} tool calls")
            
            # Execute tool calls and append results
            for call in calls:
                name = call["name"]
                args = call["arguments"]
                # Get tool_call_id if available, needed for OpenAI/OpenRouter
                tool_call_id = call.get("id", "")
                
                tool = self.tools.get(name)
                if not tool:
                    error = {"error": f"No such tool: {name}"}
                    result_content = json.dumps(error)
                else:
                    try:
                        result = tool(**args)
                        result_content = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        error = {"error": f"Tool execution failed: {str(e)}"}
                        result_content = json.dumps(error)
                        logger.error(f"Tool {name} failed: {e}")
                
                # Use provider-specific formatting for tool result messages
                if self.provider.supports_native_tool_calling():
                    # For providers like OpenAI/OpenRouter that expect tool_call_id to match a preceding tool_calls message
                    tool_msg = self.provider.build_tool_result_message(tool_call_id, name, result_content)
                    
                    # Only add to llm_history if the previous message contains tool_calls
                    prev_message = self.llm_history[-1] if self.llm_history else None
                    if prev_message and prev_message.get("role") == "assistant" and "tool_calls" in str(prev_message.get("content", "")):
                        self.llm_history.append(tool_msg)
                        logger.info(f"Added tool result to LLM history: {name} = {result_content}")
                    else:
                        # Don't send the tool message to the LLM if there's no preceding tool_calls message
                        logger.warning(f"Skipping tool message for LLM history - no preceding tool_calls found")
                        # But we still need the LLM to know about the tool result in the next response
                        # Add special system message with the tool result to ensure LLM sees it
                        self.llm_history.append({
                            "role": "system", 
                            "content": f"Tool '{name}' was called and returned the result: {result_content}"
                        })
                        logger.info(f"Added system message with tool result: {name} = {result_content}")
                else:
                    # For providers with simpler tool message format
                    tool_msg = {"role": "tool", "name": name, "content": result_content}
                    self.llm_history.append(tool_msg)
                    
                # Always add to history for the app
                self.history.append(tool_msg)
        
        # Return the last assistant message
        return self.history[-1]["content"]
    
    def chat(self, user_input: str, max_rounds: int = 3, response_format: Optional[Type] = None) -> str:
        """
        Synchronous chat method.
        Continue the conversation by adding the user_input to history,
        generating a reply, and running any tool calls.
        
        Args:
            user_input: The user's message to process
            max_rounds: Maximum number of LLM response/tool calling rounds to perform
            response_format: Optional Pydantic model class to specify the response format
        
        Returns:
            The LLM's response
        """
        # Append user message to both histories
        user_msg = {"role": "user", "content": user_input}
        self.history.append(user_msg)
        self.llm_history.append(user_msg)
        
        for round_num in range(max_rounds):
            logger.info(f"Chat round {round_num + 1}/{max_rounds}")
            
            # Get tool schemas for the provider
            tools = None
            if self.tools:
                format_type = "anthropic" if isinstance(self.provider, AnthropicProvider) else "openai"
                tools = self.get_tool_schemas(format_type=format_type)
            
            # Convert response_format to appropriate JSON schema if provided
            generation_kwargs = {}
            if response_format:
                try:
                    if isinstance(self.provider, OpenAIProvider):
                        # OpenAI only supports {"type": "json_object"}
                        response_format_schema = pydantic_model_to_json_schema(response_format, provider_type="openai")
                        generation_kwargs["response_format"] = response_format_schema
                        logger.info(f"Using response format schema for OpenAI: {response_format_schema}")
                    elif isinstance(self.provider, OpenRouterProvider):
                        # OpenRouter supports full JSON schema
                        response_format_schema = pydantic_model_to_json_schema(response_format, provider_type="openrouter")
                        generation_kwargs["response_format"] = response_format_schema
                        logger.info(f"Using response format schema for OpenRouter: {response_format_schema}")
                except ImportError as e:
                    logger.warning(f"Could not use response_format: {e}")
                except ValueError as e:
                    logger.warning(f"Invalid response_format: {e}")
            
            # Generate response (use sync method)
            if hasattr(self.provider, 'generate_sync'):
                reply = self.provider.generate_sync(self.llm_history, tools, **generation_kwargs)
            else:
                # Fallback to async in sync context
                reply = asyncio.run(self.provider.generate(self.llm_history, tools, **generation_kwargs))
            
            # Append assistant reply to both histories
            assistant_msg = {"role": "assistant", "content": reply}
            self.history.append(assistant_msg)
            self.llm_history.append(assistant_msg)
            
            # Check for tool calls in assistant reply
            calls = self.provider.extract_tool_calls(reply)
            if not calls:
                break
            
            logger.info(f"Executing {len(calls)} tool calls")
            
            # Execute tool calls and append results
            for call in calls:
                name = call["name"]
                args = call["arguments"]
                # Get tool_call_id if available, needed for OpenAI/OpenRouter
                tool_call_id = call.get("id", "")
                
                tool = self.tools.get(name)
                if not tool:
                    error = {"error": f"No such tool: {name}"}
                    result_content = json.dumps(error)
                else:
                    try:
                        result = tool(**args)
                        result_content = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        error = {"error": f"Tool execution failed: {str(e)}"}
                        result_content = json.dumps(error)
                        logger.error(f"Tool {name} failed: {e}")
                
                # Use provider-specific formatting for tool result messages
                if self.provider.supports_native_tool_calling():
                    # For providers like OpenAI/OpenRouter that expect tool_call_id to match a preceding tool_calls message
                    tool_msg = self.provider.build_tool_result_message(tool_call_id, name, result_content)
                    
                    # Only add to llm_history if the previous message contains tool_calls
                    prev_message = self.llm_history[-1] if self.llm_history else None
                    if prev_message and prev_message.get("role") == "assistant" and "tool_calls" in str(prev_message.get("content", "")):
                        self.llm_history.append(tool_msg)
                        logger.info(f"Added tool result to LLM history: {name} = {result_content}")
                    else:
                        # Don't send the tool message to the LLM if there's no preceding tool_calls message
                        logger.warning(f"Skipping tool message for LLM history - no preceding tool_calls found")
                        # But we still need the LLM to know about the tool result in the next response
                        # Add special system message with the tool result to ensure LLM sees it
                        self.llm_history.append({
                            "role": "system", 
                            "content": f"Tool '{name}' was called and returned the result: {result_content}"
                        })
                        logger.info(f"Added system message with tool result: {name} = {result_content}")
                else:
                    # For providers with simpler tool message format
                    tool_msg = {"role": "tool", "name": name, "content": result_content}
                    self.llm_history.append(tool_msg)
                    
                # Always add to history for the app
                self.history.append(tool_msg)
        
        # Return the last assistant message
        return self.history[-1]["content"]
    
    def save_history(self, filepath: Union[str, Path]) -> None:
        """Save conversation history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "system_prompt": self.system_prompt,
                "history": self.history,
                "llm_history": self.llm_history
            }, f, indent=2)
    
    def load_history(self, filepath: Union[str, Path]) -> None:
        """Load conversation history from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.system_prompt = data.get("system_prompt")
        self.history = data.get("history", [])
        self.llm_history = data.get("llm_history", [])
