"""
MLX provider for Apple Silicon optimized models.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from .base import LLMProvider
from ..utils import extract_tool_calls

logger = logging.getLogger(__name__)

# Import with fallback
try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate
    from mlx_lm.models.cache import make_prompt_cache
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = mlx_lm = load = generate = make_prompt_cache = None


class MLXProvider(LLMProvider):
    """MLX local model provider for Apple Silicon optimized inference."""
    
    def __init__(self, model_name: str, **kwargs):
        if not HAS_MLX:
            raise ImportError("MLX packages not installed. Run: pip install mlx-lm")
        
        self.model_name = model_name
        self.generation_kwargs = kwargs
        
        # Store a mapping of function names to original tool names
        self.function_to_tool_map = {}
        
        # Load model and tokenizer with MLX
        logger.info(f"Loading MLX model: {model_name}")
        try:
            self.model, self.tokenizer = load(model_name)
            logger.info(f"MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response using MLX model."""
        return self.generate_sync(messages, tools, **kwargs)
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """
        Generate response using MLX model, implementing the two-step tool calling process.
        
        The function calling process follows the workflow:
        1. Initial generation with tools to get a tool call
        2. Tool execution (handled externally)
        3. Final generation with tool result to get the complete response
        
        The tool call format is: <tool_call>JSON_OBJECT</tool_call>
        """
        try:
            # Combine default generation kwargs with any additional kwargs
            combined_kwargs = {**self.generation_kwargs, **kwargs}
            max_tokens = combined_kwargs.get("max_tokens", 2048)
            verbose = combined_kwargs.get("verbose", False)
            
            # Create prompt cache for efficiency
            prompt_cache = make_prompt_cache(self.model)
            
            # Check if the last message is a tool result
            # NOTE: This step may not be useful
            if messages and messages[-1].get("role") == "tool":
                # STEP 3: Generate final response based on tool result
                tool_name = messages[-1].get("name", "unknown")
                tool_result = messages[-1].get("content")
                logger.info(f"Generating final response based on tool result from '{tool_name}'")
                
                # For tool results, create a prompt with the tool result
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
                
                # Generate final response with MLX
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=verbose,
                    prompt_cache=prompt_cache
                )
                
                logger.info(f"Final response generated after tool execution")
                return response
            
            # STEP 1: Initial generation to get a tool call
            
            # MLX requires callable function objects for tools
            if tools:
                # Store function name mappings for later reference
                for tool in tools:
                    if callable(tool):
                        func_name = tool.__name__
                        self.function_to_tool_map[func_name] = func_name
                
                logger.info(f"Using {len(tools)} function objects for MLX: {[t.__name__ for t in tools]}")
                
                # Apply chat template with tools
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    tools=tools
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True
                )
            
            # Generate initial response with MLX
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=verbose,
                prompt_cache=prompt_cache
            )
            
            # Check if the response contains a tool call
            tool_call = self._extract_mlx_tool_call(response)
            
            # STEP 2 (preparation): Return the response with tool call to be executed externally
            if tool_call and tools:
                logger.info(f"Tool call detected: {tool_call.get('name', 'unknown')}")
                return response
            
            # No tool call detected, return the regular response
            return response
            
        except Exception as e:
            logger.error(f"MLX model error: {e}")
            return f"Error: {str(e)}"
        
    def _extract_mlx_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from MLX model response using the format from the example.
        Format: <tool_call>JSON_OBJECT</tool_call>
        """
        try:
            tool_open = "<tool_call>"
            tool_close = "</tool_call>"
            
            if tool_open in response and tool_close in response:
                start_tool = response.find(tool_open) + len(tool_open)
                end_tool = response.find(tool_close)
                tool_call_str = response[start_tool:end_tool].strip()
                
                # Parse the JSON formatted tool call
                tool_call = json.loads(tool_call_str)
                logger.info(f"Extracted tool call: {tool_call}")
                return tool_call
            return None
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from MLX model response."""
        # First try the MLX-specific tool call format
        tool_call = self._extract_mlx_tool_call(response)
        if tool_call:
            # Get the function name from the tool call
            func_name = tool_call.get('name', 'unknown')
            
            # Map the function name back to the original tool name if available
            original_tool_name = self.function_to_tool_map.get(func_name, func_name)
            
            # Log the mapping for debugging
            if func_name != original_tool_name:
                logger.info(f"Mapped function name '{func_name}' back to original tool name '{original_tool_name}'")
            
            # Create a unique ID for this tool call
            tool_call_id = f"call_{original_tool_name}"
            
            # Check if arguments is a string that needs to be parsed
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments string: {arguments}")
            
            return [{
                "id": tool_call_id,
                "name": original_tool_name,
                "arguments": arguments
            }]
        
        # Fall back to the standard extraction method
        # NOTE: The above tool call extraction may be better folded into extract_tool_calls
        return extract_tool_calls(response)
