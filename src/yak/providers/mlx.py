"""
MLX provider for Apple Silicon optimized models.
"""

import logging
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..utils import extract_tool_calls

logger = logging.getLogger(__name__)

# Import with fallback
try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = mlx_lm = load = generate = None


class MLXProvider(LLMProvider):
    """MLX local model provider for Apple Silicon optimized inference."""
    
    def __init__(self, model_name: str, **kwargs):
        if not HAS_MLX:
            raise ImportError("MLX packages not installed. Run: pip install mlx-lm")
        
        self.model_name = model_name
        self.generation_kwargs = kwargs
        
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
        """Generate response using MLX model."""
        try:
            # Combine default generation kwargs with any additional kwargs
            combined_kwargs = {**self.generation_kwargs, **kwargs}
            
            # Apply chat template to messages
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Add tools information if available
            if tools:
                tools_prompt = self._format_tools_info(tools)
                # Insert tools info before the assistant's generation prompt
                assistant_prompt_marker = "Assistant: "
                if assistant_prompt_marker in prompt:
                    insert_position = prompt.rindex(assistant_prompt_marker)
                    prompt = prompt[:insert_position] + tools_prompt + prompt[insert_position:]
                else:
                    prompt = tools_prompt + prompt
            
            # Generate with MLX
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=combined_kwargs.get("max_tokens", 512),
                # temp=combined_kwargs.get("temperature", 0.7),
                # top_p=combined_kwargs.get("top_p", 0.9),
                verbose=combined_kwargs.get("verbose", False)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"MLX model error: {e}")
            return f"Error: {str(e)}"
    
    def _format_tools_info(self, tools: List[Dict]) -> str:
        """Format tools information for inclusion in the prompt."""
        tools_text = "Available tools:\n"
        for tool in tools:
            name = tool.get('name', 'unknown')
            desc = tool.get('description', 'No description')
            tools_text += f"- {name}: {desc}\n"
            
            # Add parameter info
            if 'parameters' in tool or 'input_schema' in tool:
                params = tool.get('parameters', tool.get('input_schema', {}))
                if 'properties' in params:
                    props = params['properties']
                    param_list = []
                    for param_name, param_info in props.items():
                        param_desc = param_info.get('description', '')
                        param_type = param_info.get('type', 'string')
                        param_list.append(f"{param_name} ({param_type}): {param_desc}")
                    if param_list:
                        tools_text += f"  Parameters: {', '.join(param_list)}\n"
        
        tools_text += "\nTo use a tool, format your response like:\n"
        tools_text += "<tool_call>\n<n>tool_name</n>\n<arguments>{\"param\": \"value\"}</arguments>\n</tool_call>\n\n"
        return tools_text
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from MLX model response."""
        return extract_tool_calls(response)
