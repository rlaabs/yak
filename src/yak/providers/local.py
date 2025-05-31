"""
Local model provider using transformers.
"""

import logging
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..utils import extract_tool_calls

logger = logging.getLogger(__name__)

# Import with fallback
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = AutoModelForCausalLM = torch = None


class LocalModelProvider(LLMProvider):
    """Local model provider using transformers."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device
        self.generation_kwargs = kwargs
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **kwargs
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response using local model."""
        return self.generate_sync(messages, tools, **kwargs)
    
    def generate_sync(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response using local model."""
        try:
            # Build prompt from messages
            prompt = self._build_prompt(messages, tools)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Combine default generation kwargs with any additional kwargs
            combined_kwargs = {**self.generation_kwargs, **kwargs}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=combined_kwargs.get("max_new_tokens", 512),
                    do_sample=combined_kwargs.get("do_sample", True),
                    temperature=combined_kwargs.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in combined_kwargs.items() 
                       if k not in ["max_new_tokens", "do_sample", "temperature", "response_format"]}
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return f"Error: {str(e)}"
    
    def _build_prompt(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Build prompt from messages and tools."""
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except:
                pass
        
        # Fallback to simple format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            elif role == "tool":
                prompt += f"Tool Result: {content}\n\n"
        
        if tools:
            tools_text = "Available tools:\n"
            for tool in tools:
                tools_text += f"- {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}\n"
            prompt = tools_text + "\n" + prompt
        
        prompt += "Assistant: "
        return prompt
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from local model response."""
        return extract_tool_calls(response)
