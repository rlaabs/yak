"""
Yak - LLM Chat Framework with Tool Calling Support

A flexible framework for working with multiple LLM providers including OpenAI, 
Anthropic, OpenRouter, local models, and Apple Silicon optimized MLX models.
"""

from .core import Yak
from .providers import (
    LLMProvider,
    OpenAIProvider, 
    AnthropicProvider,
    OpenRouterProvider,
    LocalModelProvider,
    MLXProvider
)
from .utils import extract_tool_calls

__version__ = "0.1.0"
__author__ = "Yak Framework Contributors"

__all__ = [
    "Yak",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider", 
    "OpenRouterProvider",
    "LocalModelProvider",
    "MLXProvider",
    "extract_tool_calls",
]