#!/usr/bin/env python3
"""
Example script demonstrating asynchronous MLX function calling with Yak.

This example shows how to:
1. Create an MLX provider with a model that supports function calling
2. Define tools/functions for the model to use
3. Use the two-step function calling process asynchronously
   - Initial call to get the function call
   - Execute the function
   - Final call to get the response with function result
"""

import json
import logging
import asyncio
from typing import Dict, Any

from yak.providers.mlx import MLXProvider

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example function that will be called by the model
async def multiply_async(a: float, b: float) -> float:
    """
    Multiply two numbers and return the result (async version).
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    # Simulate some async processing
    await asyncio.sleep(0.1)
    result = a * b
    logger.info(f"Multiplying {a} x {b} = {result}")
    return result

async def main():
    # Specify the checkpoint - adjust to your locally available model
    checkpoint = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    
    # Initialize the MLX provider
    try:
        provider = MLXProvider(
            model_name=checkpoint,
            max_tokens=2048,
            verbose=True
        )
        logger.info(f"Successfully initialized MLX provider with model: {checkpoint}")
    except ImportError as e:
        logger.error(f"Failed to initialize MLX provider: {e}")
        logger.error("Make sure mlx and mlx-lm are installed: pip install mlx mlx-lm")
        return
    except Exception as e:
        logger.error(f"Error initializing MLX provider: {e}")
        return

    # Define the async function that will be used as a tool
    # The MLX provider now supports function objects directly, similar to OpenAI
    async def multiply_async(a: float, b: float) -> float:
        """
        A function that multiplies two numbers

        Args:
            a: The first number to multiply
            b: The second number to multiply
        """
        # Simulate some async processing
        await asyncio.sleep(0.1)
        return a * b
        
    # Pass the function directly as a tool
    tools = [multiply_async]
    
    # Create the prompt
    prompt = "What is the product of 987654321 and 123456789?"
    messages = [{"role": "user", "content": prompt}]
    
    # STEP 1: Get the initial response with tool call (async)
    logger.info("STEP 1: Generating initial response with potential tool call")
    response = await provider.generate(messages, tools)
    logger.info(f"Initial response: {response}")
    
    # Extract the tool call from the response
    tool_calls = provider.extract_tool_calls(response)
    
    if not tool_calls:
        logger.error("No tool calls were found in the response")
        return
    
    # STEP 2: Execute the function call
    logger.info("STEP 2: Executing the tool call")
    tool_call = tool_calls[0]
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]
    
    logger.info(f"Tool call: {tool_name}({arguments})")
    
    # Execute the appropriate function based on the tool name
    tool_functions = {"multiply": multiply_async}
    
    if tool_name not in tool_functions:
        logger.error(f"Unknown tool: {tool_name}")
        return
    
    # Execute the function with the provided arguments (async)
    tool_result = await tool_functions[tool_name](**arguments)
    
    # STEP 3: Get the final response with the tool result (async)
    logger.info("STEP 3: Generating final response with tool result")
    # Add the tool result to the messages
    messages.append({
        "role": "tool",
        "name": tool_name,
        "content": str(tool_result)
    })
    
    # Get the final response
    final_response = await provider.generate(messages)
    logger.info(f"Final response: {final_response}")

if __name__ == "__main__":
    asyncio.run(main())
