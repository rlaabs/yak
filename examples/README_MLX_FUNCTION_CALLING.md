# MLX Function Calling

This document explains how function calling works with the MLX provider in Yak, and how to use the provided examples.

## Overview

Function calling allows the language model to invoke functions defined by the user. The workflow consists of three steps:

1. **Initial Request**: Send a prompt to the model along with a list of available tools/functions
2. **Function Execution**: Extract and execute the function call from the model's response
3. **Final Response**: Send the function result back to the model to get a human-readable response

## Examples

Two example scripts are provided to demonstrate function calling with MLX:

- `mlx_function_calling_example.py`: Synchronous example
- `mlx_function_calling_async_example.py`: Asynchronous example

### Prerequisites

1. Install the required packages:
   ```bash
   pip install mlx mlx-lm
   ```

2. Have a compatible MLX model available. The examples use "mlx-community/Qwen2.5-32B-Instruct-4bit", but you can modify to use any MLX model that supports function calling.

### Running the Examples

Simply execute the example scripts:

```bash
# Synchronous example
python examples/mlx_function_calling_example.py

# Asynchronous example
python examples/mlx_function_calling_async_example.py
```

## Function Calling Workflow in Detail

### 1. Initial Request

```python
# Method 1: Using function objects directly (recommended)
def multiply(a: float, b: float) -> float:
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [multiply]  # Pass the function directly as a tool

# Method 2: Using dictionary-based tool definitions
tools_dict = [
    {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number to multiply"
                },
                "b": {
                    "type": "number",
                    "description": "The second number to multiply"
                }
            },
            "required": ["a", "b"]
        }
    }
]

# Create the prompt
messages = [{"role": "user", "content": "Multiply 12234585 and 48838483920."}]

# Get the initial response with potential tool call
response = provider.generate_sync(messages, tools)
```

### 2. Function Execution

```python
# Extract the tool call from the response
tool_calls = provider.extract_tool_calls(response)

# Get the tool call details
tool_call = tool_calls[0]
tool_name = tool_call["name"]
arguments = tool_call["arguments"]

# Execute the appropriate function
tool_functions = {"multiply": multiply}
tool_result = tool_functions[tool_name](**arguments)
```

### 3. Final Response

```python
# Add the tool result to the messages
messages.append({
    "role": "tool",
    "name": tool_name,
    "content": str(tool_result)
})

# Get the final response
final_response = provider.generate_sync(messages)
```

## How It Works Internally

The MLX provider uses the following process:

1. For each tool, it creates a placeholder function with proper docstring and type hints that match the format expected by MLX's function calling API.

2. When generating a response, it applies the chat template with these function definitions and passes them to the model.

3. If the model decides to use a function, it formats a response with a tool call in this format:
   ```
   <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
   ```

4. The provider extracts this tool call, which the user can then execute to get a result.

5. The user adds the tool result as a new message with role "tool" and calls the generate method again to get the final response.

## Technical Details

- **Tool Call Format**: `<tool_call>{"name": "function_name", "arguments": {...}}</tool_call>`
- **Tool Result Format**: Message with role "tool", name set to the function name, and content set to the result

## Supported Tool Formats

The MLX provider now supports three formats for defining tools:

### 1. Function Objects (Recommended)

```python
def calculate_area(width: float, height: float) -> float:
    """
    Calculate the area of a rectangle
    
    Args:
        width: Width of the rectangle
        height: Height of the rectangle
    """
    return width * height

# Pass the function directly as a tool
tools = [calculate_area]
```

### 2. OpenAI-Style Nested Function Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_area",
            "description": "Calculate the area of a rectangle",
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {
                        "type": "number",
                        "description": "Width of the rectangle"
                    },
                    "height": {
                        "type": "number",
                        "description": "Height of the rectangle"
                    }
                },
                "required": ["width", "height"]
            }
        }
    }
]
```

### 3. Flat Dictionary Tool Definitions

```python
tools = [
    {
        "name": "calculate_area",
        "description": "Calculate the area of a rectangle",
        "parameters": {
            "type": "object",
            "properties": {
                "width": {
                    "type": "number",
                    "description": "Width of the rectangle"
                },
                "height": {
                    "type": "number",
                    "description": "Height of the rectangle"
                }
            },
            "required": ["width", "height"]
        }
    }
]
```

Regardless of which format you use to define tools, the function execution part remains the same:

```python
def calculate_area(width: float, height: float) -> float:
    return width * height

# When executing the tool call
tool_functions = {"calculate_area": calculate_area}
tool_result = tool_functions[tool_name](**arguments)
```
