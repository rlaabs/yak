"""
Utility functions for the Yak framework.
"""

import json
import re
from typing import List, Dict, Any, Optional, Type

# Import pydantic with fallback
try:
    import pydantic
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    pydantic = None
    BaseModel = object


def pydantic_model_to_json_schema(model: Type, provider_type: str = "openai", schema_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Convert a Pydantic model to a JSON schema compatible with OpenAI/OpenRouter response_format.
    
    Args:
        model: A Pydantic model class
        provider_type: The provider type - "openai" or "openrouter"
        schema_name: Optional custom name for the schema (defaults to model name in lowercase)
    
    Returns:
        A dictionary containing the JSON schema or None if pydantic is not installed
    """
    if not HAS_PYDANTIC:
        raise ImportError("Pydantic is required for response_format. Install with: pip install pydantic")
    
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise ValueError("response_format must be a Pydantic model class")
    
    # Get JSON schema from the Pydantic model
    schema = model.model_json_schema()
    
    # Recursively set additionalProperties to false and ensure required fields
    def process_schema_object(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
                
                # Ensure all properties are listed in required array
                if "properties" in obj and isinstance(obj["properties"], dict):
                    # Make sure all properties are in the required array
                    obj["required"] = list(obj["properties"].keys())
                    
                    # Process each property
                    for prop in obj["properties"].values():
                        process_schema_object(prop)
            
            # Handle nested arrays with object items
            if obj.get("type") == "array" and "items" in obj and isinstance(obj["items"], dict):
                process_schema_object(obj["items"])
    
    process_schema_object(schema)
    
    # Use provided schema_name or default to model name in lowercase
    schema_name = schema_name or model.__name__.lower()
    
    # Format for all providers (same format for both OpenAI and OpenRouter)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True
        }
    }


def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from LLM response.
    Supports multiple formats including XML-style tags and JSON.
    """
    tool_calls = []
    
    # Pattern 1: XML-style tool calls
    xml_pattern = r'<tool_call>\s*<n>(.*?)</n>\s*<arguments>(.*?)</arguments>\s*</tool_call>'
    matches = re.findall(xml_pattern, response, re.DOTALL | re.IGNORECASE)
    
    # Generate unique IDs for tool calls if not provided
    import uuid
    
    for name, args_str in matches:
        try:
            # Try to parse arguments as JSON
            arguments = json.loads(args_str.strip())
        except json.JSONDecodeError:
            # If not valid JSON, treat as string
            arguments = {"input": args_str.strip()}
        
        # Generate a unique ID for this tool call
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        tool_calls.append({
            "id": tool_call_id,
            "name": name.strip(),
            "arguments": arguments
        })
    
    # Pattern 2: Function call format
    func_pattern = r'(\w+)\((.*?)\)'
    if not tool_calls:  # Only try this if XML pattern didn't match
        matches = re.findall(func_pattern, response)
        for name, args_str in matches:
            if name in ["print", "len", "str", "int", "float"]:  # Skip common Python functions
                continue
            
            try:
                # Try to evaluate as Python arguments
                arguments = eval(f"dict({args_str})")
            except:
                arguments = {"input": args_str}
            
            # Generate a unique ID for this tool call
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            tool_calls.append({
                "id": tool_call_id,
                "name": name,
                "arguments": arguments
            })
    
    return tool_calls
