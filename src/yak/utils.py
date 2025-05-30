"""
Utility functions for the Yak framework.
"""

import json
import re
from typing import List, Dict, Any


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
