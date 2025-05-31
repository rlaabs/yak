"""
Validate that the pydantic_model_to_json_schema function produces the expected format
"""
import json
from pydantic import BaseModel, Field
from typing import List

from src.yak.utils import pydantic_model_to_json_schema

# Create a model that matches the example
class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str

# Generate schema using our updated function
result = pydantic_model_to_json_schema(MathResponse, schema_name="math_response")

# Expected format based on the requirement
expected = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_response",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"}
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False
                    }
                },
                "final_answer": {"type": "string"}
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False
        },
        "strict": True
    }
}

# Print both formats for comparison
print("Generated schema:")
print(json.dumps(result, indent=2))

print("\nExpected schema:")
print(json.dumps(expected, indent=2))

# Validate structure matches required format
assert result["type"] == "json_schema"
assert "json_schema" in result
assert "name" in result["json_schema"]
assert result["json_schema"]["name"] == "math_response"
assert "schema" in result["json_schema"]
assert "strict" in result["json_schema"]
assert result["json_schema"]["strict"] is True

# Validate schema content
schema = result["json_schema"]["schema"]
assert schema["type"] == "object"
assert "properties" in schema
assert "steps" in schema["properties"]
assert "final_answer" in schema["properties"]
assert "required" in schema
assert "additionalProperties" in schema
assert schema["additionalProperties"] is False

# Validate steps array
steps = schema["properties"]["steps"]
assert steps["type"] == "array"
assert "items" in steps
assert steps["items"]["type"] == "object"
assert "properties" in steps["items"]
assert "explanation" in steps["items"]["properties"]
assert "output" in steps["items"]["properties"]
assert "required" in steps["items"]
assert "additionalProperties" in steps["items"]
assert steps["items"]["additionalProperties"] is False

print("\nValidation successful - schema matches the required format!")

# Also test with OpenRouter
openrouter_result = pydantic_model_to_json_schema(MathResponse, provider_type="openrouter", schema_name="math_response")
assert openrouter_result == result
print("OpenRouter format matches OpenAI format!")
