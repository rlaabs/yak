"""
Test the pydantic_model_to_json_schema function with the new format
"""
from pydantic import BaseModel, Field
from typing import List

from src.yak.utils import pydantic_model_to_json_schema

# Define a test model that resembles the math_response example
class Step(BaseModel):
    explanation: str
    output: str
    
    class Config:
        extra = "forbid"  # Equivalent to additionalProperties: false

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str
    
    class Config:
        extra = "forbid"  # Equivalent to additionalProperties: false

# Test with OpenAI provider
result = pydantic_model_to_json_schema(MathResponse, provider_type='openai')
print("\nOpenAI Format (MathResponse):")
print(result)

# Test with UserInfo model using a specific schema name
class UserInfo(BaseModel):
    name: str
    age: int
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    
    class Config:
        extra = "forbid"

# Using a fixed schema name 'math_response' to match the example format
user_info_result = pydantic_model_to_json_schema(UserInfo, provider_type='openai', schema_name='math_response')
print("\nUserInfo Example with fixed schema name:")
print(user_info_result)

# Compare with the expected format
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

print("\nExpected Format:")
print(expected)
