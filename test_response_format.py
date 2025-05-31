"""
Test script for the response_format feature in yak.chat.
"""

import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from src.yak import Yak

class UserInfo(BaseModel):
    """Information about a user"""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age")
    hobbies: List[str] = Field(description="List of the user's hobbies")
    email: Optional[str] = Field(description="The user's email address", default=None)

# Initialize a yak client with OpenAI
# Make sure to set your OPENAI_API_KEY environment variable or pass it directly
yak = Yak(provider="openai", model_name="gpt-3.5-turbo")

# Test with regular chat (no response_format)
print("=== Test 1: Regular chat ===")
response = yak.chat("Tell me about a fictional user named John Doe.")
print(response)
print("\n")

# Test with response_format
print("=== Test 2: Chat with response_format ===")
response = yak.chat(
    "Create a fictional user profile for John Doe who is 30 years old, likes coding and hiking. Format it according to the schema.",
    response_format=UserInfo
)
print(response)
print("\n")

# Test parsing the JSON response
try:
    print("=== Test 3: Parsing the JSON response ===")
    user_dict = json.loads(response)
    user = UserInfo(**user_dict)
    print(f"Successfully parsed user: {user}")
    print(f"User name: {user.name}")
    print(f"User hobbies: {', '.join(user.hobbies)}")
except json.JSONDecodeError:
    print("Failed to parse response as JSON")
except Exception as e:
    print(f"Error creating user model: {e}")
