"""
Test script for comparing OpenAI and OpenRouter response_format handling.
"""

import os
import json
import asyncio
import sys
import subprocess

# Check if httpx is installed, and install it if not
try:
    import httpx
except ImportError:
    print("Installing httpx package for OpenRouter JSON schema support...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    print("httpx installed successfully")

from pydantic import BaseModel, Field
from typing import List, Optional
from src.yak import Yak

class Person(BaseModel):
    """Information about a person"""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age")
    hobbies: List[str] = Field(description="List of the person's hobbies")
    email: Optional[str] = Field(description="The person's email address", default=None)

async def test_both_providers():
    # Test with OpenAI first (this should work already)
    print("=== Testing with OpenAI provider ===")
    try:
        openai_yak = Yak(provider="openai", model_name="gpt-3.5-turbo")
        openai_response = openai_yak.chat(
            "Create a fictional person profile for Jane Doe who is 28 years old, likes reading and swimming.",
            response_format=Person
        )
        
        print("OpenAI Response:")
        print(openai_response)
        print("\nParsing OpenAI response as JSON:")
        
        try:
            person_dict = json.loads(openai_response)
            person = Person(**person_dict)
            print(f"Successfully parsed person: {person}")
        except json.JSONDecodeError:
            print("Failed to parse OpenAI response as JSON")
        except Exception as e:
            print(f"Error creating person model from OpenAI response: {e}")
            
    except Exception as e:
        print(f"Error with OpenAI: {e}")
    
    print("\n")
    
    # Now test with OpenRouter (should work with our fix)
    print("=== Testing with OpenRouter provider ===")
    try:
        # You'll need an OPENROUTER_API_KEY environment variable or pass it directly
        openrouter_yak = Yak(provider="openrouter", model_name="openai/gpt-3.5-turbo")
        
        # Use chat with response_format
        openrouter_response = openrouter_yak.chat(
            "Create a fictional person profile for John Smith who is 32 years old, likes hiking and cooking.",
            response_format=Person
        )
        
        print("OpenRouter Response:")
        print(openrouter_response)
        print("\nParsing OpenRouter response as JSON:")
        
        try:
            person_dict = json.loads(openrouter_response)
            person = Person(**person_dict)
            print(f"Successfully parsed person: {person}")
        except json.JSONDecodeError:
            print("Failed to parse OpenRouter response as JSON")
        except Exception as e:
            print(f"Error creating person model from OpenRouter response: {e}")
            
    except Exception as e:
        print(f"Error with OpenRouter: {e}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_both_providers())
