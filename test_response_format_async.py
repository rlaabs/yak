"""
Test script for the response_format feature in yak.chat, including async support.
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
from typing import List, Optional, Dict, Any
from src.yak import Yak

class WeatherForecast(BaseModel):
    """Weather forecast information"""
    location: str = Field(description="The city or location name")
    temperature: float = Field(description="Current temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g., sunny, cloudy, rainy)")
    forecast: Dict[str, Any] = Field(description="Forecast for upcoming days")
    humidity: Optional[float] = Field(description="Humidity percentage", default=None)

async def test_async_response_format():
    # Initialize Yak with OpenRouter provider
    # You'll need an OPENROUTER_API_KEY environment variable or pass it directly
    print("=== Testing with OpenRouter provider (async) ===")
    
    try:
        yak = Yak(provider="openrouter", model_name="openai/gpt-3.5-turbo")
        
        # Use async chat with response_format
        response = await yak.chat_async(
            "Create a fictional weather forecast for New York. Format it according to the schema.",
            response_format=WeatherForecast
        )
        
        print("Response:")
        print(response)
        print("\n")
        
        # Try to parse the JSON response
        try:
            weather_dict = json.loads(response)
            weather = WeatherForecast(**weather_dict)
            print(f"Successfully parsed weather: {weather}")
            print(f"Location: {weather.location}")
            print(f"Temperature: {weather.temperature}°C")
            print(f"Conditions: {weather.conditions}")
        except json.JSONDecodeError:
            print("Failed to parse response as JSON")
        except Exception as e:
            print(f"Error creating weather model: {e}")
            
    except Exception as e:
        print(f"Error with OpenRouter: {e}")

# Run the async test
if __name__ == "__main__":
    asyncio.run(test_async_response_format())
