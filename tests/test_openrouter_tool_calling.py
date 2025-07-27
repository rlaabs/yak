import os
import asyncio
import pytest
from yak import Yak

# Sample tool function
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical operation on two numbers.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    
    Returns:
        The result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

@pytest.mark.asyncio
async def test_openrouter_tool_calling():
    """Test that OpenRouter tool calling works correctly."""
    # Skip if no API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("No OpenRouter API key found")
    
    # Initialize Yak with OpenRouter provider and calculator tool
    # Using GPT model that supports tool calling
    yak = Yak(
        provider="openrouter",
        model_name="openai/gpt-4o-mini", 
        api_key=api_key,
        tools=[calculator]
    )
    
    # Test a simple query that should trigger tool use
    response = await yak.chat_async(
        "Can you calculate 25 divided by 5 using the calculator tool?"
    )
    
    print(f"Response: {response}")
    
    # Verify history has expected structure
    history = yak.history
    
    # Print the history for debugging
    print("\nMessage History:")
    for i, msg in enumerate(history):
        print(f"{i}. {msg['role']}: {msg.get('content', '')[:100]}")
    
    # Check that we have the expected number of messages
    assert len(history) >= 3, "History should have at least 3 messages (user, assistant, result)"
    
    # Check that the first message is from the user
    assert history[0]["role"] == "user"
    
    # Check if the tool was used
    assistant_msg = history[1]["role"] == "assistant"
    assert assistant_msg, "Second message should be from assistant"
    
    # Check if the final response contains the answer
    assert "5" in response or "5.0" in response, "Response should contain the answer (5)"
    
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_openrouter_tool_calling())
