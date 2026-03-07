#!/usr/bin/env python3
"""
Examples showing the new **request_kwargs passthrough functionality.

This demonstrates how users can now pass temperature (and any future OpenAI params)
via **request_kwargs without changing the public API.
"""

from openai_service.chat_util import get_text, get_structured, chat

def example_basic_usage():
    """Basic usage examples with the new kwargs pattern."""
    
    # Plain call (uses default temperature from CONFIG)
    result = get_text("Hello, how are you today?")
    print("Basic call result:", result)
    
    # Set temperature explicitly (only on models that support it)
    result = get_text(
        "Write a creative haiku about hash maps",
        temperature=0.8
    )
    print("Creative call with temperature=0.8:", result)
    
    # Pass extra parameters the SDK supports now or in the future
    result = get_text(
        "Answer this carefully and precisely",
        user="session-123",
        temperature=0.1,
        presence_penalty=0.2
    )
    print("Call with multiple extra parameters:", result)

def example_structured_output():
    """Example using get_structured with kwargs."""
    
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["sentiment", "confidence"]
    }
    
    # Basic structured call
    result = get_structured(
        "I love this new feature!",
        schema,
        temperature=0.2  # Low temperature for more consistent structured output
    )
    print("Structured output with temperature:", result)

def example_chat_with_kwargs():
    """Example using chat() with kwargs."""
    
    messages = [
        {"role": "user", "content": [{"type": "input_text", "text": "Tell me a short joke"}]}
    ]
    
    result = chat(
        messages,
        temperature=0.9,  # High temperature for creative jokes
        max_output_tokens=100,
        user="joke-session"
    )
    print("Chat with kwargs:", result)

def example_forward_compatibility():
    """Demonstrates forward compatibility - new OpenAI params work without code changes."""
    
    # These params might be added by OpenAI in the future
    # Users can use them immediately without waiting for library updates
    result = get_text(
        "Explain quantum computing briefly",
        temperature=0.3,
        service_tier="default",  # hypothetical future param
        custom_instruction="Be concise",  # another hypothetical param
        # Any kwargs that aren't None will be passed through
        some_future_param=None  # This will be filtered out by _clean_kwargs
    )
    print("Forward-compatible call:", result)

if __name__ == "__main__":
    print("=== OpenAI Service **kwargs Examples ===\n")
    
    try:
        example_basic_usage()
        print("\n" + "="*50 + "\n")
        
        example_structured_output()
        print("\n" + "="*50 + "\n")
        
        example_chat_with_kwargs()
        print("\n" + "="*50 + "\n")
        
        example_forward_compatibility()
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        print("Note: You need a valid OpenAI API key and configuration to run these examples.")
