# ChatEngine

**ChatEngine** is a versatile Python module designed to interface seamlessly with the OpenAI API. It supports multi-modality interactions, enabling you to generate text, create images, and perform vision-based tasks with a unified and intuitive interface. Whether you're building chatbots, generating artwork, or analyzing images, ChatEngine provides a flexible and powerful toolkit to harness the capabilities of OpenAI's GPT models.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Getting a Text Response](#getting-a-text-response)
  - [Getting an Image Response](#getting-an-image-response)
  - [Getting a Vision Response](#getting-a-vision-response)
  - [Multi-Modality Inference](#multi-modality-inference)
  - [Streaming Responses](#streaming-responses)
- [Message History](#message-history)
- [Pre-Engineered Conversations](#pre-engineered-conversations)

## Features

- **Multi-Modality Support**: Automatically infers the type of response (text, image, vision) based on the prompt.
- **Flexible Configuration**: Customize model parameters, response formats, and more via YAML configuration files.
- **Streaming Responses**: Streamlined handling of real-time responses.
- **Comprehensive Logging**: Enable detailed logging for debugging and monitoring.
- **Easy Integration**: Simple and intuitive API design for quick integration into your projects.


## Configuration

ChatEngine can be configured using environment variables or a YAML configuration file. The primary configuration parameters include API keys, model selection, temperature settings, and more.

### Environment Variables

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-openai-api-key'
```
# Usage

## Initialization

First, import and initialize the `ChatEngine` class. You can provide various parameters such as `temperature`, `model`, and more.

```python
from gpt_utils import ChatEngine

# Initialize with default settings
chat = ChatEngine()

# Initialize with custom settings
chat = ChatEngine(
    temperature=0.5,
    model='gpt-4',
    stream=True,
    enable_logging=True
)
```

## The `get_response() Method`

_When using the `get_response` method, you can pass additional keyword arguments (`kwargs`) to customize the API request based on the selected model. These kwargs will be utilized if they are appropriate and supported by the chosen model, allowing you to fine-tune responses to better suit your application's needs._

### Getting a Text Response

Generate a text-based response using the `get_response` method.

```python
# Simple text response
response = chat.get_response("Tell me a joke.")
print(response)
```

**Example with additional parameters:**

```python
chat.get_response(
    prompt="Explain the theory of relativity.",
    temperature=0.7,
    max_tokens=150,
    stop=["\n"]
)
```

### Getting an Image Response

Generate images based on a text prompt.

```python
# Generate a single image
image_response = chat.get_response(
    prompt="A sunset over a mountain range.",
    response_type='image',
    model='dall-e-3',
    size='1024x1024',
    style='vivid'
)
print(image_response)
```

**Example with multiple images and different settings:**

```python
image_response = chat.get_response(
    prompt="A futuristic city skyline at night.",
    response_type='image',
    model='dall-e-2',
    n=3,  # Number of images
    size='512x512',
    response_format='url'
)
for img in image_response:
    print(img)
```

### Getting a Vision Response

Perform vision-based tasks by providing image prompts along with text.

```python
# Vision response with an image URL
vision_response = chat.get_response(
    prompt="Describe the scene in the image.",
    response_type='vision',
    image_prompt='https://example.com/image.jpg'
)
print(vision_response)
```

**Example with multiple image prompts and additional text:**

```python
vision_response = chat.get_response(
    prompt=[
        'https://example.com/image1.jpg',
        'https://example.com/image2.jpg'
    ],
    text_prompt="Analyze the following images and provide a summary.",
    response_type='vision',
    top_p=0.9,
    max_tokens=200
)
print(vision_response)
```

### Multi-Modality Inference

ChatEngine can automatically determine the appropriate API to call based on the prompt provided.

```python
# Text prompt
chat.get_response("What's the capital of France?")

# Image generation prompt
chat.get_response("Create an illustration of a dragon.")

# Vision analysis prompt with image URL
chat.get_response('https://example.com/object.jpg', text_prompt="What objects are in this image?")
```

### Streaming Responses

Enable streaming to receive responses in real-time. When `stream=True`, `get_response` returns a streaming object that can be processed using the `process_stream` method.

```python
# Initialize ChatEngine with streaming enabled
chat = ChatEngine(stream=True, enable_logging=True)

# Get a streaming text response
stream_obj = chat.get_response("Write a story about a brave knight.")

# Process the stream
processor = chat.process_stream(stream_obj)
processor.run()
```
You can also pass an optional `callback` function, or use one found in the 
`stream_callbacks.py` module.

```python
def my_callback(chunk):
    # Custom processing for each chunk
    print(f"Received chunk: {chunk}")

# Get a streaming response
stream_obj = chat.get_response("Explain quantum computing.", stream=True)

# Process the stream with a custom callback
processor = chat.process_stream(stream_obj, chunk_callback=my_callback)
processor.run()
```

## Messages

ChatEngine maintains a comprehensive message history that adheres to the OpenAI API's expected format. This history is stored in the `.messages` attribute, allowing you to access, review, and manage the conversation context seamlessly.

### Accessing Message History

You can access the entire message history at any point using the `.messages` attribute. This is particularly useful for reviewing past interactions or for debugging purposes.

```python
# Access the current message history
current_messages = chat.messages
print(current_messages)
```

### Message Format

Each entry in the message history is a dictionary formatted according to the OpenAI API specifications. The structure ensures that the conversation context is maintained correctly when interacting with the API.

```python
{
    "role": "system" | "user" | "assistant",
    "content": "Your message content here."
}
```

**Example:**

```python
[
    {"role": "system", "content": "You're a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm today."}
]
```

## Pre-Engineered Conversations

Pre-engineered conversations enhance the effectiveness and reliability of interactions by providing predefined dialogue structures and contexts. This ensures that the AI consistently understands the desired tone, purpose, and flow of the conversation, leading to more accurate and relevant responses.

#### Using List Prompts for Structured Conversations

The `prompt` parameter in the `get_response` method can accept a list of prompts, allowing you to define a sequence of user and assistant messages. Each pair of items in the list is treated as alternating user and assistant roles, enabling the simulation of a complete conversation history. This feature is particularly useful—especially on conjunction with the system role—for setting up initial context or guiding specific interaction scenarios.

**Example:**

```python
prompt_1 = "What is the capitol of England"
reply_1 = "London."
prompt_2 = "What is the capital of France"
pre_defined_msgs = [prompt_1, reply_1, prompt_2]

response = chat.get_response(
    prompt=pre_defined_msgs,
    system_role="You're an assistant who answers concisely.",
    stream=False,
    top_p=0.5
)
```
```text
Output: "Paris."
```
In this example:

- The `conversation_history` list contains alternating user and assistant messages.
- By passing this list to the `get_response` method, ChatEngine maintains the context of the conversation, allowing the AI to provide a coherent and contextually appropriate response.

Even with a clear system role, without sending a pre-defined conversation as in the prior example, the response would most likely be something like this:
```text
'The capital of France is Paris.'
```

**Benefits:**

- **Context Preservation:** Maintains the flow of conversation, ensuring responses are relevant to previous messages.
- **Enhanced Control:** Allows developers to set up specific dialogue scenarios, guiding the AI towards desired outcomes.
- **Improved User Experience:** Provides a more natural and engaging interaction by simulating real conversation dynamics.
