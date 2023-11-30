
# ChatEngine

## Introduction
`ChatEngine` is a Python class designed for interacting with GPT models via the OpenAI API. It provides an easy-to-use interface for generating responses from text, image, and vision models, offering a flexible and powerful tool for various AI-driven applications.

## Features
- Supports various GPT models including the latest GPT-4 models.
- Handles text, image, and vision-based responses.
- Allows customization of prompts and API parameters.
- Supports response streaming.
- Includes alias functionality for image sizes to enhance user experience.

## Installation
To use the `ChatEngine` class, you need to have Python installed on your system. Additionally, the `openai` and `yaml` Python packages are required.

You can install these dependencies using pip:
```bash
pip install openai pyyaml
```

## Usage

### Basic Setup
First, import the `ChatEngine` class from your module:

```python
from chat_engine import ChatEngine
```

Create an instance of `ChatEngine`:

```python
chat_engine = ChatEngine(api_key='your_openai_api_key')
```

### Generating Responses
To generate a text response:

```python
response = chat_engine.get_response(
    response_type='text',
    system_role="You're a comedian who tells one-liners",
    prompt='Tell me a joke.'
)
print(response)
```

For image generation:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='A scenic mountain view.',
    image_quality='hd'
)
print(response)
```

### Using Aliases
You can use predefined aliases for image sizes:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='An abastract digital wallpaper.',
    image_size='iphone'  # Alias for '1024x1792'
)
print(response)
```

### Generating/Sending Multiple Images
You can generate multiple images with a single prompt:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='A futuristic cityscape.',
    image_model='dall-e-2',
    image_count=3
)
print(response)
```

### Vision API Usage
Using the 'vision' `response_type` to send 2 images, and request to compare them using the `text_prompt` parameter:

```python
response = chat_engine.get_response(
    response_type='vision',
    image_prompt=['http://example.com/image_1.jpg', 'http://example.com/image_2.jpg']
    text_prompt='Tell me differences between these images.'
)
print(response)
```
