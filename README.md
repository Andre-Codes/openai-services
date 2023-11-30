
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

Alternatively, you can create an instance with a configuration file:

```python
chat_engine = ChatEngine(api_key='your_openai_api_key', config_path='path_to_config.yaml')
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

## Using a Configuration File

The ChatEngine class allows the use of a configuration file to define various prompts, roles, and formats for interacting with the OpenAI API. This not only enhances flexibility and usability but also enables sophisticated prompt engineering, allowing for customized and context-specific interactions based on predefined settings.

### Configuration File Structure
The configuration file, typically in YAML format, can contain settings for different operational contexts, response formats, and specialized instructions. Here's an example structure:

```yaml
role_contexts:
  general:
    instruct: "You are a general-purpose assistant."
  api_explain:
    system_role: "You're an expert teacher of code library/API documentation. You will be provided with either the library name or the library documentation itself."
        instruct: "For the API/library provide the description, signature (for classes and methods), parameters, attributes, 'returns', and code examples."
        # 'documentation' is anything you want appended to the very end of the prompt
        # typically used for controlling how detailed the generated response is and how it's concluded
        documentation: "Be very detailed with your response. Conclude with, if applicable, real-world scenarios where this will be useful and how one could implement it."

response_formats:
  markdown:
    instruct: "Respond using markdown style formatting. Use the # character for headers ..."
  html:
    use_css: True
    css: "Put any code inside <pre><code> tag using the following CSS for styling: code  font-family: 'Fira Code' ..."

custom_prompts:
  joke:
    prompt: "Tell me a joke."
  weather:
    prompt: "What's the weather like today?"
```

### Using the Config File with ChatEngine
- **Initialization with Config Path**: Instantiate `ChatEngine` with the path to your config file.
- **Defining Roles and Formats**: The class uses `role_contexts` and `response_formats` from the config for structuring prompts and formatting responses.

This setup allows for dynamic interactions with the GPT model, suitable for a variety of applications.

