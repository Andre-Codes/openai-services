
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

## Generating Responses
### Generate a basic text response:

```python
response = chat_engine.get_response(
    response_type='text',
    system_role="You're a comedian who tells one-liners",
    prompt='Tell me a joke.'
)
print(response)
```

**Example using a list of messages (user\assistant):**
```python
response = chat_engine.get_response(
    system_role="You respond with a single word.",
    prompt=["I say red, you say...", "apple", "If I say yellow, you say..."]
)
print(response)

# Output:
# Banana.
```

***Note:** The above example leaves out the `response_type` parameter. See [Smart Inference](#Smart-Inference) section below.


#### Control format style when not using a config file:

When no config file is present to control formatting, or other prompt engineering, the response will return *as is* (plain-text) from the OpenAI API.
To provide a basic level of control for how text-based responses are returned, you can use the `format_style` parameter and type in the desired format style, e.g. html, markdown, json. This will format the inputted prompt with the correct wording to receive the desired format style.


**Note:** When "json" is passed as the `format_style`
in conjunction with the 'text' `response_type`, this will enable the API's *JSON mode* "which guarantees the message the model generates is valid JSON."


```python
response = chat_engine.get_response(
    response_type='text',
    prompt="compare basic demographics between china and japan using a table",
    format_style='markdown'
)
print(response)
```
**raw response output:**
| Demographics   | China         | Japan         |
|----------------|---------------|---------------|
| Population     | 1.4 billion   | 126 million   |
| Area           | 9.6 million km²| 377 thousand km²|
| Capital        | Beijing       | Tokyo         |
| Official Language | Mandarin Chinese | Japanese |
| Life Expectancy | 76.7 years (male), 81.6 years (female) | 81.1 years (male), 87.3 years (female) |
| Literacy Rate  | 96.4%         | 99%           |
| GDP (nominal)  | $14.3 trillion | $5.2 trillion |
| Currency       | Chinese Yuan (CNY) | Japanese Yen (JPY) |


#### Example enabling "json mode":
```python
response = chat_engine.get_response(
    response_type='text',
    prompt="list the top 3 most populated countries",
    format_style='json'
)
print(response)
```
**raw response output:**
```json
{
  "countries": [
    {
      "rank": 1,
      "name": "China",
      "population": "1,412,600,000"
    },
    {
      "rank": 2,
      "name": "India",
      "population": "1,366,200,000"
    },
    {
      "rank": 3,
      "name": "United States",
      "population": "331,000,000"
    }
  ]
}
```

---

### Image generation:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='A scenic mountain view.',
    image_quality='hd'
)
```

#### Using Aliases
You can use predefined aliases for image sizes:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='An abastract digital wallpaper.',
    image_size='iphone'  # Alias for '1024x1792'
)
```

#### Generating/Sending Multiple Images
You can generate multiple images with a single prompt:

```python
response = chat_engine.get_response(
    response_type='image',
    prompt='A futuristic cityscape.',
    image_model='dall-e-2',
    image_count=3
)
```

#### Handling Image API Responses
The `ChatEngine` class is adept at handling various response formats from the image API. Depending on the user's requirements or the specified response format, it can process and return either URLs or base64 encoded representations of images.

- **URL Responses**: If the response format (`response_format`) is set to "url", the returned response will be direct links to the generated images.
- **Base64 Encodings**: For use cases requiring embedded images or additional processing, response format can be set to "b64_json", returning the images as encoded strings.

This flexibility allows for a wide range of applications, from direct image display in web applications to further processing in desktop or server-side applications.

---

### Vision API Usage
Using the 'vision' `response_type` to send 2 images, and request to compare them using the `text_prompt` parameter:

```python
response = chat_engine.get_response(
    response_type='vision',
    image_prompt=['http://example.com/image_1.jpg', 'http://example.com/image_2.jpg']
    text_prompt='Tell me differences between these images.'
)
```

---

## Intuitive `get_response` Method

The `get_response` method in the `ChatEngine` class is designed to intelligently understand and interpret the user's intent based on the provided prompt. This feature enhances user experience by reducing the need for specifying the type of response explicitly.

### Smart Inference
- **Automatic API Selection**: The method automatically determines whether to call the text, image, or vision API based on the nature of the prompt.
- **Handling Various Prompt Types**: It can process a wide range of prompt types including strings, lists of strings, URLs, and file paths, intelligently inferring the most appropriate API to use.
- **Context-Aware**: By analyzing the prompt content (such as keywords for image requests or checking if the input is a file path or URL), `get_response` smartly decides the response type, making the API interaction seamless and user-friendly.

### User-Friendly
- **Less Manual Input**: Users don't have to manually specify the API type (text, image, or vision), as `get_response` smartly infers this from the prompt.
- **Flexibility**: The method is flexible enough to handle explicit instructions as well as interpret ambiguous prompts, catering to a wide range of use cases.

This intuitive approach allows users to interact with the ChatEngine more naturally and efficiently, focusing on their requirements without worrying about the underlying API details.

---

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

