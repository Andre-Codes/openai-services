import base64
import logging
import os
import openai
from openai import OpenAI
import re
import yaml
from typing import Iterator


class ChatEngine:
    """
    A class for interacting with GPT models via the OpenAI API. It supports text, image,
    and vision responses with flexible configurations and formatting options.

    Attributes:
        api_key (str): The OpenAI API key, sourced from environment variables.
        model (str): The GPT model name to be used. Defaults to "gpt-3.5-turbo".
        role_context (str): Operational context for the GPT model, e.g., 'general', 'api_explain'.
        temperature (float): Controls randomness in output. Lower is more deterministic.

    Class Variables:
        DISPLAY_MAPPING (dict): Mappings for IPython display function names.
        MD_TABLE_STYLE (str): Default style for Markdown tables.
        MODEL_OPTIONS (dict): Options for different models, particularly for image generation.
        VALID_RESPONSE_TYPES (set): Set of valid response types for determining the API call
    """

    MD_TABLE_STYLE = "pipes"  # default format for markdown tables

    MODEL_OPTIONS = {
        'image': {
            'dall-e-3': {
                'qualities': ['standard', 'hd'],
                'sizes': ['1024x1024', '1024x1792', '1792x1024'],
                'styles': ['vivid', 'natural'],
                'max_count': 1,
                'response_formats': ['url', 'b64_json']
            },
            'dall-e-2': {
                'qualities': ['standard'],
                'sizes': ['1024x1024', '512x512', '256x256'],
                'styles': [None],
                'max_count': 10,
                'response_formats': ['url', 'b64_json']
            }
        },
        'text': {
            'gpt-3.5-turbo-1106': {

            },
            'gpt-4-1106-preview': {

            }
        }
    }

    # alias mappings for image sizes
    SIZE_ALIASES = {
        'iphone': '1024x1792',
        'portrait': '1024x1792',
        'banner': '1792x1024',
        'landscape': '1792x1024',
        'square': '1024x1024'
    }

    VALID_RESPONSE_TYPES = {'text', 'image', 'vision'}

    def __init__(self, role_context=None, system_role=None, temperature=1,
                 model="gpt-3.5-turbo", stream=False, api_key=None,
                 config_path=None, enable_logging=False):
        """
        Initializes the ChatEngine instance with various configuration settings.

        Parameters:
            role_context (str, optional): Specifies the operational context for the GPT model.
            system_role (str, optional): Describes the system's role in interactions.
            temperature (float, optional): Controls the randomness of the model's output.
            model (str, optional): Specifies the GPT model to use.
            stream (bool, optional): If True, enables streaming of responses.
            api_key (str, optional): The API key for OpenAI.
            config_path (str, optional): Path to a YAML configuration file.

        If a configuration file is provided, it sets up additional formatting and role contexts.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        if not enable_logging:
            logging.disable(logging.CRITICAL)

        if config_path:
            with open(config_path, "r") as f:
                self.CONFIG = yaml.safe_load(f)
        else:
            self.CONFIG = {}

        # Set system role
        self.system_role = system_role or "You're a helpful assistant who answers questions."

        # Set up API access key
        OpenAI.api_key = api_key

        # Turn off/on streaming of response
        self.stream = stream

        # Set the GPT model
        self.model = model

        # Validate and set role_context
        if self.CONFIG:
            available_role_contexts = self.CONFIG.get('role_contexts', {}).keys()
            self.role_context = role_context if role_context in available_role_contexts else 'general'
        else:
            self.role_context = 'general'

        # Validate and set temperature
        self.temperature = temperature

    def set_md_table_style(self, style):
        available_table_styles = self.CONFIG['response_formats']['markdown']['table_styles'].keys()
        if style not in available_table_styles:
            raise ValueError(
                f"Invalid MD_TABLE_STYLE. Available styles: {list(self.CONFIG['table_formatting'].keys())}.")
        self.MD_TABLE_STYLE = self.CONFIG['response_formats']['markdown']['table_styles'][style]

    @staticmethod
    def get_format_styles(self):
        available_formats = list(self.CONFIG['response_formats'].keys())
        return available_formats

    @staticmethod
    def get_role_contexts(self):
        available_role_contexts = list(self.CONFIG['role_contexts'].keys())
        return available_role_contexts

    def _validate_and_assign_params(self, prompt):
        if prompt is None:
            raise ValueError("Prompt can't be None.")
        self.prompt = prompt

    def _handle_format_instructions(self, format_style, prompt):
        """
        Constructs the final prompt by combining role/context-specific instructions with
        formatting instructions based on the provided format style.

        This method leverages the `_handle_role_instructions` method to append any necessary
        instructions or context to the user's prompt, based on the current role_context setting.
        It then applies additional formatting instructions as specified by the `format_style`.

        Parameters:
            format_style (str): The desired format style for the response, such as 'markdown' or 'html'.
                                This influences how the final prompt is constructed.

        Returns:
            str: The fully constructed prompt, ready for the OpenAI API call.

        This method updates `self.complete_prompt` with the final formatted prompt.
        """

        # get the adjusted prompt reconstructed with any custom instructions
        adjusted_prompt = self._handle_role_instructions(prompt)

        if self.role_context != 'general':
            response_formats = self.CONFIG.get('response_formats', {})
            style_name = response_formats.get(format_style, {})
            response_instruct = style_name.get('instruct', '')

            logging.info(f"Adding formatting style instructions from config file for: '{style_name}' ")

            if style_name.lower() == 'markdown':
                md_table_style = style_name.get('table_styles', {}).get(self.MD_TABLE_STYLE, '')
                response_instruct += md_table_style
            elif style_name.lower() == 'html':
                use_css = style_name.get('use_css', False)
                if use_css:
                    css = style_name.get('css', '')
                    response_instruct += css
            # construct and save the final prompt to be sent with API call
            self.complete_prompt = f"{response_instruct}{adjusted_prompt}"

        # If format_style is specified but no role_context found in CONFIG
        elif format_style and format_style != 'text':
            response_instruct = f"For the following request, respond in {format_style} format:\n "
            logging.info(f"Adding instructions to use style '{format_style}' from format_style parameter")
            self.complete_prompt = f"{response_instruct}{adjusted_prompt}"
        # Finally, if no format style specified save only the adjusted prompt
        else:
            self.complete_prompt = adjusted_prompt

        logging.info(f"Finished adding custom formatting instructions to text prompt:\n{self.complete_prompt[:100]}...")

    def _text_api_call(self, text_prompt, **kwargs):
        """
        Makes an API call for text-based responses using the specified prompt and additional parameters.

        This method constructs the message for the OpenAI API call by handling the provided text prompt
        and applying any additional formatting or configuration specified in kwargs. It then makes the
        API call using the OpenAI client and stores the response.

        Parameters:
            text_prompt (str | list): The input text or list of texts to serve as a basis for the OpenAI API response.

        Keyword Arguments:
            streaming (bool, optional): If set to True, the response is streamed. Default is False.
            format_style (str, optional): The desired format style in which the AI responds (e.g., 'markdown').
                Not to be confused with `response_format` which determines the format in which the API sends the data.

            Other keyword arguments may be passed and will be handled according to the OpenAI API requirements.

        Raises:
            openai.APIConnectionError: If there is an issue with the network connection.
            openai.RateLimitError: If the rate limit for the API is reached.
            openai.APIError: For other API-related errors.

        This method updates `self.response` with the response from the OpenAI API call.
        """
        logging.info("Initiating text API call.")

        # Check for streaming and format_style in kwargs
        self.stream = kwargs.get('streaming', self.stream)
        format_style = kwargs.get('format_style', 'text')

        self._handle_format_instructions(format_style=format_style, prompt=text_prompt)
        self._build_messages(prompt=text_prompt, **kwargs)

        response_format = {"type": "json_object" if format_style.lower() == 'json' else 'text'}

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=self.__messages,
                temperature=self.temperature,
                top_p=0.2,
                stream=self.stream,
                response_format=response_format
            )
            if response:
                self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _vision_api_call(self, image_prompt, **kwargs):
        """
        Makes an API call for vision-based responses using the specified image prompt and additional parameters.

        This method is designed to interact with the OpenAI API for vision-related tasks. It constructs
        the appropriate message structure for the vision API, taking into account the provided image URLs
        and any additional text prompt. It then makes the API call using the OpenAI client and stores the response.

        Parameters:
            image_prompt (str | list): A URL or a list of URLs pointing to the images to be processed.

        Keyword Arguments:
            text_prompt (str, optional): An additional text prompt to accompany the image in the API call.
            Other keyword arguments may be passed and will be handled according to the OpenAI API requirements.

        Raises:
            openai.APIConnectionError: If there is an issue with the network connection.
            openai.RateLimitError: If the rate limit for the API is reached.
            openai.APIError: For other API-related errors.

        This method updates `self.response` with the response from the OpenAI API call.
        """
        logging.info("Initiating vision API call.")

        processed_prompts = []

        if isinstance(image_prompt, list):
            for item in image_prompt:
                if not self._prompt_is_url(item) and not self._prompt_is_base64_image(item) and os.path.isfile(item):
                    logging.info(f"Encoding image file for vision API")
                    processed_prompts.append(self.encode_image_prompts(item))
                else:
                    processed_prompts.append(item)
        elif isinstance(image_prompt, str):
            if not self._prompt_is_url(image_prompt) and not self._prompt_is_base64_image(image_prompt) and os.path.isfile(image_prompt):
                logging.info(f"Encoding image file for vision API")
                processed_prompts = self.encode_image_prompts(image_prompt)
            else:
                processed_prompts = [image_prompt]
        else:
            logging.error("Invalid type for image_prompt in vision API call.")
            return

        # Get text_prompt and format_style if provided
        text_prompt = kwargs.get('text_prompt', "Describe the image(s).")
        format_style = kwargs.get('format_style', None)

        self._handle_format_instructions(format_style=format_style, prompt=text_prompt)

        # Build messages for vision API
        self._build_messages(prompt=self.complete_prompt, response_type='vision', image_prompts=processed_prompts)
        # logging.info(f"Formatted 'messages' param: {self.__messages}")
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=self.__messages,
                max_tokens=700,
            )
            if response:
                self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

        logging.info("Vision API call completed.")

    @staticmethod
    def _validate_model_option(option_value, valid_options, option_name, model=None):
        """
        Validates the given option value against a list of valid options. Ignores the option
        if the model does not support it.

        Parameters:
            option_value (str): The value to validate.
            valid_options (list): A list of valid options.
            option_name (str): The name of the option for logging purposes.
            model (str, optional): The model for which the option is being validated.

        Returns:
            str: A valid option value or None if the option is not supported.
        """

        if valid_options == [None]:
            logging.info(f"The parameter '{option_name}' is not supported by the model '{model}'. Ignoring it.")
            return 'vivid'  # something still needs to be passed to the api
        elif valid_options and option_value not in valid_options:
            default = valid_options[0]
            logging.info(f"Provided {option_name} is invalid for model '{model}'. Defaulting to {default}.")
            return default
        return option_value

    def _image_api_call(self, text_prompt, **kwargs):
        """
        Makes an API call for image generation based on the specified text prompt and additional parameters.

        This method interfaces with the OpenAI API for image-related tasks, particularly generating images
        from text prompts. It configures the request parameters based on the method arguments and additional
        keyword arguments, then makes the API call using the OpenAI client to generate images.

        Parameters:
            text_prompt (str): The text prompt based on which images will be generated.

        Keyword Arguments:
            image_model (str, optional): Specifies the image generation model to use (e.g., 'dall-e-3').
            image_count (int, optional): The number of images to generate. For **dall-e-3**, only 1 image is supported
            image_size (str, optional): The size of the generated images.
            image_quality (str, optional): The quality of the generated images.
            image_style: 'Vivid' causes the model to lean towards generating hyperreal and dramatic images.
                'Natural' causes the model to produce more natural, less hyperreal looking images.
                This param is only supported for **dall-e-3**
            revised_prompt (bool, optional): If True, uses the generated revised prompt for image generation.
            Other keyword arguments may be passed and will be handled according to the OpenAI API requirements.

        Raises:
            openai.APIConnectionError: If there is an issue with the network connection.
            openai.RateLimitError: If the rate limit for the API is reached.
            openai.APIError: For other API-related errors.

        This method updates `self.response` with the response from the OpenAI API call.
        """
        logging.info("Initiating image API call.")

        # get the adjusted prompt reconstructed with any custom instructions
        text_prompt = self._handle_role_instructions(text_prompt)

        model = kwargs.get('image_model', 'dall-e-3')
        # Extract parameters with defaults
        count = kwargs.get('image_count', 1)
        size = kwargs.get('image_size', "1024x1024").lower()
        quality = kwargs.get('image_quality', "standard").lower()
        style = kwargs.get('image_style', 'vivid').lower()
        revised_prompt = kwargs.get('revised_prompt', False)
        response_format = kwargs.get('response_format', 'url')
        # Translate size aliases
        size = self.SIZE_ALIASES.get(size.lower(), size)

        # Adjusting model for count
        if count > 1 and model != 'dall-e-2':
            model = 'dall-e-2'
            logging.info(f"Reverting to '{model}' since requested image_count > 1")

        # Fetch model options
        model_opts = self.MODEL_OPTIONS['image'].get(model, {})

        # Validate and adjust size, quality, and style
        size = self._validate_model_option(size, model_opts.get('sizes', []), "image_size", model)
        quality = self._validate_model_option(quality, model_opts.get('qualities', []), "quality", model)
        style = self._validate_model_option(style, model_opts.get('styles', []), "image_style", model)
        logging.info(f"Completed validation of selected options for model '{model}'.")

        # Validate and adjust count
        if count > model_opts.get('max_count', 1):
            count = model_opts['max_count']

        if not revised_prompt and model != 'dall-e-2':
            preface = """
            I NEED to test how the tool works with extremely simple prompts. 
            DO NOT add any detail, just use it AS-IS:
            
            """
        else:
            preface = ''
        try:
            client = OpenAI()
            response = client.images.generate(
                model=model,
                prompt=preface + text_prompt,
                n=count,
                size=size,
                quality=quality,
                style=style,
                response_format=response_format
            )
            logging.info(f"The following settings were used to produce the image:\n"
                         f"model: {model}, count: {count}, size: {size}, quality: "
                         f"{quality}, style: {style}, preface: '{revised_prompt}'")
            self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _build_messages(self, prompt, response_type='text', **kwargs):
        """
        Constructs the message list for API calls based on the given prompt, response type,
        and additional keyword arguments.

        This method prepares the message structure required by the OpenAI API, which varies
        depending on the response type (e.g., 'text', 'vision'). It ensures that the messages
        are formatted correctly, including handling image prompts for 'vision' response types.

        Parameters:
            prompt (str | list): The main text prompt or a list of prompts for the API call.
            response_type (str, optional): The type of response required ('text', 'vision').
                                           Defaults to 'text'.

        Keyword Arguments:
            image_prompts (list, optional): Used when response_type is 'vision'. A list of image URL(s).
            system_role (str, optional): The role of the system in the conversation, influencing the message structure.
            Other keyword arguments may be relevant for specific response types and are handled accordingly.

        This method updates `self.__messages` with the appropriately structured messages for the API call.
        """

        if response_type == 'vision':
            # Handle vision API message format
            image_prompts = kwargs.get('image_prompts', [])
            if isinstance(image_prompts, str):
                image_prompts = [image_prompts]  # Convert single URL to list

            vision_msgs = [{"type": "image_url", "image_url": {"url": url}} for url in image_prompts]
            user_msg = {"role": "user", "content": [{"type": "text", "text": prompt}] + vision_msgs}
            self.__messages = [user_msg]
        else:
            # Existing logic for text messages
            if not all(isinstance(item, str) for item in prompt):
                raise ValueError(f"All elements in the list should be strings {prompt}")
            # Initialize system message
            if 'system_role' in kwargs:
                self.system_role = kwargs['system_role']
            system_msg = [{"role": "system", "content": self.system_role}]

            # Determine user and assistant messages based on the length of the 'prompt'
            if isinstance(prompt, list) and len(prompt) > 1:
                user_assistant_msgs = [
                    {
                        "role": "assistant" if i % 2 else "user",
                        "content": prompt[i]
                    }
                    for i in range(len(prompt))
                ]
            else:
                user_assistant_msgs = [{"role": "user", "content": self.complete_prompt}]

            # Combine system, user, and assistant messages
            self.__messages = system_msg + user_assistant_msgs

    def _handle_role_instructions(self, prompt):
        """
        Construct the context for the prompt by adding role/context specific instructions.

        If no instructions are found in config file, only the `system_role` value
        will be supplied, as this is a necessary arg for the API call.

        Returns:
            str: The inputted prompt with role/context instructions..
        """
        logging.info(f"Adding any role/context instructions to text prompt: {prompt[:15]}...")

        default_documentation = self.CONFIG.get('role_contexts', {}) \
            .get('defaults', {}).get('documentation', '')

        default_role_instructions = self.CONFIG.get('role_contexts', {}) \
            .get('defaults', {}).get('instruct', '')

        default_system_role = self.CONFIG.get('role_contexts', {}) \
            .get('defaults', {}).get('system_role', self.system_role)

        documentation = self.CONFIG.get('role_contexts', {}) \
            .get(self.role_context, {}).get('documentation', default_documentation)

        role_instructions = self.CONFIG.get('role_contexts', {}) \
            .get(self.role_context, {}).get('instruct', default_role_instructions)

        system_role = self.CONFIG.get('role_contexts', {}) \
            .get(self.role_context, {}).get('system_role', default_system_role)

        # set the system_role class variable
        self.system_role = system_role
        # construct the prompt by prefixing any role instructions
        # and appending any documentation to the end
        prompt_with_context = f"{role_instructions}{prompt}{documentation}"
        # raise AssertionError("prompt with context:" + str(prompt_with_context))
        self.prompt = prompt_with_context

        return prompt_with_context

    def _infer_response_type(self, prompt):
        """
        Infers the response type based on the content and format of the prompt.
        """
        if isinstance(prompt, list):
            # If the prompt is a list, check each item
            if all(isinstance(item, str) and (self._prompt_is_url(item) or self._prompt_is_base64_image(item) or os.path.isfile(item)) for item in prompt):
                logging.info("Prompt is a list suitable for the vision API.")
                return 'vision'
            else:
                logging.info("Prompt is a list suitable for the text API.")
                return 'text'
        elif isinstance(prompt, str):
            # Check for URL, base64 encoded image, or file path indicative of the vision API
            if self._prompt_is_url(prompt) or self._prompt_is_base64_image(prompt) or os.path.isfile(prompt):
                logging.info("Prompt is a string suitable for the vision API.")
                return 'vision'
            # Check for keywords indicative of the image API
            elif self._prompt_suggests_image(prompt):
                logging.info("Prompt suggests an image generation request.")
                return 'image'
            else:
                logging.info("Prompt is a string suitable for the text API.")
                return 'text'
        else:
            # Default to text if the prompt type is unexpected
            logging.warning("Prompt type is unexpected. Defaulting to text API.")
            return 'text'

    @staticmethod
    def _prompt_is_url(string):
        """
        Checks if the string is a URL.
        """
        return re.match(r'https?://', string) is not None

    @staticmethod
    def _prompt_is_base64_image(string):
        """
        Checks if the string is a base64 encoded image.
        """
        return re.match(r'data:image/.+;base64,', string) is not None

    @staticmethod
    def _prompt_suggests_image(prompt):
        """
        Checks if the prompt starts with phrases that suggest an image generation request.
        """
        image_keywords = [
            'sketch of', 'paint', 'design a', 'visualize',
            'depict', 'create an image of', 'generate a picture of', 'illustration of',
            'render', 'artwork of', 'graphic of', 'visual representation of',
            'art style', 'art format', 'image context', 'image of'
        ]
        # Split the prompt into words and consider only the first five
        first_five_words = ' '.join(prompt.lower().split()[:5])
        # Check if any of the keywords match within the first five words
        return any(keyword in first_five_words for keyword in image_keywords)

    @staticmethod
    def encode_image_prompts(image_file_paths):
        """
        Encodes one or multiple image files to base64 data URLs.

        Parameters:
            image_file_paths (str or list): A single path or a list of paths to the image files.

        Returns:
            list: List of base64 encoded data URLs of the images.
        """
        # if isinstance(image_file_paths, str):
        #     # If it's a single image path, convert it to a list
        #     image_file_paths = [image_file_paths]
        #     logging.info(f"Begin encoding of files: {image_file_paths}")

        def encode_single_image(image_file_path):
            """Encodes a single image file to a base64 data URL."""
            if not os.path.isfile(image_file_path):
                logging.warning(f"File not found: {image_file_path}")
                return None

            logging.info(f"Encoding image: {image_file_path}")
            with open(image_file_path, "rb") as image_file:
                bytes_data = image_file.read()

            file_ext = image_file_path.split('.')[-1].lower()
            file_ext = mime_to_extension.get(file_ext, file_ext)
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            return f"data:image/{file_ext};base64,{base64_image}"

        mime_to_extension = {
            'jpeg': 'jpg',
            'png': 'png',
            'gif': 'gif',
            'bmp': 'bmp',
            'svg+xml': 'svg',
            'webp': 'webp'
        }

        if isinstance(image_file_paths, str):
            # Process a single image file path
            return encode_single_image(image_file_paths)
        elif isinstance(image_file_paths, list):
            # Process a list of image file paths
            encoded_images = [encode_single_image(path) for path in image_file_paths if path]
            return [img for img in encoded_images if img]  # Filter out any None values

        logging.warning("Invalid input type for image_file_paths.")
        return None

    def extract_response(self, response_type, **kwargs):
        """
        Extracts and formats the response from the OpenAI API based on the response type.

        Parameters:
            response_type (str): The type of response ('text', 'image', or 'vision').
            kwargs: Additional keyword arguments.

        Returns:
            The formatted response based on the response type.
        """
        logging.info(f"Extracting response for type: {response_type}")

        if response_type == 'text':
            return self.response.choices[0].message.content
        elif response_type == 'image':
            return self._extract_image_response(kwargs.get('revised_prompt', False))
        elif response_type == 'vision':
            return self.response.choices[0].message.content

    def _extract_image_response(self, revised_prompt):
        """
        Extracts the image response, handling URLs and base64 JSON.

        Parameters:
            revised_prompt (bool): Flag to determine the format of the response.

        Returns:
            Either a list of image URLs/b64_json or tuples of image URLs/b64_json and revised prompts.
        """

        if any(getattr(image, 'b64_json', None) for image in self.response.data):
            # Handling base64 JSON data
            logging.info("Extracting 'b64_json' image response")
            data = [getattr(image, 'b64_json', None) for image in self.response.data]
        else:
            # Handling URLs
            logging.info("Extracting 'url' image response")
            data = [getattr(image, 'url', None) for image in self.response.data]

        if revised_prompt:
            logging.info("Including revised prompts in response (creates a list of tuples).")
            revised_prompts = [getattr(image, 'revised_prompt', None) for image in self.response.data]
            return list(zip(data, revised_prompts))

        return data

    def get_response(self, response_type: str = None,
                     prompt: str | list = None,
                     raw_output: bool = False, **kwargs) -> str | dict | tuple | list | Iterator:
        """
        Retrieves a response from the OpenAI API based on the specified parameters. This is the main
        method to interact with different OpenAI API functionalities like text, image, and vision responses.

        It routes the request to the appropriate internal method based on the response type and handles
        the construction of prompts, messages, and API call parameters.

        Parameters:
            response_type (str): The type of response to generate; options include 'text', 'image', or 'vision'.
                                 Determines the OpenAI API endpoint to use. Defaults to 'text'.
            prompt (str | list): The input prompt or a list of prompts for the OpenAI API.
            raw_output (bool): Determines whether to return the raw API response or a processed result.
                               Defaults to True for raw JSON output.

        Keyword Arguments:
            Keyword arguments specific to each response type are forwarded to the respective internal methods.
            For 'text' and 'vision': these may include 'streaming', 'format_style', 'text_prompt', etc.
            For 'image': these may include 'image_model', 'image_count', 'image_size', 'image_quality', etc.

        Returns:
            Depending on the response type and 'raw_output' flag:
            - A string, dictionary, tuple, list, or an iterator containing the response data.

        Raises:
            The method may raise various exceptions based on internal validation and API call outcomes.

        Usage Example:
            get_response(response_type='text', prompt='Tell me a joke.', raw_output=False)
        """

        # Automatically determine response type if not provided
        if not response_type:
            response_type = self._infer_response_type(prompt)

        logging.info(f"Getting response: type={response_type}")

        # validate and set the instance variable for prompt
        self._validate_and_assign_params(prompt)

        # Validate the response type
        if response_type not in ChatEngine.VALID_RESPONSE_TYPES:
            raise ValueError(
                f"Invalid response type: '{response_type}'. "
                f"Valid options are: {ChatEngine.VALID_RESPONSE_TYPES}"
            )

        # Call the appropriate API based on response_type
        if response_type == 'text':
            self._text_api_call(text_prompt=prompt, **kwargs)
        elif response_type == 'image':
            self._image_api_call(text_prompt=prompt, **kwargs)
        elif response_type == 'vision':
            self._vision_api_call(image_prompt=prompt, **kwargs)

        # Return finished response from OpenAI
        if not raw_output and not self.stream:  # if raw and stream are False
            return self.extract_response(response_type, **kwargs)

        print(self.response)
        return self.response
