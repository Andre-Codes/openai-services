import base64
import logging
import os
import re
import threading
from io import BytesIO, IOBase
from urllib.parse import urlparse

import openai
import yaml
from openai import OpenAI
from typing import Iterator

from stream_processor import StreamProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


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

            },
            'gpt-4-turbo': {

            },
            'gpt-4o': {

            },
            'o1-preview': {},
            'o1-mini': {}
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

    BASE64_IMAGE_PATTERN = re.compile(r'^data:image/.+;base64,', re.IGNORECASE)

    # Allowed kwargs for API calls
    ALLOWED_TEXT_KWARGS = {'response_format', 'stream', 'top_p', 'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty'}
    ALLOWED_IMAGE_KWARGS = {'model', 'count', 'size', 'quality', 'style', 'response_format'}
    ALLOWED_VISION_KWARGS = {'text_prompt', 'stream', 'top_p', 'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty'}

    def __init__(self, role_context=None, system_role=None, temperature: float = 0.7,
                 model: str = "gpt-3.5-turbo", stream: bool = False, api_key: str = None,
                 config_path: str = None, enable_logging: bool = False):
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
        self._configure_logging(enable_logging)

        # attribute set when the API called is made (text, image, vision)
        self.api_used = None

        if config_path:
            with open(config_path, "r") as f:
                self.CONFIG = yaml.safe_load(f)
        else:
            self.CONFIG = {}

        # Set system role
        self.system_role = system_role or "You're a helpful assistant who answers questions."

        # Initialize messages and apply system role
        self.messages: list = [{"role": "system", "content": self.system_role}]

        # Set up API access key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error("OpenAI API key must be provided either via parameter or environment variable.")
            raise ValueError("OpenAI API key must be provided either via parameter or environment variable.")
        openai.api_key = self.api_key  # Set the API key for OpenAI

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.api_key)

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
        self.temperature = self._validate_temperature(temperature)

    def _configure_logging(self, enable_logging: bool) -> None:
        """
        Configures the logging settings.

        Parameters:
            enable_logging (bool): If True, enables INFO level logging. Otherwise, disables logging.
        """
        if enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
            logging.info("Logging is enabled.")
        else:
            logging.disable(logging.CRITICAL)
            logging.debug("Logging is disabled.")

    def _validate_temperature(self, temperature: float) -> float:
        """
        Validates the temperature parameter to ensure it is within acceptable bounds.

        Parameters:
            temperature (float): The desired temperature setting.

        Returns:
            float: The validated temperature.

        Raises:
            ValueError: If temperature is not within (0, 2).
        """
        if not (0 <= temperature <= 2):
            logging.warning(f"Temperature {temperature} out of bounds. Clamping to [0, 2].")
            temperature = max(0.0, min(temperature, 2))
        logging.info(f"Temperature set to {temperature}.")
        return temperature

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

        # TODO: POSSIBLY REMOVE THIS ENTIRE METHOD. BE SURE SELF.complete_prompt
        #  IS NOT USED ELSEWHERE UNINTENDED

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

    def _text_api_call(self, text_prompt, stream: bool = None, top_p: float = None, **kwargs):
        """
        Makes an API call for text-based responses using the specified prompt and additional parameters.

        Parameters:
            text_prompt (str | list): The input text or list of texts to serve as a basis for the OpenAI API response.
            format_style (str, optional): The desired format style for the response (e.g., 'markdown', 'json').
                                           Determines the response format from the API.
            stream (bool, optional): If set to True, the response is streamed. Defaults to the instance's stream setting.
            top_p (float, optional): Controls the diversity via nucleus sampling. Defaults to 1.0.

        Keyword Arguments:
            temperature (float, optional): Controls the randomness of the model's output.
            max_tokens (int, optional): The maximum number of tokens to generate in the completion.
            stop (str or list, optional): Up to 4 sequences where the API will stop generating further tokens.
            presence_penalty (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their presence.
            frequency_penalty (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their frequency.

        Returns:
            Any: The response from the OpenAI API, either streamed or as a complete object.

        Raises:
            openai.OpenAIError: For any errors related to the OpenAI API.
        """
        logging.info("Initiating text API call.")

        # Override stream if provided
        use_stream = stream if stream is not None else self.stream

        # Check for streaming and format_style in kwargs
        format_style = kwargs.get('format_style', 'text')

        self._handle_format_instructions(format_style=format_style, prompt=text_prompt)
        self._build_messages(prompt=text_prompt, **kwargs)

        response_format_mapping = {
            'string': 'string',
            'json': 'json_object',
            'text': 'text'
        }
        response_format = kwargs.pop('response_format', 'auto').lower()
        response_format = {"type": response_format_mapping.get(response_format, 'text')}
        print("#########", response_format)
        # Set top_p if provided, else default to 1.0
        top_p_value = top_p if top_p is not None else 1.0

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": top_p_value,
            "stream": use_stream,
            "response_format": response_format
        }

        # Update api_params with additional kwargs, ensuring no conflicts
        allowed_kwargs = {'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty'}
        for key in allowed_kwargs:
            if key in kwargs:
                api_params[key] = kwargs[key]

        try:
            # Make the API call using the pre-initialized client
            response = self.openai_client.chat.completions.create(**api_params)
            self.response = response
            logging.info("Text API call successful.")
            return response
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error during text call: {e}")
            raise

    def _vision_api_call(self, text_prompt, image_prompt, stream: bool = None, top_p: float = None, **kwargs):
        """
        Makes an API call for vision-based responses using the specified image prompt and additional parameters.

        This method is designed to interact with the OpenAI API for vision-related tasks. It constructs
        the appropriate message structure for the vision API, taking into account the provided image URLs
        and any additional text prompt. It then makes the API call using the OpenAI client and stores the response.

        Parameters:
            image_prompt (str | list): A URL or a list of URLs pointing to the images to be processed.
            text_prompt (str): The text prompt to be used in conjunction with the image.

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

        use_stream = stream if stream is not None else self.stream

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
                logging.info(f"Encoding image file or file-like object for vision API")
                processed_prompts = self.encode_image_prompts(image_prompt)
            else:
                processed_prompts = [image_prompt]

        # Handle single file-like object or string
        elif isinstance(image_prompt, IOBase):  # Direct file-like object check
            logging.info("Encoding file-like object for vision API")
            processed_prompts = self.encode_image_prompts(image_prompt)

        else:
            logging.error("Invalid type for image_prompt in vision API call.")
            return

        # Get text_prompt, if not found or found but value is None
        # default to standard message
        if text_prompt is None:
            text_prompt = "Describe the image(s)."

        format_style = kwargs.get('format_style', None)
        # self._handle_format_instructions(format_style=format_style, prompt=text_prompt)

        # Build messages for vision API
        self._build_messages(prompt=text_prompt, response_type='vision', image_prompts=processed_prompts)
        # logging.info(f"Formatted 'messages' param: {self.messages}")

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": top_p if top_p is not None else 1.0,
            "stream": use_stream,
        }

        # Update api_params with additional kwargs, ensuring no conflicts
        allowed_kwargs = {'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty'}
        for key in allowed_kwargs:
            if key in kwargs:
                api_params[key] = kwargs[key]

        try:
            # Make the API call using the pre-initialized client
            response = self.openai_client.chat.completions.create(**api_params)
            self.response = response
            logging.info("Vision API call successful.")
            return response
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error during vision call: {e}")
            raise

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

        # If text_prompt is a list (assuming list of message history) isolate prompt
        #  without message history. For image prompt length requirements.
        if isinstance(text_prompt, list):
            text_prompt = text_prompt[-1]

        # get the adjusted prompt reconstructed with any custom instructions
        text_prompt = self._handle_role_instructions(text_prompt)
        self._build_messages(prompt=text_prompt, **kwargs)

        model = kwargs.get('model', 'dall-e-3')
        # Extract parameters with defaults
        count = kwargs.get('count', 1)
        size = kwargs.get('size', "1024x1024").lower()
        quality = kwargs.get('quality', "standard").lower()
        style = kwargs.get('style', 'vivid').lower()
        revised_prompt = kwargs.get('revised_prompt', True)
        response_format = kwargs.get('response_format', 'url')

        # Translate size aliases
        size_pixels = self.SIZE_ALIASES.get(size.lower(), size)
        print("%%%%%%%%%%%", size_pixels)

        # Adjusting model for count
        if count > 1 and model != 'dall-e-2':
            model = 'dall-e-2'
            logging.info(f"Reverting to '{model}' since requested image_count > 1")

        # Fetch model options
        model_opts = self.MODEL_OPTIONS['image'].get(model, {})

        # Validate and adjust size, quality, and style
        size_pixels = self._validate_model_option(size_pixels, model_opts.get('sizes', []), "size", model)
        quality = self._validate_model_option(quality, model_opts.get('qualities', []), "quality", model)
        style = self._validate_model_option(style, model_opts.get('styles', []), "style", model)
        logging.info(f"Completed validation of selected options for model '{model}'.")
        print("%%%%%%%%%%%", size_pixels)

        # Validate and adjust count
        if count > model_opts.get('max_count', 1):
            count = model_opts['max_count']

        if not revised_prompt and model != 'dall-e-2':
            preface = """
            I NEED to test how the tool works with extremely simple prompts. 
            DO NOT add any detail, just use it AS-IS:
            
            """
            full_prompt = f"{preface}\n\n{text_prompt}"
        else:
            full_prompt = text_prompt

        # Prepare API call parameters
        api_params = {
            "model": model,
            "prompt": full_prompt,
            "n": count,
            "size": size_pixels,
            "quality": quality,
            "style": style,
            "response_format": response_format
        }

        # Update api_params with additional allowed kwargs
        for key in self.ALLOWED_IMAGE_KWARGS:
            if key in kwargs:
                api_params[key] = kwargs[key]

        try:
            # Make the API call using the pre-initialized client
            response = self.openai_client.images.generate(**api_params)
            self.response = response
            logging.info("Image API call successful.")
            return response
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error during image call: {e}")
            raise

    def _build_messages(self, prompt: str | list[str], response_type: str = 'text', **kwargs):
        """
        Constructs and appends the message list for API calls based on the given prompt and response type.

        Parameters:
            prompt (str | list): The main text prompt or a list of prompts for the API call.
            response_type (str, optional): The type of response required ('text', 'vision').
                                           Defaults to 'text'.

        Keyword Arguments:
            image_prompts (list, optional): Used when response_type is 'vision'. A list of image URL(s).
            system_role (str, optional): The role of the system in the conversation, influencing the message structure.

        Raises:
            ValueError: If the prompt structure is invalid.
        """

        if response_type == 'vision':
            self._build_vision_messages(prompt, **kwargs)
        elif response_type == 'text':
            self._build_text_messages(prompt)
        else:
            logging.error(f"Unsupported response type: {response_type}")
            raise ValueError(f"Unsupported response type: {response_type}")

    def _build_text_messages(self, prompt: str | list[str]) -> None:
        # Existing logic for text messages
        if not all(isinstance(item, str) for item in prompt):
            logging.error(f"All elements in the prompt list must be strings. Received: {prompt}")
            raise ValueError(f"All elements in the list should be strings {prompt}")

        # Determine user and assistant messages based on the length of the 'prompt'
        if isinstance(prompt, list) and len(prompt) > 1:
            user_msgs = [
                {
                    "role": "assistant" if i % 2 else "user",
                    "content": prompt[i]
                }
                for i in range(len(prompt))
            ]
            logging.debug(f"Appending dialogue history messages: {user_msgs}")
            self.messages.extend(user_msgs)
        else:
            user_msg = {"role": "user", "content": prompt}
            logging.debug(f"Appending single user message from list: {user_msg}")
            self.messages.append(user_msg)

    def _build_vision_messages(self, prompt: str | list[str], **kwargs) -> None:
        """
        Constructs and appends messages formatted for vision API calls.

        Parameters:
            prompt (str | list): The main text prompt for the vision API.
            image_prompts (list, optional): A list of image URLs or file paths.

        Raises:
            ValueError: If image_prompts are not in the expected format.
        """
        # Handle vision API message format

        image_prompts = kwargs.get('image_prompts', [])

        if isinstance(image_prompts, str):
            image_prompts = [image_prompts]  # Convert single URL to list
        elif not isinstance(image_prompts, list):
            logging.error("image_prompts must be a string or a list of strings.")
            raise ValueError("image_prompts must be a string or a list of strings.")

        # Construct vision messages
        vision_msgs = [{"type": "image_url", "image_url": {"url": url}} for url in image_prompts]
        text_msg = {"type": "text", "text": prompt} if isinstance(prompt, str) else {"type": "text",
                                                                                     "text": " ".join(prompt)}
        content = [text_msg] + vision_msgs

        user_msg = {"role": "user", "content": content}
        logging.debug(f"Appending vision user message: {user_msg}")
        self.messages.append(user_msg)

    def _handle_role_instructions(self, prompt):
        """
        Construct the context for the prompt by adding role/context specific instructions.

        If no instructions are found in config file, only the `system_role` value
        will be supplied, as this is a necessary arg for the API call.

        Returns:
            str: The inputted prompt with role/context instructions..
        """
        logging.info(f"Adding any role/context instructions to text prompt...")

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
        response_type: str = 'text'

        # Assume a list of AI/user messages and work with only the last item.
        # This ensures that only the last item will be treated as the most recent.
        # Necessary for vision and image API to understand.
        last_item = prompt[-1] if isinstance(prompt, list) else prompt
        if isinstance(last_item, list):
            # If the prompt is a list, check each item
            if all(isinstance(item, str) and
                   (self._prompt_is_url(item) or
                    self._prompt_is_base64_image(item) or
                    os.path.isfile(item)) for item in last_item):
                logging.info("Prompt is a list suitable for the vision API.")
                response_type = 'vision'
                prompt = last_item
            else:
                logging.info("Prompt is a list suitable for the text API.")
                response_type = 'text'

        elif isinstance(last_item, str):
            # Check for URL, base64 encoded image, or file path indicative of the vision API
            if (self._prompt_is_url(last_item) or
                    self._prompt_is_base64_image(last_item) or
                    os.path.isfile(last_item)):
                logging.info("Prompt is a string suitable for the vision API.")
                response_type = 'vision'
                prompt = last_item
            # Check for keywords indicative of the image API
            elif self._prompt_suggests_image(last_item):
                logging.info("Prompt suggests an image generation request.")
                response_type = 'image'
                prompt = last_item
            else:
                logging.info("Prompt is a string suitable for the text API.")
                response_type = 'text'
        else:
            # Default to text if the prompt type is unexpected
            logging.warning("Prompt type is unexpected. Defaulting to text API.")
        #TODO: investigate whether returning an adjusted prompt is needed
        return response_type, prompt

    @staticmethod
    def _prompt_is_url(string):
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def _prompt_is_base64_image(string):
        """
        Checks if the string is a base64 encoded image.
        """
        return bool(ChatEngine.BASE64_IMAGE_PATTERN.match(string))

    @staticmethod
    def _prompt_suggests_image(prompt):
        """
        Determines if the user's prompt suggests an image generation request,
        while minimizing false positives by ensuring the presence of action verbs.

        Parameters:
            prompt (str): The user's input prompt.

        Returns:
            bool: True if the prompt suggests an image generation request, False otherwise.
        """
        # Define action verbs that suggest image generation
        action_verbs = [
            r'\bgenerate\b', r'\bcreate\b', r'\bdesign\b', r'\bvisualize\b',
            r'\bdepict\b', r'\billustrate\b', r'\brender\b', r'\bproduce\b',
            r'\bmake\b', r'\bcompose\b', r'\bcraft\b', r'\bconstruct\b',
            r'\bdraw\b', r'\bsketch\b'
        ]

        # Define image-related keywords and phrases
        image_keywords = [
            r'\bimage of\b', r'\bpicture of\b', r'\bpicture using\b',
            r'\bimage from\b', r'\bimage using\b', r'\billustration of\b',
            r'\bgraphic of\b', r'\bvisual representation of\b',
            r'\bartwork of\b', r'\bdesign for\b', r'\blogo for\b',
            r'\bscene of\b', r'\bconcept art of\b', r'\bmodel of\b',
            r'\bblueprint of\b', r'\bschematic of\b', r'\bdigital art of\b',
            r'\bvectors of\b', r'\binfographic of\b'
        ]

        # Compile regex patterns for action verbs and image keywords
        verbs_pattern = re.compile('|'.join(action_verbs), re.IGNORECASE)
        keywords_pattern = re.compile('|'.join(image_keywords), re.IGNORECASE)

        # Search for action verbs and image keywords in the prompt
        verbs_found = verbs_pattern.search(prompt)
        keywords_found = keywords_pattern.search(prompt)

        # Debugging statements (can be removed in production)
        logging.debug(f"Verbs found: {verbs_found.group(0) if verbs_found else 'None'}")
        logging.debug(f"Image keywords found: {keywords_found.group(0) if keywords_found else 'None'}")

        # Return True only if both an action verb and an image keyword are found
        return bool(verbs_found and keywords_found)

    @staticmethod
    def encode_image_prompts(image_file_paths, file_type=None):
        """
        Encodes one or multiple image files or file-like objects to base64 data URLs.

        Parameters:
            image_file_paths (str, file-like, or list): A single path, file-like object, or a list of paths/objects.
            file_type (str): Default file extension to use if no extension is found.

        Returns:
            list: List of base64 encoded data URLs of the images.
        """

        def encode_single_image(image_file):
            """Encodes a single image file or file-like object to a base64 data URL."""
            if isinstance(image_file, IOBase):  # Check if it's a file-like object
                logging.info(
                    f"Encoding file-like image object: {getattr(image_file, 'name', 'Unnamed file-like object')}")
                bytes_data = image_file.read()
                file_extension = getattr(image_file, 'name', f".{file_type}").rsplit('.', 1)[-1].lower() or file_type
            elif os.path.isfile(image_file):  # Standard file path
                logging.info(f"Encoding image: {image_file}")
                with open(image_file, "rb") as f:
                    bytes_data = f.read()
                file_extension = image_file.rsplit('.', 1)[-1].lower() if '.' in image_file else file_type
            else:
                logging.warning(f"File or file-like object not found: {image_file}")
                return None

            # Map to the appropriate MIME type if available, or keep as is
            file_extension = mime_to_extension.get(file_extension, file_extension)
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            return f"data:image/{file_extension};base64,{base64_image}"

        mime_to_extension = {
            'jpeg': 'jpg',
            'png': 'png',
            'gif': 'gif',
            'bmp': 'bmp',
            'svg+xml': 'svg',
            'webp': 'webp'
        }

        # Handle single or multiple inputs
        if isinstance(image_file_paths, (str, IOBase)):
            # Process a single image file path or file-like object
            return encode_single_image(image_file_paths)
        elif isinstance(image_file_paths, list):
            # Process a list of image file paths or file-like objects
            encoded_images = [encode_single_image(file) for file in image_file_paths if file]
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
            return self._extract_image_response(kwargs.get('revised_prompt', True))
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

    def get_response_parallel(self, prompts: list, response_type='text', raw_output=False, **kwargs) -> dict:
        results = {}
        threads = []
        lock = threading.Lock()

        def thread_function(current_prompt, response_type, raw_output, **kwargs):
            result = self.get_response(current_prompt, response_type, raw_output, **kwargs)
            print(result)
            with lock:
                results[current_prompt] = result

        for prompt in prompts:
            thread = threading.Thread(target=thread_function, args=(prompt, response_type, raw_output), kwargs=kwargs)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def process_stream(self, stream_obj: Iterator, update_history: bool = True, chunk_callback=None) -> StreamProcessor:
        """
        Returns a StreamProcessor instance for processing the streaming object.

        Parameters:
            stream_obj (Iterator): The streaming object returned by get_response.
            update_history (bool): Whether to update self.messages with the assistant's reply.
            chunk_callback (callable, optional): A function to call with each chunk of the reply.
                                                 Defaults to None (will use default chunk callback).

        Returns:
            StreamProcessor: An iterator that yields each chunk as it is received.
        """
        return StreamProcessor(stream_obj, update_history, self.messages, chunk_callback=chunk_callback)

    def get_response(self, prompt: str | list = None,
                     response_type: str = None,
                     raw_output: bool = False, system_role: str = None, **kwargs) -> str | dict | tuple | list | Iterator:
        """
        Retrieves a response from the OpenAI API based on the specified parameters. This is the main
        method to interact with different OpenAI API functionalities like text, image, and vision responses.

        It routes the request to the appropriate internal method based on the response type and handles
        the construction of prompts, messages, and API call parameters.

        Parameters:
            response_type (str): The type of response to generate; options include 'text', 'image', or 'vision'.
                                 Determines the OpenAI API endpoint to use. Defaults to 'text'.
            prompt (str | list): The user prompt or a list of prompts representing user/assistant dialogue.
            raw_output (bool): Determines whether to return the raw API response or a processed result.
                               Defaults to True for raw JSON output.
            system_role:

        Keyword Arguments:
            Keyword arguments specific to each response type are forwarded to the respective internal methods.
            For 'text' and 'vision': these may include 'stream', 'format_style', 'text_prompt', etc.
            For 'image': these may include 'image_model', 'image_count', 'image_size', 'image_quality', etc.

        Returns:
            Depending on the response type and 'raw_output' flag:
            - A string, dictionary, tuple, list, or an iterator containing the response data.

        Raises:
            The method may raise various exceptions based on internal validation and API call outcomes.

        Usage Example:
            get_response(response_type='text', prompt='Tell me a joke.', raw_output=False)
        """

        # Override system_role if provided
        if system_role:
            # Update the system message in self.messages
            if self.messages and self.messages[0]['role'] == 'system':
                self.messages[0]['content'] = system_role
            else:
                # If self.messages is empty or doesn't start with a system message, insert it
                self.messages.insert(0, {"role": "system", "content": system_role})

        # Automatically determine response type if not provided
        if not response_type:
            #! DO NOT use prompt_adjusted until this method is optimized
            response_type, prompt_adjusted = self._infer_response_type(prompt)

        logging.info(f"Getting response: type={response_type}")

        # validate and set the instance variable for prompt
        self._validate_and_assign_params(prompt)

        # Validate the response type
        if response_type not in ChatEngine.VALID_RESPONSE_TYPES:
            raise ValueError(
                f"Invalid response type: '{response_type}'. "
                f"Valid options are: {ChatEngine.VALID_RESPONSE_TYPES}"
            )

        # Filter kwargs based on response type
        if response_type == 'text':
            allowed_kwargs = self.ALLOWED_TEXT_KWARGS
        elif response_type == 'image':
            allowed_kwargs = self.ALLOWED_IMAGE_KWARGS  # Define similarly
        elif response_type == 'vision':
            allowed_kwargs = self.ALLOWED_VISION_KWARGS  # Define similarly
        else:
            allowed_kwargs = set()

        # Extract allowed kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        # Call the appropriate API based on response_type
        if response_type == 'text':
            self.api_used = 'text'
            self._text_api_call(text_prompt=prompt, **filtered_kwargs)
        elif response_type == 'image':
            self.api_used = 'image'
            self._image_api_call(text_prompt=prompt, **filtered_kwargs)
        elif response_type == 'vision':
            self.api_used = 'vision'
            self._vision_api_call(image_prompt=prompt, **filtered_kwargs)

        # If streaming is enabled, return the streaming object
        if self.stream and response_type in ['text', 'vision']:
            return self.response

        # Non-streaming response handling
        if not raw_output:
            assistant_reply = self.extract_response(response_type, **kwargs)
            if response_type != 'image':
                self.messages.append({"role": "assistant", "content": assistant_reply})
            elif response_type == 'image':
                if kwargs.get('revised_prompt', True):
                    self.messages.append({"role": "assistant", "content": f'here is a description of the image i created for you: {assistant_reply[0][1]}'})
                else:
                    self.messages.append({"role": "assistant", "content": '[GENERATED IMAGE]'})

            return assistant_reply
        else:
            # raw_output is True
            assistant_reply = self.extract_response(response_type, **kwargs)
            self.messages.append({"role": "assistant", "content": assistant_reply})
            return self.response
