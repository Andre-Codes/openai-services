import openai
from openai import OpenAI
import yaml
from typing import Union, Iterator


class ChatEngine:
    """
    A class for interacting with GPT models via the OpenAI API.
    
    Attributes:
        api_key (str): The OpenAI API key, sourced from environment variables.
        model (str): The GPT model name to be used. Defaults to "gpt-3.5-turbo".
        role_context (str): Operational context for the GPT model, e.g., 'general', 'api_explain'.
        temperature (float): Controls randomness in output. Lower is more deterministic.
        
    Class Variables:
        DISPLAY_MAPPING (dict): Mappings for IPython display function names.
        MD_TABLE_STYLE (str): Default style for Markdown tables.
        
    Methods:
        __init__(): Initializes class attributes.
        set_md_table_style(): Sets Markdown table style.
        get_format_styles(): Prints available response formats.
        get_role_contexts(): Prints available role contexts.
        _validate_and_assign_params(): Validates and assigns prompt and format_style.
        _handle_format_instructions(): Constructs the complete prompt for OpenAI API call.
        _text_api_call(): Makes the API call and stores the response.
        _handle_output(): Handles saving and displaying the response.
        get_response(): Main function to get a response from the GPT model.
        _handle_role_instructions(): Constructs role-specific instructions for the prompt.
        show(): Displays the generated content.
    """

    # Class variables
    # DISPLAY_MAPPING = { # mappings for IPython.display function names
    #     'html': HTML,
    #     'markdown': Markdown
    # }

    MD_TABLE_STYLE = "pipes"  # default format for markdown tables

    MODEL_OPTIONS = {
        'image': {
            'dall-e-3': {
                'qualities': ['standard', 'hd'],
                'sizes': ['1024x1024', '1024x1792', '1792x1024'],
                'max_count': 1
            },
            'dall-e-2': {
                'qualities': ['standard'],
                'sizes': ['1024x1024'],
                'max_count': 10
            }
        }
    }

    def __init__(
            self,
            role_context=None,
            system_role=None,
            temperature=1,
            model="gpt-3.5-turbo",
            stream=False,
            api_key=None,  # os.environ['OPENAI_API_KEY']
            config_path=None):
        """
        Initializes the GPTService class with settings to control the prompt and response.

        # Parameters
        ----------
            role_context (str, optional): Operational context for GPT. This directly controls
            what information is sent to the GPT model in addition to the user's prompt.
            Use the `get_role_contexts()` method to view the available roles. Defaults to 'general'.

            comment_level (str, optional): Level of comment verbosity. Defaults to 'normal'.

            explain_level (str, optional): Level of explanation verbosity. Defaults to 'concise'.

            temperature (float, optional): Controls randomness in output. Defaults to 0.

            model (str, optional): The GPT model name to use. Defaults to "gpt-3.5-turbo".
        """

        if config_path:
            with open(config_path, "r") as f:
                self.CONFIG = yaml.safe_load(f)
        else:
            self.CONFIG = {}

        # Set system role
        self.system_role = system_role or "You're a helpful assistant who answers questions."

        # Set up API access
        self.api_key = api_key

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

    def _handle_format_instructions(self, format_style):
        """
        Construct the final prompt by combining the role/context specific instructions
        (built from `_handle_role_instructions`) with formatting instructions.

        This method first acquires role/context-specific instructions from the `_handle_role_instructions` method.

        Returns:
            str: The fully constructed prompt, ready to for the API call.

        """

        # get the adjusted prompt reconstructed with any custom instructions
        adjusted_prompt = self._handle_role_instructions(self.prompt)

        if self.role_context != 'general':
            response_formats = self.CONFIG.get('response_formats', {})
            style_name = response_formats.get(format_style, {})
            response_instruct = style_name.get('instruct', '')

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
        elif format_style:
            response_instruct = f"Respond in the following {format_style} format \n\n "
            self.complete_prompt = f"{response_instruct}{adjusted_prompt}"
        # Finally, if no format style specified save only the adjusted prompt
        else:
            self.complete_prompt = adjusted_prompt

    def _text_api_call(self, text_prompt, **kwargs):
        if 'streaming' in kwargs:
            self.stream = kwargs['streaming']
        if 'format_style' in kwargs:
            format_style = kwargs['format_style']
        else:
            format_style = None

        self._handle_format_instructions(format_style)
        self._build_messages(prompt=text_prompt, **kwargs)
        print(self.__messages)
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=self.__messages,
                temperature=self.temperature,
                top_p=0.2,
                stream=self.stream
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
        if 'text_prompt' in kwargs:
            text_prompt = kwargs['text_prompt']
        else:
            text_prompt = "Describe this image."

        self._build_messages(
            prompt=text_prompt,
            response_type='vision',
            image_prompts=image_prompt
        )

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

    def _image_api_call(self, text_prompt, **kwargs):
        # get the adjusted prompt reconstructed with any custom instructions
        text_prompt = self._handle_role_instructions(text_prompt)

        # Extracting parameters with defaults
        model = kwargs.get('image_model', 'dall-e-3')
        count = kwargs.get('image_count', 1)
        size = kwargs.get('image_size', "1024x1024")
        quality = kwargs.get('image_q', "standard")
        revised_prompt = kwargs.get('revised_prompt', False)

        # Adjusting model for count
        if count > 1 and model != 'dall-e-2':
            model = 'dall-e-2'

        # Validate and adjust size and quality
        model_opts = ChatEngine.MODEL_OPTIONS['image'][model]
        if size not in model_opts['sizes']:
            size = model_opts['sizes'][0]  # Default to first available size
        if quality not in model_opts['qualities']:
            quality = model_opts['qualities'][0]  # Default to first available quality

        # Validate and adjust count
        if count > model_opts['max_count']:
            count = model_opts['max_count']

        if not revised_prompt and model != 'dall-e-2':
            preface = """
            I NEED to test how the tool works with extremely simple prompts. 
            DO NOT add any detail, just use it AS-IS:
            
            """
        else:
            preface = ""
        try:
            client = OpenAI()
            response = client.images.generate(
                model=model,
                prompt=preface + text_prompt,
                n=count,
                size=size,
                quality=quality
            )
            print(f"model: {model}, preface: '{preface}', count: {count}, size: {size}, quality: {quality}")
            self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _build_messages(self, prompt, response_type='text', **kwargs):
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

    def _handle_role_instructions(self, user_prompt):
        """
        Construct the context for the prompt by adding role/context specific instructions.

        If no instructions are found in config file, only the `system_role` value
        will be supplied, as this is a necessary arg for the API call.

        Returns:
            str: The inputted prompt with role/context instructions..
        """

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
        prompt_with_context = f"{role_instructions}{user_prompt}{documentation}"
        # raise AssertionError("prompt with context:" + str(prompt_with_context))
        self.prompt = prompt_with_context

        return prompt_with_context

    def get_response(self, response_type: str = 'text',
                     prompt: str | list = None,
                     raw_output: bool = True, **kwargs) -> Union[str, dict, tuple, Iterator]:
        """
        Retrieve a response from the OpenAI API in the
         desired format based on the provided parameters.

        Parameters:

        - response_type (str): The type of response to generate; either 'text' or 'image'.
                               This determines whether to use the `ChatCompletion`
                               or the `Image` API.
                               Default is 'text'.
        - prompt (str): The input text to serve as a basis for the OpenAI API response.
        - format_style (str): The format to use for the response, such as 'markdown'.
                              Default is 'markdown'. This will be ignored if the 'Image'
                              API is chosen since `_handle_format_instructions` will not be called.
        - raw_output (bool): Whether to return the raw JSON response or the extracted content
                             found within: `self.response['choices'][0]['message']['content']`
                             Default is True for raw JSON output.
        - **kwargs: Additional keyword arguments for more advanced configurations.
                    e.g. `_build_messages` can accept `system_role`

        Returns:

        - If raw_output is False and response_type is not 'image', returns the 'content' field
          from the 'choices' list in the API response.
        - Otherwise, returns the raw JSON response.
            In the case of the 'image' api an image URL is returned.

        Raises:
        - May raise exceptions based on internal validation methods and API calls.

        Usage Example:

            get_response(response_type='text',
            system_role="You're a comedian",
            prompt='Tell me a joke.',
            raw_output=False)
        """

        # validate and set the instance variable for prompt
        self._validate_and_assign_params(prompt)
        openai.api_key = self.api_key

        if response_type == 'text':
            self._text_api_call(text_prompt=prompt, **kwargs)
        elif response_type == 'image':
            self._image_api_call(text_prompt=prompt, **kwargs)
        elif response_type == 'vision':
            self._vision_api_call(image_prompt=prompt, **kwargs)
        # Return finished response from OpenAI
        if not raw_output and not self.stream:  # if raw and stream are False
            if response_type == 'text':
                return self.response.choices[0].message.content
            elif response_type == 'image':
                # Extract URLs into a list of 1 or more
                image_urls = [image.url for image in self.response.data]
                # Extract revised prompts (only for dall-e-3) into list
                # will be list of 'Nones' for older models
                revised_prompts = [image.revised_prompt for image in self.response.data]
                # If revised_prompt param is used return the urls and revisions
                # revised_prompt = kwargs.get('revised_prompt', False)
                return image_urls, revised_prompts
            elif response_type == 'vision':
                return self.response.choices[0].message.content

        return self.response
