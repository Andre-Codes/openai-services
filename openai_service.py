import openai
import yaml


class CodeTutor:
    """
    A class for interacting with GPT models via the OpenAI API.
    
    Attributes:
        api_key (str): The OpenAI API key, sourced from environment variables.
        model (str): The GPT model name to be used. Defaults to "gpt-3.5-turbo".
        role_context (str): Operational context for the GPT model, e.g., 'basic', 'api_explain'.
        comment_level (str): Level of comment verbosity.
        explain_level (str): Level of explanation verbosity.
        temperature (float): Controls randomness in output. Lower is more deterministic.
        
    Class Variables:
        DISPLAY_MAPPING (dict): Mappings for IPython.display function names.
        MD_TABLE_STYLE (str): Default style for Markdown tables.
        
    Methods:
        __init__(): Initializes class attributes.
        set_md_table_style(): Sets Markdown table style.
        get_format_styles(): Prints available response formats.
        get_role_contexts(): Prints available role contexts.
        _validate_and_assign_params(): Validates and assigns prompt and format_style.
        _build_prompt(): Constructs the complete prompt for OpenAI API call.
        _make_openai_call(): Makes the API call and stores the response.
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
    
    MD_TABLE_STYLE = "pipes" # default format for markdown tables
    
    def __init__(
        self, 
        role_context=None,
        comment_level=None,
        explain_level=None,
        temperature=None,
        model="gpt-3.5-turbo",
        stream=False,
        api_key=None, # os.environ['OPENAI_API_KEY']
        config_path=None):
        """
        Initializes the GPTService class with settings to control the prompt and response.

        # Parameters
        ----------
            role_context (str, optional): Operational context for GPT. This directly control \
                what is sent to the GPT model in addition to the user inputted prompt. \
                    Use the `get_role_contexts()` method to view the available roles. \
                        Defaults to 'basic'.
            comment_level (str, optional): Level of comment verbosity. Defaults to 'normal'.
            explain_level (str, optional): Level of explanation verbosity. Defaults to 'concise'.
            temperature (float, optional): Controls randomness in output. Defaults to 0.
            model (str, optional): The GPT model name to use. Defaults to "gpt-3.5-turbo".
        """
        
        with open(config_path, "r") as f:
            self.CONFIG = yaml.safe_load(f)
        
        # Set up API access
        self.api_key = api_key
        
        # Turn off/on streaming of response
        self.stream = stream
                
        # Set the GPT model
        self.model = model
        
        # Validate and set role_context
        available_role_contexts = self.CONFIG.get('role_contexts', {}).keys()
        self.role_context = role_context if role_context in available_role_contexts else 'basic'
        
        # Validate and set comment_level
        comment_levels = self.CONFIG['comment_levels']
        self.comment_level = comment_level if comment_level in comment_levels \
            or comment_level is None else 'normal'
        
        # Validate and set explain_level
        explain_levels = self.CONFIG['explain_levels']
        self.explain_level = explain_level if explain_level in explain_levels \
            or explain_level is None else 'concise'
        
        # Validate and set temperature
        self.temperature = temperature

    
    def set_md_table_style(self, style):
        available_table_styles = self.CONFIG['response_formats']['markdown']['table_styles'].keys()
        if style not in available_table_styles:
            raise ValueError(f"Invalid MD_TABLE_STYLE. Available styles: {list(self.CONFIG['table_formatting'].keys())}.")
        self.MD_TABLE_STYLE = self.CONFIG['response_formats']['markdown']['table_styles'][style]
        
    def get_format_styles(self):
        available_formats = list(self.CONFIG['response_formats'].keys())
        print("Available response formats:", available_formats)
         
    def get_role_contexts(self):
        available_role_contexts = list(self.CONFIG['role_contexts'].keys())
        return available_role_contexts

    def _validate_and_assign_params(self, prompt, format_style):
        if prompt is None:
            raise ValueError("Prompt can't be None.")
        self.prompt = prompt
        self.format_style = format_style.lower()

    def _build_prompt(self):
        self.system_role, user_content = self._handle_role_instructions(self.prompt)

        response_instruct = self.CONFIG['response_formats'][self.format_style]['instruct']
        if self.format_style == 'markdown':
            response_instruct += self.CONFIG['response_formats']['markdown']['table_styles'][self.MD_TABLE_STYLE]
        elif self.format_style == 'html':
            response_instruct += self.CONFIG['response_formats']['html']['css']

        self.complete_prompt = f"{response_instruct}; {user_content}"

    def _make_openai_call(self):
        try:
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model = self.model,
                messages = self.__messages,
                temperature = self.temperature,
                top_p = 0.2,
                stream = self.stream
            )
        except Exception as e:
            return "Connection to API failed - Verify internet connection or API key"
        if response:
            self.response = response
    
    def _build_messages(self, prompt):
        # Validate that all items in 'prompt' are strings
        if not all(isinstance(item, str) for item in prompt):
            raise ValueError("All elements in the list should be strings")
        
        # Initialize system message
        system_msg = [{"role": "system", "content": self.system_role}]
        
        # Determine user and assistant messages based on the length of the 'prompt'
        if len(prompt) > 1:
            user_assistant_msgs = [
                {
                    "role": "assistant" if i % 2 == 0 else "user", 
                    "content": prompt[i]
                }
                for i in range(len(prompt))
            ]
        else:
            user_assistant_msgs = [{"role": "user", "content": self.complete_prompt}]
        
        # Combine system, user, and assistant messages
        self.__messages = system_msg + user_assistant_msgs
    
    def _handle_role_instructions(self, user_prompt):
        if self.role_context != 'basic':
            comment_level = f"Provide {self.comment_level}" if self.comment_level is not None else "Do not add any"
            explain_level = f"Provide {self.explain_level}" if self.explain_level is not None else "Do not give any"
            default_documentation = (
                f"{comment_level} comments and {explain_level} explanation of your response."
            )

            documentation = (
                self.CONFIG.get('role_contexts', {})
                            .get(self.role_context, {})
                            .get('documentation', default_documentation)
            )

            instructions = (
                f"{self.CONFIG['role_contexts'][self.role_context]['instruct']}"
            )
            user_content = f"{instructions}; Request: {user_prompt}; {documentation}"

            system_role = self.CONFIG['role_contexts'][self.role_context]['system_role']
        else:
            system_role = "You're a helpful assistant who answers my questions."
            user_content = user_prompt

        return system_role, user_content

    def get_response(self, prompt=None, format_style='markdown'):
        # _build_messages requires prompt to be a list
        # convert prompt to a list if it is not already
        prompt = [prompt] if not isinstance(prompt, list) else prompt
        self._validate_and_assign_params(prompt, format_style)
        self._build_prompt()
        self._build_messages(prompt)
        self._make_openai_call()
        # Return finished response from OpenAI
        return self.response