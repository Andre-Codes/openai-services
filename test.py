import os

import gpt_utils as gpt

gpt = gpt.ChatEngine(api_key=os.environ["OPENAI_API_KEY"])

# system role
role = """
You reply in reverse.
"""

# For passing a list of assistant|user messages
prompt_1 = "What is your favorite color"
reply_1 = "blue like i"
prompt_2 = "what is a volcano?"

image_prompt = """
https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Mandel_zoom_00_mandelbrot_set.jpg/322px-Mandel_zoom_00_mandelbrot_set.jpg
"""

response = gpt.get_response(
    response_type='vision',
    prompt=image_prompt,
    text_prompt='what is this famous shape.',
    raw_output=False,
    system_role=role
)

if not gpt.stream:
    print(response)
else:
    for i in response:
        print(i.choices[0].delta.content)
