import gpt_service as gpt
import streamlit as st
import random


# get main app title information
app_title = (
    gpt.INSTRUCTIONS['app_ui'].get('app_title', 'AI Assistant')
)
title_emoji = (
    gpt.INSTRUCTIONS['app_ui'].get('title_emoji', '')
)
# set page configuration
st.set_page_config(page_title=app_title, page_icon=title_emoji)

# initalize the class with role context
ct = gpt.CodeTutor()

#@st.cache
def generate_response(prompt):
    with st.spinner('...creating a lesson :thought_balloon:'):
        return ct.get_response(
            prompt = prompt,
            format_style = 'markdown'
        )

def display_response(response, assistant):
    # st.text(ct.response)
    # st.markdown(ct.complete_prompt)

    st.divider()
    
    # Create a placeholder for the markdown
    markdown_placeholder = st.empty()
    
    collected_chunks = []
    collected_responses = []
    # iterate through the stream of events
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        if chunk['choices'][0]['finish_reason'] != 'stop':
            content_chunk = chunk['choices'][0]['delta']['content']  # extract the response
            if content_chunk:
                collected_responses.append(content_chunk)  # save the response
                formatted_response = ''.join(collected_responses)
                markdown_placeholder.markdown(f"{formatted_response}\n\n") #display the formatted chunk on the webpage
    
    response_file = handle_file_output(formatted_response)
    
    if assistant:
        create_download(response_file)

def handle_file_output(responses):
    all_response_content.append(f"{responses} \n\n")
    combined_responses = ''.join(all_response_content)
    return combined_responses

def create_download(response):
    # with col1:
    st.download_button(
        label=":green[Download] :floppy_disk:",
        data=response,
        file_name=f'{selected_friendly_role}.md',
        mime='text/markdown'
    )  

def extra_lesson(user_prompt, role_context):
    with st.spinner('Next lesson...'):
        # get second instruction set for continuing previous converstaion
        instruct_2 = gpt.INSTRUCTIONS['role_contexts'][role_context].get('instruct_2', 'Provide additional details.')
        prompt2 = instruct_2
        messages = [user_prompt, formatted_response, prompt2]
        return messages

# BEGIN WIDGETS
# Side bar controls
# Open API Key
ct.api_key = st.sidebar.text_input(
    label = "Open API Key :key:", 
    type = "password",
    help = "Get your API key from https://openai.com/"
) or ct.api_key

# Advanced settings expander
adv_settings = st.sidebar.expander(
    label = "Advanced Settings :gear:", 
    expanded = False
)

# Add Open API key and Advanced Settings widgets to the expander
with adv_settings:
    ct.model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], help="Account must be authorized for gpt-4")
    ct.temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.2, 0.1
    )
    ct.temperature = round(ct.temperature * 10) / 10


# Sidebar with dropdown for mapping json names to friendly names
json_roles = gpt.CodeTutor.get_role_contexts()
roles = {gpt.INSTRUCTIONS['role_contexts'][role]['display_name']: role for role in json_roles}

selected_friendly_role = st.sidebar.selectbox(
    'Lesson Context :memo:', 
    roles.keys()
)

# get the role context name from json
selected_json_role = roles[selected_friendly_role]
# set the class variable to json name
ct.role_context = selected_json_role
# get the button phrase based on selected role
button_phrase = (
    gpt.INSTRUCTIONS['role_contexts'][selected_json_role].get('display_name', 'Enter')
)

# get other app title information
subheader = (
    gpt.INSTRUCTIONS['app_ui'].get('subheader', 'How can I help you?')
) 

# configure app title information
st.title(app_title)
st.subheader(f":{title_emoji}: {subheader}")
prompt_box = st.empty()

# Create two columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    answer_button = st.button(
        f":blue[{button_phrase}] :sparkles:", 
        help="Generate an answer"
    )
with col2:
    extra_lesson_toggle = st.toggle(
        "Extra Details", 
        help="Provide additional, detailed information. Toggle this _before_ getting an answer.",
        key='extra_lesson',
        value=False
    )
    
prompt_placeholder = (
    gpt.INSTRUCTIONS['role_contexts'][selected_json_role].get('prompt_placeholder', 'Enter your prompt...')
)

user_prompt = prompt_box.text_area(
    label="How can I help ?",
    label_visibility = "hidden",
    height=185,
    placeholder=prompt_placeholder, 
    key='prompt'
) or None

 
if answer_button:
    # control whether the download button gets created
    # based on whether or not a second response will be generated
    activate_assistant = False if extra_lesson_toggle else True

    formatted_response = ''
    all_response_content = []
    
    # get the response from openai
    try:
        # set initial actions based on user selected settings
        if ct.model == 'gpt-4':
            st.toast('Be patient. Responses from GPT-4 can be slower ...', icon="⏳")
        if user_prompt is None:
            st.info("Not sure what to ask? Creating a random lesson!", icon="🎲")
            user_prompt = random.choice(gpt.INSTRUCTIONS['python_modules'])
            ct.role_context = 'random'
            extra_lesson_toggle = True
        response = generate_response(user_prompt)
        display_response(response, assistant=activate_assistant) 
        
        if extra_lesson_toggle:
            prompt_messages = extra_lesson(user_prompt, ct.role_context)
            extra_response = generate_response(prompt_messages)
            display_response(extra_response, assistant=True)
        
        st.toast(':teacher: Lesson Complete!', icon='✅')
        
    except Exception:
        st.error("Connection to API failed \n\nVerify internet connection or API key", icon='🚨')
