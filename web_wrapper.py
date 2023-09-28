import streamlit as st
import os
import random
from gpt_service_helpers import display_response, generate_response
import gpt_service as gpt
import traceback


# Load instructions from JSON file
path_web = "/mount/src/code-tutor/web_app/config.yaml" # streamlit server path
path_local = "config.yaml"
config_path = path_web if os.path.exists(path_web) else path_local

# Set value for API Key
api_key = os.environ['OPENAI_API_KEY']

# initalize the GPT class
app = gpt.CodeTutor(config_path=config_path, api_key=api_key, stream=True)

# get main app title information
app_title = (
    app.CONFIG['app_ui'].get('title', 'App Title')
)

title_emoji = (
    app.CONFIG['app_ui'].get('title_emoji', 'question')
)

page_title = (
    app.CONFIG['app_ui'].get('page_title', 'Streamlit App')
)

# set page configuration
st.set_page_config(page_title=page_title, page_icon=title_emoji)

def extra_lesson(user_prompt, role_context, response):
    with st.spinner('Next lesson...'):
        # get second instruction set for continuing previous converstaion
        instruct_2 = app.CONFIG['role_contexts'][role_context].get('instruct_2', 'Provide additional details.')
        prompt2 = instruct_2
        messages = [user_prompt, response, prompt2]
        return messages

# BEGIN WIDGETS
# Side bar controls
# Open API Key
app.api_key = st.sidebar.text_input(
    label = "OpenAI API Key :key:", 
    type = "password",
    help = "Get your API key from https://openai.com/"
) or app.api_key

# Advanced settings expander
adv_settings = st.sidebar.expander(
    label = "Advanced Settings :gear:", 
    expanded = False
)

# Add Open API key and Advanced Settings widgets to the expander
with adv_settings:
    app.model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], help="Account must be authorized for gpt-4")
    app.temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.2, 0.1
    )
    app.temperature = round(app.temperature * 10) / 10


#### Sidebar with dropdown of friendly role names ###

# Get all enabled roles and create dictionary with display names
json_roles = gpt.CodeTutor.get_role_contexts(app)

roles = {
    settings.get('display_name', 'Unknown Role'): role 
    for role, settings in app.CONFIG['role_contexts'].items() 
    if settings.get('enable', False)
}

selected_friendly_role = st.sidebar.selectbox(
    'Lesson Context :memo:', 
    roles.keys()
)

# get the role context name from json
selected_json_role = roles[selected_friendly_role]
# set the class variable to json name
app.role_context = selected_json_role
# get the button phrase based on selected role
button_phrase = (
    app.CONFIG['role_contexts'][selected_json_role].get('display_name', 'Enter')
)

# get other app title information
subheader = (
    app.CONFIG['app_ui'].get('subheader', 'How can I help you?')
) 

# configure app title information
st.title(f":{title_emoji}: {app_title}")
st.subheader(subheader)
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
    app.CONFIG['role_contexts'][selected_json_role].get('prompt_placeholder', 'Enter your prompt...')
)

user_prompt = prompt_box.text_area(
    label="How can I help ?",
    label_visibility = "hidden",
    height=185,
    placeholder=prompt_placeholder, 
    key='prompt'
) or None

if answer_button:
    try:
        allow_download = not extra_lesson_toggle
        all_response_content = []
        
        if app.model == 'gpt-4':
            st.toast('Be patient. Responses from GPT-4 can be slower ...', icon="⏳")
            
        if user_prompt is None:
            if app.CONFIG['allow_null_prompt']:
                user_prompt = 'Tell me anything.'
                app.role_context = 'random'
                extra_lesson_toggle = False
            else:
                st.info('Please provide a prompt...', icon='😑')

        response = generate_response(app, user_prompt)
        
        displayed_response = display_response(
            response, 
            assistant=allow_download, 
            all_response_content=all_response_content,
            selected_friendly_role=selected_friendly_role,
            streaming=app.stream
        )
        
        if extra_lesson_toggle:
            prompt_messages = extra_lesson(user_prompt, app.role_context, displayed_response)
            extra_response = generate_response(app, prompt_messages)
            display_response(
                response, 
                assistant=True, 
                all_response_content=all_response_content,
                selected_friendly_role=selected_friendly_role,
                streaming=app.stream
            )
        
        st.toast(':teacher: Lesson Complete!', icon='✅')
        
    except Exception as e:
        traceback.print_exc()
        st.error(f"""There was an error while the response was being generated.
                 Possible issues: \n
                 -Incorrect or missing API key\n-No internet connection  \n\n 
                 {e}
                 """, icon='🚨')