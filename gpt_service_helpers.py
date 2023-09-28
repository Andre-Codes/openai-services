import streamlit as st
import gpt_service as gpt


def generate_response(ct, prompt):
    with st.spinner('...creating a lesson :thought_balloon:'):
        return ct.get_response(prompt=prompt, format_style='markdown')

def handle_file_output(responses, all_response_content):
    all_response_content.append(f"{responses} \n\n")
    file_data = ''.join(all_response_content)
    return file_data

def create_download(response, role_name):
    st.download_button(
        label=":green[Download] :floppy_disk:",
        data=response,
        file_name=f'{role_name}.md',
        mime='text/markdown'
    )  

def display_response(response, assistant, all_response_content, selected_friendly_role, streaming):
    st.divider()
    markdown_placeholder = st.empty()
    collected_responses = []

    if streaming:
        for chunk in response:
            if chunk['choices'][0]['finish_reason'] != 'stop':
                content_chunk = chunk['choices'][0]['delta']['content']
                if content_chunk:
                    collected_responses.append(content_chunk)
                    response_content = ''.join(collected_responses)
                    markdown_placeholder.markdown(f"{response_content}\n\n")
        
        file_data = handle_file_output(response_content, all_response_content)
    else:
        response_content = response['choices'][0]['message']['content']
        markdown_placeholder.markdown(response_content)
        file_data = response_content
    
    if assistant:
        create_download(file_data, selected_friendly_role)
        
    return response_content
