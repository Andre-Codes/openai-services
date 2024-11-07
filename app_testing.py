import streamlit as st
import os
import gpt_utils as gpt
from streamlit.runtime.uploaded_file_manager import UploadedFile

if 'chat' not in st.session_state:
    chat = gpt.ChatEngine(
        api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-4o',
        enable_logging=True,
        stream=True
    )
    st.session_state['chat'] = chat

stream_container = []


def stream(chunk, container: list):
    container.append(chunk)  # Append each new chunk
    st.session_state.stream_placeholder.markdown("".join(container))  # Update display with accumulated content


text = st.text_area("Prompt")
file = st.file_uploader("Upload Image")

text_prompt = None
if st.button("Go"):
    if file and text:
        api = "vision"
        prompt = file
        text_prompt = text
    elif file:
        api = "vision"
        prompt = file
    else:
        api = None
        prompt = text
    st.session_state.stream_placeholder = st.empty()
    reply = st.session_state['chat'].get_response(
        prompt=prompt, response_type=api, text_prompt=text_prompt
    )
    if st.session_state['chat'].api_used != "image":
        streamer = st.session_state['chat'].process_stream(
            reply, chunk_callback=lambda x: stream(x, stream_container)
        )
        streamer.run()
    else:
        # revised_prompt defaults to True
        st.image(reply[0][0])
        st.write(reply[0][1])

st.divider()

st.write(st.session_state['chat'].messages)
