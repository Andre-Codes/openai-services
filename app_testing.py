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
file = st.file_uploader("Upload Image", accept_multiple_files=True)

files = [
    r"C:\Users\havok\OneDrive\Pictures\AI Art\aragorn_arwen_impres_fantasy.png",
    r"C:\Users\havok\OneDrive\Pictures\AI Art\artsy_fibonacci.png"
]
urls = [
    "https://images.ctfassets.net/cnu0m8re1exe/4luBczkxnxMMVO4gTiz0vD/6f290be6f5340264ddc1fe0977134a90/Untitled_design_-_2023-06-20T145027.117.png?fm=jpg&fl=progressive&w=660&h=433&fit=fill",
    "https://i.natgeofe.com/n/5d00b0cc-ab95-4522-ad13-7c65b7589e6b/NationalGeographic_748483.jpg?w=636&h=424"
]

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
        if isinstance(reply, list):
            for img in reply:
                st.image(img[0])
        else:
            st.image(reply[0][0])
            st.write(reply[0][1])

st.divider()

st.write(st.session_state['chat'].messages)
