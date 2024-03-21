from AssistantEngine import AssistantEngine

assistant_engine = AssistantEngine()
assistant_engine.get_assistants()
###################################
# CREATE AND RUN AN ASSISTANT
###################################

# Follow steps 1-4 for the initial setup of an assistant and to
# send and receive a message
# After message is received from the assistant, only steps 3 and 4 need
# to be repeated to continue the conversation.
# At any time, the process_thread_messages() and download_files() can be
# used to view the thread of messages and download any generated files

# 1. Create an Assistant
files = [r"C:\GitHub\Streamlit Gen\documentation\Streamlit_App_Generation_Guide.md"
         #r"C:\GitHub\Streamlit Gen\streamlit_app_gen\app_gen.py",
         #r"C:\GitHub\Streamlit Gen\streamlit_app_gen\config\config_main_adv_refs.yaml"
         ]
assistant = assistant_engine.create_assistant(
    name="Streamlit_Design_Tools_Analysis",
    instructions="You're an expert in documentation and code for a Python module. You are"
                 "provided 3 files; First, a markdown file with the documentation. "
                 "Use the MD file for how-to, documentation related questions. "
                 "Second, the main .py file containing the code. Third, a YAML"
                 " file containing an example configuration that the .py file "
                 "processes to create a Streamlit app.",
    tools=["retrieval"],
    files=files
)

# 2. Create a Thread
thread = assistant_engine.create_thread()

# 3. Add Messages to the Thread
prompt = ("create a small .yaml file for me with a configuration example based "
          "using the documenation to guide how it's created.")
message = assistant_engine.create_message(prompt)

# 4. Run the Assistant on the Thread to trigger responses
run = assistant_engine.create_run()  # thread.id, assistant.id
# Check the run status (https://platform.openai.com/docs/assistants/how-it-works/run-lifecycle)
assistant_engine.check_run_status(run_object=run, continuous=True)
# If Run is stuck or taking too long:
# assistant_engine.cancel_run(run_object=run)

###################################
# PROCESS MESSAGES AND FILES
###################################

# Process the thread messages
messages_response = assistant_engine.process_thread_messages(
    thread.id,
    print_content=True,
    order='asc',
    role='assistant'
)
print("#" * 50)
print(messages_response[0]['text'])
for citation in messages_response[0]['citations']:
    print(citation['index'], citation['quote'], citation['source'], '\n\n')

assistant_engine.get_messages()

# List all the files from each message if exist
for message in messages_response:
    if message['files']:
        print(list(message['files']))

# Extract code from the text value
# pattern = r"```python\n(.*?)```"
# code_snippet = re.findall(pattern, messages_response['text'], re.DOTALL)

# assumes the assistant returned file information in its response
assistant_engine.download_files(
    messages_response  # None to download all files
)
assistant_engine.download_file('test_file.yaml', 'file-ERvp11Cmo0EBSQuQyclqz8a5')
###################################
###################################

# Delete all assistants created in this instance
assistant_engine.delete_assistants()

# Delete all uploaded files
for file in assistant_engine.client.files.list().data:
    assistant_engine.client.files.delete(file.id)
