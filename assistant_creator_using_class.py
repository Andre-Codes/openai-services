from AssistantEngine import AssistantEngine

assistant_engine = AssistantEngine()

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
assistant = assistant_engine.create_assistant(
    name="DataAnalysis",
    instructions="You're an assistant who helps with creating data visualizations.",
    tools=[{"type": "code_interpreter"}]
)

# 2. Create a Thread
thread = assistant_engine.create_thread()

# 3. Add Messages to the Thread
prompt = "I need a csv file of a public dataset. Then create two data visualization" \
         "image files for the data."
message = assistant_engine.create_message(prompt)

# 4. Run the Assistant on the Thread to trigger responses
run = assistant_engine.create_run()  # thread.id, assistant.id
# Check the run status (https://platform.openai.com/docs/assistants/how-it-works/run-lifecycle)
assistant_engine.check_run_status(run_object=run, continuous=True)
# If Run is stuck or taking too long:
# assistant_engine.cancel_run(run_object=run)

###################################
###################################

###################################
# PROCESS MESSAGES AND FILES
###################################

# Process the thread messages
messages_response = assistant_engine.process_thread_messages(
    thread.id,
    print_content=True,
    order='asc'
)
print(messages_response)

# List all the files from each message if exist
# for message in messages_response:
#     if message['files']:
#         print(list(message['files'].keys()))

# Extract code from the text value
# pattern = r"```python\n(.*?)```"
# code_snippet = re.findall(pattern, messages_response['text'], re.DOTALL)

# assumes the assistant returned file information in its response
assistant_engine.download_files(
    messages_response  # None to download all files
)

###################################
###################################

# Delete all assistants created in this instance
assistant_engine.delete_assistants()
