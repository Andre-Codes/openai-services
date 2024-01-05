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
files = [r"C:\Users\havok\Downloads\thoelogy_pdfs\Ice, Thomas - Has Bible Prophecy Already Been Fulfilled.pdf",
r"C:\Users\havok\Downloads\thoelogy_pdfs\Ice, Thomas - Evaluation of Theonomic Neopostmillennialism.pdf"]

assistant = assistant_engine.create_assistant(
    name="Theology",
    instructions="You're scholar of theology. You answer questions related to"
                 "scholarly work.",
    tools=["retrieval"],
    files=files
)

# 2. Create a Thread
thread = assistant_engine.create_thread()

# 3. Add Messages to the Thread
prompt = "There's some dispute on how to interpret prophecy, especially prophecy that" \
         "seemingly has not come to pass. According to these documents, how should" \
         "prophetic events be interpreted and should we interpret events surrounding Jesus's" \
         "return literally, e.g. the period of 'tribulation'? Pull from each document. Provide citations."
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
    order='asc',
    role='assistant'
)
print("#"*50)
print(messages_response)
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

###################################
###################################

# Delete all assistants created in this instance
assistant_engine.delete_assistants()

# Delete all uploaded files
for file in assistant_engine.client.files.list().data:
    assistant_engine.client.files.delete(file.id)