from openai import OpenAI
from pathlib import Path


def create_assistant(client, name, instructions, model, tools):
    return client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model
    )


def create_thread(client):
    return client.beta.threads.create()


def retrieve_thread(client, thread_id):
    return client.beta.threads.retrieve(thread_id=thread_id)


def create_message(client, thread_id, content, role="user"):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content
    )


def list_messages(client, thread_id):
    return client.beta.threads.messages.list(
        thread_id=thread_id
    )


def retrieve_message(client, thread_id):
    message_object = list_messages(client, thread_id)
    return client.beta.threads.messages.retrieve(
        thread_id=thread_id,
        message_id=message_object.data[0].id
    )


def infer_file_extension(file_path):
    # Basic extension inference based on common file format indicators
    if '.csv' in file_path.lower():
        return 'csv'
    elif '.xlsx' in file_path.lower() or '.xls' in file_path.lower():
        return 'xlsx'
    elif '.txt' in file_path.lower():
        return 'txt'
    # Add more conditions as needed
    else:
        return 'unknown'


def process_message_content(client, message):
    for content in message.content:
        if content.type == 'text':
            print("Text:", content.text.value)
            for annotation in content.text.annotations:
                if annotation.type == 'file_path':
                    file_id = annotation.file_path.file_id
                    file_extension = infer_file_extension(annotation.text)
                    download_file(client, file_id, f"./file_{file_id}.{file_extension}")

        elif content.type == 'image_file':
            file_id = content.image_file.file_id
            download_file(client, file_id, f"./image_{file_id}.png")


def process_message(client, message, file_name_id_dict, print_text=True):
    print(f"Message ID: {message.id}, "
          f"Role: {message.role}, "
          f"Created At: {message.created_at}")
    print('#'*40)

    for content_idx, content in enumerate(message.content):
        if content.type == 'text':
            if print_text:
                print("Text:", content.text.value)
            for annot_idx, annotation in enumerate(content.text.annotations, start=1):
                file_citation = getattr(annotation, 'file_citation', None)
                file_path = getattr(annotation, 'file_path', None)

                if file_citation:
                    cited_file = client.files.retrieve(file_citation.file_id)
                    file_name_id_dict[f'{annot_idx}_{cited_file.filename}'] = file_citation.file_id
                    print(f'[{annot_idx}] {file_citation.quote} from {cited_file.filename}')

                elif file_path:
                    file_id = file_path.file_id
                    file_info = client.files.retrieve(file_id)
                    file_name = f"{Path(file_info.filename).stem}" \
                                f"_{annot_idx}{Path(file_info.filename).suffix}"
                    file_name_id_dict[file_name] = file_id

        elif content.type == 'image_file':
            file_id = content.image_file.file_id
            file_info = client.files.retrieve(file_id)
            file_name = f"{Path(file_info.filename).stem}" \
                        f"_{content_idx}{Path(file_info.filename).suffix}"
            file_name_id_dict[file_name] = file_id


def process_thread_messages(client, thread_id, message_id=None,
                            index=None, role=None, **kwargs):

    print_text = kwargs.get('print_text', False)
    messages = list_messages(client, thread_id).data
    if role:
        if role not in ['assistant', 'user']:
            raise ValueError(f"Invalid role: '{role}'.")
        else:
            messages = [message for message in messages if message.role == role]

    file_name_id_dict = {}  # Dictionary to store filename:file_id pairs
    if message_id:
        message = next((msg for msg in messages if msg.id == message_id), None)
        if message:
            process_message(client, message, file_name_id_dict, print_text=print_text)
        else:
            print("Message ID not found.")
    elif index is not None:
        if 0 <= index < len(messages):
            process_message(client, messages[index], file_name_id_dict, print_text=print_text)
        else:
            print("Index out of range.")
    else:
        for message in messages:
            process_message(client, message, file_name_id_dict, print_text=print_text)

    return file_name_id_dict


def create_run(client, thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )


def check_run_status(client, thread_id, run_id):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status


def download_file(client, file_id, file_path):
    file_content = client.files.content(file_id).read()
    with open(file_path, "wb") as file:
        file.write(file_content)


def get_assistants(client):
    return client.beta.assistants.list(
        order="desc",
        limit=100,
    )


def delete_assistants(client, specific_id=None):
    assistants = get_assistants(client)
    for assistant in assistants.data:
        if specific_id is None or assistant.id == specific_id:
            client.beta.assistants.delete(assistant.id)
            print(f"Deleted Assistant: {assistant.id}")
            if specific_id is not None:
                break  # Stop after deleting the specific assistant