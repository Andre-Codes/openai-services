import os
import threading
import time
from pathlib import Path

import openai


class AssistantEngine:
    def __init__(self, api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-1106-preview"):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.threads = {}  # Store threads created
        self.assistants = {}
        self.assistants_id_to_name = {}
        self.threads = {}
        self.runs = {}

    def create_assistant(self, name, instructions, model=None, tools=None):
        tools = tools or []
        model = model or self.model
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model
        )
        self.assistants[name] = assistant
        self.assistants_id_to_name[assistant.id] = name
        return assistant

    def create_thread(self):
        thread = self.client.beta.threads.create()
        self.threads[thread.id] = thread
        return thread

    def create_message(self, thread_id, content):
        if thread_id in self.threads:
            return self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=content
            )
        else:
            raise ValueError("Thread ID not found.")

    def get_messages(self, thread_id, order='asc', **kwargs):
        order = kwargs.get('order', order)
        return self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order=order
        )

    def process_message(self, message, print_content=True, **kwargs):
        print(f"{'#' * 40}\n{'#' * 40}")
        print(f"Message ID: {message.id}, "
              f"Role: {message.role}, "
              f"Created At: {message.created_at}")
        print('_' * 40)

        print_content = kwargs.get('print_content', print_content)
        content_text_values = ''
        file_name_id_dict = {}  # Dictionary storing filename:file_id pairs
        for content_idx, content in enumerate(message.content):
            if content.type == 'text':
                content_text_values += content.text.value
                if print_content:
                    print("Text:", content.text.value)

                for annot_idx, annotation in enumerate(content.text.annotations, start=1):
                    file_citation = getattr(annotation, 'file_citation', None)
                    file_path = getattr(annotation, 'file_path', None)

                    if file_citation:
                        cited_file_id = file_citation.file_id
                        file_name, _ = self.process_file_info(cited_file_id, annot_idx)
                        file_name_id_dict[file_name] = cited_file_id
                        print(f'[{annot_idx}] {file_citation.quote} from {file_name}')

                    elif file_path:
                        file_id = file_path.file_id
                        file_name, _ = self.process_file_info(file_id, annot_idx)
                        if file_name not in file_name_id_dict:
                            file_name_id_dict[file_name] = file_id

            elif content.type == 'image_file':
                file_id = content.image_file.file_id
                file_name, _ = self.process_file_info(file_id, content_idx)
                if file_name not in file_name_id_dict:
                    file_name_id_dict[file_name] = file_id

            return {'files': file_name_id_dict, 'text': content_text_values}

    def process_file_info(self, file_id, identifier):
        """Helper function to parse file info."""
        file_info = self.client.files.retrieve(file_id)
        full_filename = Path(file_info.filename).name
        file_stem = Path(file_info.filename).stem
        file_ext = Path(file_info.filename).suffix
        file_name_edit = f"{file_stem}_{identifier}{file_ext}"
        return full_filename, file_name_edit

    def process_thread_messages(self, thread_id, message_id=None,
                                index=None, role=None, **kwargs):

        if thread_id not in self.threads:
            raise ValueError("Thread ID not found.")

        message_objects = self.get_messages(thread_id, **kwargs).data
        response = None
        # Filter messages for specified role
        if role:
            if role not in ['assistant', 'user']:
                raise ValueError(f"Invalid role: '{role}'.")
            else:
                message_objects = [m for m in message_objects if m.role == role]

        if message_id:
            message = next((m for m in message_objects if m.id == message_id), None)
            if message:
                response = self.process_message(message, **kwargs)
            else:
                print("Message ID not found.")
        elif index is not None:
            if 0 <= abs(index) < len(message_objects):
                response = self.process_message(message_objects[index], **kwargs)
            else:
                print("Index out of range.")
        else:
            response = []
            for message in message_objects:
                message_content = self.process_message(message, **kwargs)
                response.append(message_content)

        return response

    def create_run(self, thread_id, assistant_id):
        if thread_id in self.threads and assistant_id in self.assistants_id_to_name:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id, assistant_id=assistant_id
            )
            self.runs[run.id] = run
            return run
        else:
            raise ValueError("Invalid thread or assistant ID.")

    def check_run_status(self, run_object, thread_id=None, run_id=None, continuous=False):

        if run_object is not None:
            thread_id = run_object.thread_id
            run_id = run_object.id

        def continuous_check():
            while True:
                thread_run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run_id
                )
                run_status = thread_run.status
                print(f"Run Status: {run_status}")
                if run_status == 'completed':
                    break
                elif run_status != 'in_progress':
                    print(f"Run Status: {run_status}")
                    break
                time.sleep(5)

        if continuous:
            status_thread = threading.Thread(target=continuous_check)
            status_thread.daemon = True  # Allows main program to exit even if thread is still running
            status_thread.start()
        else:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run_id
            )
            return run.status

    def cancel_run(self, run_object, thread_id=None, run_id=None):
        """
        Cancels an ongoing run for a given thread.

        Parameters:
            run_object:
            thread_id (str): ID of the thread where the run is ongoing.
            run_id (str): ID of the run to be cancelled.

        Returns:
            A confirmation message or an error message if cancellation fails.
        """
        if run_object is not None:
            thread_id = run_object.thread_id
            run_id = run_object.id

        try:
            self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            return f"Run {run_id} in thread {thread_id} has been cancelled."
        except Exception as e:
            return f"Error cancelling run: {e}"

    def get_assistants(self):
        return self.client.beta.assistants.list(
            order="desc",
            limit=100,
        )

    def delete_assistant(self, name):
        if name in self.assistants:
            self.delete_assistants(self.client, self.assistants[name].id)
            del self.assistants[name]

    def delete_assistants(self, specific_id=None):
        assistants = self.get_assistants()
        for assistant in assistants.data:
            if specific_id is None or assistant.id == specific_id:
                self.client.beta.assistants.delete(assistant.id)
                print(f"Deleted Assistant: {assistant.id}")
                if specific_id is not None:
                    break  # Stop after deleting the specific assistant

    def download_file(self, file_path, file_id):
        file_content = self.client.files.content(file_id).read()
        with open(file_path, "wb") as file:
            file.write(file_content)

    def download_files(self, message_content, file_names=None):
        """
        Downloads files based on a dictionary of filenames and file IDs.
        If file_names is None, all files will be downloaded.

        Parameters:
            message_content (dict): A dictionary with structure {file_name: file_id}
            file_names (list, optional): List of specific file names to download. Default is None.
        """

        if isinstance(message_content, list):
            for message in message_content:
                files_dict = message['files']
                if file_names is None:
                    # Download all files in the dictionary
                    for name, _id in files_dict.items():
                        self.download_file(name, _id)
                else:
                    file_names = [file_names] if not isinstance(file_names, list) else file_names
                    for name in file_names:
                        if name in files_dict:
                            self.download_file(name, files_dict[name])
                        else:
                            print(f"File '{name}' not found in the provided message content.")
        else:
            files_dict = message_content['files']
            for filename, file_id in files_dict.items():
                self.download_file(filename, file_id)

