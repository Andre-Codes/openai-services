import os
import threading
import time
import warnings
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
        self.processed_messages = []

    def create_assistant(self, name, instructions, files=None, **kwargs):

        if files is not None:
            if not isinstance(kwargs['tools'], list):
                raise TypeError("The 'files' parameter must be a list.")
            kwargs['file_ids'] = self.create_file(files, key='id')

        if 'tools' in kwargs:
            if not isinstance(kwargs['tools'], list):
                raise TypeError("The 'tools' parameter must be a list.")

            kwargs['tools'] = [{"type": tool} for tool in kwargs['tools']]
            if 'retrieval' in kwargs['tools'] and files is None:
                warnings.warn(
                    "The 'Retrieval' tool was found in the 'tools' parameter "
                    "without any 'files' provided.", UserWarning
                )

        # Ensure 'model' has a default value if not provided
        kwargs.setdefault('model', self.model)

        # Call the create method with name, instructions, and other parameters
        assistant = self.client.beta.assistants.create(name=name, instructions=instructions, **kwargs)
        self.assistants[name] = assistant
        self.assistants_id_to_name[assistant.id] = name
        return assistant

    def create_file(self, files, key=None):
        MAX_FILES = 20
        MAX_SIZE_MB = 512
        MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

        def process_single_file(path):
            # Check file size
            if os.path.getsize(path) > MAX_SIZE_BYTES:
                raise ValueError(f"File '{path}' exceeds the maximum size of {MAX_SIZE_MB} MB.")

            with open(path, "rb") as file:
                file_obj = self.client.files.create(
                    file=open(path, "rb"),
                    purpose='assistants'
                )
            return getattr(file_obj, key, None) if key else file_obj

        # Ensure filepath is a list
        if not isinstance(files, list):
            files = [files]

        if len(files) > MAX_FILES:
            raise ValueError(f"You can attach a maximum of {MAX_FILES} files per Assistant.")

        # Process files
        return [process_single_file(path) for path in files]

    def create_thread(self):
        thread = self.client.beta.threads.create()
        self.threads[thread.id] = thread
        return thread

    def create_message(self, content, thread_id=None):

        thread_id = list(self.threads)[0] if thread_id is None else thread_id
        if thread_id in self.threads:
            return self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=content
            )
        else:
            raise ValueError("Thread ID not found.")

    def get_messages(self, thread_id=None, order='asc', **kwargs):
        order = kwargs.get('order', order)
        thread_id = list(self.threads)[0] if thread_id is None else thread_id
        return self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order=order
        )

    def parse_message_object(self, message, print_content=True, **kwargs):
        print(f"{'#' * 40}\n{'#' * 40}")
        print(f"Message ID: {message.id}, "
              f"Role: {message.role}, "
              f"Created At: {message.created_at}")
        print('_' * 40)

        print_content = kwargs.get('print_content', print_content)
        content_text_values = ''
        file_name_id_dict = {}  # Dictionary storing filename:file_id pairs
        file_citations_list = []
        for content_idx, content in enumerate(message.content):
            if content.type == 'text':

                content_text_values += content.text.value
                last_citation_value = None
                footnote_index = 0
                for annot_idx, annotation in enumerate(content.text.annotations, start=1):
                    file_citation = getattr(annotation, 'file_citation', None)
                    file_path = getattr(annotation, 'file_path', None)

                    footnote_index += 1
                    current_citation_value = annotation.text
                    content_text_values = content_text_values.replace(current_citation_value, f' [{footnote_index}]')
                    if file_citation:
                        print(annotation.text)
                        cited_file_id = file_citation.file_id
                        file_name, _ = self.process_file_info(cited_file_id, annot_idx, 'file_citation')
                        file_name_id_dict[file_name] = cited_file_id
                        if current_citation_value != last_citation_value:
                            file_citations_list.append(
                                {'index': annot_idx, 'quote': file_citation.quote, 'source': file_name})
                        else:
                            footnote_index -= 1
                            print(f"current: {current_citation_value} == last: {last_citation_value}")
                        last_citation_value = current_citation_value

                    elif file_path:
                        file_id = file_path.file_id
                        file_name, _ = self.process_file_info(file_id, annot_idx, 'file_path')
                        if file_name not in file_name_id_dict:
                            file_name_id_dict[file_name] = file_id

                if print_content:
                    print(f"Text: {content_text_values}")
                    print(f"Citations: {file_citations_list}")

            elif content.type == 'image_file':
                file_id = content.image_file.file_id
                file_name, _ = self.process_file_info(file_id, content_idx, 'image_file')
                print(file_name)
                if file_name not in file_name_id_dict:
                    file_name_id_dict[file_name] = file_id

            extracted_content = {
                'text': content_text_values,
                'citations': file_citations_list,
                'files': file_name_id_dict
            }

            self.processed_messages.append(extracted_content)

            return extracted_content

    def process_file_info(self, file_id, identifier, file_type):
        """
        Helper function to pass the appropriate message object to `process_thread_messages()`
        """
        file_info = self.client.files.retrieve(file_id)
        full_filename = Path(file_info.filename).name
        file_stem = Path(file_info.filename).stem
        file_ext = Path(file_info.filename).suffix
        file_ext = '.png' if file_ext == '' and file_type == 'image_file' else file_ext
        file_name_edit = f"{file_stem}_{identifier}{file_ext}"
        return full_filename, file_name_edit

    def process_thread_messages(self, thread_id, message_id=None,
                                index=None, role=None, **kwargs) -> list | dict:
        """
        Processes messages within a specified thread, optionally filtering by message ID,
        index, role, or other criteria provided via `kwargs`.

        Args:
            thread_id (str): The ID of the thread whose messages are to be processed.
            message_id (str, optional): Specific message ID to process. If specified,
                only this message will be processed. Defaults to None.
            index (int, optional): Index of the message in the thread to process.
                If specified, only the message at this index is processed. Negative
                indices are supported, following Python's list indexing conventions.
                Defaults to None.
            role (str, optional): Role to filter messages by before processing. Only
                messages with this role will be processed. Valid roles include 'assistant'
                and 'user'. Defaults to None.
            **kwargs: Additional keyword arguments that will be passed to the message
                processing function.

        Returns:
            list | dict: If `index` is specified, returns a single dictionary representing
            the processed message; text, citations, & files. Otherwise, returns a list of dictionaries, each
            representing a processed message. The content of the returned
            objects depend on the response from the API & processing performed by `parse_message_object()`.

        Raises:
            ValueError: If `thread_id` is not found or if an invalid `role` is specified.
    """
        if thread_id not in self.threads:
            raise ValueError("Thread ID not found.")

        # Process all message in the active thread
        message_objects = self.get_messages(thread_id, **kwargs).data

        # Filter messages for specified role
        if role:
            if role not in ['assistant', 'user']:
                raise ValueError(f"Invalid role: '{role}'.")
            else:
                message_objects = [m for m in message_objects if m.role == role]

        response = None
        if message_id:
            message = next((m for m in message_objects if m.id == message_id), None)
            if message:
                response = self.parse_message_object(message, **kwargs)
            else:
                print("Message ID not found.")
        elif index is not None:
            if 0 <= abs(index) < len(message_objects):
                response = self.parse_message_object(message_objects[index], **kwargs)
            else:
                print("Index out of range.")
        else:
            response = []
            for message in message_objects:
                message_content = self.parse_message_object(message, **kwargs)
                response.append(message_content)

        return response

    def stream_text_response(self, response):
        for event in response:
            # Check if the event is a ThreadMessageDelta
            if event.event == 'thread.message.delta':
                for content_block in event.data.delta.content:
                    if content_block.type == 'text':
                        yield content_block.text.value

    def get_response(self, thread_id=None, assistant_id=None, stream=False):
        """
        Creates a 'run' and returns the run object or a stream of the message
        response if `stream=True`

        Args:
            thread_id:
            assistant_id:
            stream:

        Returns:

        """
        thread_id = (
            list(self.threads)[0] if thread_id is None else thread_id
        )
        assistant_id = (
            list(self.assistants_id_to_name)[0] if assistant_id is None else assistant_id
        )

        # TODO: store active threads and assistants from api into their appropriate
        #  class vars upon instantiation. For now, check for asst id is commented out,
        #  so an existing asst id can be passed.
        if thread_id in self.threads:  # and assistant_id in self.assistants_id_to_name
            response = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                stream=stream
            )

            # Create iterator for stream of response text
            if stream:
                return self.stream_text_response(response)
            else:
                return response

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
                print(
                    f"Deleted Assistant: {assistant.id} "
                    f"({assistant.name})"
                )
                # Remove from assistant dict
                if assistant.id in self.assistants_id_to_name:
                    del self.assistants_id_to_name[assistant.id]

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
