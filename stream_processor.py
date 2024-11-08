from typing import Iterator
import time

from stream_callbacks import default_chunk_callback


class StreamProcessor:
    def __init__(self, stream_obj: Iterator, update_history: bool, messages: list, chunk_callback=None):
        """
        Initializes the StreamProcessor.

        Parameters:
            stream_obj (Iterator): The streaming object returned by get_response.
            update_history (bool): Whether to update messages with the assistant's reply.
            messages (list): Reference to the ChatEngine messages list for history updates.
            chunk_callback (callable, optional): A function to call with each chunk of the reply.
                                                 Defaults to a function that prints the chunk.
        """
        self.stream_obj = stream_obj
        self.update_history = update_history
        self.messages = messages
        self.assistant_reply = ''  # Accumulated assistant reply content
        self.chunk_callback = chunk_callback if chunk_callback is not None else default_chunk_callback

    def __iter__(self):
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self.stream_obj)
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                self.assistant_reply += delta_content  # Accumulate the reply content
                # Call the chunk_callback
                self.chunk_callback(delta_content)
                return delta_content  # Return each chunk content
            else:
                return ''
        except StopIteration:
            # When streaming ends, update the conversation history
            if self.update_history:
                self.messages.append({"role": "assistant", "content": self.assistant_reply})
            raise StopIteration

    def get_full_reply(self) -> str:
        """Returns the full accumulated reply of the assistant."""
        return self.assistant_reply

    def run(self):
        """
        Processes the entire stream internally.
        """
        for _ in self:
            pass  # The chunk_callback handles the processing


