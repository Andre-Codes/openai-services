import time


def default_chunk_callback(chunk):
    """Prints each chunk to the console."""
    print(chunk, end='', flush=True)


def log_chunk_to_file(chunk, filename="stream_log.txt"):
    """Logs each chunk to a specified file."""
    with open(filename, "a") as f:
        f.write(chunk + "\n")


def buffered_display(chunk, update_frequency=5):
    """
    Accumulates chunks in a buffer and prints them together every `update_frequency` chunks.

    Parameters:
        chunk (str): The current chunk of content.
        update_frequency (int): Number of chunks to accumulate before printing.
    """
    # Initialize the buffer attribute on the first call
    if not hasattr(buffered_display, "buffer"):
        buffered_display.buffer = []  # Create a buffer attribute on the function itself

    buffered_display.buffer.append(chunk)
    if len(buffered_display.buffer) >= update_frequency:
        print("".join(buffered_display.buffer))
        buffered_display.buffer.clear()  # Clear the buffer after printing


def typing_effect_display(chunk, delay=0.05):
    """
    Displays each chunk with a typing effect.

    Parameters:
        chunk (str): The current chunk of content.
        delay (float): Delay in seconds between each character.
    """
    for char in chunk:
        print(char, end='', flush=True)
        time.sleep(delay)


def count_keywords(chunk, keywords: dict):
    """
    Counts the occurrence of specified keywords in each chunk and prints the running total.

    Parameters:
        chunk (str): The current chunk of content.
        keywords (dict): Dictionary of keywords and their counts.
    """
    for keyword in keywords.keys():
        keywords[keyword] += chunk.lower().count(keyword.lower())
    print("Current keyword counts:", keywords)
