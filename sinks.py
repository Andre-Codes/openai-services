from __future__ import annotations
from typing import Callable, Optional

def terminal_sink() -> Callable[[str], None]:
    """Stream to terminal without adding extra newlines; flush per chunk."""
    import sys
    def _write(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    return _write

def jupyter_sink(display_id: Optional[str] = None) -> Callable[[str], None]:
    """Stream in Jupyter by incrementally updating a single output cell, rendering as Markdown."""
    from IPython.display import display, update_display, Markdown

    buf = []
    _id = [display_id]

    def _write(chunk: str) -> None:
        buf.append(chunk)
        text = "".join(buf)
        if _id[0] is None:
            handle = display(Markdown(text), display_id=True)
            _id[0] = handle.display_id
        else:
            update_display(Markdown(text), display_id=_id[0])

    return _write

def streamlit_sink(placeholder=None) -> Callable[[str], None]:
    """
    Stream in Streamlit. Pass an st.empty() placeholder or we create one.
    Usage:
        import streamlit as st
        sink = streamlit_sink(st.empty())
    """
    import streamlit as st
    ph = placeholder or st.empty()
    buf = []
    def _write(chunk: str) -> None:
        buf.append(chunk)
        ph.markdown("".join(buf))
    return _write

def buffer_sink(container: list) -> Callable[[str], None]:
    """Append deltas to a list you provide (for GUI frameworks like Shiny)."""
    def _write(chunk: str) -> None:
        container.append(chunk)
    return _write
