from __future__ import annotations
from typing import Any, Dict, List, Union
from .errors import NormalizationError

Message = Dict[str, Any]
ContentPart = Dict[str, Any]
InputType = Union[str, List[Any], Dict[str, Any], List[Dict[str, Any]]]

def _to_input_text(text: str) -> ContentPart:
    return {"type": "input_text", "text": text}

def _to_input_image(obj: Any) -> ContentPart:
    # URL string
    if isinstance(obj, str) and (obj.startswith("http://") or obj.startswith("https://")):
        return {"type": "input_image", "image_url": obj}  # ← string, not {"url": ...}

    # Dict passthroughs
    if isinstance(obj, dict):
        # allow direct string: {"image_url": "https://..."}
        if isinstance(obj.get("image_url"), str):
            return {"type": "input_image", "image_url": obj["image_url"]}
        # base64 form: {"image_data": {"b64": "...", "mime_type": "image/png"}}
        if "image_data" in obj:
            return {"type": "input_image", "image_data": obj["image_data"]}

    raise NormalizationError(f"Unrecognized image input: {obj!r}")


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def _coalesce(parts: List[ContentPart]) -> List[ContentPart]:
    # Already normalized for Responses API content (text & images mix)
    return parts

def _merge_adjacent_user(messages: List[Message]) -> List[Message]:
    out: List[Message] = []
    for m in messages:
        if out and m.get("role") == "user" and out[-1].get("role") == "user":
            out[-1]["content"].extend(m["content"])
        else:
            out.append(m)
    return out

def format_messages(input_obj: InputType) -> List[Message]:
    """
    Convert flexible input into Responses-ready messages[].
    Rules:
      - str -> single user turn
      - list[str] -> each entry becomes a user turn
      - list[mixed with images] -> one user turn with mixed parts
      - list[{role, content}] -> validated passthrough
      - dict with keys {system, developer, user, images, history, tool_results} -> normalized
    """
    if isinstance(input_obj, str):
        return [{"role": "user", "content": [_to_input_text(input_obj)]}]

    if isinstance(input_obj, list):
        # If list of dicts with roles -> passthrough (validate shapes)
        if input_obj and isinstance(input_obj[0], dict) and "role" in input_obj[0]:
            for m in input_obj:
                if m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                    raise NormalizationError(f"Unknown role in message: {m}")
                if not isinstance(m.get("content"), list):
                    raise NormalizationError("message.content must be a list of content parts.")
            return _merge_adjacent_user(input_obj)

        # Mixed text+image -> fold into one user turn
        parts: List[ContentPart] = []
        for item in input_obj:
            if isinstance(item, str) and (item.startswith("http://") or item.startswith("https://")):
                print(f"Detected image URL: {item}")
                parts.append(_to_input_image(item))
            elif isinstance(item, str):
                parts.append(_to_input_text(item))
            elif isinstance(item, dict) and ("image_url" in item or "image_data" in item):
                parts.append(_to_input_image(item))
            else:
                raise NormalizationError(f"Unsupported list item: {item!r}")
        return [{"role": "user", "content": _coalesce(parts)}]

    if isinstance(input_obj, dict):
        messages: List[Message] = []
        if sys := input_obj.get("system"):
            messages.append({"role": "system", "content": [_to_input_text(sys)]})
        if dev := input_obj.get("developer"):
            messages.append({"role": "developer", "content": [_to_input_text(dev)]})
        if hist := input_obj.get("history"):
            if not isinstance(hist, list):
                raise NormalizationError("history must be a list of message dicts.")
            messages.extend(hist)

        # User turn (text/images)
        parts: List[ContentPart] = []
        if usr := input_obj.get("user"):
            parts.append(_to_input_text(usr))
        for img in _ensure_list(input_obj.get("images", [])):
            parts.append(_to_input_image(img))
        if parts:
            messages.append({"role": "user", "content": _coalesce(parts)})

        # Tool results (must be message(s) with role='tool')
        for tr in _ensure_list(input_obj.get("tool_results", [])):
            if not (isinstance(tr, dict) and tr.get("role") == "tool"):
                raise NormalizationError("tool_results entries must be messages with role='tool'.")
            if not isinstance(tr.get("content"), list):
                raise NormalizationError("tool message content must be a list.")
            messages.append(tr)

        # Final validation & merge
        return _merge_adjacent_user(messages)

    raise NormalizationError(f"Unsupported input type: {type(input_obj)!r}")
