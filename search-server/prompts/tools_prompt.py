import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_prompt_file_path = os.path.join(_current_dir, "tools_prompt.md")

with open(_prompt_file_path, "r", encoding="utf-8") as _f:
    TOOL_CALLING_SYSTEM_PROMPT = _f.read()
