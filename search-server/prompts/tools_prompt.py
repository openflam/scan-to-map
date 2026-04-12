import os
import json

_current_dir = os.path.dirname(os.path.abspath(__file__))
_prompt_file_path = os.path.join(_current_dir, "tools_prompt.md")
_answers_file_path = os.path.join(_current_dir, "unique_answers.json")

with open(_prompt_file_path, "r", encoding="utf-8") as _f:
    _prompt_template = _f.read()

with open(_answers_file_path, "r", encoding="utf-8") as _f:
    _unique_answers = json.load(_f)

# Inject the allowed answers into the prompt template
TOOL_CALLING_SYSTEM_PROMPT = _prompt_template.replace("{{UNIQUE_ANSWERS}}", json.dumps(_unique_answers))
