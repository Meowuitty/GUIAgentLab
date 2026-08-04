"""MAI-UI prompt and benchmark-dataset helpers."""

# ruff: noqa: E501 -- prompt lines are kept verbatim for checkpoint compatibility.

from __future__ import annotations

from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- Available Apps: `["桌面","Contacts","Settings","设置","Clock","Maps","Chrome","Calendar","files","Gallery","淘店","Taodian","Mattermost","Mastodon","Mail","SMS","Camera"]`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.""".strip()


def normalize_goal(value: object) -> str:
    """Remove source-code line-end padding without changing task wording."""
    return "\n".join(line.rstrip() for line in str(value).splitlines()).strip()


def read_task_specs(path: str | Path) -> pd.DataFrame:
    """Read task metadata embedded in a GUIAgentLab parquet dataset."""
    frame = pd.read_parquet(path, columns=["extra_info"])
    rows = []
    for extra_info in frame["extra_info"]:
        if not isinstance(extra_info, dict):
            raise ValueError("dataset extra_info entries must be mappings")
        rows.append(
            {
                "task_name": str(extra_info.get("task_name", "")),
                "goal": str(extra_info.get("goal", "")),
                "requires_external_network": bool(
                    extra_info.get("requires_external_network", False)
                ),
            }
        )
    tasks = pd.DataFrame(rows)
    if (
        tasks.empty
        or tasks.task_name.duplicated().any()
        or (tasks.task_name == "").any()
    ):
        raise ValueError("task_name values must be nonempty and unique")
    return tasks
