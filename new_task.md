How to Add a NEW Task (Example: headline)
Create agents/tasks/headline_task.py:

Python

# agents/tasks/headline_task.py
from agents.task_config import TaskConfig

SYSTEM_PROMPT = """You write ad headlines..."""

def build_batch_prompt(rows):
    # ... your prompt logic
    pass

def build_single_prompt(row, variation=0):
    pass

def build_fallback(row):
    return {
        "headline": f"Best {row['object_detected']}",
        "subheadline": f"Shop {row['theme']} now",
    }

HEADLINE_TASK = TaskConfig(
    name="headline",
    output_columns=["headline", "subheadline"],  # ← Any column names
    primary_column="headline",
    min_length=5,
    system_prompt=SYSTEM_PROMPT,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
)
Then in agents/tasks/__init__.py:

Python

from .headline_task import HEADLINE_TASK
TaskRegistry.register(HEADLINE_TASK)
Done. Now you can run:

Bash

python main.py run headline dataset.csv