# agents/tasks/headline_task.py
# just example task config for headline generation
from agents.task_config import TaskConfig

SYSTEM_PROMPT = """You write ad headlines..."""

def build_batch_prompt(rows):
    # ... your prompt logic
    pass

def build_single_prompt(row, variation=0):
    pass

def build_fallback(row):
    pass

HEADLINE_TASK = TaskConfig(
    name="headline",
    output_columns=["headline", "subheadline"],  # ← Any column names
    primary_column="headline",
    min_length=5,
    system_prompt=SYSTEM_PROMPT,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
    deduplicate=False,  # Headlines can repeat
)