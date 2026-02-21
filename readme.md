project/
├── config.py                    # (keep as-is, minor cleanup)
├── utils.py                     # (keep as-is)
├── main.py                      # Simplified CLI
│
├── core/
│   ├── __init__.py
│   ├── client.py                # Single API client (extracted)
│   ├── parser.py                # JSON parser (5 strategies, parameterized)
│   ├── skeleton.py              # (moved from root)
│   └── validator.py             # (moved from root)
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Generic 5-phase engine
│   ├── task_config.py           # TaskConfig dataclass
│   └── tasks/
│       ├── __init__.py
│       ├── text_task.py         # text + keywords
│       ├── img_desc_task.py     # img_desc
│       ├── cta_task.py          # monetary_mention + call_to_action
│       └── audience_task.py     # target_audience
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py          # Unified DatasetPipeline
│   ├── progress.py              # ProgressManager (save/load/resume)
│   └── display.py               # Rich progress display
│
└── generated_datasets/



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





# Build skeleton only
python main.py skeleton

# Run single task
python main.py run text
python main.py run img_desc
python main.py run cta
python main.py run audience

# Run task on existing CSV
python main.py run audience dataset.csv

# Full pipeline (all tasks in sequence)
python main.py generate

# Full pipeline, ignore saved progress
python main.py generate --fresh

# Check progress
python main.py status

# Clean up progress files
python main.py clean

# Validate a dataset
python main.py validate generated_datasets/dataset.csv

# List available tasks
python main.py tasks