# agents/tasks/__init__.py
"""
Task definitions.
Import and register all tasks here.
"""

from agents.task_config import TaskRegistry

# Import all task configs
from .text_task import TEXT_TASK
from .img_desc_task import IMG_DESC_TASK
from .cta_task import CTA_TASK
from .audience_task import AUDIENCE_TASK

# Register all tasks
TaskRegistry.register(TEXT_TASK)
TaskRegistry.register(IMG_DESC_TASK)
TaskRegistry.register(CTA_TASK)
TaskRegistry.register(AUDIENCE_TASK)

# Export for convenience
__all__ = [
    "TEXT_TASK",
    "IMG_DESC_TASK",
    "CTA_TASK",
    "AUDIENCE_TASK",
    "TaskRegistry",
]

from .headline_task import HEADLINE_TASK
TaskRegistry.register(HEADLINE_TASK)