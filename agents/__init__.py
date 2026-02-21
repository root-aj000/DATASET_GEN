# agents/__init__.py
from .base_agent import BaseAgent
from .task_config import TaskConfig, TaskRegistry

# Import tasks to register them
from . import tasks