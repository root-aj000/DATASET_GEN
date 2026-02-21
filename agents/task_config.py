# agents/task_config.py
"""
TaskConfig — defines a generation task with dynamic columns.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any


@dataclass
class TaskConfig:
    """
    Configuration for a single generation task.

    Change column names here — the engine adapts automatically.
    Add 1, 2, 3, or more output columns — all dynamic.
    """

    # ── Identity ──
    name: str  # e.g., "text_gen", "img_desc", "cta"

    # ── Output columns (DYNAMIC) ──
    output_columns: List[str]
    # e.g., ["text", "keywords"]
    # e.g., ["img_desc"]
    # e.g., ["monetary_mention", "call_to_action"]

    # ── Primary column (for validation) ──
    primary_column: str = ""  # Auto-set to first output_column if empty

    # ── Minimum length for "filled" check ──
    min_length: int = 10

    # ── Prompts ──
    system_prompt: str = ""
    system_prompt_single: str = ""  # Optional override for Phase 4

    # ── Prompt builders (MUST be set) ──
    # Signature: fn(rows: List[dict]) -> str
    batch_prompt_builder: Optional[Callable[[List[Dict]], str]] = None

    # Signature: fn(row: dict, variation: int) -> str
    single_prompt_builder: Optional[Callable[[Dict, int], str]] = None

    # ── Fallback builder (Phase 5) ──
    # Signature: fn(row: dict) -> dict {col_name: value}
    fallback_builder: Optional[Callable[[Dict], Dict[str, str]]] = None

    # ── Deduplication ──
    deduplicate: bool = True  # Track used values to avoid duplicates

    # ── API overrides ──
    temperature: Optional[float] = None  # Override config temperature
    max_retries: Optional[int] = None
    retry_delay: Optional[float] = None

    def __post_init__(self):
        # Set primary column to first output column if not specified
        if not self.primary_column and self.output_columns:
            self.primary_column = self.output_columns[0]

        # Validate
        if not self.output_columns:
            raise ValueError(f"Task '{self.name}' must have at least one output_column")

        if self.primary_column not in self.output_columns:
            raise ValueError(
                f"Task '{self.name}': primary_column '{self.primary_column}' "
                f"not in output_columns {self.output_columns}"
            )

    def get_system_prompt_single(self) -> str:
        """Get system prompt for single generation (Phase 4)."""
        return self.system_prompt_single or self.system_prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TaskRegistry:
    """
    Registry of all available tasks.
    Add new tasks here — no changes needed elsewhere.
    """

    _tasks: Dict[str, TaskConfig] = {}

    @classmethod
    def register(cls, task: TaskConfig):
        """Register a task."""
        cls._tasks[task.name] = task

    @classmethod
    def get(cls, name: str) -> Optional[TaskConfig]:
        """Get task by name."""
        return cls._tasks.get(name)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered task names."""
        return list(cls._tasks.keys())

    @classmethod
    def all(cls) -> Dict[str, TaskConfig]:
        """Get all tasks."""
        return cls._tasks.copy()