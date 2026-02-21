# pipeline/progress.py
"""
Progress manager for save/load/resume functionality.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List

from config import OUTPUT_DIR, TOTAL_ROWS, TEXT_BATCH_SIZE, MODEL_NAME, ACTIVE_PROFILE, BASE_UNIT


class ProgressManager:
    """Manages progress save/load/resume for any task."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.progress_file = os.path.join(OUTPUT_DIR, f"_progress_{task_name}.json")
        self.progress_csv = os.path.join(OUTPUT_DIR, f"_progress_{task_name}.csv")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save(
        self,
        df: pd.DataFrame,
        completed_batches: int,
        total_batches: int,
        primary_column: str,
        min_length: int = 3,
    ):
        """Save current progress."""
        # Count filled rows
        filled = sum(
            1 for _, r in df.iterrows()
            if r.get(primary_column) and len(str(r.get(primary_column, ""))) >= min_length
        )

        progress = {
            "task_name": self.task_name,
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "completed_rows": filled,
            "total_rows": len(df),
            "batch_size": TEXT_BATCH_SIZE,
            "model": MODEL_NAME,
            "profile": ACTIVE_PROFILE,
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "primary_column": primary_column,
        }

        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        df.to_csv(self.progress_csv, index=False)

    def load(self) -> Optional[Dict]:
        """Load saved progress if exists and valid."""
        if not os.path.exists(self.progress_file):
            return None

        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)

            # Validate
            if progress.get("task_name") != self.task_name:
                print(f"  ⚠ Progress is for different task: {progress.get('task_name')}")
                return None

            if progress.get("completed_batches", 0) >= progress.get("total_batches", 0):
                print(f"  ⚠ Progress shows task complete. Starting fresh...")
                return None

            if not os.path.exists(self.progress_csv):
                print(f"  ⚠ Progress JSON exists but CSV missing.")
                return None

            return progress

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ Corrupted progress file: {e}")
            return None

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        """Load progress DataFrame."""
        if os.path.exists(self.progress_csv):
            return pd.read_csv(self.progress_csv)
        return None

    def cleanup(self):
        """Remove progress files after successful completion."""
        for f in [self.progress_file, self.progress_csv]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  ✓ Cleaned up: {os.path.basename(f)}")

    def display_progress(self, progress: Dict):
        """Display saved progress info."""
        completed = progress.get("completed_batches", 0)
        total = progress.get("total_batches", 0)
        rows_done = progress.get("completed_rows", 0)
        total_rows = progress.get("total_rows", 0)

        print(f"\n  ╔{'═'*50}╗")
        print(f"  ║  SAVED PROGRESS: {self.task_name.upper():32}║")
        print(f"  ╠{'═'*50}╣")
        print(f"  ║  Batches    : {completed}/{total}{' '*(36-len(f'{completed}/{total}'))}║")
        print(f"  ║  Rows       : {rows_done}/{total_rows}{' '*(36-len(f'{rows_done}/{total_rows}'))}║")
        print(f"  ║  Model      : {progress.get('model', '?')[:36]:<36}║")
        print(f"  ║  Last saved : {progress.get('last_saved', '?')[:36]:<36}║")
        print(f"  ╚{'═'*50}╝")