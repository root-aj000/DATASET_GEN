# # pipeline/orchestrator.py
# """
# Unified DatasetPipeline — runs any combination of tasks.
# """

# import os
# import time
# import pandas as pd
# from datetime import datetime
# from typing import List, Dict, Optional

# from config import (
#     OUTPUT_DIR, TOTAL_ROWS, TEXT_BATCH_SIZE,
#     DELAY_BETWEEN_BATCHES, MODEL_NAME, ACTIVE_PROFILE,
# )
# from core.skeleton import build_skeleton, verify_skeleton, print_skeleton_summary
# from core.validator import validate_dataset, print_full_report
# from agents.base_agent import BaseAgent
# from agents.task_config import TaskConfig, TaskRegistry
# from pipeline.progress import ProgressManager


# class DatasetPipeline:
#     """
#     Unified pipeline that runs any task on a dataset.

#     Usage:
#         pipeline = DatasetPipeline()

#         # Run single task
#         df = pipeline.run_task("text", seed=42)

#         # Run multiple tasks in sequence
#         df = pipeline.run_tasks(["text", "img_desc", "cta"], seed=42)

#         # Run task on existing CSV
#         df = pipeline.run_task("audience", input_csv="dataset.csv")
#     """

#     def __init__(self, verbose: bool = True):
#         self.verbose = verbose
#         os.makedirs(OUTPUT_DIR, exist_ok=True)

#     def _log(self, msg: str):
#         if self.verbose:
#             print(msg)

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  MAIN ENTRY POINTS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def run_task(
#         self,
#         task_name: str,
#         input_csv: Optional[str] = None,
#         seed: int = 42,
#         force_restart: bool = False,
#     ) -> pd.DataFrame:
#         """
#         Run a single task.

#         Args:
#             task_name: Name of task (must be registered)
#             input_csv: Optional input CSV (otherwise builds skeleton)
#             seed: Random seed for skeleton
#             force_restart: Ignore saved progress

#         Returns:
#             DataFrame with task columns filled
#         """
#         task = TaskRegistry.get(task_name)
#         if not task:
#             available = TaskRegistry.list_tasks()
#             raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

#         self._banner(task)

#         # Check for resume
#         progress_mgr = ProgressManager(task_name)
#         progress = None if force_restart else progress_mgr.load()

#         if progress:
#             return self._resume_task(task, progress_mgr, progress)
#         else:
#             return self._fresh_task(task, progress_mgr, input_csv, seed)

#     def run_tasks(
#         self,
#         task_names: List[str],
#         seed: int = 42,
#         force_restart: bool = False,
#     ) -> pd.DataFrame:
#         """
#         Run multiple tasks in sequence.

#         Args:
#             task_names: List of task names to run
#             seed: Random seed
#             force_restart: Ignore all saved progress

#         Returns:
#             Final DataFrame with all columns filled
#         """
#         df = None

#         for task_name in task_names:
#             if df is None:
#                 # First task — build skeleton
#                 df = self.run_task(task_name, seed=seed, force_restart=force_restart)
#             else:
#                 # Subsequent tasks — use previous output
#                 temp_csv = os.path.join(OUTPUT_DIR, f"_temp_{task_name}.csv")
#                 df.to_csv(temp_csv, index=False)
#                 df = self.run_task(task_name, input_csv=temp_csv, force_restart=force_restart)
#                 os.remove(temp_csv)

#         return df

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  FRESH / RESUME
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _fresh_task(
#         self,
#         task: TaskConfig,
#         progress_mgr: ProgressManager,
#         input_csv: Optional[str],
#         seed: int,
#     ) -> pd.DataFrame:
#         """Run task from scratch."""

#         # Step 1: Get or build skeleton
#         if input_csv:
#             self._log(f"\n  Loading: {input_csv}")
#             df = pd.read_csv(input_csv)
#         else:
#             self._log(f"\n  Building skeleton ({TOTAL_ROWS} rows)")
#             df = build_skeleton(seed=seed)
#             errs = verify_skeleton(df)
#             if errs:
#                 for e in errs:
#                     self._log(f"    ✗ {e}")
#                 return pd.DataFrame()
#             self._log(f"  ✓ Skeleton verified")

#         # Step 2: Run task
#         t0 = time.time()
#         filled_df = self._run_batched(task, df, progress_mgr, start_batch=0)
#         elapsed = time.time() - t0

#         self._log(f"\n  ✓ Task '{task.name}' complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")

#         # Step 3: Save
#         filepath = self._save_final(filled_df, task.name)
#         progress_mgr.cleanup()

#         return filled_df

#     def _resume_task(
#         self,
#         task: TaskConfig,
#         progress_mgr: ProgressManager,
#         progress: Dict,
#     ) -> pd.DataFrame:
#         """Resume task from saved progress."""

#         progress_mgr.display_progress(progress)

#         # Ask user
#         print(f"\n  [R] Resume from batch {progress['completed_batches'] + 1}")
#         print(f"  [F] Fresh start")
#         try:
#             choice = input("\n  Choice [R/F]: ").strip().upper()
#         except (EOFError, KeyboardInterrupt):
#             choice = "R"

#         if choice == "F":
#             self._log("  Starting fresh...")
#             progress_mgr.cleanup()
#             return self._fresh_task(task, progress_mgr, None, seed=42)

#         # Load progress
#         df = progress_mgr.load_dataframe()
#         if df is None:
#             self._log("  ✗ Could not load progress CSV")
#             return pd.DataFrame()

#         # Resume
#         start_batch = progress["completed_batches"]
#         self._log(f"\n  Resuming from batch {start_batch + 1}")

#         t0 = time.time()
#         filled_df = self._run_batched(task, df, progress_mgr, start_batch=start_batch)
#         elapsed = time.time() - t0

#         self._log(f"\n  ✓ Task '{task.name}' complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")

#         # Save
#         filepath = self._save_final(filled_df, task.name)
#         progress_mgr.cleanup()

#         return filled_df

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  BATCHED EXECUTION
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _run_batched(
#         self,
#         task: TaskConfig,
#         df: pd.DataFrame,
#         progress_mgr: ProgressManager,
#         start_batch: int = 0,
#     ) -> pd.DataFrame:
#         """Run task in batches with progress saving."""

#         all_rows = df.to_dict("records")
#         total = len(all_rows)
#         n_batches = -(-total // TEXT_BATCH_SIZE)

#         # Create agent
#         agent = BaseAgent(task, verbose=self.verbose)

#         # Load existing values for deduplication (resume case)
#         if start_batch > 0 and task.deduplicate:
#             existing = df[task.primary_column].fillna("").astype(str).tolist()
#             agent.load_existing_values(existing)

#         filled_rows = all_rows.copy()

#         for b in range(start_batch, n_batches):
#             batch_start = b * TEXT_BATCH_SIZE
#             batch_end = min(batch_start + TEXT_BATCH_SIZE, total)
#             batch_rows = all_rows[batch_start:batch_end]

#             # Check if batch already done
#             batch_done = all(
#                 self._is_filled(r, task.primary_column, task.min_length)
#                 for r in batch_rows
#             )
#             if batch_done:
#                 continue

#             self._log(f"\n  ━━━ Batch {b+1}/{n_batches} (rows {batch_start+1}-{batch_end}) ━━━")

#             t_batch = time.time()
#             result = agent.generate(batch_rows, batch_label=f"batch{b+1:03d}")
#             elapsed_batch = time.time() - t_batch

#             # Update filled_rows
#             for i, row in enumerate(result):
#                 filled_rows[batch_start + i] = row

#             # Progress
#             filled_count = sum(
#                 1 for r in filled_rows
#                 if self._is_filled(r, task.primary_column, task.min_length)
#             )
#             self._log(f"    ✓ Batch {b+1}: {elapsed_batch:.1f}s | Total filled: {filled_count}/{total}")

#             # Save progress
#             progress_df = pd.DataFrame(filled_rows)
#             progress_mgr.save(
#                 progress_df,
#                 completed_batches=b + 1,
#                 total_batches=n_batches,
#                 primary_column=task.primary_column,
#                 min_length=task.min_length,
#             )

#             if b < n_batches - 1:
#                 time.sleep(DELAY_BETWEEN_BATCHES)

#         # Final DataFrame
#         final_df = pd.DataFrame(filled_rows)

#         # Ensure column order matches original
#         for col in df.columns:
#             if col not in final_df.columns:
#                 final_df[col] = df[col]

#         return final_df[df.columns.tolist() + [c for c in task.output_columns if c not in df.columns]]

#     def _is_filled(self, row: Dict, column: str, min_length: int) -> bool:
#         """Check if row's column is filled."""
#         val = row.get(column, "")
#         return isinstance(val, str) and len(val) >= min_length

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  SAVE
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _save_final(self, df: pd.DataFrame, task_name: str) -> str:
#         """Save final dataset."""
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"dataset_{task_name}_{len(df)}rows_{ts}.csv"
#         path = os.path.join(OUTPUT_DIR, filename)

#         df.to_csv(path, index=False)
#         size_kb = os.path.getsize(path) / 1024

#         self._log(f"\n  ✓ Saved: {path}")
#         self._log(f"    {len(df)} rows × {len(df.columns)} cols ({size_kb:.1f} KB)")

#         return path

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  BANNER
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _banner(self, task: TaskConfig):
#         print(f"\n  {'═'*55}")
#         print(f"  ║  TASK: {task.name.upper():<46}║")
#         print(f"  ╠{'═'*55}╣")
#         print(f"  ║  Columns : {str(task.output_columns):<43}║")
#         print(f"  ║  Model   : {MODEL_NAME[:43]:<43}║")
#         print(f"  ║  Profile : {ACTIVE_PROFILE[:43]:<43}║")
#         print(f"  ╚{'═'*55}╝")

# pipeline/orchestrator.py
# pipeline/orchestrator.py
"""
Unified DatasetPipeline — runs any combination of tasks.
Complete implementation with all helper methods.
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from config import (
    OUTPUT_DIR, TOTAL_ROWS, TEXT_BATCH_SIZE,
    DELAY_BETWEEN_BATCHES, MODEL_NAME, ACTIVE_PROFILE,
)
from core.skeleton import build_skeleton, verify_skeleton
from core.validator import validate_dataset, print_full_report
from agents.base_agent import BaseAgent
from agents.task_config import TaskConfig, TaskRegistry
from pipeline.progress import ProgressManager
from pipeline.display import DisplayManager, RICH_AVAILABLE

try:
    from rich.progress import Progress
except ImportError:
    Progress = None


class DatasetPipeline:
    """
    Unified pipeline that runs any task on a dataset.
    Features: Rich CLI, Resume support, Proper completion detection.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.display = DisplayManager(verbose=verbose)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  COMPLETION DETECTION HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _is_task_complete(self, df: pd.DataFrame, task: TaskConfig) -> bool:
        """
        Check if a task is already complete (all rows filled).
        """
        if task.primary_column not in df.columns:
            return False

        filled_count = 0
        for _, row in df.iterrows():
            val = row.get(task.primary_column, "")
            if pd.notna(val) and isinstance(val, str) and len(val.strip()) >= task.min_length:
                filled_count += 1

        return filled_count == len(df)

    def _count_filled(self, df: pd.DataFrame, task: TaskConfig) -> int:
        """Count how many rows have the primary column filled."""
        if task.primary_column not in df.columns:
            return 0

        count = 0
        for _, row in df.iterrows():
            val = row.get(task.primary_column, "")
            if pd.notna(val) and isinstance(val, str) and len(val.strip()) >= task.min_length:
                count += 1
        return count

    def _is_row_filled(self, row: Dict, column: str, min_length: int) -> bool:
        """Check if a row's column is filled."""
        val = row.get(column, "")
        if pd.isna(val):
            return False
        if not isinstance(val, str):
            val = str(val)
        return len(val.strip()) >= min_length

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN ENTRY POINTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run_task(
        self,
        task_name: str,
        input_csv: Optional[str] = None,
        seed: int = 42,
        force_restart: bool = False,
    ) -> pd.DataFrame:
        """
        Run a single task.
        Returns immediately if task is already complete.
        """
        task = TaskRegistry.get(task_name)
        if not task:
            available = TaskRegistry.list_tasks()
            self.display.error(f"Unknown task '{task_name}'. Available: {available}")
            return pd.DataFrame()

        # ═══════════════════════════════════════════════════
        # CHECK IF TASK ALREADY COMPLETE (before banner)
        # ═══════════════════════════════════════════════════
        if input_csv and os.path.exists(input_csv):
            df = pd.read_csv(input_csv)
            if self._is_task_complete(df, task):
                self.display.success(f"Task '{task_name}' already complete in {input_csv}")
                self.display.info(f"All {len(df)} rows have '{task.primary_column}' filled")
                return df

        # Display task banner
        total_rows = TOTAL_ROWS
        if input_csv and os.path.exists(input_csv):
            try:
                total_rows = len(pd.read_csv(input_csv))
            except:
                pass

        self.display.task_banner(
            task_name=task.name,
            columns=task.output_columns,
            model=MODEL_NAME,
            profile=ACTIVE_PROFILE,
            total_rows=total_rows,
        )

        # Check for resume
        progress_mgr = ProgressManager(task_name)
        progress = None if force_restart else progress_mgr.load()

        if progress:
            return self._resume_task(task, progress_mgr, progress, seed)
        else:
            return self._fresh_task(task, progress_mgr, input_csv, seed)

    def run_tasks(
        self,
        task_names: List[str],
        seed: int = 42,
        force_restart: bool = False,
        input_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run multiple tasks in sequence.
        Skips tasks that are already complete.
        """
        start_time = time.time()
        df = None
        completed_tasks = []
        skipped_tasks = []

        # Load initial CSV if provided
        if input_csv and os.path.exists(input_csv):
            df = pd.read_csv(input_csv)
            self.display.file_loaded(input_csv, len(df), len(df.columns))

        for i, task_name in enumerate(task_names):
            task = TaskRegistry.get(task_name)
            if not task:
                self.display.error(f"Unknown task: {task_name}")
                continue

            self.display.header(f"Task {i+1}/{len(task_names)}: {task_name.upper()}")

            # Determine input for this task
            if df is None:
                task_input_csv = None
            else:
                # Check if already complete
                if self._is_task_complete(df, task):
                    self.display.success(f"Task '{task_name}' already complete — skipping")
                    skipped_tasks.append(task_name)
                    continue

                # Save current df as temp input
                temp_csv = os.path.join(OUTPUT_DIR, f"_temp_{task_name}.csv")
                df.to_csv(temp_csv, index=False)
                task_input_csv = temp_csv

            # Run task
            df = self.run_task(task_name, input_csv=task_input_csv, seed=seed, force_restart=force_restart)

            # Cleanup temp
            if task_input_csv and os.path.exists(task_input_csv):
                os.remove(task_input_csv)

            if df.empty:
                self.display.error(f"Task '{task_name}' failed")
                return df

            completed_tasks.append(task_name)

        total_time = time.time() - start_time

        if df is None or df.empty:
            self.display.error("No tasks completed")
            return pd.DataFrame()

        # Save final
        filepath = self._save_final(df, "full_pipeline")

        # Validate
        self.display.header("Final Validation")
        ok, errs = validate_dataset(df)
        self.display.validation_result(ok, errs)

        # Summary
        self.display.final_summary(
            tasks_completed=completed_tasks,
            total_time=total_time,
            total_rows=len(df),
            output_path=filepath,
            validation_passed=ok,
        )

        if skipped_tasks:
            self.display.info(f"Skipped (already complete): {', '.join(skipped_tasks)}")

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FRESH / RESUME
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fresh_task(
        self,
        task: TaskConfig,
        progress_mgr: ProgressManager,
        input_csv: Optional[str],
        seed: int,
    ) -> pd.DataFrame:
        """Run task from scratch."""

        # Get or build skeleton
        if input_csv and os.path.exists(input_csv):
            self.display.info(f"Loading: {input_csv}")
            df = pd.read_csv(input_csv)
            self.display.file_loaded(input_csv, len(df), len(df.columns))

            # Double-check completion after loading
            if self._is_task_complete(df, task):
                self.display.success(f"Task already complete!")
                return df
        else:
            self.display.subheader(f"Building skeleton ({TOTAL_ROWS:,} rows)")

            with self.display.spinner("Building skeleton"):
                df = build_skeleton(seed=seed)
                errs = verify_skeleton(df)

            if errs:
                for e in errs:
                    self.display.error(e)
                return pd.DataFrame()

            self.display.success(f"Skeleton verified: {len(df):,} rows")

        # Count already filled
        already_filled = self._count_filled(df, task)
        if already_filled > 0:
            self.display.info(f"Found {already_filled}/{len(df)} rows already filled")

        # Run task
        t0 = time.time()
        filled_df = self._run_batched(task, df, progress_mgr, start_batch=0)
        elapsed = time.time() - t0

        filled_count = self._count_filled(filled_df, task)

        # Save
        filepath = self._save_final(filled_df, task.name)
        progress_mgr.cleanup()

        # Display completion
        self.display.task_complete(
            task_name=task.name,
            total_rows=len(filled_df),
            elapsed=elapsed,
            filled=filled_count,
            output_path=filepath,
        )

        return filled_df

    def _resume_task(
        self,
        task: TaskConfig,
        progress_mgr: ProgressManager,
        progress: Dict,
        seed: int,
    ) -> pd.DataFrame:
        """Resume task from saved progress."""

        # Check if actually complete
        if progress.get("completed_batches", 0) >= progress.get("total_batches", 0):
            self.display.success(f"Task already complete according to progress file")
            progress_mgr.cleanup()

            df = progress_mgr.load_dataframe()
            if df is not None:
                return df
            else:
                return self._fresh_task(task, progress_mgr, None, seed)

        choice = self.display.resume_prompt(progress)

        if choice == "F":
            self.display.info("Starting fresh...")
            progress_mgr.cleanup()
            return self._fresh_task(task, progress_mgr, None, seed)

        # Load progress
        df = progress_mgr.load_dataframe()
        if df is None:
            self.display.error("Could not load progress CSV")
            return pd.DataFrame()

        # Check if actually complete
        if self._is_task_complete(df, task):
            self.display.success(f"Task already complete!")
            progress_mgr.cleanup()
            return df

        self.display.file_loaded(progress_mgr.progress_csv, len(df), len(df.columns))

        # Resume
        start_batch = progress["completed_batches"]
        self.display.info(f"Resuming from batch {start_batch + 1}")

        t0 = time.time()
        filled_df = self._run_batched(task, df, progress_mgr, start_batch=start_batch)
        elapsed = time.time() - t0

        filled_count = self._count_filled(filled_df, task)

        # Save
        filepath = self._save_final(filled_df, task.name)
        progress_mgr.cleanup()

        # Display completion
        self.display.task_complete(
            task_name=task.name,
            total_rows=len(filled_df),
            elapsed=elapsed,
            filled=filled_count,
            output_path=filepath,
        )

        return filled_df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BATCHED EXECUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run_batched(
        self,
        task: TaskConfig,
        df: pd.DataFrame,
        progress_mgr: ProgressManager,
        start_batch: int = 0,
    ) -> pd.DataFrame:
        """Run task in batches with progress bar."""

        all_rows = df.to_dict("records")
        total = len(all_rows)
        n_batches = -(-total // TEXT_BATCH_SIZE)

        # Check if already complete
        already_complete = all(
            self._is_row_filled(r, task.primary_column, task.min_length)
            for r in all_rows
        )
        if already_complete:
            self.display.success(f"All {total} rows already filled — nothing to do")
            return df

        # Create agent
        agent = BaseAgent(task, verbose=False)

        # Load existing values for deduplication
        if task.deduplicate:
            if task.primary_column in df.columns:
                existing = df[task.primary_column].fillna("").astype(str).tolist()
                agent.load_existing_values(existing)
                self.display.info(f"Loaded {len(agent._used_values)} existing values for dedup")

        filled_rows = all_rows.copy()

        # Count already completed
        already_completed_rows = sum(
            1 for r in all_rows
            if self._is_row_filled(r, task.primary_column, task.min_length)
        )

        if RICH_AVAILABLE:
            return self._run_with_rich_progress(
                task, agent, all_rows, filled_rows,
                start_batch, n_batches, total, progress_mgr, df,
                already_completed_rows
            )
        else:
            return self._run_with_basic_progress(
                task, agent, all_rows, filled_rows,
                start_batch, n_batches, total, progress_mgr, df
            )

    def _run_with_rich_progress(
        self,
        task: TaskConfig,
        agent: BaseAgent,
        all_rows: List[Dict],
        filled_rows: List[Dict],
        start_batch: int,
        n_batches: int,
        total: int,
        progress_mgr: ProgressManager,
        original_df: pd.DataFrame,
        already_completed_rows: int = 0,
    ) -> pd.DataFrame:
        """Run with rich progress bar."""

        progress = self.display.create_progress()
        batches_skipped = 0

        with progress:
            overall = progress.add_task(
                "[cyan]Overall Progress",
                total=total,
                completed=already_completed_rows,
            )

            batch_task = progress.add_task(
                "[yellow]Current Batch",
                total=TEXT_BATCH_SIZE,
                completed=0,
            )

            for b in range(start_batch, n_batches):
                batch_start = b * TEXT_BATCH_SIZE
                batch_end = min(batch_start + TEXT_BATCH_SIZE, total)
                batch_rows = all_rows[batch_start:batch_end]
                batch_size = len(batch_rows)

                # Check if batch done
                batch_done = all(
                    self._is_row_filled(r, task.primary_column, task.min_length)
                    for r in batch_rows
                )

                if batch_done:
                    batches_skipped += 1
                    continue

                # Reset batch progress
                progress.reset(batch_task, total=batch_size, completed=0)
                progress.update(batch_task, description=f"[yellow]Batch {b+1}/{n_batches}")

                t_batch = time.time()
                result = agent.generate(batch_rows, batch_label=f"b{b+1:03d}")
                elapsed_batch = time.time() - t_batch

                # Update filled_rows
                for i, row in enumerate(result):
                    filled_rows[batch_start + i] = row

                # Update progress
                new_filled = sum(
                    1 for r in result
                    if self._is_row_filled(r, task.primary_column, task.min_length)
                )
                progress.update(batch_task, completed=batch_size)
                progress.update(overall, advance=new_filled)

                # Count total filled
                filled_count = sum(
                    1 for r in filled_rows
                    if self._is_row_filled(r, task.primary_column, task.min_length)
                )

                # Save progress
                progress_df = pd.DataFrame(filled_rows)
                progress_mgr.save(
                    progress_df,
                    completed_batches=b + 1,
                    total_batches=n_batches,
                    primary_column=task.primary_column,
                    min_length=task.min_length,
                )

                # Display
                self.display.batch_summary(
                    batch_num=b + 1,
                    total_batches=n_batches,
                    batch_size=batch_size,
                    elapsed=elapsed_batch,
                    filled_total=filled_count,
                    total_rows=total,
                )

                if b < n_batches - 1:
                    time.sleep(DELAY_BETWEEN_BATCHES)

        if batches_skipped > 0:
            self.display.info(f"Skipped {batches_skipped} already-complete batches")

        return self._build_final_df(filled_rows, original_df, task)

    def _run_with_basic_progress(
        self,
        task: TaskConfig,
        agent: BaseAgent,
        all_rows: List[Dict],
        filled_rows: List[Dict],
        start_batch: int,
        n_batches: int,
        total: int,
        progress_mgr: ProgressManager,
        original_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run with basic progress (no rich)."""

        self.display.info(f"Processing {total:,} rows in {n_batches} batches")

        for b in range(start_batch, n_batches):
            batch_start = b * TEXT_BATCH_SIZE
            batch_end = min(batch_start + TEXT_BATCH_SIZE, total)
            batch_rows = all_rows[batch_start:batch_end]
            batch_size = len(batch_rows)

            # Check if batch done
            batch_done = all(
                self._is_row_filled(r, task.primary_column, task.min_length)
                for r in batch_rows
            )
            if batch_done:
                continue

            pct = (batch_end / total) * 100
            bar_width = 40
            filled_width = int(bar_width * batch_end / total)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)

            print(f"\r  [{bar}] {pct:5.1f}% | Batch {b+1}/{n_batches}", end="", flush=True)

            t_batch = time.time()
            result = agent.generate(batch_rows, batch_label=f"b{b+1:03d}")
            elapsed_batch = time.time() - t_batch

            for i, row in enumerate(result):
                filled_rows[batch_start + i] = row

            progress_df = pd.DataFrame(filled_rows)
            progress_mgr.save(
                progress_df,
                completed_batches=b + 1,
                total_batches=n_batches,
                primary_column=task.primary_column,
                min_length=task.min_length,
            )

            print(f" ✓ {elapsed_batch:.1f}s")

            if b < n_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)

        print()
        return self._build_final_df(filled_rows, original_df, task)

    def _build_final_df(
        self,
        filled_rows: List[Dict],
        original_df: pd.DataFrame,
        task: TaskConfig,
    ) -> pd.DataFrame:
        """Build final DataFrame with proper column order."""
        final_df = pd.DataFrame(filled_rows)

        # Ensure all original columns exist
        for col in original_df.columns:
            if col not in final_df.columns:
                final_df[col] = original_df[col]

        # Build column order
        final_columns = list(original_df.columns)
        for col in task.output_columns:
            if col not in final_columns:
                final_columns.append(col)

        final_columns = [c for c in final_columns if c in final_df.columns]

        return final_df[final_columns]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SAVE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_final(self, df: pd.DataFrame, task_name: str) -> str:
        """Save final dataset."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{task_name}_{len(df)}rows_{ts}.csv"
        path = os.path.join(OUTPUT_DIR, filename)

        df.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024

        self.display.file_saved(path, len(df), len(df.columns), size_kb)

        return path