# generator.py
"""
Orchestrator: skeleton → LLM text → validate → save.
SUPPORTS RESUME — picks up from where it stopped.
ADVANCED CLI PROGRESS BAR
"""

import os
import time
import json
import glob
import pandas as pd
from datetime import datetime
from typing import List, Optional

from config import (
    TOTAL_ROWS, TEXT_BATCH_SIZE,
    DELAY_BETWEEN_BATCHES, OUTPUT_DIR,
    THEME_OBJECTS, USE_OPENAI_SDK,
    MODEL_NAME, ACTIVE_PROFILE,
    BASE_UNIT, NUM_CYCLES,
)
from skeleton import build_skeleton, verify_skeleton, print_skeleton_summary
from text_generator import TextGenerator
from validator import validate_dataset, print_full_report

# Rich imports for advanced progress bar
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ Install 'rich' for advanced progress bar: pip install rich")


# Progress tracking file
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_progress.json")
PROGRESS_CSV = os.path.join(OUTPUT_DIR, "_progress_data.csv")


class ProgressTracker:
    """Advanced progress tracking with rich UI."""

    def __init__(self, total_rows: int, total_batches: int, start_batch: int = 0):
        self.total_rows = total_rows
        self.total_batches = total_batches
        self.start_batch = start_batch
        self.current_batch = start_batch
        self.completed_rows = start_batch * TEXT_BATCH_SIZE
        self.start_time = time.time()
        self.batch_times: List[float] = []
        self.errors = 0
        self.retries = 0

        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def _create_progress_table(self) -> Table:
        """Create a stats table for the progress display."""
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=25)

        elapsed = time.time() - self.start_time
        rows_done = self.completed_rows
        rows_remaining = self.total_rows - rows_done

        # Calculate speed
        if elapsed > 0 and rows_done > 0:
            rows_per_sec = rows_done / elapsed
            rows_per_min = rows_per_sec * 60
            eta_seconds = rows_remaining / rows_per_sec if rows_per_sec > 0 else 0
        else:
            rows_per_sec = 0
            rows_per_min = 0
            eta_seconds = 0

        # Format times
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)

        # Average batch time
        if self.batch_times:
            avg_batch = sum(self.batch_times) / len(self.batch_times)
            avg_batch_str = f"{avg_batch:.1f}s"
        else:
            avg_batch_str = "calculating..."

        table.add_row("📊 Batch", f"{self.current_batch}/{self.total_batches}")
        table.add_row("📝 Rows Done", f"{rows_done:,}/{self.total_rows:,}")
        table.add_row("⏱️  Elapsed", elapsed_str)
        table.add_row("⏳ ETA", eta_str)
        table.add_row("🚀 Speed", f"{rows_per_min:.1f} rows/min")
        table.add_row("📦 Avg Batch", avg_batch_str)
        table.add_row("🔄 Retries", str(self.retries))
        table.add_row("❌ Errors", str(self.errors))

        return table

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def get_progress_bar(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

    def update(self, batch_num: int, batch_time: float, rows_in_batch: int):
        """Update progress after a batch completes."""
        self.current_batch = batch_num
        self.completed_rows += rows_in_batch
        self.batch_times.append(batch_time)

    def add_retry(self):
        self.retries += 1

    def add_error(self):
        self.errors += 1

    def print_stats(self):
        """Print current stats (for non-rich fallback)."""
        elapsed = time.time() - self.start_time
        pct = (self.completed_rows / self.total_rows) * 100
        speed = self.completed_rows / elapsed if elapsed > 0 else 0

        print(f"\r    ├─ Progress: {self.completed_rows}/{self.total_rows} "
              f"({pct:.1f}%) | {speed:.1f} rows/sec | "
              f"Batch {self.current_batch}/{self.total_batches}", end="", flush=True)


class DatasetGenerator:
    """
    Pipeline with RESUME support + Advanced Progress Bar:
    1. Check for existing progress
    2. Build skeleton (or load from progress)
    3. Fill texts starting from last completed batch
    4. Save after every batch
    5. Final save + cleanup
    """

    def __init__(self):
        self.text_gen = TextGenerator()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.console = Console() if RICH_AVAILABLE else None

    def generate(self, seed: int = 42, force_restart: bool = False) -> pd.DataFrame:
        """
        Full pipeline with resume.

        Args:
            seed: Random seed for skeleton
            force_restart: If True, ignore saved progress and start fresh
        """
        self._banner()

        # ── Check for existing progress ──
        progress = None
        if not force_restart:
            progress = self._load_progress()

        if progress:
            return self._resume_generation(progress, seed)
        else:
            return self._fresh_generation(seed)

    def _fresh_generation(self, seed: int) -> pd.DataFrame:
        """Start generation from scratch."""

        # Step 1: Build skeleton
        self._print_step("STEP 1: Building Skeleton", f"{TOTAL_ROWS} rows")

        skeleton = build_skeleton(seed=seed)
        errs = verify_skeleton(skeleton)

        if errs:
            self._print_error("SKELETON ERRORS:")
            for e in errs:
                print(f"    • {e}")
            return pd.DataFrame()

        self._print_success(f"Skeleton: {len(skeleton)} rows — ALL PERFECT")
        print_skeleton_summary(skeleton)

        # Save skeleton
        skeleton_path = os.path.join(OUTPUT_DIR, "_skeleton.csv")
        skeleton.to_csv(skeleton_path, index=False)
        self._print_info(f"Skeleton saved: {skeleton_path}")

        # Step 2: Fill texts
        self._print_step("STEP 2: Generating Texts via LLM", "")

        t0 = time.time()
        filled = self._fill_all_texts_with_progress(skeleton, start_batch=0)
        elapsed = time.time() - t0

        self._print_success(f"Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Step 3: Save final
        self._print_step("STEP 3: Save Final Dataset", "")

        filepath = self._save_final(filled)
        self._cleanup_progress()

        return filled

    # def _resume_generation(self, progress: dict, seed: int) -> pd.DataFrame:
    #     """Resume from saved progress."""

    #     completed_batches = progress["completed_batches"]
    #     total_batches = progress["total_batches"]
    #     completed_rows = progress["completed_rows"]

    #     self._print_step(
    #         f"♻ RESUMING FROM BATCH {completed_batches + 1}/{total_batches}",
    #         f"Rows: {completed_rows}/{TOTAL_ROWS}"
    #     )

    #     # Load the progress CSV
    #     if not os.path.exists(PROGRESS_CSV):
    #         self._print_error(f"Progress CSV not found: {PROGRESS_CSV}")
    #         return self._fresh_generation(seed)

    #     df = pd.read_csv(PROGRESS_CSV)
    #     self._print_info(f"Loaded {len(df)} rows from progress file")

    #     if len(df) != TOTAL_ROWS:
    #         self._print_error(f"Progress CSV has {len(df)} rows, expected {TOTAL_ROWS}")
    #         return self._fresh_generation(seed)

    #     # Reload used texts to avoid duplicates
    #     existing_texts = df["text"].astype(str).tolist()
    #     for text in existing_texts:
    #         if text and len(text) >= 8 and text != "":
    #             self.text_gen._used_texts.add(text)
    #     self._print_info(f"Loaded {len(self.text_gen._used_texts)} existing texts for dedup")

    #     # Continue filling texts
    #     self._print_step("CONTINUING TEXT GENERATION", "")

    #     t0 = time.time()
    #     filled = self._fill_all_texts_with_progress(df, start_batch=completed_batches)
    #     elapsed = time.time() - t0

    #     self._print_success(f"Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    #     # Save final
    #     self._print_step("SAVING FINAL DATASET", "")

    #     filepath = self._save_final(filled)
    #     self._cleanup_progress()

    #     return filled
    def _resume_generation(self, progress: dict, seed: int) -> pd.DataFrame:
        """Resume from saved progress."""

        completed_batches = progress["completed_batches"]
        total_batches = progress["total_batches"]
        completed_rows = progress["completed_rows"]

        self._print_step(
            f"♻ RESUMING FROM BATCH {completed_batches + 1}/{total_batches}",
            f"Rows: {completed_rows}/{TOTAL_ROWS}"
        )

        # Load the progress CSV
        if not os.path.exists(PROGRESS_CSV):
            self._print_error(f"Progress CSV not found: {PROGRESS_CSV}")
            return self._fresh_generation(seed)

        df = pd.read_csv(PROGRESS_CSV)
        self._print_info(f"Loaded {len(df)} rows from progress file")

        if len(df) != TOTAL_ROWS:
            self._print_error(f"Progress CSV has {len(df)} rows, expected {TOTAL_ROWS}")
            return self._fresh_generation(seed)

        # Reload used texts to avoid duplicates - FIX HERE
        # Handle NaN values properly
        existing_texts = df["text"].fillna("").astype(str).tolist()  # Fill NaN first
        valid_count = 0
        for text in existing_texts:
            # Check if text is valid (not empty, not "nan", and long enough)
            if (text and 
                isinstance(text, str) and 
                text.lower() not in ["", "nan", "none"] and 
                len(text) >= 8):
                self.text_gen._used_texts.add(text)
                valid_count += 1

        self._print_info(f"Loaded {valid_count} existing texts for dedup (out of {len(df)} rows)")

        # Continue filling texts
        self._print_step("CONTINUING TEXT GENERATION", "")

        t0 = time.time()
        filled = self._fill_all_texts_with_progress(df, start_batch=completed_batches)
        elapsed = time.time() - t0

        self._print_success(f"Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Save final
        self._print_step("SAVING FINAL DATASET", "")

        filepath = self._save_final(filled)
        self._cleanup_progress()

        return filled

    def _fill_all_texts_with_progress(
        self,
        skeleton: pd.DataFrame,
        start_batch: int = 0,
    ) -> pd.DataFrame:
        """
        Fill text + keywords with advanced progress bar.
        """
        all_rows = skeleton.to_dict("records")
        total = len(all_rows)
        n_batches = -(-total // TEXT_BATCH_SIZE)

        # Initialize tracker
        tracker = ProgressTracker(total, n_batches, start_batch)

        # Pre-fill completed rows count
        completed_before = sum(
            1 for r in all_rows[:start_batch * TEXT_BATCH_SIZE]
            if r.get("text") and len(str(r.get("text", ""))) >= 8
        )
        tracker.completed_rows = completed_before

        filled_rows = all_rows.copy()

        if RICH_AVAILABLE:
            return self._fill_with_rich_progress(
                all_rows, filled_rows, start_batch, n_batches, total, tracker, skeleton
            )
        else:
            return self._fill_with_basic_progress(
                all_rows, filled_rows, start_batch, n_batches, total, tracker, skeleton
            )

    def _fill_with_rich_progress(
        self,
        all_rows: List[dict],
        filled_rows: List[dict],
        start_batch: int,
        n_batches: int,
        total: int,
        tracker: ProgressTracker,
        skeleton: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill texts with rich progress display."""

        console = Console()

        # Create the main progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50, complete_style="green", finished_style="bright_green"),
            TextColumn("[bold]{task.percentage:>5.1f}%"),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
            transient=False,
        )

        # Calculate already completed
        already_completed = start_batch * TEXT_BATCH_SIZE

        with progress:
            # Add tasks
            overall_task = progress.add_task(
                "[cyan]Overall Progress",
                total=total,
                completed=already_completed,
            )
            batch_task = progress.add_task(
                "[yellow]Current Batch",
                total=TEXT_BATCH_SIZE,
                completed=0,
            )

            for b in range(start_batch, n_batches):
                start = b * TEXT_BATCH_SIZE
                end = min(start + TEXT_BATCH_SIZE, total)
                batch_rows = all_rows[start:end]
                batch_size = len(batch_rows)

                # Check if batch is done
                batch_already_done = all(
                    row.get("text") and len(str(row.get("text", ""))) >= 8
                    for row in batch_rows
                )

                if batch_already_done:
                    progress.update(overall_task, advance=batch_size)
                    continue

                # Reset batch progress
                progress.reset(batch_task, total=batch_size, completed=0)
                progress.update(batch_task, description=f"[yellow]Batch {b+1}/{n_batches}")

                batch_start_time = time.time()

                # Generate texts
                result = self.text_gen.generate_texts(
                    batch_rows, batch_label=f"batch{b+1:02d}"
                )

                batch_time = time.time() - batch_start_time

                # Update filled_rows
                for i, row in enumerate(result):
                    filled_rows[start + i] = row

                # Update progress
                progress.update(batch_task, completed=batch_size)
                progress.update(overall_task, advance=batch_size)
                tracker.update(b + 1, batch_time, batch_size)

                # Save progress
                self._save_progress(filled_rows, b + 1, n_batches)

                # Print batch summary below progress bar
                console.print(
                    f"    ✓ Batch {b+1}/{n_batches} completed in {batch_time:.1f}s "
                    f"({batch_size/batch_time:.1f} rows/sec)",
                    style="dim"
                )

                if b < n_batches - 1:
                    time.sleep(DELAY_BETWEEN_BATCHES)

        # Print final summary
        self._print_final_summary(tracker)

        filled_df = pd.DataFrame(filled_rows)
        filled_df = filled_df[skeleton.columns]
        return filled_df

    def _fill_with_basic_progress(
        self,
        all_rows: List[dict],
        filled_rows: List[dict],
        start_batch: int,
        n_batches: int,
        total: int,
        tracker: ProgressTracker,
        skeleton: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill texts with basic progress (no rich library)."""

        print(f"\n  Rows: {total} | Batch: {TEXT_BATCH_SIZE} | "
              f"Batches: {n_batches} | Starting from: {start_batch + 1}")
        print()

        for b in range(start_batch, n_batches):
            start = b * TEXT_BATCH_SIZE
            end = min(start + TEXT_BATCH_SIZE, total)
            batch_rows = all_rows[start:end]
            batch_size = len(batch_rows)

            # Check if batch is done
            batch_already_done = all(
                row.get("text") and len(str(row.get("text", ""))) >= 8
                for row in batch_rows
            )

            if batch_already_done:
                continue

            pct = end / total * 100
            bar_width = 40
            filled_width = int(bar_width * (end / total))
            bar = "█" * filled_width + "░" * (bar_width - filled_width)

            print(f"\r  [{bar}] {pct:5.1f}% | Batch {b+1}/{n_batches} | "
                  f"Rows {start+1}-{end}", end="", flush=True)

            batch_start_time = time.time()

            result = self.text_gen.generate_texts(
                batch_rows, batch_label=f"batch{b+1:02d}"
            )

            batch_time = time.time() - batch_start_time

            for i, row in enumerate(result):
                filled_rows[start + i] = row

            tracker.update(b + 1, batch_time, batch_size)
            self._save_progress(filled_rows, b + 1, n_batches)

            print(f" ✓ {batch_time:.1f}s")

            if b < n_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)

        print()
        filled_df = pd.DataFrame(filled_rows)
        filled_df = filled_df[skeleton.columns]
        return filled_df

    def _print_final_summary(self, tracker: ProgressTracker):
        """Print a beautiful final summary."""
        if not RICH_AVAILABLE:
            return

        elapsed = time.time() - tracker.start_time
        avg_batch = sum(tracker.batch_times) / len(tracker.batch_times) if tracker.batch_times else 0
        speed = tracker.completed_rows / elapsed if elapsed > 0 else 0

        table = Table(
            title="📊 Generation Complete",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Rows", f"{tracker.completed_rows:,}")
        table.add_row("Total Batches", str(tracker.current_batch))
        table.add_row("Total Time", tracker._format_time(elapsed))
        table.add_row("Avg Batch Time", f"{avg_batch:.1f}s")
        table.add_row("Speed", f"{speed * 60:.1f} rows/min")
        table.add_row("Retries", str(tracker.retries))
        table.add_row("Errors", str(tracker.errors))

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELPER PRINT METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _print_step(self, title: str, subtitle: str = ""):
        """Print a step header."""
        if RICH_AVAILABLE:
            text = f"[bold cyan]{title}[/bold cyan]"
            if subtitle:
                text += f" [dim]({subtitle})[/dim]"
            self.console.print()
            self.console.rule(text, style="cyan")
        else:
            print(f"\n{'━'*60}")
            print(f"  {title}" + (f" ({subtitle})" if subtitle else ""))
            print(f"{'━'*60}")

    def _print_success(self, message: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [green]✓[/green] {message}")
        else:
            print(f"  ✓ {message}")

    def _print_error(self, message: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [red]✗[/red] {message}")
        else:
            print(f"  ✗ {message}")

    def _print_info(self, message: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [blue]ℹ[/blue] {message}")
        else:
            print(f"  💾 {message}")

    def _print_warning(self, message: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [yellow]⚠[/yellow] {message}")
        else:
            print(f"  ⚠ {message}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PROGRESS SAVE / LOAD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_progress(
        self,
        filled_rows: List[dict],
        completed_batches: int,
        total_batches: int,
    ):
        """Save current progress to disk."""

        completed_rows = sum(
            1 for r in filled_rows
            if r.get("text") and len(str(r.get("text", ""))) >= 8
        )

        progress = {
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "completed_rows": completed_rows,
            "total_rows": TOTAL_ROWS,
            "batch_size": TEXT_BATCH_SIZE,
            "model": MODEL_NAME,
            "profile": ACTIVE_PROFILE,
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_unit": BASE_UNIT,
            "num_cycles": NUM_CYCLES,
        }

        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)

        df = pd.DataFrame(filled_rows)
        df.to_csv(PROGRESS_CSV, index=False)

    def _load_progress(self) -> Optional[dict]:
        """Load saved progress if it exists."""

        if not os.path.exists(PROGRESS_FILE):
            return None

        try:
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)

            if progress.get("total_rows") != TOTAL_ROWS:
                self._print_warning(
                    f"Saved progress is for {progress.get('total_rows')} rows, "
                    f"but current config is {TOTAL_ROWS} rows."
                )
                return None

            if progress.get("base_unit") != BASE_UNIT:
                self._print_warning(
                    f"Saved progress has base_unit={progress.get('base_unit')}, "
                    f"but current config has {BASE_UNIT}."
                )
                return None

            completed = progress.get("completed_batches", 0)
            total = progress.get("total_batches", 0)

            if completed >= total:
                self._print_warning("Progress shows all batches complete. Starting fresh...")
                return None

            if not os.path.exists(PROGRESS_CSV):
                self._print_warning("Progress JSON exists but CSV missing. Starting fresh...")
                return None

            # Display saved progress
            self._display_saved_progress(progress, completed, total)

            # Ask user
            print(f"\n  Options:")
            print(f"    [R] Resume from batch {completed + 1}")
            print(f"    [F] Fresh start (discard progress)")
            print(f"    [Enter] Resume (default)")

            try:
                choice = input("\n  Your choice [R/F]: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                choice = "R"

            if choice == "F":
                print(f"  Starting fresh...")
                return None

            return progress

        except (json.JSONDecodeError, KeyError) as e:
            self._print_warning(f"Corrupted progress file: {e}")
            return None

    def _display_saved_progress(self, progress: dict, completed: int, total: int):
        """Display the saved progress info."""

        if RICH_AVAILABLE:
            table = Table(
                title="💾 Saved Progress Found",
                box=box.DOUBLE_EDGE,
                show_header=False,
                title_style="bold yellow",
            )
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Batches", f"{completed}/{total}")
            table.add_row("Rows", f"{progress.get('completed_rows', '?')}/{TOTAL_ROWS}")
            table.add_row("Model", str(progress.get('model', '?')))
            table.add_row("Last Saved", str(progress.get('last_saved', '?')))

            self.console.print()
            self.console.print(table)
        else:
            batches_str = f"{completed}/{total}"
            rows_str = f"{progress.get('completed_rows', '?')}/{TOTAL_ROWS}"

            print(f"\n  ╔{'═'*50}╗")
            print(f"  ║  SAVED PROGRESS FOUND                          ║")
            print(f"  ╠{'═'*50}╣")
            print(f"  ║  Batches : {batches_str}{' '*(38-len(batches_str))}║")
            print(f"  ║  Rows    : {rows_str}{' '*(38-len(rows_str))}║")
            print(f"  ║  Model   : {progress.get('model', '?')[:37]:<37}║")
            print(f"  ║  Saved   : {progress.get('last_saved', '?')[:37]:<37}║")
            print(f"  ╚{'═'*50}╝")

    def _cleanup_progress(self):
        """Remove progress files after successful completion."""
        for f in [PROGRESS_FILE, PROGRESS_CSV]:
            if os.path.exists(f):
                os.remove(f)
                self._print_info(f"Cleaned up: {os.path.basename(f)}")

        skeleton_path = os.path.join(OUTPUT_DIR, "_skeleton.csv")
        if os.path.exists(skeleton_path):
            os.remove(skeleton_path)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SAVE FINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_final(self, df: pd.DataFrame, filename: str = None) -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_{len(df)}rows_{ts}.csv"

        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024

        if RICH_AVAILABLE:
            panel = Panel(
                f"[green]{path}[/green]\n"
                f"[dim]{len(df):,} rows × {len(df.columns)} cols ({size_kb:.1f} KB)[/dim]",
                title="✓ Dataset Saved",
                border_style="green",
            )
            self.console.print(panel)
        else:
            print(f"\n  ✓ SAVED: {path}")
            print(f"    {len(df)} rows × {len(df.columns)} cols ({size_kb:.1f} KB)")

        return path

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BANNER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _banner(self):
        total_objs = sum(len(v) for v in THEME_OBJECTS.values())
        mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"

        if RICH_AVAILABLE:
            table = Table(
                title="🚀 Dataset Generator v5",
                box=box.DOUBLE_EDGE,
                show_header=False,
                title_style="bold magenta",
                border_style="blue",
            )
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("API Mode", mode)
            table.add_row("Model", MODEL_NAME)
            table.add_row("Profile", ACTIVE_PROFILE)
            table.add_row("Total Rows", f"{TOTAL_ROWS:,}")
            table.add_row("Themes", str(len(THEME_OBJECTS)))
            table.add_row("Objects", str(total_objs))
            table.add_row("Rows/Object", str(TOTAL_ROWS // total_objs))
            table.add_row("Batch Size", str(TEXT_BATCH_SIZE))
            table.add_row("Resume", "✓ Enabled")

            self.console.print()
            self.console.print(table)
        else:
            print(f"\n{'#'*60}")
            print(f"  DATASET GENERATOR v5 (with Advanced Progress)")
            print(f"{'#'*60}")
            print(f"  API Mode     : {mode}")
            print(f"  Model        : {MODEL_NAME}")
            print(f"  Profile      : {ACTIVE_PROFILE}")
            print(f"  Total rows   : {TOTAL_ROWS}")
            print(f"  Themes       : {len(THEME_OBJECTS)}")
            print(f"  Objects      : {total_objs}")
            print(f"  Rows/object  : {TOTAL_ROWS // total_objs}")
            print(f"  Batch size   : {TEXT_BATCH_SIZE}")
            print(f"  Resume       : ENABLED")
            print(f"{'#'*60}")