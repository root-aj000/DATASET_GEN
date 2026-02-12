# # img_generator.py
# """
# Orchestrator for Image Search Queries: load dataset → generate img_desc → save.
# SUPPORTS RESUME — picks up from where it stopped.
# ADVANCED CLI PROGRESS BAR
# Mirror of generator.py but for img_desc column.
# """

# import os
# import time
# import json
# import glob
# import pandas as pd
# from datetime import datetime
# from typing import List, Optional

# from config import (
#     OUTPUT_DIR, USE_OPENAI_SDK,
#     MODEL_NAME, ACTIVE_PROFILE,
#     MAX_RETRIES, DELAY_BETWEEN_BATCHES,
#     TEXT_BATCH_SIZE
# )
# from search_agent import SearchAgent

# # Rich imports for advanced progress bar
# try:
#     from rich.console import Console
#     from rich.progress import (
#         Progress, SpinnerColumn, TextColumn, BarColumn,
#         TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
#         MofNCompleteColumn
#     )
#     from rich.panel import Panel
#     from rich.table import Table
#     from rich import box
#     RICH_AVAILABLE = True
# except ImportError:
#     RICH_AVAILABLE = False
#     print("⚠ Install 'rich' for advanced progress bar: pip install rich")


# # Configuration
# IMG_BATCH_SIZE = TEXT_BATCH_SIZE  # Rows per batch for image queries
# IMG_PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_img_progress.json")
# IMG_PROGRESS_CSV = os.path.join(OUTPUT_DIR, "_img_progress_data.csv")


# class ImageProgressTracker:
#     """Progress tracking for image query generation."""

#     def __init__(self, total_rows: int, total_batches: int, start_batch: int = 0):
#         self.total_rows = total_rows
#         self.total_batches = total_batches
#         self.start_batch = start_batch
#         self.current_batch = start_batch
#         self.completed_rows = start_batch * IMG_BATCH_SIZE
#         self.start_time = time.time()
#         self.batch_times: List[float] = []
#         self.errors = 0
#         self.retries = 0

#         self.console = Console() if RICH_AVAILABLE else None

#     def _format_time(self, seconds: float) -> str:
#         """Format seconds into human readable time."""
#         if seconds < 60:
#             return f"{seconds:.0f}s"
#         elif seconds < 3600:
#             mins = int(seconds // 60)
#             secs = int(seconds % 60)
#             return f"{mins}m {secs}s"
#         else:
#             hours = int(seconds // 3600)
#             mins = int((seconds % 3600) // 60)
#             return f"{hours}h {mins}m"

#     def update(self, batch_num: int, batch_time: float, rows_in_batch: int):
#         """Update progress after a batch completes."""
#         self.current_batch = batch_num
#         self.completed_rows += rows_in_batch
#         self.batch_times.append(batch_time)

#     def add_retry(self):
#         self.retries += 1

#     def add_error(self):
#         self.errors += 1


# class ImageDescGenerator:
#     """
#     Pipeline for img_desc with RESUME support + Advanced Progress Bar:
#     1. Load dataset CSV
#     2. Check for existing img_desc progress
#     3. Fill img_desc starting from last completed batch
#     4. Save after every batch
#     5. Final save + cleanup
#     """

#     def __init__(self):
#         self.search_agent = SearchAgent()
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         self.console = Console() if RICH_AVAILABLE else None

#     def generate(self, filepath: str = None, force_restart: bool = False) -> pd.DataFrame:
#         """
#         Full pipeline with resume.

#         Args:
#             filepath: Path to dataset CSV. If None, finds latest dataset_*rows*.csv
#             force_restart: If True, ignore saved progress and start fresh
#         """
        
#         # Find input file
#         if filepath is None:
#             files = glob.glob(os.path.join(OUTPUT_DIR, "dataset_*rows*.csv"))
#             if not files:
#                 self._print_error("No dataset found. Run text generation first.")
#                 return pd.DataFrame()
#             filepath = max(files, key=os.path.getctime)
#             self._print_info(f"Using latest dataset: {os.path.basename(filepath)}")
        
#         if not os.path.exists(filepath):
#             self._print_error(f"File not found: {filepath}")
#             return pd.DataFrame()

#         self._banner(filepath)

#         # Load dataset
#         df = pd.read_csv(filepath)
#         df.columns = df.columns.str.strip()
        
#         # Ensure img_desc column exists
#         if "img_desc" not in df.columns:
#             df["img_desc"] = ""
        
#         # Check for existing progress
#         progress = None
#         if not force_restart:
#             progress = self._load_progress(filepath)

#         if progress:
#             return self._resume_generation(df, filepath, progress)
#         else:
#             return self._fresh_generation(df, filepath)

#     def _fresh_generation(self, df: pd.DataFrame, filepath: str) -> pd.DataFrame:
#         """Start generation from scratch."""
        
#         # Clear any existing img_desc
#         df["img_desc"] = ""
        
#         self._print_step("STEP 1: Analyzing Dataset", f"{len(df)} rows")
        
#         # Check required columns
#         required = ['object_detected', 'theme', 'emotion', 'text']
#         missing = [c for c in required if c not in df.columns]
#         if missing:
#             self._print_error(f"Missing required columns: {missing}")
#             return pd.DataFrame()
        
#         self._print_success(f"Dataset loaded: {len(df)} rows")
        
#         # Count rows needing processing
#         needs_processing = df["img_desc"].isna() | (df["img_desc"].str.len() < 10)
#         to_process = needs_processing.sum()
        
#         self._print_info(f"Rows needing img_desc: {to_process}/{len(df)}")
        
#         # Step 2: Generate queries
#         self._print_step("STEP 2: Generating Search Queries via LLM", "")
        
#         t0 = time.time()
#         filled = self._fill_all_queries_with_progress(df, start_batch=0, filepath=filepath)
#         elapsed = time.time() - t0
        
#         self._print_success(f"Query generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        
#         # Step 3: Save final
#         self._print_step("STEP 3: Save Updated Dataset", "")
        
#         self._save_final(filled, filepath)
#         self._cleanup_progress()
        
#         return filled

#     def _resume_generation(self, df: pd.DataFrame, filepath: str, progress: dict) -> pd.DataFrame:
#         """Resume from saved progress."""
        
#         completed_batches = progress["completed_batches"]
#         total_batches = progress["total_batches"]
#         completed_queries = progress.get("completed_queries", 0)
        
#         self._print_step(
#             f"♻ RESUMING FROM BATCH {completed_batches + 1}/{total_batches}",
#             f"Queries: {completed_queries}/{len(df)}"
#         )
        
#         # Load the progress CSV
#         if not os.path.exists(IMG_PROGRESS_CSV):
#             self._print_error(f"Progress CSV not found: {IMG_PROGRESS_CSV}")
#             return self._fresh_generation(df, filepath)
        
#         progress_df = pd.read_csv(IMG_PROGRESS_CSV)
#         self._print_info(f"Loaded {len(progress_df)} rows from progress file")
        
#         if len(progress_df) != len(df):
#             self._print_error(f"Progress CSV has {len(progress_df)} rows, expected {len(df)}")
#             return self._fresh_generation(df, filepath)
        
#         # Use progress data
#         df = progress_df
        
#         # Reload used queries to avoid duplicates
#         existing_queries = df["img_desc"].fillna("").astype(str).tolist()
#         valid_count = 0
#         for query in existing_queries:
#             if query and isinstance(query, str) and len(query) >= 10:
#                 self.search_agent._used_queries.add(query)
#                 valid_count += 1
        
#         self._print_info(f"Loaded {valid_count} existing queries for dedup")
        
#         # Continue filling queries
#         self._print_step("CONTINUING QUERY GENERATION", "")
        
#         t0 = time.time()
#         filled = self._fill_all_queries_with_progress(df, start_batch=completed_batches, filepath=filepath)
#         elapsed = time.time() - t0
        
#         self._print_success(f"Query generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        
#         # Save final
#         self._print_step("SAVING UPDATED DATASET", "")
        
#         self._save_final(filled, filepath)
#         self._cleanup_progress()
        
#         return filled

#     def _fill_all_queries_with_progress(
#         self,
#         df: pd.DataFrame,
#         start_batch: int = 0,
#         filepath: str = None,
#     ) -> pd.DataFrame:
#         """Fill img_desc with advanced progress bar."""
        
#         all_rows = df.to_dict("records")
#         total = len(all_rows)
#         n_batches = -(-total // IMG_BATCH_SIZE)
        
#         # Initialize tracker
#         tracker = ImageProgressTracker(total, n_batches, start_batch)
        
#         # Pre-fill completed rows count
#         completed_before = sum(
#             1 for r in all_rows
#             if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= 10
#         )
#         tracker.completed_rows = completed_before
        
#         filled_rows = all_rows.copy()
        
#         if RICH_AVAILABLE:
#             filled_df = self._fill_with_rich_progress(
#                 all_rows, filled_rows, start_batch, n_batches, total, tracker, df, filepath
#             )
#         else:
#             filled_df = self._fill_with_basic_progress(
#                 all_rows, filled_rows, start_batch, n_batches, total, tracker, df, filepath
#             )
        
#         return filled_df

#     def _fill_with_rich_progress(
#         self,
#         all_rows: List[dict],
#         filled_rows: List[dict],
#         start_batch: int,
#         n_batches: int,
#         total: int,
#         tracker: ImageProgressTracker,
#         original_df: pd.DataFrame,
#         filepath: str,
#     ) -> pd.DataFrame:
#         """Fill queries with rich progress display."""
        
#         console = Console()
        
#         # Create progress bar
#         progress = Progress(
#             SpinnerColumn(),
#             TextColumn("[bold blue]{task.description}"),
#             BarColumn(bar_width=50, complete_style="green", finished_style="bright_green"),
#             TextColumn("[bold]{task.percentage:>5.1f}%"),
#             TextColumn("•"),
#             MofNCompleteColumn(),
#             TextColumn("•"),
#             TimeElapsedColumn(),
#             TextColumn("→"),
#             TimeRemainingColumn(),
#             console=console,
#             expand=False,
#             transient=False,
#         )
        
#         # Calculate already completed
#         already_completed = sum(
#             1 for r in all_rows
#             if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= 10
#         )
        
#         with progress:
#             # Add tasks
#             overall_task = progress.add_task(
#                 "[cyan]Search Queries",
#                 total=total,
#                 completed=already_completed,
#             )
#             batch_task = progress.add_task(
#                 "[yellow]Current Batch",
#                 total=IMG_BATCH_SIZE,
#                 completed=0,
#             )
            
#             for b in range(start_batch, n_batches):
#                 start = b * IMG_BATCH_SIZE
#                 end = min(start + IMG_BATCH_SIZE, total)
#                 batch_rows = all_rows[start:end]
#                 batch_size = len(batch_rows)
                
#                 # Check if batch is done
#                 batch_already_done = all(
#                     row.get("img_desc") and len(str(row.get("img_desc", ""))) >= 10
#                     for row in batch_rows
#                 )
                
#                 if batch_already_done:
#                     progress.update(overall_task, advance=batch_size)
#                     continue
                
#                 # Reset batch progress
#                 progress.reset(batch_task, total=batch_size, completed=0)
#                 progress.update(batch_task, description=f"[yellow]Batch {b+1}/{n_batches}")
                
#                 batch_start_time = time.time()
                
#                 # Generate queries using SearchAgent
#                 result = self.search_agent.generate_queries(
#                     batch_rows, batch_label=f"img_batch{b+1:02d}"
#                 )
                
#                 batch_time = time.time() - batch_start_time
                
#                 # Update filled_rows
#                 for i, row in enumerate(result):
#                     filled_rows[start + i] = row
                
#                 # Update progress
#                 progress.update(batch_task, completed=batch_size)
#                 progress.update(overall_task, advance=batch_size)
#                 tracker.update(b + 1, batch_time, batch_size)
                
#                 # Save progress
#                 self._save_progress(filled_rows, b + 1, n_batches, filepath)
                
#                 # Print batch summary
#                 filled_count = sum(
#                     1 for row in result
#                     if row.get("img_desc") and len(str(row.get("img_desc", ""))) >= 10
#                 )
#                 console.print(
#                     f"    ✓ Batch {b+1}/{n_batches}: {batch_time:.1f}s | "
#                     f"Filled: {filled_count}/{batch_size} queries",
#                     style="dim"
#                 )
                
#                 if b < n_batches - 1:
#                     time.sleep(DELAY_BETWEEN_BATCHES)
        
#         # Print final summary
#         self._print_final_summary(tracker)
        
#         filled_df = pd.DataFrame(filled_rows)
#         # Preserve column order from original
#         filled_df = filled_df[original_df.columns]
#         return filled_df

#     def _fill_with_basic_progress(
#         self,
#         all_rows: List[dict],
#         filled_rows: List[dict],
#         start_batch: int,
#         n_batches: int,
#         total: int,
#         tracker: ImageProgressTracker,
#         original_df: pd.DataFrame,
#         filepath: str,
#     ) -> pd.DataFrame:
#         """Fill queries with basic progress (no rich library)."""
        
#         print(f"\n  Rows: {total} | Batch: {IMG_BATCH_SIZE} | "
#               f"Batches: {n_batches} | Starting from: {start_batch + 1}")
#         print()
        
#         for b in range(start_batch, n_batches):
#             start = b * IMG_BATCH_SIZE
#             end = min(start + IMG_BATCH_SIZE, total)
#             batch_rows = all_rows[start:end]
#             batch_size = len(batch_rows)
            
#             # Check if batch is done
#             batch_already_done = all(
#                 row.get("img_desc") and len(str(row.get("img_desc", ""))) >= 10
#                 for row in batch_rows
#             )
            
#             if batch_already_done:
#                 continue
            
#             pct = end / total * 100
#             bar_width = 40
#             filled_width = int(bar_width * (end / total))
#             bar = "█" * filled_width + "░" * (bar_width - filled_width)
            
#             print(f"\r  [{bar}] {pct:5.1f}% | Batch {b+1}/{n_batches} | "
#                   f"Rows {start+1}-{end}", end="", flush=True)
            
#             batch_start_time = time.time()
            
#             result = self.search_agent.generate_queries(
#                 batch_rows, batch_label=f"img_batch{b+1:02d}"
#             )
            
#             batch_time = time.time() - batch_start_time
            
#             for i, row in enumerate(result):
#                 filled_rows[start + i] = row
            
#             tracker.update(b + 1, batch_time, batch_size)
#             self._save_progress(filled_rows, b + 1, n_batches, filepath)
            
#             print(f" ✓ {batch_time:.1f}s")
            
#             if b < n_batches - 1:
#                 time.sleep(DELAY_BETWEEN_BATCHES)
        
#         print()
#         filled_df = pd.DataFrame(filled_rows)
#         filled_df = filled_df[original_df.columns]
#         return filled_df

#     def _print_final_summary(self, tracker: ImageProgressTracker):
#         """Print a beautiful final summary."""
#         if not RICH_AVAILABLE:
#             return
        
#         elapsed = time.time() - tracker.start_time
#         avg_batch = sum(tracker.batch_times) / len(tracker.batch_times) if tracker.batch_times else 0
#         speed = tracker.completed_rows / elapsed if elapsed > 0 else 0
        
#         table = Table(
#             title="📊 Query Generation Complete",
#             box=box.DOUBLE_EDGE,
#             show_header=True,
#             header_style="bold magenta",
#         )
#         table.add_column("Metric", style="cyan", justify="left")
#         table.add_column("Value", style="green", justify="right")
        
#         table.add_row("Total Queries", f"{tracker.completed_rows:,}")
#         table.add_row("Total Batches", str(tracker.current_batch))
#         table.add_row("Total Time", tracker._format_time(elapsed))
#         table.add_row("Avg Batch Time", f"{avg_batch:.1f}s")
#         table.add_row("Speed", f"{speed * 60:.1f} queries/min")
#         table.add_row("Retries", str(tracker.retries))
#         table.add_row("Errors", str(tracker.errors))
        
#         self.console.print()
#         self.console.print(table)
#         self.console.print()

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  HELPER PRINT METHODS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _print_step(self, title: str, subtitle: str = ""):
#         """Print a step header."""
#         if RICH_AVAILABLE:
#             text = f"[bold cyan]{title}[/bold cyan]"
#             if subtitle:
#                 text += f" [dim]({subtitle})[/dim]"
#             self.console.print()
#             self.console.rule(text, style="cyan")
#         else:
#             print(f"\n{'━'*60}")
#             print(f"  {title}" + (f" ({subtitle})" if subtitle else ""))
#             print(f"{'━'*60}")

#     def _print_success(self, message: str):
#         if RICH_AVAILABLE:
#             self.console.print(f"  [green]✓[/green] {message}")
#         else:
#             print(f"  ✓ {message}")

#     def _print_error(self, message: str):
#         if RICH_AVAILABLE:
#             self.console.print(f"  [red]✗[/red] {message}")
#         else:
#             print(f"  ✗ {message}")

#     def _print_info(self, message: str):
#         if RICH_AVAILABLE:
#             self.console.print(f"  [blue]ℹ[/blue] {message}")
#         else:
#             print(f"  💾 {message}")

#     def _print_warning(self, message: str):
#         if RICH_AVAILABLE:
#             self.console.print(f"  [yellow]⚠[/yellow] {message}")
#         else:
#             print(f"  ⚠ {message}")

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  PROGRESS SAVE / LOAD
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _save_progress(
#         self,
#         filled_rows: List[dict],
#         completed_batches: int,
#         total_batches: int,
#         filepath: str,
#     ):
#         """Save current progress to disk."""
        
#         completed_queries = sum(
#             1 for r in filled_rows
#             if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= 10
#         )
        
#         progress = {
#             "filepath": filepath,
#             "completed_batches": completed_batches,
#             "total_batches": total_batches,
#             "completed_queries": completed_queries,
#             "total_rows": len(filled_rows),
#             "batch_size": IMG_BATCH_SIZE,
#             "model": MODEL_NAME,
#             "profile": ACTIVE_PROFILE,
#             "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         }
        
#         with open(IMG_PROGRESS_FILE, "w") as f:
#             json.dump(progress, f, indent=2)
        
#         df = pd.DataFrame(filled_rows)
#         df.to_csv(IMG_PROGRESS_CSV, index=False)

#     def _load_progress(self, filepath: str) -> Optional[dict]:
#         """Load saved progress if it exists."""
        
#         if not os.path.exists(IMG_PROGRESS_FILE):
#             return None
        
#         try:
#             with open(IMG_PROGRESS_FILE, "r") as f:
#                 progress = json.load(f)
            
#             # Check if progress is for the same file
#             if progress.get("filepath") != filepath:
#                 self._print_warning(
#                     f"Saved progress is for different file: {progress.get('filepath')}"
#                 )
#                 return None
            
#             completed = progress.get("completed_batches", 0)
#             total = progress.get("total_batches", 0)
            
#             if completed >= total:
#                 self._print_warning("Progress shows all batches complete. Starting fresh...")
#                 return None
            
#             if not os.path.exists(IMG_PROGRESS_CSV):
#                 self._print_warning("Progress JSON exists but CSV missing. Starting fresh...")
#                 return None
            
#             # Display saved progress
#             self._display_saved_progress(progress, completed, total)
            
#             # Ask user
#             print(f"\n  Options:")
#             print(f"    [R] Resume from batch {completed + 1}")
#             print(f"    [F] Fresh start (discard progress)")
#             print(f"    [Enter] Resume (default)")
            
#             try:
#                 choice = input("\n  Your choice [R/F]: ").strip().upper()
#             except (EOFError, KeyboardInterrupt):
#                 choice = "R"
            
#             if choice == "F":
#                 print(f"  Starting fresh...")
#                 return None
            
#             return progress
            
#         except (json.JSONDecodeError, KeyError) as e:
#             self._print_warning(f"Corrupted progress file: {e}")
#             return None

#     def _display_saved_progress(self, progress: dict, completed: int, total: int):
#         """Display the saved progress info."""
        
#         if RICH_AVAILABLE:
#             table = Table(
#                 title="💾 Saved Image Query Progress Found",
#                 box=box.DOUBLE_EDGE,
#                 show_header=False,
#                 title_style="bold yellow",
#             )
#             table.add_column("Field", style="cyan")
#             table.add_column("Value", style="green")
            
#             table.add_row("Batches", f"{completed}/{total}")
#             table.add_row("Queries", f"{progress.get('completed_queries', '?')}/{progress.get('total_rows', '?')}")
#             table.add_row("File", os.path.basename(progress.get('filepath', '?')))
#             table.add_row("Last Saved", str(progress.get('last_saved', '?')))
            
#             self.console.print()
#             self.console.print(table)
#         else:
#             batches_str = f"{completed}/{total}"
#             queries_str = f"{progress.get('completed_queries', '?')}/{progress.get('total_rows', '?')}"
            
#             print(f"\n  ╔{'═'*50}╗")
#             print(f"  ║  SAVED IMG QUERY PROGRESS                      ║")
#             print(f"  ╠{'═'*50}╣")
#             print(f"  ║  Batches : {batches_str}{' '*(38-len(batches_str))}║")
#             print(f"  ║  Queries : {queries_str}{' '*(38-len(queries_str))}║")
#             print(f"  ║  File    : {os.path.basename(progress.get('filepath', '?'))[:38]:<38}║")
#             print(f"  ║  Saved   : {progress.get('last_saved', '?')[:38]:<38}║")
#             print(f"  ╚{'═'*50}╝")

#     def _cleanup_progress(self):
#         """Remove progress files after successful completion."""
#         for f in [IMG_PROGRESS_FILE, IMG_PROGRESS_CSV]:
#             if os.path.exists(f):
#                 os.remove(f)
#                 self._print_info(f"Cleaned up: {os.path.basename(f)}")

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  SAVE FINAL
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _save_final(self, df: pd.DataFrame, original_filepath: str) -> str:
#         """Save the updated dataset with img_desc filled."""
        
#         # Overwrite the original file
#         df.to_csv(original_filepath, index=False)
#         size_kb = os.path.getsize(original_filepath) / 1024
        
#         # Count filled queries
#         filled_count = sum(
#             1 for _, row in df.iterrows()
#             if row.get("img_desc") and len(str(row.get("img_desc", ""))) >= 10
#         )
        
#         if RICH_AVAILABLE:
#             panel = Panel(
#                 f"[green]{original_filepath}[/green]\n"
#                 f"[dim]{len(df):,} rows × {len(df.columns)} cols ({size_kb:.1f} KB)[/dim]\n"
#                 f"[cyan]img_desc filled: {filled_count}/{len(df)}[/cyan]",
#                 title="✓ Dataset Updated",
#                 border_style="green",
#             )
#             self.console.print(panel)
#         else:
#             print(f"\n  ✓ UPDATED: {original_filepath}")
#             print(f"    {len(df)} rows × {len(df.columns)} cols ({size_kb:.1f} KB)")
#             print(f"    img_desc filled: {filled_count}/{len(df)}")
        
#         return original_filepath

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  BANNER
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _banner(self, filepath: str):
#         mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
        
#         if RICH_AVAILABLE:
#             table = Table(
#                 title="🔍 Image Query Generator v1",
#                 box=box.DOUBLE_EDGE,
#                 show_header=False,
#                 title_style="bold magenta",
#                 border_style="blue",
#             )
#             table.add_column("Setting", style="cyan")
#             table.add_column("Value", style="green")
            
#             table.add_row("API Mode", mode)
#             table.add_row("Model", MODEL_NAME)
#             table.add_row("Profile", ACTIVE_PROFILE)
#             table.add_row("Input File", os.path.basename(filepath))
#             table.add_row("Batch Size", str(IMG_BATCH_SIZE))
#             table.add_row("Resume", "✓ Enabled")
            
#             self.console.print()
#             self.console.print(table)
#         else:
#             print(f"\n{'#'*60}")
#             print(f"  IMAGE QUERY GENERATOR v1 (with Advanced Progress)")
#             print(f"{'#'*60}")
#             print(f"  API Mode     : {mode}")
#             print(f"  Model        : {MODEL_NAME}")
#             print(f"  Profile      : {ACTIVE_PROFILE}")
#             print(f"  Input File   : {os.path.basename(filepath)}")
#             print(f"  Batch size   : {IMG_BATCH_SIZE}")
#             print(f"  Resume       : ENABLED")
#             print(f"{'#'*60}")


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #  MAIN ENTRY POINT
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# if __name__ == "__main__":
#     import sys
    
#     # Parse arguments
#     filepath = None
#     force_restart = False
    
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "--fresh":
#             force_restart = True
#             if len(sys.argv) > 2:
#                 filepath = sys.argv[2]
#         else:
#             filepath = sys.argv[1]
    
#     # Run generator
#     gen = ImageDescGenerator()
#     result = gen.generate(filepath=filepath, force_restart=force_restart)
    
#     if not result.empty:
#         print(f"\n✓ Success! Dataset updated with img_desc column.")
#     else:
#         print(f"\n✗ Failed to generate queries.")
#         sys.exit(1)


# img_generator.py
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Advanced Image Search Query Generator v3.0                            ║
║  ──────────────────────────────────────────────────────────────────     ║
║  Orchestrator: load dataset → generate img_desc → validate → save     ║
║                                                                        ║
║  Features:                                                             ║
║   • Full resume support with versioned checkpoints                     ║
║   • Live Rich dashboard with real-time metrics                         ║
║   • Quality scoring & validation pipeline                              ║
║   • Adaptive batch sizing based on API latency                         ║
║   • Graceful shutdown (Ctrl+C safe)                                    ║
║   • Structured audit logging to file                                   ║
║   • Post-generation analytics report                                   ║
║   • Smart deduplication with similarity detection                      ║
║   • Pre-flight system & dependency checks                              ║
║   • Multiple export formats (CSV, JSON, Parquet)                       ║
║   • Dry-run mode for previewing workload                               ║
║   • Memory-efficient streaming for large datasets                      ║
║   • Retry with exponential backoff + jitter                            ║
║   • Per-row error isolation (one bad row won't kill a batch)           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import sys
import csv
import time
import json
import math
import glob
import signal
import hashlib
import logging
import platform
import traceback
import threading
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Tuple, Any,
    Callable, Set, Iterator, Union
)
from collections import Counter, defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
import numpy as np

from config import (
    OUTPUT_DIR, USE_OPENAI_SDK,
    MODEL_NAME, ACTIVE_PROFILE,
    MAX_RETRIES, DELAY_BETWEEN_BATCHES,
    TEXT_BATCH_SIZE,
)
from search_agent import SearchAgent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RICH IMPORTS — graceful fallback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from rich.console import Console, Group
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn, TransferSpeedColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.rule import Rule
    from rich.markup import escape
    from rich import box
    from rich.logging import RichHandler
    from rich.spinner import Spinner
    from rich.emoji import Emoji
    from rich.style import Style

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSION = "3.0.0"
APP_NAME = "Image Query Generator"

# Batch configuration
IMG_BATCH_SIZE = TEXT_BATCH_SIZE
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = IMG_BATCH_SIZE * 4
ADAPTIVE_BATCH_ENABLED = True

# Quality thresholds
MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 500
IDEAL_QUERY_LENGTH_RANGE = (20, 200)
QUALITY_SCORE_THRESHOLD = 0.5          # minimum acceptable quality 0-1
DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # Jaccard threshold

# Retry configuration
RETRY_BASE_DELAY = 2.0     # seconds
RETRY_MAX_DELAY = 120.0    # seconds
RETRY_JITTER_FACTOR = 0.3  # ±30 % jitter

# File paths
IMG_PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_img_progress.json")
IMG_PROGRESS_CSV = os.path.join(OUTPUT_DIR, "_img_progress_data.csv")
IMG_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "_img_checkpoints")
IMG_AUDIT_LOG = os.path.join(OUTPUT_DIR, "_img_audit.log")
IMG_ANALYTICS_REPORT = os.path.join(OUTPUT_DIR, "_img_analytics_report.json")

# Required source columns
REQUIRED_COLUMNS = ["object_detected", "theme", "emotion", "text"]

# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RowStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    LOW_QUALITY = auto()
    DUPLICATE = auto()
    ERROR = auto()
    SKIPPED = auto()


class BatchStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    PARTIAL = auto()
    FAILED = auto()
    SKIPPED = auto()


class ExportFormat(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "xlsx"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class QualityScore:
    """Score breakdown for a single generated query."""
    length_score: float = 0.0       # 0-1 based on ideal length range
    specificity_score: float = 0.0  # 0-1 based on keyword richness
    relevance_score: float = 0.0    # 0-1 alignment with source row
    uniqueness_score: float = 0.0   # 0-1 no near-duplicates
    overall: float = 0.0           # weighted average

    WEIGHTS = {
        "length": 0.20,
        "specificity": 0.25,
        "relevance": 0.30,
        "uniqueness": 0.25,
    }

    def compute_overall(self) -> float:
        w = self.WEIGHTS
        self.overall = (
            self.length_score * w["length"]
            + self.specificity_score * w["specificity"]
            + self.relevance_score * w["relevance"]
            + self.uniqueness_score * w["uniqueness"]
        )
        return self.overall


@dataclass
class RowResult:
    """Result for a single row's query generation."""
    row_index: int
    status: RowStatus = RowStatus.PENDING
    query: str = ""
    quality: QualityScore = field(default_factory=QualityScore)
    attempts: int = 0
    error_message: str = ""
    generation_time_ms: float = 0.0


@dataclass
class BatchResult:
    """Aggregated result for a batch."""
    batch_id: int
    status: BatchStatus = BatchStatus.PENDING
    rows: List[RowResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    retry_count: int = 0

    @property
    def elapsed(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.rows if r.status == RowStatus.SUCCESS)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.rows if r.status == RowStatus.ERROR)

    @property
    def success_rate(self) -> float:
        if not self.rows:
            return 0.0
        return self.success_count / len(self.rows)


@dataclass
class SessionMetrics:
    """Aggregated session-level metrics."""
    session_id: str = ""
    started_at: str = ""
    total_rows: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    rows_skipped: int = 0
    rows_low_quality: int = 0
    rows_duplicate: int = 0
    total_retries: int = 0
    total_api_calls: int = 0
    total_time_s: float = 0.0
    avg_batch_time_s: float = 0.0
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0
    queries_per_minute: float = 0.0
    batch_times: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    error_categories: Dict[str, int] = field(default_factory=dict)
    theme_distribution: Dict[str, int] = field(default_factory=dict)
    emotion_distribution: Dict[str, int] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    adaptive_batch_adjustments: int = 0

    def compute_derived(self):
        if self.batch_times:
            self.avg_batch_time_s = sum(self.batch_times) / len(self.batch_times)
        if self.quality_scores:
            self.avg_quality_score = sum(self.quality_scores) / len(self.quality_scores)
            self.min_quality_score = min(self.quality_scores)
            self.max_quality_score = max(self.quality_scores)
        if self.total_time_s > 0:
            self.queries_per_minute = (self.rows_succeeded / self.total_time_s) * 60


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _setup_logger(name: str = "img_generator") -> logging.Logger:
    """Create structured logger with file + optional Rich console handler."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # already set up

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # File handler — always active
    fh = logging.FileHandler(IMG_AUDIT_LOG, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT))
    logger.addHandler(fh)

    # Console handler
    if RICH_AVAILABLE:
        rh = RichHandler(
            console=Console(stderr=True),
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            level=logging.WARNING,
        )
        logger.addHandler(rh)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT))
        logger.addHandler(ch)

    return logger


logger = _setup_logger()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_time(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


def _format_bytes(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _file_hash(path: str, algo: str = "sha256", chunk: int = 8192) -> str:
    """Quick hash of a file for integrity checks."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()[:16]


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _get_memory_mb() -> float:
    """Current process RSS in MB (cross-platform best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            # Linux fallback
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
    return 0.0


def _retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
    jitter: float = RETRY_JITTER_FACTOR,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs,
) -> Any:
    """
    Execute *func* with exponential backoff + jitter.
    Calls on_retry(attempt, exception, delay) before each sleep.
    """
    import random

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1.0 + random.uniform(-jitter, jitter)
            if on_retry:
                on_retry(attempt, exc, delay)
            logger.warning(
                "Retry %d/%d after %.1fs — %s: %s",
                attempt, max_retries, delay, type(exc).__name__, exc,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"All {max_retries} retries exhausted. Last error: {last_exc}"
    ) from last_exc


def _sparkline(values: List[float], width: int = 20) -> str:
    """Tiny Unicode sparkline for terminal."""
    if not values:
        return "—"
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn or 1.0
    # down-sample to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    return "".join(blocks[min(int((v - mn) / rng * 8), 8)] for v in sampled)


def _truncate(text: str, length: int = 60, suffix: str = "…") -> str:
    if len(text) <= length:
        return text
    return text[: length - len(suffix)] + suffix


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QUALITY ANALYZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QueryQualityAnalyzer:
    """
    Scores generated image-search queries on multiple dimensions:
    length, specificity, relevance to source row, uniqueness vs corpus.
    """

    # Common filler words that lower specificity
    STOPWORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "and", "but", "or", "nor", "not", "so", "yet", "for", "at",
        "by", "to", "of", "in", "on", "with", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "about", "up", "it", "its", "this", "that",
        "these", "those", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their",
        "what", "which", "who", "whom", "image", "photo", "picture",
        "show", "showing", "search", "find", "looking",
    }

    # Descriptive words that boost specificity
    QUALITY_MARKERS: Set[str] = {
        "vibrant", "serene", "dramatic", "cinematic", "minimalist",
        "abstract", "detailed", "realistic", "surreal", "vintage",
        "modern", "rustic", "elegant", "panoramic", "close-up",
        "macro", "wide-angle", "aerial", "underwater", "silhouette",
        "golden-hour", "sunset", "sunrise", "moody", "ethereal",
        "textured", "layered", "contrasting", "symmetrical",
        "professional", "candid", "editorial", "conceptual",
        "illustrative", "artistic", "photorealistic", "hdr",
        "bokeh", "long-exposure", "high-contrast", "low-key",
    }

    def __init__(self):
        self._corpus: List[str] = []
        self._corpus_word_sets: List[Set[str]] = []
        self._lock = threading.Lock()

    def add_to_corpus(self, query: str):
        """Thread-safe addition to duplicate-detection corpus."""
        with self._lock:
            self._corpus.append(query.lower())
            self._corpus_word_sets.append(set(query.lower().split()))

    def score(self, query: str, source_row: dict) -> QualityScore:
        """Full quality assessment of a generated query."""
        qs = QualityScore()

        if not query or not isinstance(query, str):
            return qs

        clean = query.strip()

        # ── length ──
        qs.length_score = self._score_length(clean)

        # ── specificity ──
        qs.specificity_score = self._score_specificity(clean)

        # ── relevance to source row ──
        qs.relevance_score = self._score_relevance(clean, source_row)

        # ── uniqueness vs corpus ──
        qs.uniqueness_score = self._score_uniqueness(clean)

        qs.compute_overall()
        return qs

    # ── individual scorers ──────────────────────────────────────

    def _score_length(self, query: str) -> float:
        length = len(query)
        if length < MIN_QUERY_LENGTH:
            return length / MIN_QUERY_LENGTH * 0.4   # harsh penalty
        lo, hi = IDEAL_QUERY_LENGTH_RANGE
        if lo <= length <= hi:
            return 1.0
        if length < lo:
            return 0.4 + 0.6 * ((length - MIN_QUERY_LENGTH) / max(lo - MIN_QUERY_LENGTH, 1))
        # longer than ideal but still acceptable
        if length <= MAX_QUERY_LENGTH:
            return max(0.5, 1.0 - (length - hi) / max(MAX_QUERY_LENGTH - hi, 1) * 0.5)
        return 0.3  # way too long

    def _score_specificity(self, query: str) -> float:
        words = re.findall(r"[a-zA-Z]+", query.lower())
        if not words:
            return 0.0
        content_words = [w for w in words if w not in self.STOPWORDS]
        ratio = len(content_words) / len(words) if words else 0
        marker_hits = sum(1 for w in content_words if w in self.QUALITY_MARKERS)
        marker_bonus = min(marker_hits * 0.08, 0.3)
        unique_ratio = len(set(content_words)) / len(content_words) if content_words else 0
        return min(1.0, ratio * 0.5 + unique_ratio * 0.3 + marker_bonus + 0.1)

    def _score_relevance(self, query: str, source: dict) -> float:
        """Check that query reflects the row's theme / emotion / objects."""
        query_lower = query.lower()
        source_terms: List[str] = []
        for col in ("object_detected", "theme", "emotion"):
            val = str(source.get(col, ""))
            source_terms.extend(re.findall(r"[a-zA-Z]{3,}", val.lower()))
        if not source_terms:
            return 0.5  # can't evaluate — neutral
        hits = sum(1 for t in source_terms if t in query_lower)
        return min(1.0, hits / max(len(set(source_terms)), 1))

    def _score_uniqueness(self, query: str) -> float:
        """1.0 = completely unique vs existing corpus."""
        query_words = set(query.lower().split())
        with self._lock:
            if not self._corpus_word_sets:
                return 1.0
            max_sim = 0.0
            for ws in self._corpus_word_sets:
                if not ws:
                    continue
                inter = len(query_words & ws)
                union = len(query_words | ws)
                sim = inter / union if union else 0
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= 1.0:
                    break
            return max(0.0, 1.0 - max_sim)

    def corpus_size(self) -> int:
        with self._lock:
            return len(self._corpus)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROGRESS TRACKER (Advanced)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImageProgressTracker:
    """Thread-safe progress tracker with rich metric aggregation."""

    def __init__(
        self,
        total_rows: int,
        total_batches: int,
        start_batch: int = 0,
    ):
        self.total_rows = total_rows
        self.total_batches = total_batches
        self.start_batch = start_batch
        self.current_batch = start_batch
        self.completed_rows = 0
        self.start_time = time.time()

        self.batch_results: List[BatchResult] = []
        self.batch_times: List[float] = []
        self.quality_scores: List[float] = []
        self.errors = 0
        self.retries = 0
        self.rows_succeeded = 0
        self.rows_failed = 0
        self.rows_low_quality = 0
        self.rows_duplicate = 0
        self.api_calls = 0
        self.adaptive_adjustments = 0
        self.current_batch_size = IMG_BATCH_SIZE
        self.peak_memory_mb = 0.0
        self.error_categories: Dict[str, int] = defaultdict(int)

        self._lock = threading.Lock()
        self.console = Console() if RICH_AVAILABLE else None

    # ── thread-safe updaters ─────────────────────────────────

    def update_batch(self, result: BatchResult):
        with self._lock:
            self.batch_results.append(result)
            self.batch_times.append(result.elapsed)
            self.current_batch = result.batch_id
            self.retries += result.retry_count
            self.api_calls += 1 + result.retry_count
            for rr in result.rows:
                if rr.status == RowStatus.SUCCESS:
                    self.rows_succeeded += 1
                    self.quality_scores.append(rr.quality.overall)
                elif rr.status == RowStatus.ERROR:
                    self.rows_failed += 1
                    self.errors += 1
                elif rr.status == RowStatus.LOW_QUALITY:
                    self.rows_low_quality += 1
                elif rr.status == RowStatus.DUPLICATE:
                    self.rows_duplicate += 1
            self.completed_rows = (
                self.rows_succeeded + self.rows_failed
                + self.rows_low_quality + self.rows_duplicate
            )
            mem = _get_memory_mb()
            if mem > self.peak_memory_mb:
                self.peak_memory_mb = mem

    def add_error_category(self, category: str):
        with self._lock:
            self.error_categories[category] += 1

    def record_adaptive_adjustment(self, new_size: int):
        with self._lock:
            self.adaptive_adjustments += 1
            self.current_batch_size = new_size

    # ── derived stats ────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_batch_time(self) -> float:
        with self._lock:
            return (sum(self.batch_times) / len(self.batch_times)) if self.batch_times else 0.0

    @property
    def speed_per_min(self) -> float:
        e = self.elapsed
        return (self.rows_succeeded / e * 60) if e > 0 else 0.0

    @property
    def eta_seconds(self) -> float:
        remaining = self.total_rows - self.completed_rows
        if self.completed_rows <= 0 or self.elapsed <= 0:
            return 0.0
        return (remaining / self.completed_rows) * self.elapsed

    @property
    def avg_quality(self) -> float:
        with self._lock:
            return (sum(self.quality_scores) / len(self.quality_scores)) if self.quality_scores else 0.0

    @property
    def success_rate(self) -> float:
        total = self.rows_succeeded + self.rows_failed
        return (self.rows_succeeded / total * 100) if total > 0 else 0.0

    def to_metrics(self) -> SessionMetrics:
        m = SessionMetrics(
            total_rows=self.total_rows,
            total_batches=self.total_batches,
            completed_batches=self.current_batch,
            rows_succeeded=self.rows_succeeded,
            rows_failed=self.rows_failed,
            rows_low_quality=self.rows_low_quality,
            rows_duplicate=self.rows_duplicate,
            total_retries=self.retries,
            total_api_calls=self.api_calls,
            total_time_s=self.elapsed,
            batch_times=list(self.batch_times),
            quality_scores=list(self.quality_scores),
            error_categories=dict(self.error_categories),
            peak_memory_mb=self.peak_memory_mb,
            adaptive_batch_adjustments=self.adaptive_adjustments,
        )
        m.compute_derived()
        return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADAPTIVE BATCH SIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveBatchSizer:
    """
    Adjusts batch size based on observed API latency & error rate.
    Tries to keep batch processing between a target window.
    """

    TARGET_BATCH_TIME_LOW = 10.0   # seconds
    TARGET_BATCH_TIME_HIGH = 60.0  # seconds
    ERROR_RATE_THRESHOLD = 0.20    # shrink if > 20 % errors

    def __init__(self, initial: int = IMG_BATCH_SIZE, enabled: bool = True):
        self.current = initial
        self.initial = initial
        self.enabled = enabled
        self.history: List[Tuple[int, float, float]] = []  # (size, time, error_rate)

    def suggest(self, last_time: float, error_rate: float) -> int:
        """Return suggested batch size for next iteration."""
        if not self.enabled:
            return self.current

        self.history.append((self.current, last_time, error_rate))

        if error_rate > self.ERROR_RATE_THRESHOLD:
            # too many errors — shrink aggressively
            new = max(MIN_BATCH_SIZE, self.current // 2)
        elif last_time < self.TARGET_BATCH_TIME_LOW and error_rate < 0.05:
            # fast & clean — can grow
            new = min(MAX_BATCH_SIZE, int(self.current * 1.5))
        elif last_time > self.TARGET_BATCH_TIME_HIGH:
            # slow — shrink gently
            new = max(MIN_BATCH_SIZE, int(self.current * 0.75))
        else:
            new = self.current

        self.current = new
        return new


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHECKPOINT MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CheckpointManager:
    """
    Manages versioned checkpoints for crash recovery.
    Keeps last N checkpoints, rotates old ones.
    """

    MAX_CHECKPOINTS = 5

    def __init__(self, checkpoint_dir: str = IMG_CHECKPOINT_DIR):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(
        self,
        filled_rows: List[dict],
        completed_batches: int,
        total_batches: int,
        filepath: str,
        metrics: Optional[SessionMetrics] = None,
    ) -> str:
        """Save a versioned checkpoint. Returns checkpoint path."""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cp_name = f"cp_b{completed_batches:04d}_{ts}"
        cp_dir = os.path.join(self.dir, cp_name)
        os.makedirs(cp_dir, exist_ok=True)

        # Save data CSV
        df = pd.DataFrame(filled_rows)
        data_path = os.path.join(cp_dir, "data.csv")
        df.to_csv(data_path, index=False)

        # Save metadata
        meta = {
            "checkpoint_name": cp_name,
            "filepath": filepath,
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "total_rows": len(filled_rows),
            "completed_queries": sum(
                1 for r in filled_rows
                if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= MIN_QUERY_LENGTH
            ),
            "batch_size": IMG_BATCH_SIZE,
            "model": MODEL_NAME,
            "profile": ACTIVE_PROFILE,
            "created_at": datetime.now().isoformat(),
            "data_hash": _file_hash(data_path),
        }
        if metrics:
            meta["metrics"] = asdict(metrics)

        meta_path = os.path.join(cp_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        # Also write to the flat progress files for backward compat
        with open(IMG_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        df.to_csv(IMG_PROGRESS_CSV, index=False)

        # Rotate old checkpoints
        self._rotate()

        logger.info("Checkpoint saved: %s (%d rows)", cp_name, len(filled_rows))
        return cp_dir

    # def load_latest(self, filepath: str) -> Optional[Tuple[dict, pd.DataFrame]]:
    #     """Load the most recent valid checkpoint for the given filepath."""
    #     checkpoints = self._list_checkpoints()
    #     for cp_dir in reversed(checkpoints):
    #         try:
    #             meta_path = os.path.join(cp_dir, "meta.json")
    #             data_path = os.path.join(cp_dir, "data.csv")
    #             if not os.path.exists(meta_path) or not os.path.exists(data_path):
    #                 continue
    #             with open(meta_path, "r", encoding="utf-8") as f:
    #                 meta = json.load(f)
    #             if meta.get("filepath") != filepath:
    #                 continue
    #             # Verify integrity
    #             actual_hash = _file_hash(data_path)
    #             if meta.get("data_hash") and meta["data_hash"] != actual_hash:
    #                 logger.warning("Checkpoint %s has hash mismatch, skipping", cp_dir)
    #                 continue
    #             df = pd.read_csv(data_path)
    #             return meta, df
    #         except Exception as e:
    #             logger.warning("Failed to load checkpoint %s: %s", cp_dir, e)
    #             continue
    #     # Fallback to flat files
    #     return self._load_flat(filepath)

    def load_latest(self, filepath: str) -> Optional[Tuple[dict, pd.DataFrame]]:
        """Load the most recent valid checkpoint for the given filepath."""
        current_basename = os.path.basename(filepath)  # ← Add this

        checkpoints = self._list_checkpoints()
        for cp_dir in reversed(checkpoints):
            try:
                meta_path = os.path.join(cp_dir, "meta.json")
                data_path = os.path.join(cp_dir, "data.csv")
                if not os.path.exists(meta_path) or not os.path.exists(data_path):
                    continue
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # ═══ FIX: Compare basenames ═══
                saved_basename = os.path.basename(meta.get("filepath", ""))
                if saved_basename != current_basename:
                    continue
                
                # Verify integrity
                actual_hash = _file_hash(data_path)
                if meta.get("data_hash") and meta["data_hash"] != actual_hash:
                    logger.warning("Checkpoint %s has hash mismatch, skipping", cp_dir)
                    continue
                df = pd.read_csv(data_path)
                return meta, df
            except Exception as e:
                logger.warning("Failed to load checkpoint %s: %s", cp_dir, e)
                continue
            
        # Fallback to flat files
        return self._load_flat(filepath)

    # def _load_flat(self, filepath: str) -> Optional[Tuple[dict, pd.DataFrame]]:
    #     """Fallback: load from the flat progress files."""
    #     if not os.path.exists(IMG_PROGRESS_FILE) or not os.path.exists(IMG_PROGRESS_CSV):
    #         return None
    #     try:
    #         with open(IMG_PROGRESS_FILE, "r") as f:
    #             meta = json.load(f)
    #         if meta.get("filepath") != filepath:
    #             return None
    #         df = pd.read_csv(IMG_PROGRESS_CSV)
    #         return meta, df
    #     except Exception:
    #         return None
    def _load_flat(self, filepath: str) -> Optional[Tuple[dict, pd.DataFrame]]:
        """Fallback: load from the flat progress files."""
        if not os.path.exists(IMG_PROGRESS_FILE) or not os.path.exists(IMG_PROGRESS_CSV):
            return None
        try:
            with open(IMG_PROGRESS_FILE, "r") as f:
                meta = json.load(f)

            # ═══ FIX: Compare basenames, not full paths ═══
            saved_path = meta.get("filepath", "")
            saved_basename = os.path.basename(saved_path)
            current_basename = os.path.basename(filepath)

            if saved_basename != current_basename:
                logger.debug(
                    "Progress file mismatch: saved=%s, current=%s",
                    saved_basename, current_basename
                )
                return None

            df = pd.read_csv(IMG_PROGRESS_CSV)
            return meta, df
        except Exception as e:
            logger.warning("Failed to load flat progress: %s", e)
            return None
    
    def cleanup(self):
        """Remove all checkpoint data."""
        import shutil
        for cp in self._list_checkpoints():
            shutil.rmtree(cp, ignore_errors=True)
        for f in [IMG_PROGRESS_FILE, IMG_PROGRESS_CSV]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(self.dir):
            try:
                os.rmdir(self.dir)
            except OSError:
                pass

    def _list_checkpoints(self) -> List[str]:
        if not os.path.exists(self.dir):
            return []
        dirs = sorted(
            [
                os.path.join(self.dir, d)
                for d in os.listdir(self.dir)
                if os.path.isdir(os.path.join(self.dir, d)) and d.startswith("cp_")
            ]
        )
        return dirs

    def _rotate(self):
        checkpoints = self._list_checkpoints()
        import shutil
        while len(checkpoints) > self.MAX_CHECKPOINTS:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            logger.debug("Rotated old checkpoint: %s", oldest)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PRE-FLIGHT CHECKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PreFlightResult:
    passed: bool = True
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))
        if not ok:
            self.passed = False


def run_preflight_checks(filepath: str) -> PreFlightResult:
    """Validate environment, dependencies, config, and input data."""

    result = PreFlightResult()

    # 1. Python version
    py_ver = platform.python_version()
    ok = sys.version_info >= (3, 8)
    result.add("Python ≥ 3.8", ok, py_ver)

    # 2. pandas
    try:
        pd_ver = pd.__version__
        result.add("pandas installed", True, pd_ver)
    except Exception:
        result.add("pandas installed", False, "missing")

    # 3. Rich
    result.add("Rich UI library", RICH_AVAILABLE, "rich" if RICH_AVAILABLE else "not installed — basic UI")

    # 4. numpy
    try:
        np_ver = np.__version__
        result.add("numpy installed", True, np_ver)
    except Exception:
        result.add("numpy installed", False, "missing")

    # 5. Output directory
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_file = os.path.join(OUTPUT_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        result.add("Output directory writable", True, OUTPUT_DIR)
    except Exception as e:
        result.add("Output directory writable", False, str(e))

    # 6. Input file
    if filepath and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        result.add("Input file exists", True, f"{size_mb:.2f} MB")
        # Check CSV readability
        try:
            df_test = pd.read_csv(filepath, nrows=5)
            cols = df_test.columns.str.strip().tolist()
            missing = [c for c in REQUIRED_COLUMNS if c not in cols]
            result.add(
                "Required columns present",
                not missing,
                f"missing: {missing}" if missing else f"all {len(REQUIRED_COLUMNS)} found",
            )
        except Exception as e:
            result.add("CSV readable", False, str(e))
    else:
        result.add("Input file exists", False, filepath or "no path given")

    # 7. Model config
    result.add("Model configured", bool(MODEL_NAME), MODEL_NAME or "empty")
    result.add("Profile set", bool(ACTIVE_PROFILE), ACTIVE_PROFILE or "empty")

    # 8. SearchAgent instantiation
    try:
        _ = SearchAgent()
        result.add("SearchAgent OK", True, "instantiated")
    except Exception as e:
        result.add("SearchAgent OK", False, str(e)[:80])

    # 9. Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(OUTPUT_DIR)
        free_gb = free / (1024 ** 3)
        result.add("Disk space ≥ 100 MB", free_gb > 0.1, f"{free_gb:.2f} GB free")
    except Exception:
        result.add("Disk space check", True, "skipped")

    # 10. Memory
    mem = _get_memory_mb()
    if mem > 0:
        result.add("Memory usage", True, f"{mem:.0f} MB RSS")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST-GENERATION ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PostAnalytics:
    """Generate comprehensive analytics report after generation."""

    @staticmethod
    def generate_report(
        df: pd.DataFrame,
        metrics: SessionMetrics,
        quality_analyzer: QueryQualityAnalyzer,
    ) -> dict:

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "version": VERSION,
        }

        # ── dataset stats ──
        report["dataset"] = {
            "total_rows": len(df),
            "columns": list(df.columns),
        }

        # ── img_desc coverage ──
        filled = df["img_desc"].fillna("").astype(str)
        valid_mask = filled.str.len() >= MIN_QUERY_LENGTH
        report["coverage"] = {
            "filled": int(valid_mask.sum()),
            "empty": int((~valid_mask).sum()),
            "fill_rate": round(valid_mask.mean() * 100, 2),
        }

        # ── length stats ──
        lengths = filled[valid_mask].str.len()
        if len(lengths) > 0:
            report["query_lengths"] = {
                "mean": round(lengths.mean(), 1),
                "median": round(lengths.median(), 1),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std": round(lengths.std(), 1),
                "p10": round(lengths.quantile(0.10), 1),
                "p25": round(lengths.quantile(0.25), 1),
                "p75": round(lengths.quantile(0.75), 1),
                "p90": round(lengths.quantile(0.90), 1),
            }

        # ── word count ──
        word_counts = filled[valid_mask].str.split().str.len()
        if len(word_counts) > 0:
            report["word_counts"] = {
                "mean": round(word_counts.mean(), 1),
                "median": round(word_counts.median(), 1),
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
            }

        # ── quality ──
        if metrics.quality_scores:
            qs = metrics.quality_scores
            report["quality"] = {
                "mean": round(np.mean(qs), 4),
                "median": round(np.median(qs), 4),
                "std": round(np.std(qs), 4),
                "min": round(min(qs), 4),
                "max": round(max(qs), 4),
                "below_threshold": sum(1 for s in qs if s < QUALITY_SCORE_THRESHOLD),
                "above_0.8": sum(1 for s in qs if s >= 0.8),
                "distribution_histogram": PostAnalytics._histogram(qs, bins=10),
            }

        # ── distribution of themes/emotions ──
        for col in ("theme", "emotion", "object_detected"):
            if col in df.columns:
                counts = df[col].fillna("unknown").value_counts().head(20).to_dict()
                report[f"{col}_distribution"] = {str(k): int(v) for k, v in counts.items()}

        # ── performance ──
        report["performance"] = {
            "total_time": _format_time(metrics.total_time_s),
            "total_time_s": round(metrics.total_time_s, 2),
            "queries_per_minute": round(metrics.queries_per_minute, 2),
            "avg_batch_time_s": round(metrics.avg_batch_time_s, 2),
            "total_api_calls": metrics.total_api_calls,
            "total_retries": metrics.total_retries,
            "peak_memory_mb": round(metrics.peak_memory_mb, 1),
            "adaptive_adjustments": metrics.adaptive_batch_adjustments,
            "batch_time_sparkline": _sparkline(metrics.batch_times),
        }

        # ── errors ──
        report["errors"] = {
            "total": metrics.rows_failed,
            "categories": metrics.error_categories,
        }

        # ── top frequent words in queries ──
        all_words = " ".join(filled[valid_mask].tolist()).lower().split()
        word_freq = Counter(
            w for w in all_words
            if w not in QueryQualityAnalyzer.STOPWORDS and len(w) > 2
        )
        report["top_query_words"] = dict(word_freq.most_common(30))

        # ── sample queries ──
        samples = filled[valid_mask].sample(min(10, valid_mask.sum())).tolist()
        report["sample_queries"] = samples

        return report

    @staticmethod
    def _histogram(values: list, bins: int = 10) -> Dict[str, int]:
        counts, edges = np.histogram(values, bins=bins, range=(0, 1))
        result = {}
        for i in range(len(counts)):
            label = f"{edges[i]:.1f}-{edges[i+1]:.1f}"
            result[label] = int(counts[i])
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RICH UI RENDERER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RichUI:
    """
    All Rich-based rendering consolidated here.
    Falls back to plain text when Rich is unavailable.
    """

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    # ── generic helpers ──────────────────────────────────────

    def banner(self, filepath: str, total_rows: int, resume_from: int = 0):
        mode = "OpenAI SDK" if USE_OPENAI_SDK else "HTTP Requests"

        if RICH_AVAILABLE:
            table = Table(
                title=f"🔍 {APP_NAME} v{VERSION}",
                box=box.HEAVY_EDGE,
                show_header=False,
                title_style="bold magenta",
                border_style="bright_blue",
                padding=(0, 2),
            )
            table.add_column("Setting", style="cyan", min_width=22)
            table.add_column("Value", style="bright_green")

            table.add_row("API Mode", mode)
            table.add_row("Model", MODEL_NAME)
            table.add_row("Profile", ACTIVE_PROFILE)
            table.add_row("Input File", os.path.basename(filepath))
            table.add_row("Total Rows", f"{total_rows:,}")
            table.add_row("Batch Size", str(IMG_BATCH_SIZE))
            table.add_row("Adaptive Batching", "✓" if ADAPTIVE_BATCH_ENABLED else "✗")
            table.add_row("Quality Scoring", "✓ Enabled")
            table.add_row("Resume", "✓ Enabled")
            if resume_from > 0:
                table.add_row("Resuming from batch", str(resume_from + 1), style="bold yellow")
            table.add_row("Max Retries", str(MAX_RETRIES))
            table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            self.console.print()
            self.console.print(table)
        else:
            print(f"\n{'═'*66}")
            print(f"  {APP_NAME} v{VERSION}")
            print(f"{'═'*66}")
            print(f"  API Mode         : {mode}")
            print(f"  Model            : {MODEL_NAME}")
            print(f"  Profile          : {ACTIVE_PROFILE}")
            print(f"  Input File       : {os.path.basename(filepath)}")
            print(f"  Total Rows       : {total_rows:,}")
            print(f"  Batch Size       : {IMG_BATCH_SIZE}")
            print(f"  Adaptive Batch   : {'Yes' if ADAPTIVE_BATCH_ENABLED else 'No'}")
            print(f"  Resume           : Enabled")
            if resume_from > 0:
                print(f"  Resuming from    : batch {resume_from + 1}")
            print(f"  Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'═'*66}")

    def preflight_report(self, result: PreFlightResult):
        if RICH_AVAILABLE:
            table = Table(
                title="🛫 Pre-Flight Checks",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold white",
                border_style="dim",
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Check", style="cyan", min_width=28)
            table.add_column("Status", justify="center", width=8)
            table.add_column("Detail", style="dim")

            for i, (name, ok, detail) in enumerate(result.checks, 1):
                icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
                d = _truncate(detail, 50) if detail else ""
                table.add_row(str(i), name, icon, d)

            self.console.print()
            self.console.print(table)

            if result.passed:
                self.console.print(
                    Panel("[bold green]All checks passed ✓[/bold green]",
                          border_style="green", padding=(0, 2))
                )
            else:
                self.console.print(
                    Panel("[bold red]Some checks failed ✗  — review above[/bold red]",
                          border_style="red", padding=(0, 2))
                )
        else:
            print(f"\n  PRE-FLIGHT CHECKS")
            print(f"  {'─'*50}")
            for i, (name, ok, detail) in enumerate(result.checks, 1):
                icon = "✓" if ok else "✗"
                print(f"  {i:>2}. [{icon}] {name:<30} {detail}")
            status = "PASSED" if result.passed else "FAILED"
            print(f"  {'─'*50}")
            print(f"  Result: {status}\n")

    def step(self, title: str, subtitle: str = ""):
        if RICH_AVAILABLE:
            text = f"[bold cyan]{title}[/bold cyan]"
            if subtitle:
                text += f"  [dim]({subtitle})[/dim]"
            self.console.print()
            self.console.rule(text, style="cyan")
        else:
            print(f"\n{'━'*60}")
            full = f"  {title}" + (f" ({subtitle})" if subtitle else "")
            print(full)
            print(f"{'━'*60}")

    def success(self, msg: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [green]✓[/green] {msg}")
        else:
            print(f"  ✓ {msg}")

    def error(self, msg: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [red]✗[/red] {msg}")
        else:
            print(f"  ✗ {msg}")

    def info(self, msg: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [blue]ℹ[/blue] {msg}")
        else:
            print(f"  ℹ {msg}")

    def warning(self, msg: str):
        if RICH_AVAILABLE:
            self.console.print(f"  [yellow]⚠[/yellow] {msg}")
        else:
            print(f"  ⚠ {msg}")

    def dim(self, msg: str):
        if RICH_AVAILABLE:
            self.console.print(f"    {msg}", style="dim")
        else:
            print(f"    {msg}")

    # ── batch summary line ───────────────────────────────────

    def batch_summary(self, batch_result: BatchResult, tracker: ImageProgressTracker):
        b = batch_result
        sr = b.success_rate * 100
        avg_q = (
            sum(r.quality.overall for r in b.rows if r.status == RowStatus.SUCCESS)
            / max(b.success_count, 1)
        )
        color = "green" if sr >= 90 else "yellow" if sr >= 70 else "red"

        if RICH_AVAILABLE:
            self.console.print(
                f"    [dim]Batch {b.batch_id:>3}[/dim]  "
                f"[{color}]{b.success_count}/{len(b.rows)}[/{color}] ok  "
                f"[dim]|[/dim]  Q̄ {avg_q:.2f}  "
                f"[dim]|[/dim]  {b.elapsed:.1f}s  "
                f"[dim]|[/dim]  ETA {_format_time(tracker.eta_seconds)}  "
                f"[dim]|[/dim]  {tracker.speed_per_min:.0f} q/min"
            )
        else:
            print(
                f"    Batch {b.batch_id:>3}: "
                f"{b.success_count}/{len(b.rows)} ok | "
                f"Q̄={avg_q:.2f} | "
                f"{b.elapsed:.1f}s | "
                f"ETA {_format_time(tracker.eta_seconds)} | "
                f"{tracker.speed_per_min:.0f} q/min"
            )

    # ── saved-progress display ───────────────────────────────

    def display_saved_progress(self, meta: dict):
        completed = meta.get("completed_batches", 0)
        total = meta.get("total_batches", 0)
        queries = meta.get("completed_queries", "?")
        rows = meta.get("total_rows", "?")

        if RICH_AVAILABLE:
            table = Table(
                title="💾 Saved Progress Detected",
                box=box.DOUBLE_EDGE,
                show_header=False,
                title_style="bold yellow",
                border_style="yellow",
            )
            table.add_column("Field", style="cyan", min_width=18)
            table.add_column("Value", style="green")
            table.add_row("Batches completed", f"{completed}/{total}")
            table.add_row("Queries generated", f"{queries}/{rows}")
            table.add_row("File", os.path.basename(meta.get("filepath", "?")))
            table.add_row("Model", meta.get("model", "?"))
            table.add_row("Profile", meta.get("profile", "?"))
            table.add_row("Last saved", meta.get("created_at", meta.get("last_saved", "?")))
            if meta.get("data_hash"):
                table.add_row("Data hash", meta["data_hash"])
            self.console.print()
            self.console.print(table)
        else:
            print(f"\n  ╔{'═'*54}╗")
            print(f"  ║  💾 SAVED PROGRESS FOUND                            ║")
            print(f"  ╠{'═'*54}╣")
            print(f"  ║  Batches  : {completed}/{total}{' '*(40-len(f'{completed}/{total}'))}║")
            print(f"  ║  Queries  : {queries}/{rows}{' '*(40-len(f'{queries}/{rows}'))}║")
            print(f"  ║  Saved    : {str(meta.get('created_at','?'))[:40]:<40}║")
            print(f"  ╚{'═'*54}╝")

    # ── final summary dashboard ──────────────────────────────

    def final_dashboard(self, metrics: SessionMetrics):
        if RICH_AVAILABLE:
            self._rich_final_dashboard(metrics)
        else:
            self._plain_final_dashboard(metrics)

    def _rich_final_dashboard(self, m: SessionMetrics):
        # ── main metrics table ──
        main = Table(
            title="📊 Generation Complete",
            box=box.HEAVY_EDGE,
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            padding=(0, 1),
        )
        main.add_column("Metric", style="cyan", min_width=26)
        main.add_column("Value", style="bright_green", justify="right", min_width=20)

        main.add_row("✅ Rows succeeded", f"{m.rows_succeeded:,}")
        main.add_row("❌ Rows failed", f"{m.rows_failed:,}")
        main.add_row("⚠️  Low quality", f"{m.rows_low_quality:,}")
        main.add_row("🔁 Duplicates flagged", f"{m.rows_duplicate:,}")
        main.add_row("", "")
        main.add_row("Total batches", f"{m.completed_batches:,}")
        main.add_row("Total time", _format_time(m.total_time_s))
        main.add_row("Avg batch time", f"{m.avg_batch_time_s:.1f}s")
        main.add_row("Speed", f"{m.queries_per_minute:.1f} queries/min")
        main.add_row("", "")
        main.add_row("API calls", f"{m.total_api_calls:,}")
        main.add_row("Retries", f"{m.total_retries:,}")
        main.add_row("Adaptive adjustments", f"{m.adaptive_batch_adjustments:,}")
        main.add_row("Peak memory", f"{m.peak_memory_mb:.1f} MB")

        self.console.print()
        self.console.print(main)

        # ── quality table ──
        if m.quality_scores:
            qt = Table(
                title="🎯 Quality Scores",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold",
                border_style="green",
            )
            qt.add_column("Stat", style="cyan")
            qt.add_column("Value", style="green", justify="right")

            qt.add_row("Mean", f"{m.avg_quality_score:.4f}")
            qt.add_row("Min", f"{m.min_quality_score:.4f}")
            qt.add_row("Max", f"{m.max_quality_score:.4f}")
            qt.add_row("Std Dev", f"{np.std(m.quality_scores):.4f}")
            qt.add_row("≥ 0.8 (high)", f"{sum(1 for s in m.quality_scores if s >= 0.8):,}")
            qt.add_row(
                f"< {QUALITY_SCORE_THRESHOLD} (low)",
                f"{sum(1 for s in m.quality_scores if s < QUALITY_SCORE_THRESHOLD):,}",
            )
            qt.add_row("Sparkline", _sparkline(m.quality_scores, 30))

            self.console.print(qt)

        # ── batch time sparkline ──
        if m.batch_times:
            bt = Table(
                title="⏱ Batch Times",
                box=box.SIMPLE,
                show_header=False,
                border_style="dim",
            )
            bt.add_column("", style="cyan")
            bt.add_column("")
            bt.add_row("Sparkline", _sparkline(m.batch_times, 40))
            bt.add_row("Fastest", f"{min(m.batch_times):.1f}s")
            bt.add_row("Slowest", f"{max(m.batch_times):.1f}s")
            self.console.print(bt)

        # ── errors breakdown ──
        if m.error_categories:
            et = Table(
                title="🔥 Error Breakdown",
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold red",
            )
            et.add_column("Category", style="red")
            et.add_column("Count", style="yellow", justify="right")
            for cat, cnt in sorted(m.error_categories.items(), key=lambda x: -x[1]):
                et.add_row(cat, str(cnt))
            self.console.print(et)

        self.console.print()

    def _plain_final_dashboard(self, m: SessionMetrics):
        print(f"\n{'═'*60}")
        print(f"  GENERATION COMPLETE")
        print(f"{'═'*60}")
        print(f"  Rows succeeded      : {m.rows_succeeded:,}")
        print(f"  Rows failed         : {m.rows_failed:,}")
        print(f"  Low quality         : {m.rows_low_quality:,}")
        print(f"  Duplicates flagged  : {m.rows_duplicate:,}")
        print(f"  Total time          : {_format_time(m.total_time_s)}")
        print(f"  Avg batch time      : {m.avg_batch_time_s:.1f}s")
        print(f"  Speed               : {m.queries_per_minute:.1f} queries/min")
        print(f"  API calls           : {m.total_api_calls:,}")
        print(f"  Retries             : {m.total_retries:,}")
        print(f"  Avg quality         : {m.avg_quality_score:.4f}")
        print(f"  Peak memory         : {m.peak_memory_mb:.1f} MB")
        if m.batch_times:
            print(f"  Batch times         : {_sparkline(m.batch_times)}")
        if m.quality_scores:
            print(f"  Quality scores      : {_sparkline(m.quality_scores)}")
        print(f"{'═'*60}")

    # ── save confirmation ────────────────────────────────────

    def save_confirmation(self, path: str, df: pd.DataFrame, filled: int):
        size_kb = os.path.getsize(path) / 1024
        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[green bold]{path}[/green bold]\n"
                    f"[dim]{len(df):,} rows × {len(df.columns)} cols "
                    f"({_format_bytes(size_kb * 1024)})[/dim]\n"
                    f"[cyan]img_desc filled: {filled:,}/{len(df):,} "
                    f"({filled/len(df)*100:.1f}%)[/cyan]",
                    title="✓ Dataset Saved",
                    border_style="green",
                    padding=(0, 2),
                )
            )
        else:
            print(f"\n  ✓ SAVED: {path}")
            print(f"    {len(df):,} rows × {len(df.columns)} cols ({size_kb:.1f} KB)")
            print(f"    img_desc filled: {filled:,}/{len(df):,} ({filled/len(df)*100:.1f}%)")

    # ── analytics report display ─────────────────────────────

    def analytics_summary(self, report: dict, report_path: str):
        if RICH_AVAILABLE:
            tree = Tree(
                f"[bold magenta]📈 Analytics Report[/bold magenta]  "
                f"[dim]({os.path.basename(report_path)})[/dim]"
            )

            # Coverage
            cov = report.get("coverage", {})
            cov_node = tree.add("[cyan]Coverage[/cyan]")
            cov_node.add(f"Filled: {cov.get('filled', '?')} / {cov.get('filled', 0) + cov.get('empty', 0)}")
            cov_node.add(f"Fill rate: {cov.get('fill_rate', 0)}%")

            # Lengths
            ql = report.get("query_lengths", {})
            if ql:
                len_node = tree.add("[cyan]Query Lengths[/cyan]")
                len_node.add(f"Mean: {ql.get('mean', '?')} | Median: {ql.get('median', '?')}")
                len_node.add(f"Range: {ql.get('min', '?')} → {ql.get('max', '?')}")
                len_node.add(f"P10/P90: {ql.get('p10', '?')} / {ql.get('p90', '?')}")

            # Quality
            qual = report.get("quality", {})
            if qual:
                q_node = tree.add("[cyan]Quality[/cyan]")
                q_node.add(f"Mean: {qual.get('mean', '?')} | Median: {qual.get('median', '?')}")
                q_node.add(f"Below threshold: {qual.get('below_threshold', '?')}")
                q_node.add(f"High quality (≥0.8): {qual.get('above_0.8', '?')}")

            # Performance
            perf = report.get("performance", {})
            if perf:
                p_node = tree.add("[cyan]Performance[/cyan]")
                p_node.add(f"Total time: {perf.get('total_time', '?')}")
                p_node.add(f"Speed: {perf.get('queries_per_minute', '?')} q/min")
                p_node.add(f"Peak memory: {perf.get('peak_memory_mb', '?')} MB")

            # Top words
            top = report.get("top_query_words", {})
            if top:
                w_node = tree.add("[cyan]Top Words in Queries[/cyan]")
                top5 = list(top.items())[:10]
                for word, count in top5:
                    w_node.add(f"{word}: {count:,}")

            self.console.print()
            self.console.print(tree)
            self.console.print()
        else:
            cov = report.get("coverage", {})
            print(f"\n  ANALYTICS REPORT → {report_path}")
            print(f"  Coverage : {cov.get('fill_rate', '?')}%")
            perf = report.get("performance", {})
            print(f"  Speed    : {perf.get('queries_per_minute', '?')} q/min")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GRACEFUL SHUTDOWN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GracefulShutdown:
    """
    Catches SIGINT / SIGTERM so we can save a checkpoint before exiting.
    Usage:
        shutdown = GracefulShutdown()
        while not shutdown.requested:
            ...
    """

    def __init__(self):
        self.requested = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        if self.requested:
            # second signal — hard exit
            sys.exit(1)
        self.requested = True
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — finishing current batch then saving checkpoint...", sig_name)
        if RICH_AVAILABLE:
            Console().print(
                f"\n  [bold yellow]⚡ {sig_name} received — "
                f"graceful shutdown after current batch…[/bold yellow]"
            )
        else:
            print(f"\n  ⚡ {sig_name} received — saving after current batch…")

    def restore(self):
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN GENERATOR CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImageDescGenerator:
    """
    Advanced pipeline for img_desc generation.

    Flow:
        1. Pre-flight checks
        2. Load dataset
        3. Detect / offer resume from checkpoint
        4. Generate image-search queries via SearchAgent
           - Adaptive batch sizing
           - Quality scoring per row
           - Checkpoint after each batch
           - Graceful shutdown on Ctrl+C
        5. Post-generation analytics
        6. Save final dataset + analytics report
        7. Cleanup checkpoints
    """

    # def __init__(self):
    #     self.search_agent = SearchAgent()
    #     self.quality_analyzer = QueryQualityAnalyzer()
    #     self.checkpoint_mgr = CheckpointManager()
    #     self.batch_sizer = AdaptiveBatchSizer(
    #         initial=IMG_BATCH_SIZE, enabled=ADAPTIVE_BATCH_ENABLED
    #     )
    #     self.ui = RichUI()
    #     self.shutdown = GracefulShutdown()

    #     os.makedirs(OUTPUT_DIR, exist_ok=True)

    #     logger.info("ImageDescGenerator initialized (v%s, model=%s)", VERSION, MODEL_NAME)
    def __init__(self):
        # Suppress SearchAgent prints when used from img_generator
        import search_agent as _sa_mod
        _sa_mod.VERBOSE = True

        self.search_agent = SearchAgent()
        self.quality_analyzer = QueryQualityAnalyzer()
        self.checkpoint_mgr = CheckpointManager()
        self.batch_sizer = AdaptiveBatchSizer(
            initial=IMG_BATCH_SIZE, enabled=ADAPTIVE_BATCH_ENABLED
        )
        self.ui = RichUI()
        self.shutdown = GracefulShutdown()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("ImageDescGenerator initialized (v%s, model=%s)", VERSION, MODEL_NAME)
    # ═══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def generate(
        self,
        filepath: str = None,
        force_restart: bool = False,
        dry_run: bool = False,
        export_formats: Optional[List[ExportFormat]] = None,
        skip_preflight: bool = False,
        skip_analytics: bool = False,
    ) -> pd.DataFrame:
        """
        Full generation pipeline.

        Args:
            filepath:        Path to input CSV (auto-detects latest if None).
            force_restart:   Discard saved progress and start fresh.
            dry_run:         Preview workload without calling the LLM.
            export_formats:  Additional export formats beyond CSV.
            skip_preflight:  Skip system checks.
            skip_analytics:  Skip post-generation analytics.

        Returns:
            Updated DataFrame with img_desc column filled.
        """
        session_start = time.time()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("═" * 60)
        logger.info("Session %s started", session_id)

        # ── resolve filepath ─────────────────────────────────
        if filepath is None:
            filepath = self._find_latest_dataset()
            if filepath is None:
                self.ui.error("No dataset found in output directory. Run text generation first.")
                return pd.DataFrame()
            self.ui.info(f"Auto-detected dataset: {os.path.basename(filepath)}")

        filepath = os.path.abspath(filepath)

        if not os.path.exists(filepath):
            self.ui.error(f"File not found: {filepath}")
            return pd.DataFrame()

        # ── pre-flight checks ────────────────────────────────
        if not skip_preflight:
            self.ui.step("PRE-FLIGHT CHECKS")
            pf = run_preflight_checks(filepath)
            self.ui.preflight_report(pf)
            if not pf.passed:
                self.ui.error("Pre-flight checks failed. Fix issues above or use --skip-preflight.")
                return pd.DataFrame()

        # ── load dataset ─────────────────────────────────────
        self.ui.step("LOAD DATASET", os.path.basename(filepath))
        df = self._load_dataset(filepath)
        if df is None:
            return pd.DataFrame()

        total_rows = len(df)

        # ── banner ───────────────────────────────────────────
        self.ui.banner(filepath, total_rows)

        # ── dry run ──────────────────────────────────────────
        if dry_run:
            return self._dry_run(df, filepath)

        # ── check resume ─────────────────────────────────────
        start_batch = 0
        if not force_restart:
            resume_result = self._check_resume(df, filepath)
            if resume_result is not None:
                df, start_batch = resume_result

        # ── seed quality corpus with existing queries ────────
        self._seed_corpus(df)

        # ── generate queries ─────────────────────────────────
        self.ui.step(
            "GENERATE IMAGE SEARCH QUERIES",
            f"batches {start_batch+1}→{-(-total_rows // IMG_BATCH_SIZE)}",
        )

        filled_df = self._run_generation(df, filepath, start_batch)

        if self.shutdown.requested:
            self.ui.warning("Shutdown requested — progress has been saved.")
            self.shutdown.restore()
            return filled_df

        # ── save final ───────────────────────────────────────
        self.ui.step("SAVE FINAL DATASET")
        self._save_final(filled_df, filepath, export_formats or [])

        # ── analytics ────────────────────────────────────────
        if not skip_analytics:
            self.ui.step("POST-GENERATION ANALYTICS")
            metrics = self._current_tracker.to_metrics() if self._current_tracker else SessionMetrics()
            metrics.session_id = session_id
            metrics.started_at = datetime.fromtimestamp(session_start).isoformat()
            self._generate_analytics(filled_df, metrics)

        # ── final dashboard ──────────────────────────────────
        if self._current_tracker:
            metrics = self._current_tracker.to_metrics()
            self.ui.final_dashboard(metrics)

        # ── cleanup ──────────────────────────────────────────
        self.ui.step("CLEANUP")
        self.checkpoint_mgr.cleanup()
        self.ui.success("Checkpoint files removed.")

        # ── restore signal handlers ──────────────────────────
        self.shutdown.restore()

        elapsed = time.time() - session_start
        logger.info("Session %s completed in %s", session_id, _format_time(elapsed))

        return filled_df

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — DATASET LOADING
    # ═══════════════════════════════════════════════════════════

    def _find_latest_dataset(self) -> Optional[str]:
        patterns = [
            os.path.join(OUTPUT_DIR, "dataset_*rows*.csv"),
            os.path.join(OUTPUT_DIR, "dataset_*.csv"),
            os.path.join(OUTPUT_DIR, "*.csv"),
        ]
        for pattern in patterns:
            files = glob.glob(pattern)
            # Exclude internal progress files
            files = [
                f for f in files
                if not os.path.basename(f).startswith("_")
            ]
            if files:
                return max(files, key=os.path.getctime)
        return None

    def _load_dataset(self, filepath: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()

            if "img_desc" not in df.columns:
                df["img_desc"] = ""
                self.ui.info("Created 'img_desc' column.")

            # Validate required columns
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                self.ui.error(f"Missing required columns: {missing}")
                self.ui.info(f"Available columns: {list(df.columns)}")
                return None

            # Data summary
            needs = df["img_desc"].isna() | (df["img_desc"].astype(str).str.len() < MIN_QUERY_LENGTH)
            done = (~needs).sum()
            self.ui.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
            self.ui.info(f"img_desc coverage: {done:,}/{len(df):,} ({done/len(df)*100:.1f}%) already filled")

            # Column data quality
            for col in REQUIRED_COLUMNS:
                empty = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
                if empty > 0:
                    self.ui.warning(f"Column '{col}' has {empty:,} empty values")

            return df

        except Exception as e:
            self.ui.error(f"Failed to load CSV: {e}")
            logger.exception("CSV load error")
            return None

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — RESUME
    # ═══════════════════════════════════════════════════════════

    def _check_resume(
        self, df: pd.DataFrame, filepath: str
    ) -> Optional[Tuple[pd.DataFrame, int]]:
        """Check for saved progress. Returns (updated_df, start_batch) or None."""

        result = self.checkpoint_mgr.load_latest(filepath)
        if result is None:
            return None

        meta, progress_df = result

        completed_batches = meta.get("completed_batches", 0)
        total_batches = meta.get("total_batches", 0)

        if completed_batches >= total_batches:
            self.ui.info("Previous run completed all batches. Starting fresh.")
            return None

        if len(progress_df) != len(df):
            self.ui.warning(
                f"Checkpoint has {len(progress_df)} rows but dataset has {len(df)}. "
                f"Starting fresh."
            )
            return None

        # Show saved progress
        self.ui.display_saved_progress(meta)

        # Ask user
        print(f"\n  Options:")
        print(f"    [R/Enter] Resume from batch {completed_batches + 1}")
        print(f"    [F]       Fresh start (discard saved progress)")
        print()

        try:
            choice = input("  Your choice [R/F]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "R"

        if choice == "F":
            self.ui.info("Discarding saved progress. Starting fresh.")
            return None

        self.ui.success(f"Resuming from batch {completed_batches + 1}/{total_batches}")

        # Use the checkpoint data
        progress_df.columns = progress_df.columns.str.strip()
        if "img_desc" not in progress_df.columns:
            progress_df["img_desc"] = ""

        return progress_df, completed_batches

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — CORPUS SEEDING
    # ═══════════════════════════════════════════════════════════

    def _seed_corpus(self, df: pd.DataFrame):
        """Load existing queries into the quality analyzer corpus + search agent dedup."""
        existing = df["img_desc"].fillna("").astype(str)
        count = 0
        for q in existing:
            if q and len(q) >= MIN_QUERY_LENGTH:
                self.quality_analyzer.add_to_corpus(q)
                self.search_agent._used_queries.add(q)
                count += 1
        if count > 0:
            self.ui.info(f"Seeded corpus with {count:,} existing queries for dedup")

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — DRY RUN
    # ═══════════════════════════════════════════════════════════

    def _dry_run(self, df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        """Preview what would be processed without making API calls."""

        self.ui.step("DRY RUN — Preview Only", "no API calls")

        needs = df["img_desc"].isna() | (df["img_desc"].astype(str).str.len() < MIN_QUERY_LENGTH)
        to_process = needs.sum()
        already_done = (~needs).sum()
        n_batches = -(-to_process // IMG_BATCH_SIZE)

        est_time_per_batch = 15.0  # rough estimate
        est_total = est_time_per_batch * n_batches

        if RICH_AVAILABLE:
            table = Table(
                title="🔍 Dry Run Summary",
                box=box.ROUNDED,
                show_header=False,
                border_style="yellow",
            )
            table.add_column("", style="cyan", min_width=30)
            table.add_column("", style="green", justify="right")

            table.add_row("Total rows", f"{len(df):,}")
            table.add_row("Already filled", f"{already_done:,}")
            table.add_row("Need processing", f"{to_process:,}")
            table.add_row("Estimated batches", f"{n_batches:,}")
            table.add_row("Batch size", f"{IMG_BATCH_SIZE}")
            table.add_row("Est. time (rough)", _format_time(est_total))
            table.add_row("", "")

            # Show theme / emotion distribution of pending rows
            pending_df = df[needs]
            for col in ("theme", "emotion"):
                if col in pending_df.columns:
                    top3 = pending_df[col].value_counts().head(3)
                    for val, cnt in top3.items():
                        table.add_row(f"  Top {col}: {_truncate(str(val), 20)}", f"{cnt:,}")

            self.ui.console.print()
            self.ui.console.print(table)
            self.ui.console.print(
                Panel(
                    "[yellow]No API calls were made. "
                    "Remove --dry-run to execute.[/yellow]",
                    border_style="yellow",
                )
            )
        else:
            print(f"\n  DRY RUN SUMMARY")
            print(f"  Total rows       : {len(df):,}")
            print(f"  Already filled   : {already_done:,}")
            print(f"  Need processing  : {to_process:,}")
            print(f"  Est. batches     : {n_batches:,}")
            print(f"  Est. time        : {_format_time(est_total)}")
            print(f"\n  (No API calls made. Remove --dry-run to execute.)")

        return df

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — CORE GENERATION LOOP
    # ═══════════════════════════════════════════════════════════

    _current_tracker: Optional[ImageProgressTracker] = None

    def _run_generation(
        self, df: pd.DataFrame, filepath: str, start_batch: int
    ) -> pd.DataFrame:
        """Core batch loop with progress, quality, adaptive sizing, graceful shutdown."""

        all_rows = df.to_dict("records")
        total = len(all_rows)
        n_batches = -(-total // IMG_BATCH_SIZE)

        tracker = ImageProgressTracker(total, n_batches, start_batch)
        self._current_tracker = tracker

        filled_rows = all_rows.copy()

        if RICH_AVAILABLE:
            return self._generation_loop_rich(
                filled_rows, start_batch, n_batches, total, tracker, df, filepath
            )
        else:
            return self._generation_loop_plain(
                filled_rows, start_batch, n_batches, total, tracker, df, filepath
            )

    # ── Rich progress loop ───────────────────────────────────

    # def _generation_loop_rich(
    #     self,
    #     filled_rows: List[dict],
    #     start_batch: int,
    #     n_batches: int,
    #     total: int,
    #     tracker: ImageProgressTracker,
    #     original_df: pd.DataFrame,
    #     filepath: str,
    # ) -> pd.DataFrame:

    #     console = Console()

    #     already_done = sum(
    #         1 for r in filled_rows
    #         if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= MIN_QUERY_LENGTH
    #     )

    #     progress = Progress(
    #         SpinnerColumn("dots"),
    #         TextColumn("[bold blue]{task.description}"),
    #         BarColumn(bar_width=50, complete_style="green", finished_style="bright_green"),
    #         TextColumn("[bold]{task.percentage:>5.1f}%"),
    #         TextColumn("•"),
    #         MofNCompleteColumn(),
    #         TextColumn("•"),
    #         TimeElapsedColumn(),
    #         TextColumn("→"),
    #         TimeRemainingColumn(),
    #         console=console,
    #         expand=False,
    #         transient=False,
    #     )

    #     with progress:
    #         overall = progress.add_task(
    #             "[cyan]Overall",
    #             total=total,
    #             completed=already_done,
    #         )
    #         batch_task = progress.add_task(
    #             "[yellow]Batch",
    #             total=IMG_BATCH_SIZE,
    #             completed=0,
    #         )

    #         current_batch_size = IMG_BATCH_SIZE

    #         for b in range(start_batch, n_batches):
    #             if self.shutdown.requested:
    #                 self.ui.warning("Shutdown requested — saving progress…")
    #                 self._save_checkpoint(filled_rows, b, n_batches, filepath, tracker)
    #                 break

    #             start_idx = b * IMG_BATCH_SIZE
    #             end_idx = min(start_idx + IMG_BATCH_SIZE, total)
    #             batch_rows = filled_rows[start_idx:end_idx]
    #             batch_size = len(batch_rows)

    #             # Skip already-done batches
    #             if self._batch_is_complete(batch_rows):
    #                 progress.update(overall, advance=batch_size)
    #                 continue

    #             progress.reset(batch_task, total=batch_size, completed=0)
    #             progress.update(
    #                 batch_task,
    #                 description=f"[yellow]Batch {b+1}/{n_batches} (size={batch_size})",
    #             )

    #             # ── process batch ──
    #             batch_result = self._process_batch(
    #                 batch_rows, b + 1, n_batches, tracker
    #             )

    #             # ── merge results back ──
    #             for i, rr in enumerate(batch_result.rows):
    #                 if rr.status in (RowStatus.SUCCESS, RowStatus.LOW_QUALITY):
    #                     filled_rows[start_idx + i]["img_desc"] = rr.query

    #             # ── update trackers ──
    #             tracker.update_batch(batch_result)

    #             progress.update(batch_task, completed=batch_size)
    #             progress.update(overall, advance=batch_size)

    #             # ── batch summary ──
    #             self.ui.batch_summary(batch_result, tracker)

    #             # ── save checkpoint ──
    #             self._save_checkpoint(filled_rows, b + 1, n_batches, filepath, tracker)

    #             # ── adaptive sizing ──
    #             error_rate = batch_result.error_count / max(len(batch_result.rows), 1)
    #             suggested = self.batch_sizer.suggest(batch_result.elapsed, error_rate)
    #             if suggested != current_batch_size:
    #                 tracker.record_adaptive_adjustment(suggested)
    #                 self.ui.dim(
    #                     f"  ↻ Adaptive batch size: {current_batch_size} → {suggested}"
    #                 )
    #                 current_batch_size = suggested

    #             # ── delay ──
    #             if b < n_batches - 1 and not self.shutdown.requested:
    #                 time.sleep(DELAY_BETWEEN_BATCHES)

    #     result_df = pd.DataFrame(filled_rows)
    #     for col in original_df.columns:
    #         if col not in result_df.columns:
    #             result_df[col] = ""
    #     result_df = result_df[original_df.columns]
    #     return result_df
    def _generation_loop_rich(
        self,
        filled_rows: List[dict],
        start_batch: int,
        n_batches: int,
        total: int,
        tracker: ImageProgressTracker,
        original_df: pd.DataFrame,
        filepath: str,
    ) -> pd.DataFrame:

        console = Console()

        already_done = sum(
            1 for r in filled_rows
            if r.get("img_desc") and len(str(r.get("img_desc", ""))) >= MIN_QUERY_LENGTH
        )

        progress = Progress(
            SpinnerColumn("dots"),
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

        # ═══ Collect batch summaries to print AFTER progress context ═══
        batch_summaries: List[str] = []

        with progress:
            overall = progress.add_task(
                "[cyan]Overall",
                total=total,
                completed=already_done,
            )
            batch_task = progress.add_task(
                "[yellow]Batch",
                total=IMG_BATCH_SIZE,
                completed=0,
            )

            current_batch_size = IMG_BATCH_SIZE

            for b in range(start_batch, n_batches):
                if self.shutdown.requested:
                    self.ui.warning("Shutdown requested — saving progress…")
                    self._save_checkpoint(filled_rows, b, n_batches, filepath, tracker)
                    break

                start_idx = b * IMG_BATCH_SIZE
                end_idx = min(start_idx + IMG_BATCH_SIZE, total)
                batch_rows = filled_rows[start_idx:end_idx]
                batch_size = len(batch_rows)

                if self._batch_is_complete(batch_rows):
                    progress.update(overall, advance=batch_size)
                    continue

                progress.reset(batch_task, total=batch_size, completed=0)
                progress.update(
                    batch_task,
                    description=f"[yellow]Batch {b+1}/{n_batches} (size={batch_size})",
                )
                # ── pause progress so streaming output is visible ──
                # progress.stop()
                batch_result = self._process_batch(
                    batch_rows, b + 1, n_batches, tracker
                )
                 # ── resume progress bar ──
                # progress.start()

                for i, rr in enumerate(batch_result.rows):
                    if rr.status in (RowStatus.SUCCESS, RowStatus.LOW_QUALITY):
                        filled_rows[start_idx + i]["img_desc"] = rr.query

                tracker.update_batch(batch_result)

                progress.update(batch_task, completed=batch_size)
                progress.update(overall, advance=batch_size)

                # ═══ FIX: Store summary instead of printing inside progress ═══
                avg_q = (
                    sum(r.quality.overall for r in batch_result.rows if r.status == RowStatus.SUCCESS)
                    / max(batch_result.success_count, 1)
                )
                summary = (
                    f"    Batch {batch_result.batch_id:>3}  "
                    f"{batch_result.success_count}/{len(batch_result.rows)} ok  |  "
                    f"Q̄ {avg_q:.2f}  |  "
                    f"{batch_result.elapsed:.1f}s  |  "
                    f"ETA {_format_time(tracker.eta_seconds)}  |  "
                    f"{tracker.speed_per_min:.0f} q/min"
                )
                batch_summaries.append(summary)
                
                self._save_checkpoint(filled_rows, b + 1, n_batches, filepath, tracker)

                error_rate = batch_result.error_count / max(len(batch_result.rows), 1)
                suggested = self.batch_sizer.suggest(batch_result.elapsed, error_rate)
                if suggested != current_batch_size:
                    tracker.record_adaptive_adjustment(suggested)
                    batch_summaries.append(f"      ↻ Adaptive: {current_batch_size} → {suggested}")
                    current_batch_size = suggested

                if b < n_batches - 1 and not self.shutdown.requested:
                    time.sleep(DELAY_BETWEEN_BATCHES)

        # ═══ FIX: Print all summaries AFTER progress bar is done ═══
        if batch_summaries:
            console.print()  # blank line
            for summary in batch_summaries:
                console.print(summary, style="dim")

        result_df = pd.DataFrame(filled_rows)
        for col in original_df.columns:
            if col not in result_df.columns:
                result_df[col] = ""
        result_df = result_df[original_df.columns]
        return result_df

    # # ── Plain (no Rich) progress loop ────────────────────────

    def _generation_loop_plain(
        self,
        filled_rows: List[dict],
        start_batch: int,
        n_batches: int,
        total: int,
        tracker: ImageProgressTracker,
        original_df: pd.DataFrame,
        filepath: str,
    ) -> pd.DataFrame:

        print(f"\n  Rows: {total:,} | Batch: {IMG_BATCH_SIZE} | "
              f"Batches: {n_batches} | Start: {start_batch + 1}\n")

        for b in range(start_batch, n_batches):
            if self.shutdown.requested:
                print(f"\n  ⚡ Saving progress before exit…")
                self._save_checkpoint(filled_rows, b, n_batches, filepath, tracker)
                break

            start_idx = b * IMG_BATCH_SIZE
            end_idx = min(start_idx + IMG_BATCH_SIZE, total)
            batch_rows = filled_rows[start_idx:end_idx]
            batch_size = len(batch_rows)

            if self._batch_is_complete(batch_rows):
                continue

            pct = end_idx / total * 100
            bar_w = 40
            filled_w = int(bar_w * end_idx / total)
            bar = "█" * filled_w + "░" * (bar_w - filled_w)

            print(
                f"\r  [{bar}] {pct:5.1f}% | Batch {b+1}/{n_batches} | "
                f"Rows {start_idx+1}-{end_idx}",
                end="", flush=True,
            )

            batch_result = self._process_batch(batch_rows, b + 1, n_batches, tracker)

            for i, rr in enumerate(batch_result.rows):
                if rr.status in (RowStatus.SUCCESS, RowStatus.LOW_QUALITY):
                    filled_rows[start_idx + i]["img_desc"] = rr.query

            tracker.update_batch(batch_result)

            sr = batch_result.success_rate * 100
            print(
                f" ✓ {batch_result.elapsed:.1f}s "
                f"({batch_result.success_count}/{batch_size} ok, "
                f"Q̄={tracker.avg_quality:.2f})"
            )

            self._save_checkpoint(filled_rows, b + 1, n_batches, filepath, tracker)

            if b < n_batches - 1 and not self.shutdown.requested:
                time.sleep(DELAY_BETWEEN_BATCHES)

        print()
        result_df = pd.DataFrame(filled_rows)
        for col in original_df.columns:
            if col not in result_df.columns:
                result_df[col] = ""
        result_df = result_df[original_df.columns]
        return result_df

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — SINGLE BATCH PROCESSING
    # ═══════════════════════════════════════════════════════════

    def _process_batch(
        self,
        batch_rows: List[dict],
        batch_id: int,
        total_batches: int,
        tracker: ImageProgressTracker,
    ) -> BatchResult:
        """
        Process a single batch:
          1. Call SearchAgent with retry
          2. Score quality of each generated query
          3. Return BatchResult with per-row status
        """

        result = BatchResult(batch_id=batch_id)
        result.start_time = time.time()

        # Attempt generation with retry
        generated: Optional[List[dict]] = None

        def _attempt():
            return self.search_agent.generate_queries(
                batch_rows, batch_label=f"img_batch{batch_id:03d}"
            )

        def _on_retry(attempt: int, exc: Exception, delay: float):
            result.retry_count += 1
            tracker.add_error_category(type(exc).__name__)
            self.ui.dim(
                f"  ⟳ Retry {attempt}/{MAX_RETRIES} in {delay:.1f}s — "
                f"{type(exc).__name__}: {_truncate(str(exc), 60)}"
            )

        try:
            generated = _retry_with_backoff(
                _attempt,
                max_retries=MAX_RETRIES,
                on_retry=_on_retry,
            )
        except Exception as e:
            logger.error("Batch %d failed after all retries: %s", batch_id, e)
            # Mark all rows as errors
            for i, row in enumerate(batch_rows):
                rr = RowResult(
                    row_index=i,
                    status=RowStatus.ERROR,
                    error_message=str(e),
                )
                result.rows.append(rr)
            result.status = BatchStatus.FAILED
            result.end_time = time.time()
            return result

        # ── score each row ───────────────────────────────────
        for i, row_data in enumerate(generated or batch_rows):
            rr = RowResult(row_index=i, attempts=1 + result.retry_count)

            query = str(row_data.get("img_desc", "")).strip()
            rr.query = query

            if not query or len(query) < MIN_QUERY_LENGTH:
                rr.status = RowStatus.ERROR
                rr.error_message = f"Query too short ({len(query)} chars)"
            else:
                # Quality scoring
                source_row = batch_rows[i] if i < len(batch_rows) else {}
                rr.quality = self.quality_analyzer.score(query, source_row)

                if rr.quality.overall < QUALITY_SCORE_THRESHOLD:
                    rr.status = RowStatus.LOW_QUALITY
                    logger.debug(
                        "Low quality (%.2f) for row %d: %s",
                        rr.quality.overall, i, _truncate(query, 40),
                    )
                elif rr.quality.uniqueness_score < (1.0 - DUPLICATE_SIMILARITY_THRESHOLD):
                    rr.status = RowStatus.DUPLICATE
                else:
                    rr.status = RowStatus.SUCCESS
                    self.quality_analyzer.add_to_corpus(query)

            rr.generation_time_ms = (time.time() - result.start_time) * 1000 / max(i + 1, 1)
            result.rows.append(rr)

        # ── determine batch status ───────────────────────────
        if all(r.status == RowStatus.SUCCESS for r in result.rows):
            result.status = BatchStatus.COMPLETED
        elif result.success_count > 0:
            result.status = BatchStatus.PARTIAL
        else:
            result.status = BatchStatus.FAILED

        result.end_time = time.time()
        return result

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL — HELPERS
    # ═══════════════════════════════════════════════════════════

    def _batch_is_complete(self, batch_rows: List[dict]) -> bool:
        return all(
            row.get("img_desc") and len(str(row.get("img_desc", ""))) >= MIN_QUERY_LENGTH
            for row in batch_rows
        )

    def _save_checkpoint(
        self,
        filled_rows: List[dict],
        completed_batches: int,
        total_batches: int,
        filepath: str,
        tracker: ImageProgressTracker,
    ):
        try:
            metrics = tracker.to_metrics()
            self.checkpoint_mgr.save(
                filled_rows, completed_batches, total_batches,
                filepath, metrics,
            )
        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            self.ui.warning(f"Checkpoint save failed: {e}")

    def _save_final(
        self,
        df: pd.DataFrame,
        filepath: str,
        extra_formats: List[ExportFormat],
    ):
        """Save the final dataset + optional exports."""

        # Primary CSV
        df.to_csv(filepath, index=False)

        filled = (
            df["img_desc"].fillna("").astype(str).str.len() >= MIN_QUERY_LENGTH
        ).sum()

        self.ui.save_confirmation(filepath, df, filled)

        # Extra exports
        base = os.path.splitext(filepath)[0]
        for fmt in extra_formats:
            try:
                if fmt == ExportFormat.JSON:
                    out = base + ".json"
                    df.to_json(out, orient="records", indent=2, force_ascii=False)
                    self.ui.success(f"Exported JSON: {os.path.basename(out)}")
                elif fmt == ExportFormat.PARQUET:
                    out = base + ".parquet"
                    df.to_parquet(out, index=False, engine="pyarrow")
                    self.ui.success(f"Exported Parquet: {os.path.basename(out)}")
                elif fmt == ExportFormat.EXCEL:
                    out = base + ".xlsx"
                    df.to_excel(out, index=False, engine="openpyxl")
                    self.ui.success(f"Exported Excel: {os.path.basename(out)}")
            except ImportError as ie:
                self.ui.warning(f"Cannot export {fmt.value}: {ie}")
            except Exception as e:
                self.ui.warning(f"Export {fmt.value} failed: {e}")

    def _generate_analytics(self, df: pd.DataFrame, metrics: SessionMetrics):
        """Generate and save the analytics report."""
        try:
            report = PostAnalytics.generate_report(
                df, metrics, self.quality_analyzer
            )

            # Save to JSON
            report_path = IMG_ANALYTICS_REPORT
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            self.ui.success(f"Analytics report saved: {os.path.basename(report_path)}")
            self.ui.analytics_summary(report, report_path)

        except Exception as e:
            logger.exception("Analytics generation failed")
            self.ui.warning(f"Analytics failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI ARGUMENT PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args(argv: List[str] = None) -> dict:
    """Parse CLI arguments into a config dict."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="img_generator",
        description=f"{APP_NAME} v{VERSION} — Generate image search queries for dataset rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python img_generator.py                          # auto-detect dataset, resume if possible
  python img_generator.py data.csv                 # specify input file
  python img_generator.py --fresh                  # discard progress, start fresh
  python img_generator.py --dry-run                # preview without API calls
  python img_generator.py --export json parquet    # also export JSON + Parquet
  python img_generator.py --skip-preflight         # skip system checks
  python img_generator.py --skip-analytics         # skip post-generation report
        """,
    )

    parser.add_argument(
        "filepath", nargs="?", default=None,
        help="Path to input CSV (auto-detects latest if omitted).",
    )
    parser.add_argument(
        "--fresh", "-f", action="store_true",
        help="Discard saved progress and start from scratch.",
    )
    parser.add_argument(
        "--dry-run", "-d", action="store_true",
        help="Preview workload without making any API calls.",
    )
    parser.add_argument(
        "--export", "-e", nargs="+", choices=["json", "parquet", "xlsx"],
        default=[],
        help="Additional export formats.",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight system checks.",
    )
    parser.add_argument(
        "--skip-analytics", action="store_true",
        help="Skip post-generation analytics report.",
    )
    parser.add_argument(
        "--version", "-v", action="version",
        version=f"{APP_NAME} v{VERSION}",
    )

    args = parser.parse_args(argv)

    export_map = {
        "json": ExportFormat.JSON,
        "parquet": ExportFormat.PARQUET,
        "xlsx": ExportFormat.EXCEL,
    }

    return {
        "filepath": args.filepath,
        "force_restart": args.fresh,
        "dry_run": args.dry_run,
        "export_formats": [export_map[e] for e in args.export],
        "skip_preflight": args.skip_preflight,
        "skip_analytics": args.skip_analytics,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    config = parse_args()

    if RICH_AVAILABLE:
        console = Console()
        console.print()
        console.print(
            Rule(
                f"[bold magenta]{APP_NAME} v{VERSION}[/bold magenta]",
                style="bright_blue",
            )
        )
    else:
        print(f"\n{'━'*60}")
        print(f"  {APP_NAME} v{VERSION}")
        print(f"{'━'*60}")

    gen = ImageDescGenerator()
    result = gen.generate(**config)

    if not result.empty:
        filled = (
            result["img_desc"].fillna("").astype(str).str.len() >= MIN_QUERY_LENGTH
        ).sum()
        if RICH_AVAILABLE:
            Console().print(
                Panel(
                    f"[bold green]✅ Generation Complete![/bold green]\n"
                    f"[dim]{len(result):,} rows processed[/dim]\n"
                    f"[cyan]img_desc filled: {filled:,}/{len(result):,} "
                    f"({filled/len(result)*100:.1f}%)[/cyan]",
                    title="🎉 Success",
                    border_style="bright_green",
                    padding=(0, 2),
                )
            )
        else:
            print(f"\n  ✅ Success! {filled:,}/{len(result):,} img_desc queries generated.")
    else:
        if RICH_AVAILABLE:
            Console().print(
                Panel(
                    "[bold red]Generation failed or produced empty result.[/bold red]\n"
                    "[dim]Check logs for details.[/dim]",
                    title="✗ Failed",
                    border_style="red",
                )
            )
        else:
            print(f"\n  ✗ Failed to generate queries. Check {IMG_AUDIT_LOG}")
        sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STANDALONE UTILITIES — callable from other modules or CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DatasetInspector:
    """
    Standalone utility to inspect and validate a dataset's img_desc
    column without running generation.  Useful for debugging or
    auditing an existing output file.

    Usage:
        inspector = DatasetInspector()
        inspector.inspect("output/dataset_500rows.csv")
    """

    def __init__(self):
        self.ui = RichUI()
        self.quality_analyzer = QueryQualityAnalyzer()

    def inspect(self, filepath: str, sample_size: int = 20) -> dict:
        """
        Full inspection of a dataset's img_desc quality.

        Returns:
            dict with inspection results (also printed to console).
        """
        self.ui.step("DATASET INSPECTION", os.path.basename(filepath))

        if not os.path.exists(filepath):
            self.ui.error(f"File not found: {filepath}")
            return {}

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        if "img_desc" not in df.columns:
            self.ui.error("No 'img_desc' column found.")
            return {"error": "missing_column"}

        results: Dict[str, Any] = {
            "filepath": filepath,
            "total_rows": len(df),
            "inspected_at": datetime.now().isoformat(),
        }

        # ── coverage analysis ────────────────────────────────
        filled_mask = (
            df["img_desc"].fillna("").astype(str).str.len() >= MIN_QUERY_LENGTH
        )
        filled_count = int(filled_mask.sum())
        empty_count = int((~filled_mask).sum())
        fill_rate = filled_count / len(df) * 100 if len(df) > 0 else 0

        results["coverage"] = {
            "filled": filled_count,
            "empty": empty_count,
            "fill_rate_pct": round(fill_rate, 2),
        }

        self.ui.info(f"Coverage: {filled_count:,}/{len(df):,} ({fill_rate:.1f}%)")

        # ── length distribution ──────────────────────────────
        queries = df.loc[filled_mask, "img_desc"].astype(str)
        if len(queries) > 0:
            lengths = queries.str.len()
            word_counts = queries.str.split().str.len()

            results["lengths"] = {
                "mean": round(float(lengths.mean()), 1),
                "median": round(float(lengths.median()), 1),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std": round(float(lengths.std()), 1),
            }
            results["word_counts"] = {
                "mean": round(float(word_counts.mean()), 1),
                "median": round(float(word_counts.median()), 1),
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
            }

            self.ui.info(
                f"Query length: mean={lengths.mean():.0f}, "
                f"median={lengths.median():.0f}, "
                f"range=[{lengths.min()}, {lengths.max()}]"
            )
            self.ui.info(
                f"Word count:   mean={word_counts.mean():.1f}, "
                f"median={word_counts.median():.0f}"
            )

        # ── quality scoring on sample ────────────────────────
        sample_n = min(sample_size, len(queries))
        if sample_n > 0:
            sample_indices = queries.sample(sample_n, random_state=42).index
            scores = []
            low_quality_samples = []
            high_quality_samples = []

            for idx in sample_indices:
                row = df.loc[idx].to_dict()
                query = str(row.get("img_desc", ""))
                qs = self.quality_analyzer.score(query, row)
                scores.append(qs.overall)
                self.quality_analyzer.add_to_corpus(query)

                if qs.overall < QUALITY_SCORE_THRESHOLD:
                    low_quality_samples.append({
                        "query": _truncate(query, 80),
                        "score": round(qs.overall, 3),
                        "breakdown": {
                            "length": round(qs.length_score, 3),
                            "specificity": round(qs.specificity_score, 3),
                            "relevance": round(qs.relevance_score, 3),
                            "uniqueness": round(qs.uniqueness_score, 3),
                        },
                    })
                elif qs.overall >= 0.8:
                    high_quality_samples.append({
                        "query": _truncate(query, 80),
                        "score": round(qs.overall, 3),
                    })

            results["quality_sample"] = {
                "sample_size": sample_n,
                "mean_score": round(float(np.mean(scores)), 4),
                "median_score": round(float(np.median(scores)), 4),
                "min_score": round(float(min(scores)), 4),
                "max_score": round(float(max(scores)), 4),
                "below_threshold": len(low_quality_samples),
                "above_0.8": len(high_quality_samples),
            }

            self.ui.info(
                f"Quality (n={sample_n}): "
                f"mean={np.mean(scores):.3f}, "
                f"min={min(scores):.3f}, max={max(scores):.3f}"
            )

            # ── display tables if Rich ───────────────────────
            if RICH_AVAILABLE and low_quality_samples:
                tbl = Table(
                    title=f"⚠ Low Quality Samples (< {QUALITY_SCORE_THRESHOLD})",
                    box=box.SIMPLE_HEAVY,
                    show_header=True,
                    header_style="bold red",
                    border_style="red",
                )
                tbl.add_column("Score", style="red", width=7, justify="right")
                tbl.add_column("Query", style="dim", ratio=1)
                tbl.add_column("L", width=5, justify="right")
                tbl.add_column("S", width=5, justify="right")
                tbl.add_column("R", width=5, justify="right")
                tbl.add_column("U", width=5, justify="right")

                for s in low_quality_samples[:8]:
                    bd = s["breakdown"]
                    tbl.add_row(
                        f"{s['score']:.3f}",
                        s["query"],
                        f"{bd['length']:.2f}",
                        f"{bd['specificity']:.2f}",
                        f"{bd['relevance']:.2f}",
                        f"{bd['uniqueness']:.2f}",
                    )
                self.ui.console.print(tbl)

            if RICH_AVAILABLE and high_quality_samples:
                tbl = Table(
                    title="✅ High Quality Samples (≥ 0.8)",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold green",
                    border_style="green",
                )
                tbl.add_column("Score", style="green", width=7, justify="right")
                tbl.add_column("Query", style="bright_white", ratio=1)
                for s in high_quality_samples[:8]:
                    tbl.add_row(f"{s['score']:.3f}", s["query"])
                self.ui.console.print(tbl)

        # ── duplicate detection ──────────────────────────────
        if len(queries) > 1:
            exact_dupes = int(queries.duplicated().sum())
            results["duplicates"] = {"exact": exact_dupes}
            if exact_dupes > 0:
                self.ui.warning(f"Exact duplicate queries: {exact_dupes:,}")
            else:
                self.ui.success("No exact duplicate queries found.")

            # Near-duplicate sampling (expensive, so cap it)
            near_dupe_count = 0
            check_limit = min(200, len(queries))
            query_list = queries.head(check_limit).tolist()
            for i in range(len(query_list)):
                for j in range(i + 1, len(query_list)):
                    sim = _jaccard_similarity(query_list[i], query_list[j])
                    if sim >= DUPLICATE_SIMILARITY_THRESHOLD:
                        near_dupe_count += 1
            results["duplicates"]["near_duplicates_in_sample"] = near_dupe_count
            results["duplicates"]["sample_checked"] = check_limit
            if near_dupe_count > 0:
                self.ui.warning(
                    f"Near-duplicate pairs (Jaccard≥{DUPLICATE_SIMILARITY_THRESHOLD}): "
                    f"{near_dupe_count} in first {check_limit} queries"
                )

        # ── theme/emotion coverage ───────────────────────────
        for col in ("theme", "emotion", "object_detected"):
            if col in df.columns:
                vc = df[col].fillna("unknown").value_counts()
                results[f"{col}_counts"] = vc.head(10).to_dict()

                # Check if any theme/emotion has very low fill rate
                grouped = df.groupby(col).apply(
                    lambda g: (
                        g["img_desc"].fillna("").astype(str).str.len()
                        >= MIN_QUERY_LENGTH
                    ).mean()
                )
                worst = grouped.nsmallest(3)
                if len(worst) > 0 and worst.iloc[0] < 0.5:
                    self.ui.warning(
                        f"Lowest fill rates by '{col}': "
                        + ", ".join(
                            f"{k}={v:.0%}" for k, v in worst.items()
                        )
                    )

        self.ui.success("Inspection complete.")
        return results


class DatasetRepairer:
    """
    Attempts to repair low-quality or missing img_desc entries
    by re-generating only the problematic rows.

    Usage:
        repairer = DatasetRepairer()
        fixed_df = repairer.repair("output/dataset_500rows.csv", quality_threshold=0.6)
    """

    def __init__(self):
        self.ui = RichUI()
        self.search_agent = SearchAgent()
        self.quality_analyzer = QueryQualityAnalyzer()
        self.shutdown = GracefulShutdown()

    def repair(
        self,
        filepath: str,
        quality_threshold: float = QUALITY_SCORE_THRESHOLD,
        max_repair_batches: int = 50,
        rescore_all: bool = False,
    ) -> pd.DataFrame:
        """
        Identify and re-generate low-quality / missing img_desc rows.

        Args:
            filepath:           Input CSV path.
            quality_threshold:  Re-generate rows below this score.
            max_repair_batches: Cap on how many batches to process.
            rescore_all:        If True, re-score every row first.

        Returns:
            Updated DataFrame.
        """
        self.ui.step("REPAIR MODE", f"threshold={quality_threshold}")

        if not os.path.exists(filepath):
            self.ui.error(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        if "img_desc" not in df.columns:
            self.ui.error("No 'img_desc' column.")
            return df

        # ── identify rows needing repair ─────────────────────
        repair_indices: List[int] = []

        for idx in range(len(df)):
            query = str(df.at[idx, "img_desc"]) if pd.notna(df.at[idx, "img_desc"]) else ""

            if len(query) < MIN_QUERY_LENGTH:
                repair_indices.append(idx)
                continue

            if rescore_all:
                row = df.iloc[idx].to_dict()
                qs = self.quality_analyzer.score(query, row)
                if qs.overall < quality_threshold:
                    repair_indices.append(idx)
                else:
                    self.quality_analyzer.add_to_corpus(query)
            else:
                self.quality_analyzer.add_to_corpus(query)

        self.ui.info(
            f"Rows needing repair: {len(repair_indices):,}/{len(df):,} "
            f"({len(repair_indices)/len(df)*100:.1f}%)"
        )

        if not repair_indices:
            self.ui.success("Nothing to repair! All rows pass quality threshold.")
            return df

        # ── batch repair ─────────────────────────────────────
        batches = [
            repair_indices[i:i + IMG_BATCH_SIZE]
            for i in range(0, len(repair_indices), IMG_BATCH_SIZE)
        ]
        batches = batches[:max_repair_batches]

        self.ui.info(f"Repair batches: {len(batches)} (capped at {max_repair_batches})")

        repaired_count = 0
        failed_count = 0

        for b_idx, batch_idx_list in enumerate(batches):
            if self.shutdown.requested:
                self.ui.warning("Shutdown during repair — saving partial results.")
                break

            batch_rows = [df.iloc[i].to_dict() for i in batch_idx_list]

            # Clear old img_desc so SearchAgent generates fresh
            for row in batch_rows:
                row["img_desc"] = ""

            try:
                result = _retry_with_backoff(
                    lambda: self.search_agent.generate_queries(
                        batch_rows,
                        batch_label=f"repair_batch{b_idx+1:03d}",
                    ),
                    max_retries=MAX_RETRIES,
                )

                for j, original_idx in enumerate(batch_idx_list):
                    if j < len(result):
                        new_query = str(result[j].get("img_desc", "")).strip()
                        if len(new_query) >= MIN_QUERY_LENGTH:
                            # Re-score
                            source = df.iloc[original_idx].to_dict()
                            qs = self.quality_analyzer.score(new_query, source)
                            if qs.overall >= quality_threshold:
                                df.at[original_idx, "img_desc"] = new_query
                                self.quality_analyzer.add_to_corpus(new_query)
                                repaired_count += 1
                            else:
                                # Still low quality but maybe better than before
                                old_q = str(df.at[original_idx, "img_desc"])
                                if len(old_q) < MIN_QUERY_LENGTH:
                                    df.at[original_idx, "img_desc"] = new_query
                                    repaired_count += 1
                                else:
                                    failed_count += 1
                        else:
                            failed_count += 1

            except Exception as e:
                self.ui.warning(f"Repair batch {b_idx+1} failed: {e}")
                failed_count += len(batch_idx_list)

            self.ui.dim(
                f"  Repair batch {b_idx+1}/{len(batches)}: "
                f"+{repaired_count} repaired, {failed_count} still failing"
            )

            if b_idx < len(batches) - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)

        # ── save ─────────────────────────────────────────────
        df.to_csv(filepath, index=False)

        total_filled = (
            df["img_desc"].fillna("").astype(str).str.len() >= MIN_QUERY_LENGTH
        ).sum()

        self.ui.success(
            f"Repair complete: {repaired_count:,} rows fixed, "
            f"{failed_count:,} still problematic"
        )
        self.ui.save_confirmation(filepath, df, int(total_filled))

        self.shutdown.restore()
        return df


class BatchRetryManager:
    """
    Manages per-row retry isolation: if a single row in a batch
    causes a parse error, this isolates and retries individual rows
    so one bad apple doesn't kill the entire batch.

    Used internally by ImageDescGenerator._process_batch when
    a batch partially fails.
    """

    def __init__(self, search_agent: SearchAgent, ui: RichUI):
        self.agent = search_agent
        self.ui = ui

    def retry_failed_rows(
        self,
        failed_rows: List[Tuple[int, dict]],
        batch_label: str,
    ) -> Dict[int, str]:
        """
        Retry each failed row individually.

        Args:
            failed_rows: List of (original_index, row_dict) tuples.
            batch_label: Label for logging.

        Returns:
            Dict mapping original_index → generated query string.
        """
        results: Dict[int, str] = {}

        for orig_idx, row in failed_rows:
            try:
                single_result = self.agent.generate_queries(
                    [row], batch_label=f"{batch_label}_retry_row{orig_idx}"
                )
                if single_result and len(single_result) > 0:
                    query = str(single_result[0].get("img_desc", "")).strip()
                    if len(query) >= MIN_QUERY_LENGTH:
                        results[orig_idx] = query
            except Exception as e:
                logger.debug(
                    "Individual row retry failed (idx=%d): %s", orig_idx, e
                )
                continue

        return results


class ExportManager:
    """
    Handles multi-format export with validation and compression options.
    """

    def __init__(self, ui: RichUI):
        self.ui = ui

    def export_all(
        self,
        df: pd.DataFrame,
        base_path: str,
        formats: List[ExportFormat],
        compress: bool = False,
    ) -> List[str]:
        """
        Export dataset in multiple formats.

        Returns:
            List of successfully exported file paths.
        """
        exported: List[str] = []
        base = os.path.splitext(base_path)[0]

        for fmt in formats:
            try:
                path = self._export_single(df, base, fmt, compress)
                if path:
                    exported.append(path)
                    size = _format_bytes(os.path.getsize(path))
                    self.ui.success(f"Exported {fmt.value}: {os.path.basename(path)} ({size})")
            except ImportError as ie:
                self.ui.warning(f"Cannot export {fmt.value} — missing dependency: {ie}")
            except Exception as e:
                self.ui.warning(f"Export {fmt.value} failed: {e}")
                logger.exception("Export %s failed", fmt.value)

        return exported

    def _export_single(
        self, df: pd.DataFrame, base: str, fmt: ExportFormat, compress: bool
    ) -> Optional[str]:
        """Export to a single format."""

        if fmt == ExportFormat.CSV:
            path = base + (".csv.gz" if compress else ".csv")
            compression = "gzip" if compress else None
            df.to_csv(path, index=False, compression=compression)
            return path

        elif fmt == ExportFormat.JSON:
            path = base + (".json.gz" if compress else ".json")
            compression = "gzip" if compress else None
            df.to_json(
                path, orient="records", indent=2,
                force_ascii=False, compression=compression,
            )
            return path

        elif fmt == ExportFormat.PARQUET:
            path = base + ".parquet"
            df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
            return path

        elif fmt == ExportFormat.EXCEL:
            path = base + ".xlsx"
            df.to_excel(path, index=False, engine="openpyxl")
            return path

        return None


class ProgressDashboard:
    """
    Real-time Rich Live dashboard that updates in-place during generation.
    Shows a compact multi-panel view with metrics, sparklines, and status.

    Used optionally when --live-dashboard flag is passed (future extension).
    """

    def __init__(self, tracker: ImageProgressTracker):
        self.tracker = tracker
        self.console = Console() if RICH_AVAILABLE else None
        self._live: Optional[Live] = None

    def _build_layout(self) -> Panel:
        """Build the dashboard panel for Rich Live display."""
        t = self.tracker

        # ── header ──
        header_text = Text.assemble(
            ("🔍 ", ""),
            (f"{APP_NAME} v{VERSION}", "bold magenta"),
            ("  |  ", "dim"),
            (f"Batch {t.current_batch}/{t.total_batches}", "cyan"),
            ("  |  ", "dim"),
            (f"{t.completed_rows:,}/{t.total_rows:,} rows", "green"),
            ("  |  ", "dim"),
            (f"ETA {_format_time(t.eta_seconds)}", "yellow"),
        )

        # ── stats table ──
        stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        stats.add_column("", style="cyan", min_width=20)
        stats.add_column("", style="bright_green", justify="right", min_width=15)

        elapsed = t.elapsed
        stats.add_row("Elapsed", _format_time(elapsed))
        stats.add_row("Speed", f"{t.speed_per_min:.1f} q/min")
        stats.add_row("Success rate", f"{t.success_rate:.1f}%")
        stats.add_row("Avg quality", f"{t.avg_quality:.3f}")
        stats.add_row("Retries", str(t.retries))
        stats.add_row("Errors", str(t.errors))
        stats.add_row("Memory", f"{_get_memory_mb():.0f} MB")
        stats.add_row("Batch size", str(t.current_batch_size))

        if t.batch_times:
            stats.add_row("Batch times", _sparkline(t.batch_times, 25))
        if t.quality_scores:
            stats.add_row("Quality", _sparkline(t.quality_scores[-50:], 25))

        # ── progress bar ──
        pct = t.completed_rows / max(t.total_rows, 1) * 100
        bar_w = 50
        filled_w = int(bar_w * pct / 100)
        bar_str = "█" * filled_w + "░" * (bar_w - filled_w)
        progress_text = Text.assemble(
            ("  [", "dim"),
            (bar_str[:filled_w], "green"),
            (bar_str[filled_w:], "dim"),
            ("]", "dim"),
            (f" {pct:5.1f}%", "bold"),
        )

        group = Group(
            header_text,
            Rule(style="dim"),
            stats,
            Rule(style="dim"),
            progress_text,
        )

        return Panel(
            group,
            border_style="bright_blue",
            padding=(0, 1),
        )

    @contextmanager
    def live_context(self):
        """Context manager for Rich Live dashboard."""
        if not RICH_AVAILABLE:
            yield
            return

        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=2,
            transient=False,
        )
        with self._live:
            yield

    def refresh(self):
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_layout())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXTENDED CLI COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_inspect(argv: List[str]):
    """CLI: inspect a dataset's img_desc column."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="img_generator inspect",
        description="Inspect and audit img_desc quality in a dataset.",
    )
    parser.add_argument("filepath", help="Path to CSV file.")
    parser.add_argument(
        "--sample-size", "-n", type=int, default=50,
        help="Number of rows to sample for quality scoring.",
    )
    args = parser.parse_args(argv)

    inspector = DatasetInspector()
    results = inspector.inspect(args.filepath, sample_size=args.sample_size)

    # Save inspection report
    report_path = os.path.splitext(args.filepath)[0] + "_inspection.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    if RICH_AVAILABLE:
        Console().print(f"  [dim]Report saved: {report_path}[/dim]")
    else:
        print(f"  Report saved: {report_path}")


def cmd_repair(argv: List[str]):
    """CLI: repair low-quality img_desc entries."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="img_generator repair",
        description="Re-generate low-quality or missing img_desc entries.",
    )
    parser.add_argument("filepath", help="Path to CSV file.")
    parser.add_argument(
        "--threshold", "-t", type=float, default=QUALITY_SCORE_THRESHOLD,
        help=f"Quality threshold (default: {QUALITY_SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-batches", "-m", type=int, default=50,
        help="Maximum repair batches (default: 50).",
    )
    parser.add_argument(
        "--rescore", action="store_true",
        help="Re-score all existing queries (slow but thorough).",
    )
    args = parser.parse_args(argv)

    repairer = DatasetRepairer()
    repairer.repair(
        args.filepath,
        quality_threshold=args.threshold,
        max_repair_batches=args.max_batches,
        rescore_all=args.rescore,
    )


def cmd_analytics(argv: List[str]):
    """CLI: generate analytics report for a completed dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="img_generator analytics",
        description="Generate a post-hoc analytics report.",
    )
    parser.add_argument("filepath", help="Path to CSV file.")
    args = parser.parse_args(argv)

    ui = RichUI()
    ui.step("GENERATING ANALYTICS REPORT")

    if not os.path.exists(args.filepath):
        ui.error(f"File not found: {args.filepath}")
        return

    df = pd.read_csv(args.filepath)
    df.columns = df.columns.str.strip()

    if "img_desc" not in df.columns:
        ui.error("No 'img_desc' column found.")
        return

    # Build a dummy metrics object
    metrics = SessionMetrics(total_rows=len(df))

    qa = QueryQualityAnalyzer()
    filled_mask = df["img_desc"].fillna("").astype(str).str.len() >= MIN_QUERY_LENGTH
    for idx in df[filled_mask].index:
        row = df.iloc[idx].to_dict()
        query = str(row.get("img_desc", ""))
        qs = qa.score(query, row)
        metrics.quality_scores.append(qs.overall)
        qa.add_to_corpus(query)

    metrics.rows_succeeded = int(filled_mask.sum())
    metrics.rows_failed = int((~filled_mask).sum())
    metrics.compute_derived()

    report = PostAnalytics.generate_report(df, metrics, qa)

    report_path = os.path.splitext(args.filepath)[0] + "_analytics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    ui.success(f"Report saved: {report_path}")
    ui.analytics_summary(report, report_path)


def cmd_status(argv: List[str]):
    """CLI: show current progress/checkpoint status."""
    ui = RichUI()
    ui.step("CHECKPOINT STATUS")

    if os.path.exists(IMG_PROGRESS_FILE):
        try:
            with open(IMG_PROGRESS_FILE, "r") as f:
                meta = json.load(f)
            ui.display_saved_progress(meta)
        except Exception as e:
            ui.error(f"Cannot read progress file: {e}")
    else:
        ui.info("No saved progress found.")

    # List checkpoints
    cp_mgr = CheckpointManager()
    checkpoints = cp_mgr._list_checkpoints()

    if checkpoints:
        if RICH_AVAILABLE:
            tbl = Table(
                title="📁 Available Checkpoints",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold",
            )
            tbl.add_column("#", width=3)
            tbl.add_column("Name", style="cyan")
            tbl.add_column("Size", justify="right")
            tbl.add_column("Modified", style="dim")

            for i, cp in enumerate(checkpoints, 1):
                name = os.path.basename(cp)
                data_file = os.path.join(cp, "data.csv")
                size = _format_bytes(os.path.getsize(data_file)) if os.path.exists(data_file) else "?"
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(cp)
                ).strftime("%Y-%m-%d %H:%M:%S")
                tbl.add_row(str(i), name, size, mtime)

            Console().print(tbl)
        else:
            print(f"\n  Checkpoints ({len(checkpoints)}):")
            for cp in checkpoints:
                print(f"    • {os.path.basename(cp)}")
    else:
        ui.info("No checkpoint directories found.")


def cmd_clean(argv: List[str]):
    """CLI: remove all progress/checkpoint files."""
    ui = RichUI()
    ui.step("CLEANUP")

    cp_mgr = CheckpointManager()
    cp_mgr.cleanup()
    ui.success("All checkpoint and progress files removed.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SUB-COMMAND DISPATCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBCOMMANDS = {
    "inspect": cmd_inspect,
    "repair": cmd_repair,
    "analytics": cmd_analytics,
    "status": cmd_status,
    "clean": cmd_clean,
}


def dispatch():
    """
    Dispatch to subcommand or default generation mode.

    Usage:
        python img_generator.py                     # default: generate
        python img_generator.py inspect data.csv    # inspect
        python img_generator.py repair data.csv     # repair
        python img_generator.py analytics data.csv  # analytics
        python img_generator.py status              # show progress
        python img_generator.py clean               # remove checkpoints
    """
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        subcmd = sys.argv[1]
        SUBCOMMANDS[subcmd](sys.argv[2:])
    else:
        main()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    dispatch()