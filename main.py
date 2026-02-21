# # # main.py
# # """
# # Simplified CLI for dataset generation pipeline.
# # """

# # import sys
# # import os
# # from datetime import datetime

# # from config import OUTPUT_DIR, TOTAL_ROWS, MODEL_NAME, ACTIVE_PROFILE


# # def main():
# #     """Main entry point."""

# #     if len(sys.argv) < 2:
# #         show_help()
# #         return

# #     cmd = sys.argv[1].lower()

# #     # ── Task commands ──
# #     if cmd == "run":
# #         if len(sys.argv) < 3:
# #             print("Usage: python main.py run <task> [input.csv]")
# #             print("Tasks:", get_task_list())
# #             return
# #         task_name = sys.argv[2]
# #         input_csv = sys.argv[3] if len(sys.argv) > 3 else None
# #         run_task(task_name, input_csv)

# #     elif cmd == "generate":
# #         # Full pipeline — text → img_desc → cta → audience
# #         run_full_pipeline(force_restart="--fresh" in sys.argv)

# #     elif cmd == "skeleton":
# #         run_skeleton()

# #     elif cmd == "validate":
# #         if len(sys.argv) < 3:
# #             print("Usage: python main.py validate <csv_file>")
# #             return
# #         run_validate(sys.argv[2])

# #     elif cmd == "status":
# #         run_status()

# #     elif cmd == "clean":
# #         run_clean()

# #     elif cmd == "tasks":
# #         print("Available tasks:", get_task_list())

# #     else:
# #         show_help()


# # def get_task_list() -> list:
# #     """Get list of registered tasks."""
# #     from agents.tasks import TaskRegistry
# #     return TaskRegistry.list_tasks()


# # def run_task(task_name: str, input_csv: str = None):
# #     """Run a single task."""
# #     from pipeline.orchestrator import DatasetPipeline

# #     print(f"\n{'═'*60}")
# #     print(f"  Running task: {task_name}")
# #     print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# #     print(f"{'═'*60}")

# #     pipeline = DatasetPipeline()
# #     df = pipeline.run_task(task_name, input_csv=input_csv)

# #     if df.empty:
# #         print("✗ Failed")
# #         sys.exit(1)

# #     print(f"\n✓ Complete: {len(df)} rows")


# # def run_full_pipeline(force_restart: bool = False):
# #     """Run full pipeline with all tasks."""
# #     from pipeline.orchestrator import DatasetPipeline

# #     print(f"\n{'═'*60}")
# #     print(f"  Full Pipeline — {TOTAL_ROWS} Rows")
# #     print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# #     print(f"  Model: {MODEL_NAME}")
# #     print(f"{'═'*60}")

# #     pipeline = DatasetPipeline()

# #     # Run tasks in order
# #     tasks = ["text", "img_desc", "cta", "audience"]
# #     df = pipeline.run_tasks(tasks, force_restart=force_restart)

# #     if df.empty:
# #         print("✗ Failed")
# #         sys.exit(1)

# #     # Validate
# #     from core.validator import validate_dataset, print_full_report
# #     ok, errs = validate_dataset(df)
# #     print_full_report(df)

# #     if ok:
# #         print("✓ ALL VALIDATIONS PASSED")
# #     else:
# #         print(f"⚠ {len(errs)} issue(s)")


# # def run_skeleton():
# #     """Build skeleton only."""
# #     from core.skeleton import build_skeleton, verify_skeleton, print_skeleton_summary

# #     print("Building skeleton...")
# #     df = build_skeleton(seed=42)
# #     errs = verify_skeleton(df)

# #     if errs:
# #         print(f"\n✗ {len(errs)} error(s):")
# #         for e in errs:
# #             print(f"  • {e}")
# #     else:
# #         print("✓ ALL DISTRIBUTIONS PERFECT")

# #     print_skeleton_summary(df)

# #     # Save
# #     os.makedirs(OUTPUT_DIR, exist_ok=True)
# #     path = os.path.join(OUTPUT_DIR, f"skeleton_{TOTAL_ROWS}.csv")
# #     df.to_csv(path, index=False)
# #     print(f"Saved: {path}")


# # def run_validate(filepath: str):
# #     """Validate CSV file."""
# #     import pandas as pd
# #     from core.validator import validate_dataset, print_full_report

# #     print(f"Validating: {filepath}")
# #     df = pd.read_csv(filepath)
# #     ok, errs = validate_dataset(df)
# #     print_full_report(df)

# #     if ok:
# #         print("✓ VALID")
# #     else:
# #         print(f"✗ {len(errs)} issue(s)")


# # def run_status():
# #     """Show progress status for all tasks."""
# #     import json
# #     import glob

# #     progress_files = glob.glob(os.path.join(OUTPUT_DIR, "_progress_*.json"))

# #     if not progress_files:
# #         print("No saved progress found.")
# #         return

# #     for pf in progress_files:
# #         with open(pf, "r") as f:
# #             progress = json.load(f)

# #         task = progress.get("task_name", "?")
# #         completed = progress.get("completed_batches", 0)
# #         total = progress.get("total_batches", 0)
# #         rows = progress.get("completed_rows", 0)
# #         total_rows = progress.get("total_rows", 0)

# #         pct = rows / total_rows * 100 if total_rows > 0 else 0
# #         bar_width = 30
# #         filled = int(bar_width * pct / 100)
# #         bar = "█" * filled + "░" * (bar_width - filled)

# #         print(f"\n  {task.upper()}")
# #         print(f"    [{bar}] {pct:.0f}%")
# #         print(f"    Batches: {completed}/{total} | Rows: {rows}/{total_rows}")
# #         print(f"    Last saved: {progress.get('last_saved', '?')}")


# # def run_clean():
# #     """Remove all progress files."""
# #     import glob

# #     patterns = [
# #         os.path.join(OUTPUT_DIR, "_progress_*.json"),
# #         os.path.join(OUTPUT_DIR, "_progress_*.csv"),
# #         os.path.join(OUTPUT_DIR, "_skeleton.csv"),
# #         os.path.join(OUTPUT_DIR, "_temp_*.csv"),
# #     ]

# #     removed = 0
# #     for pattern in patterns:
# #         for f in glob.glob(pattern):
# #             os.remove(f)
# #             print(f"  Removed: {f}")
# #             removed += 1

# #     print(f"\n  ✓ Cleaned {removed} file(s)" if removed else "  No files to clean")


# # def show_help():
# #     print("""
# # Dataset Generator — Production Pipeline

# # USAGE:
# #     python main.py <command> [args]

# # COMMANDS:
# #     run <task> [csv]    Run single task (optionally on existing CSV)
# #     generate            Run full pipeline (text → img_desc → cta → audience)
# #     generate --fresh    Full pipeline, ignore saved progress
# #     skeleton            Build skeleton only (no LLM)
# #     validate <csv>      Validate CSV file
# #     status              Show progress for all tasks
# #     clean               Remove all progress files
# #     tasks               List available tasks

# # EXAMPLES:
# #     python main.py run text                     # Generate text column
# #     python main.py run img_desc dataset.csv    # Add img_desc to existing CSV
# #     python main.py generate                     # Full pipeline
# #     python main.py status                       # Check progress
# # """)


# # if __name__ == "__main__":
# #     main()

# # main.py
# """
# Rich CLI for dataset generation pipeline.
# FIXED: No re-running of completed tasks.
# """

# import sys
# import os
# import json
# import glob
# from datetime import datetime
# from typing import List

# from config import OUTPUT_DIR, TOTAL_ROWS, MODEL_NAME, ACTIVE_PROFILE

# from pipeline.display import DisplayManager, RICH_AVAILABLE

# display = DisplayManager(verbose=True)


# def main():
#     """Main entry point."""

#     if len(sys.argv) < 2:
#         show_help([])
#         return

#     cmd = sys.argv[1].lower()
#     args = sys.argv[2:]

#     commands = {
#         "run": cmd_run,
#         "generate": cmd_generate,
#         "skeleton": cmd_skeleton,
#         "validate": cmd_validate,
#         "status": cmd_status,
#         "clean": cmd_clean,
#         "tasks": cmd_tasks,
#         "help": show_help,
#         "--help": show_help,
#         "-h": show_help,
#     }

#     if cmd in commands:
#         commands[cmd](args)
#     else:
#         display.error(f"Unknown command: {cmd}")
#         show_help([])


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #  COMMANDS
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# def cmd_run(args: List[str]):
#     """Run a single task."""
#     if not args:
#         display.error("Usage: python main.py run <task> [input.csv]")
#         cmd_tasks([])
#         return

#     task_name = args[0]
#     input_csv = args[1] if len(args) > 1 else None

#     # Validate input file
#     if input_csv and not os.path.exists(input_csv):
#         display.error(f"File not found: {input_csv}")
#         return

#     # Check for --fresh flag
#     force_restart = "--fresh" in args or "-f" in args

#     from pipeline.orchestrator import DatasetPipeline

#     pipeline = DatasetPipeline()
#     df = pipeline.run_task(task_name, input_csv=input_csv, force_restart=force_restart)

#     if df.empty:
#         display.error("Task failed")
#         sys.exit(1)
#     else:
#         display.success(f"Task complete: {len(df)} rows")


# def cmd_generate(args: List[str]):
#     """Run full pipeline."""
#     force_restart = "--fresh" in args or "-f" in args

#     display.banner(
#         title="🚀 Dataset Generator",
#         subtitle=f"Full Pipeline — {TOTAL_ROWS:,} Rows",
#         data={
#             "Model": MODEL_NAME,
#             "Profile": ACTIVE_PROFILE,
#             "Tasks": "text → img_desc → cta → audience",
#             "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         },
#         style="magenta",
#     )

#     if force_restart:
#         display.warning("Fresh start mode — ignoring saved progress")

#     from pipeline.orchestrator import DatasetPipeline

#     pipeline = DatasetPipeline()
#     tasks = ["text", "img_desc", "cta", "audience"]
    
#     df = pipeline.run_tasks(tasks, force_restart=force_restart)

#     if df.empty:
#         display.error("Pipeline failed")
#         sys.exit(1)


# def cmd_skeleton(args: List[str]):
#     """Build skeleton only."""
#     display.banner(
#         title="📊 Skeleton Builder",
#         data={"Total Rows": f"{TOTAL_ROWS:,}", "Output": OUTPUT_DIR},
#         style="cyan",
#     )

#     from core.skeleton import build_skeleton, verify_skeleton

#     with display.spinner("Building skeleton"):
#         df = build_skeleton(seed=42)
#         errs = verify_skeleton(df)

#     if errs:
#         display.error(f"{len(errs)} error(s):")
#         for e in errs:
#             display.log(f"• {e}", style="red")
#         sys.exit(1)

#     display.success("All distributions verified")

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, f"skeleton_{TOTAL_ROWS}.csv")
#     df.to_csv(path, index=False)

#     display.file_saved(path, len(df), len(df.columns), os.path.getsize(path) / 1024)


# def cmd_validate(args: List[str]):
#     """Validate CSV file."""
#     if not args:
#         display.error("Usage: python main.py validate <csv_file>")
#         return

#     filepath = args[0]
#     if not os.path.exists(filepath):
#         display.error(f"File not found: {filepath}")
#         return

#     display.banner(title="🔍 Dataset Validator", data={"File": filepath}, style="blue")

#     import pandas as pd
#     from core.validator import validate_dataset, print_full_report

#     with display.spinner("Loading and validating"):
#         df = pd.read_csv(filepath)
#         ok, errs = validate_dataset(df)

#     display.file_loaded(filepath, len(df), len(df.columns))
#     display.validation_result(ok, errs)
#     print_full_report(df)


# def cmd_status(args: List[str]):
#     """Show progress status."""
#     display.header("Progress Status")

#     progress_files = glob.glob(os.path.join(OUTPUT_DIR, "_progress_*.json"))

#     if not progress_files:
#         display.warning("No saved progress found")
#         display.info("Run 'python main.py generate' to start")
#         return

#     progress_list = []
#     for pf in progress_files:
#         try:
#             with open(pf, "r") as f:
#                 progress_list.append(json.load(f))
#         except (json.JSONDecodeError, IOError):
#             continue

#     display.show_status(progress_list)


# def cmd_clean(args: List[str]):
#     """Remove all progress files."""
#     display.header("Cleaning Progress Files")

#     patterns = [
#         os.path.join(OUTPUT_DIR, "_progress_*.json"),
#         os.path.join(OUTPUT_DIR, "_progress_*.csv"),
#         os.path.join(OUTPUT_DIR, "_skeleton.csv"),
#         os.path.join(OUTPUT_DIR, "_temp_*.csv"),
#     ]

#     removed = 0
#     for pattern in patterns:
#         for f in glob.glob(pattern):
#             os.remove(f)
#             display.success(f"Removed: {os.path.basename(f)}")
#             removed += 1

#     if removed == 0:
#         display.info("No progress files to clean")
#     else:
#         display.success(f"Cleaned {removed} file(s)")


# def cmd_tasks(args: List[str]):
#     """List available tasks."""
#     from agents.tasks import TaskRegistry
#     tasks = TaskRegistry.all()
#     display.show_tasks(tasks)


# def show_help(args: List[str] = None):
#     """Show help."""
#     display.show_help()


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #  ENTRY POINT
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         display.warning("\nInterrupted by user")
#         sys.exit(130)
#     except Exception as e:
#         display.error(f"Unexpected error: {e}")
#         if RICH_AVAILABLE:
#             from rich.console import Console
#             Console().print_exception()
#         sys.exit(1)


# main.py
"""
Rich CLI for dataset generation pipeline.
Features: CSV auto-detection, path resolution, interactive selection.
"""

import sys
import os
import json
import glob
from datetime import datetime
from typing import List, Optional

from config import OUTPUT_DIR, TOTAL_ROWS, MODEL_NAME, ACTIVE_PROFILE

from pipeline.display import DisplayManager, RICH_AVAILABLE

display = DisplayManager(verbose=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSV RESOLUTION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_csv(csv_input: Optional[str] = None) -> Optional[str]:
    """
    Find CSV file from input.
    
    Resolution order:
    1. If csv_input is a valid path → use it
    2. If csv_input is a filename → search in OUTPUT_DIR, current dir
    3. If csv_input is None → list available CSVs for selection
    
    Returns:
        Resolved CSV path or None
    """
    if csv_input:
        # Direct path exists
        if os.path.exists(csv_input):
            return os.path.abspath(csv_input)
        
        # Check in OUTPUT_DIR
        output_path = os.path.join(OUTPUT_DIR, csv_input)
        if os.path.exists(output_path):
            return os.path.abspath(output_path)
        
        # Check in current directory
        current_path = os.path.join(os.getcwd(), csv_input)
        if os.path.exists(current_path):
            return os.path.abspath(current_path)
        
        # Try adding .csv extension
        if not csv_input.endswith('.csv'):
            return find_csv(csv_input + '.csv')
        
        # Not found
        display.error(f"CSV not found: {csv_input}")
        display.info(f"Searched in:")
        display.log(f"  • {os.path.abspath(csv_input)}")
        display.log(f"  • {os.path.abspath(output_path)}")
        display.log(f"  • {os.path.abspath(current_path)}")
        
        # Show available CSVs
        show_available_csvs()
        return None
    
    else:
        # No input — show selection
        return select_csv_interactive()


def get_available_csvs() -> List[dict]:
    """
    Get list of available CSV files with metadata.
    Searches: OUTPUT_DIR, current directory.
    """
    csvs = []
    search_dirs = [
        OUTPUT_DIR,
        os.getcwd(),
    ]
    
    seen_paths = set()
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        pattern = os.path.join(search_dir, "*.csv")
        for filepath in glob.glob(pattern):
            abs_path = os.path.abspath(filepath)
            
            # Skip duplicates
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            
            # Skip progress files
            basename = os.path.basename(filepath)
            if basename.startswith("_progress") or basename.startswith("_temp"):
                continue
            
            # Get file info
            try:
                stat = os.stat(filepath)
                size_kb = stat.st_size / 1024
                mtime = datetime.fromtimestamp(stat.st_mtime)
                
                # Try to get row count (quick check)
                with open(filepath, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
                
                csvs.append({
                    "path": abs_path,
                    "name": basename,
                    "dir": os.path.dirname(abs_path),
                    "size_kb": size_kb,
                    "rows": row_count,
                    "modified": mtime,
                })
            except Exception:
                csvs.append({
                    "path": abs_path,
                    "name": basename,
                    "dir": os.path.dirname(abs_path),
                    "size_kb": 0,
                    "rows": "?",
                    "modified": None,
                })
    
    # Sort by modification time (newest first)
    csvs.sort(key=lambda x: x["modified"] or datetime.min, reverse=True)
    
    return csvs


def show_available_csvs():
    """Display available CSV files."""
    csvs = get_available_csvs()
    
    if not csvs:
        display.warning("No CSV files found")
        display.info(f"Searched in: {OUTPUT_DIR}, {os.getcwd()}")
        return
    
    display.header("Available CSV Files")
    
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box
        
        table = Table(
            title="📂 CSV Files",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Filename", style="green")
        table.add_column("Rows", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Modified", style="dim")
        table.add_column("Directory", style="dim", max_width=30)
        
        for i, csv in enumerate(csvs[:15], 1):  # Show max 15
            mtime_str = csv["modified"].strftime("%Y-%m-%d %H:%M") if csv["modified"] else "?"
            table.add_row(
                str(i),
                csv["name"],
                str(csv["rows"]),
                f"{csv['size_kb']:.1f} KB",
                mtime_str,
                csv["dir"][-30:] if len(csv["dir"]) > 30 else csv["dir"],
            )
        
        display.console.print(table)
    else:
        print(f"\n  Available CSVs:")
        for i, csv in enumerate(csvs[:15], 1):
            print(f"    {i}. {csv['name']} ({csv['rows']} rows, {csv['size_kb']:.1f} KB)")


def select_csv_interactive() -> Optional[str]:
    """Interactive CSV selection."""
    csvs = get_available_csvs()
    
    if not csvs:
        display.warning("No CSV files found")
        display.info(f"Run 'python main.py skeleton' to create one")
        return None
    
    show_available_csvs()
    
    if len(csvs) == 1:
        display.info(f"Auto-selecting: {csvs[0]['name']}")
        return csvs[0]["path"]
    
    print()
    display.info("Enter number to select, or filename/path:")
    
    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    
    if not choice:
        # Default to most recent
        display.info(f"Using most recent: {csvs[0]['name']}")
        return csvs[0]["path"]
    
    # Try as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(csvs):
            return csvs[idx]["path"]
    except ValueError:
        pass
    
    # Try as filename/path
    return find_csv(choice)


def get_latest_dataset() -> Optional[str]:
    """Get the most recent dataset CSV (excluding skeleton)."""
    csvs = get_available_csvs()
    
    for csv in csvs:
        # Skip skeleton files
        if "skeleton" in csv["name"].lower():
            continue
        return csv["path"]
    
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN & COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help([])
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    commands = {
        "run": cmd_run,
        "generate": cmd_generate,
        "skeleton": cmd_skeleton,
        "validate": cmd_validate,
        "status": cmd_status,
        "clean": cmd_clean,
        "tasks": cmd_tasks,
        "list": cmd_list,
        "ls": cmd_list,
        "help": show_help,
        "--help": show_help,
        "-h": show_help,
    }

    if cmd in commands:
        commands[cmd](args)
    else:
        display.error(f"Unknown command: {cmd}")
        show_help([])


def cmd_run(args: List[str]):
    """
    Run a single task.
    
    Usage:
        python main.py run <task>                    # Interactive CSV selection
        python main.py run <task> <csv>              # Specify CSV
        python main.py run <task> <csv> --fresh      # Force restart
        python main.py run <task> --latest           # Use most recent dataset
    """
    if not args:
        display.error("Usage: python main.py run <task> [csv] [--fresh] [--latest]")
        cmd_tasks([])
        return

    task_name = args[0]
    
    # Parse flags
    force_restart = "--fresh" in args or "-f" in args
    use_latest = "--latest" in args or "-l" in args
    
    # Remove flags from args
    remaining_args = [a for a in args[1:] if not a.startswith("-")]
    
    # Determine input CSV
    if use_latest:
        input_csv = get_latest_dataset()
        if not input_csv:
            display.error("No dataset found. Run 'python main.py generate' first.")
            return
        display.info(f"Using latest dataset: {os.path.basename(input_csv)}")
    elif remaining_args:
        input_csv = find_csv(remaining_args[0])
        if not input_csv:
            return  # Error already shown
    else:
        # Interactive selection or skeleton
        display.info("No CSV specified. Select input or press Enter to build skeleton:")
        input_csv = select_csv_interactive()
        # None means build skeleton (that's OK)

    from pipeline.orchestrator import DatasetPipeline

    pipeline = DatasetPipeline()
    df = pipeline.run_task(task_name, input_csv=input_csv, force_restart=force_restart)

    if df.empty:
        display.error("Task failed")
        sys.exit(1)
    else:
        display.success(f"Task complete: {len(df)} rows")


def cmd_generate(args: List[str]):
    """Run full pipeline."""
    force_restart = "--fresh" in args or "-f" in args
    
    # Check for input CSV
    remaining_args = [a for a in args if not a.startswith("-")]
    input_csv = None
    
    if remaining_args:
        input_csv = find_csv(remaining_args[0])
        if not input_csv:
            return

    display.banner(
        title="🚀 Dataset Generator",
        subtitle=f"Full Pipeline — {TOTAL_ROWS:,} Rows",
        data={
            "Model": MODEL_NAME,
            "Profile": ACTIVE_PROFILE,
            "Tasks": "text → img_desc → cta → audience",
            "Input": os.path.basename(input_csv) if input_csv else "New skeleton",
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        style="magenta",
    )

    if force_restart:
        display.warning("Fresh start mode — ignoring saved progress")

    from pipeline.orchestrator import DatasetPipeline

    pipeline = DatasetPipeline()
    tasks = ["text", "img_desc", "cta", "audience"]
    
    df = pipeline.run_tasks(tasks, force_restart=force_restart, input_csv=input_csv)

    if df.empty:
        display.error("Pipeline failed")
        sys.exit(1)


def cmd_skeleton(args: List[str]):
    """Build skeleton only."""
    display.banner(
        title="📊 Skeleton Builder",
        data={"Total Rows": f"{TOTAL_ROWS:,}", "Output": OUTPUT_DIR},
        style="cyan",
    )

    from core.skeleton import build_skeleton, verify_skeleton

    with display.spinner("Building skeleton"):
        df = build_skeleton(seed=42)
        errs = verify_skeleton(df)

    if errs:
        display.error(f"{len(errs)} error(s):")
        for e in errs:
            display.log(f"• {e}", style="red")
        sys.exit(1)

    display.success("All distributions verified")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"skeleton_{TOTAL_ROWS}.csv")
    df.to_csv(path, index=False)

    display.file_saved(path, len(df), len(df.columns), os.path.getsize(path) / 1024)


def cmd_validate(args: List[str]):
    """Validate CSV file."""
    # Auto-find CSV
    csv_input = args[0] if args else None
    filepath = find_csv(csv_input)
    
    if not filepath:
        if not csv_input:
            display.error("Usage: python main.py validate <csv_file>")
        return

    display.banner(title="🔍 Dataset Validator", data={"File": filepath}, style="blue")

    import pandas as pd
    from core.validator import validate_dataset, print_full_report

    with display.spinner("Loading and validating"):
        df = pd.read_csv(filepath)
        ok, errs = validate_dataset(df)

    display.file_loaded(filepath, len(df), len(df.columns))
    display.validation_result(ok, errs)
    print_full_report(df)


def cmd_status(args: List[str]):
    """Show progress status."""
    display.header("Progress Status")

    progress_files = glob.glob(os.path.join(OUTPUT_DIR, "_progress_*.json"))

    if not progress_files:
        display.warning("No saved progress found")
        display.info("Run 'python main.py generate' to start")
        return

    progress_list = []
    for pf in progress_files:
        try:
            with open(pf, "r") as f:
                progress_list.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue

    display.show_status(progress_list)


def cmd_clean(args: List[str]):
    """Remove all progress files."""
    display.header("Cleaning Progress Files")

    patterns = [
        os.path.join(OUTPUT_DIR, "_progress_*.json"),
        os.path.join(OUTPUT_DIR, "_progress_*.csv"),
        os.path.join(OUTPUT_DIR, "_skeleton.csv"),
        os.path.join(OUTPUT_DIR, "_temp_*.csv"),
    ]

    removed = 0
    for pattern in patterns:
        for f in glob.glob(pattern):
            os.remove(f)
            display.success(f"Removed: {os.path.basename(f)}")
            removed += 1

    if removed == 0:
        display.info("No progress files to clean")
    else:
        display.success(f"Cleaned {removed} file(s)")


def cmd_tasks(args: List[str]):
    """List available tasks."""
    from agents.tasks import TaskRegistry
    tasks = TaskRegistry.all()
    display.show_tasks(tasks)


def cmd_list(args: List[str]):
    """List available CSV files."""
    show_available_csvs()


def show_help(args: List[str] = None):
    """Show help."""
    help_text = """
Dataset Generator — Production Pipeline

USAGE:
    python main.py <command> [args]

COMMANDS:
    run <task> [csv]         Run single task
    run <task> --latest      Run task on most recent dataset
    generate [csv]           Run full pipeline
    generate --fresh         Full pipeline, ignore progress
    skeleton                 Build skeleton only
    validate [csv]           Validate dataset
    status                   Show progress
    list / ls                List available CSV files
    clean                    Remove progress files
    tasks                    List available tasks
    help                     Show this help

EXAMPLES:
    python main.py run text                      # Interactive CSV selection
    python main.py run text dataset.csv         # Specify CSV
    python main.py run audience 01.csv          # Just filename (auto-find)
    python main.py run cta --latest             # Use most recent dataset
    python main.py generate                     # Full pipeline
    python main.py generate skeleton.csv        # Start from skeleton
    python main.py list                         # Show available CSVs
    python main.py validate                     # Validate (interactive select)

CSV RESOLUTION:
    The system searches for CSV files in:
    1. Exact path provided
    2. generated_datasets/ directory
    3. Current working directory
    
    You can provide just the filename without path.
"""
    if RICH_AVAILABLE:
        from rich.markdown import Markdown
        from rich.panel import Panel
        display.console.print(Panel(Markdown(help_text), title="📖 Help", border_style="blue"))
    else:
        print(help_text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        display.warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        display.error(f"Unexpected error: {e}")
        if RICH_AVAILABLE:
            from rich.console import Console
            Console().print_exception()
        sys.exit(1)