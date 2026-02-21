# pipeline/display.py
"""
Rich CLI display manager — beautiful terminal output.
"""

import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn, ProgressColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    from rich.columns import Columns
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich import box
    from rich.spinner import Spinner
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class SpeedColumn(ProgressColumn):
    """Custom column showing rows/sec."""
    
    def render(self, task) -> Text:
        if task.elapsed is None or task.elapsed == 0:
            return Text("--.-/s", style="cyan")
        speed = task.completed / task.elapsed
        return Text(f"{speed:.1f}/s", style="cyan")


class DisplayManager:
    """
    Rich display manager for beautiful terminal output.
    Falls back to basic print if rich is not available.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE else None
        self._start_time = time.time()
        self._batch_times: List[float] = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BANNERS & HEADERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def banner(
        self,
        title: str,
        subtitle: str = "",
        data: Dict[str, str] = None,
        style: str = "blue",
    ):
        """Display a fancy banner with optional data table."""
        if not RICH_AVAILABLE:
            print(f"\n{'═'*60}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            if data:
                for k, v in data.items():
                    print(f"    {k}: {v}")
            print(f"{'═'*60}\n")
            return

        # Build content
        content_parts = []
        
        if subtitle:
            content_parts.append(Text(subtitle, style="dim"))
        
        if data:
            table = Table(
                show_header=False,
                box=None,
                padding=(0, 2),
                collapse_padding=True,
            )
            table.add_column("Key", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            for k, v in data.items():
                table.add_row(k, str(v))
            
            content_parts.append(table)

        # Create panel
        panel = Panel(
            Columns(content_parts, expand=True) if len(content_parts) > 1 
            else (content_parts[0] if content_parts else ""),
            title=f"[bold]{title}[/bold]",
            border_style=style,
            padding=(1, 2),
        )
        
        self.console.print()
        self.console.print(panel)

    def header(self, text: str, style: str = "bold cyan"):
        """Print a section header."""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.rule(f"[{style}]{text}[/{style}]", style=style)
        else:
            print(f"\n{'━'*60}")
            print(f"  {text}")
            print(f"{'━'*60}")

    def subheader(self, text: str, style: str = "cyan"):
        """Print a subsection header."""
        if RICH_AVAILABLE:
            self.console.print(f"\n  [{style}]▸ {text}[/{style}]")
        else:
            print(f"\n  ▸ {text}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STATUS MESSAGES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def success(self, message: str):
        """Print success message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [green]✓[/green] {message}")
        else:
            print(f"  ✓ {message}")

    def error(self, message: str):
        """Print error message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [red]✗[/red] {message}")
        else:
            print(f"  ✗ {message}")

    def warning(self, message: str):
        """Print warning message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [yellow]⚠[/yellow] {message}")
        else:
            print(f"  ⚠ {message}")

    def info(self, message: str):
        """Print info message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [blue]ℹ[/blue] {message}")
        else:
            print(f"  ℹ {message}")

    def log(self, message: str, style: str = "dim"):
        """Print log message."""
        if not self.verbose:
            return
        if RICH_AVAILABLE:
            self.console.print(f"    [{style}]{message}[/{style}]")
        else:
            print(f"    {message}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PROGRESS BARS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_progress(self) -> Progress:
        """Create a rich progress bar."""
        if not RICH_AVAILABLE:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            SpeedColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
            transient=False,
        )

    def create_simple_progress(self, description: str = "Processing") -> Progress:
        """Create a simpler progress bar."""
        if not RICH_AVAILABLE:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TABLES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def table(
        self,
        title: str,
        columns: List[str],
        rows: List[List[Any]],
        show_header: bool = True,
        box_style: Any = None,
    ):
        """Display a table."""
        if not RICH_AVAILABLE:
            print(f"\n  {title}")
            if show_header:
                print("    " + " | ".join(str(c) for c in columns))
                print("    " + "-" * (sum(len(str(c)) for c in columns) + 3 * len(columns)))
            for row in rows:
                print("    " + " | ".join(str(c) for c in row))
            return

        table = Table(
            title=title,
            box=box_style or box.ROUNDED,
            show_header=show_header,
            header_style="bold magenta",
        )
        
        for col in columns:
            table.add_column(col)
        
        for row in rows:
            table.add_row(*[str(c) for c in row])
        
        self.console.print(table)

    def key_value_table(
        self,
        title: str,
        data: Dict[str, Any],
        style: str = "cyan",
    ):
        """Display a key-value table."""
        if not RICH_AVAILABLE:
            print(f"\n  {title}")
            for k, v in data.items():
                print(f"    {k}: {v}")
            return

        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=False,
            title_style=f"bold {style}",
            border_style=style,
        )
        table.add_column("Key", style="cyan", no_wrap=True, width=20)
        table.add_column("Value", style="green")
        
        for k, v in data.items():
            table.add_row(str(k), str(v))
        
        self.console.print(table)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TASK DISPLAYS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def task_banner(
        self,
        task_name: str,
        columns: List[str],
        model: str,
        profile: str,
        total_rows: int,
    ):
        """Display task start banner."""
        data = {
            "Columns": ", ".join(columns),
            "Model": model,
            "Profile": profile,
            "Total Rows": f"{total_rows:,}",
            "Started": datetime.now().strftime("%H:%M:%S"),
        }
        self.banner(
            title=f"🚀 Task: {task_name.upper()}",
            data=data,
            style="cyan",
        )

    def task_complete(
        self,
        task_name: str,
        total_rows: int,
        elapsed: float,
        filled: int,
        output_path: str,
    ):
        """Display task completion summary."""
        speed = total_rows / elapsed if elapsed > 0 else 0
        
        if not RICH_AVAILABLE:
            print(f"\n  ✓ Task '{task_name}' complete")
            print(f"    Rows: {filled}/{total_rows}")
            print(f"    Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"    Speed: {speed:.1f} rows/sec")
            print(f"    Output: {output_path}")
            return

        data = {
            "Rows Filled": f"{filled:,} / {total_rows:,}",
            "Time": f"{elapsed:.1f}s ({elapsed/60:.1f} min)",
            "Speed": f"{speed:.1f} rows/sec",
            "Output": output_path,
        }
        
        self.banner(
            title=f"✓ Task Complete: {task_name.upper()}",
            data=data,
            style="green",
        )

    def batch_summary(
        self,
        batch_num: int,
        total_batches: int,
        batch_size: int,
        elapsed: float,
        filled_total: int,
        total_rows: int,
    ):
        """Display batch completion summary."""
        speed = batch_size / elapsed if elapsed > 0 else 0
        pct = (filled_total / total_rows) * 100 if total_rows > 0 else 0
        
        if RICH_AVAILABLE:
            self.console.print(
                f"    [green]✓[/green] Batch {batch_num}/{total_batches} | "
                f"[cyan]{elapsed:.1f}s[/cyan] | "
                f"[yellow]{speed:.1f}/s[/yellow] | "
                f"Total: [green]{filled_total:,}[/green]/{total_rows:,} "
                f"([magenta]{pct:.0f}%[/magenta])"
            )
        else:
            print(
                f"    ✓ Batch {batch_num}/{total_batches} | "
                f"{elapsed:.1f}s | {speed:.1f}/s | "
                f"Total: {filled_total}/{total_rows} ({pct:.0f}%)"
            )

    def phase_status(self, phase: int, name: str, filled: int, total: int, complete: bool = False):
        """Display phase completion status."""
        pct = (filled / total) * 100 if total > 0 else 0
        
        if RICH_AVAILABLE:
            status = "[green]✓ COMPLETE[/green]" if complete else f"[yellow]{filled}/{total}[/yellow]"
            self.console.print(
                f"      Phase {phase} ({name}): {status} "
                f"[dim]({pct:.0f}%)[/dim]"
            )
        else:
            status = "✓ COMPLETE" if complete else f"{filled}/{total}"
            print(f"      Phase {phase} ({name}): {status} ({pct:.0f}%)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PROGRESS TRACKING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def progress_saved(self, batch: int, total_batches: int, rows_done: int, total_rows: int):
        """Display progress saved notification."""
        if RICH_AVAILABLE:
            self.console.print(
                f"    [dim]💾 Progress saved: batch {batch}/{total_batches}, "
                f"{rows_done:,}/{total_rows:,} rows[/dim]"
            )
        else:
            print(f"    💾 Progress saved: batch {batch}/{total_batches}")

    def resume_prompt(self, progress: Dict) -> str:
        """Display resume prompt and get user choice."""
        completed = progress.get("completed_batches", 0)
        total = progress.get("total_batches", 0)
        rows_done = progress.get("completed_rows", 0)
        total_rows = progress.get("total_rows", 0)
        pct = rows_done / total_rows * 100 if total_rows > 0 else 0

        if RICH_AVAILABLE:
            # Progress bar visualization
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            table = Table(
                title="💾 Saved Progress Found",
                box=box.DOUBLE_EDGE,
                show_header=False,
                title_style="bold yellow",
                border_style="yellow",
            )
            table.add_column("Field", style="cyan", width=15)
            table.add_column("Value", style="green", width=35)

            table.add_row("Task", progress.get("task_name", "?"))
            table.add_row("Batches", f"{completed} / {total}")
            table.add_row("Rows", f"{rows_done:,} / {total_rows:,}")
            table.add_row("Progress", f"[yellow]{bar}[/yellow] {pct:.0f}%")
            table.add_row("Model", progress.get("model", "?"))
            table.add_row("Last Saved", progress.get("last_saved", "?"))

            self.console.print()
            self.console.print(table)
            self.console.print()
            self.console.print("  [bold]Options:[/bold]")
            self.console.print(f"    [green][R][/green] Resume from batch {completed + 1}")
            self.console.print("    [red][F][/red] Fresh start (discard progress)")
            self.console.print("    [dim][Enter][/dim] Resume (default)")
            self.console.print()

        else:
            print(f"\n  ╔{'═'*50}╗")
            print(f"  ║  SAVED PROGRESS FOUND                          ║")
            print(f"  ╠{'═'*50}╣")
            print(f"  ║  Batches : {completed}/{total}{' '*(38-len(f'{completed}/{total}'))}║")
            print(f"  ║  Rows    : {rows_done}/{total_rows}{' '*(38-len(f'{rows_done}/{total_rows}'))}║")
            print(f"  ╚{'═'*50}╝")
            print(f"\n  [R] Resume  |  [F] Fresh start")

        try:
            choice = input("  Your choice [R/F]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "R"

        return choice

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  VALIDATION DISPLAYS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validation_result(self, is_valid: bool, errors: List[str]):
        """Display validation results."""
        if is_valid:
            if RICH_AVAILABLE:
                panel = Panel(
                    "[green]All validations passed[/green]",
                    title="✓ Valid Dataset",
                    border_style="green",
                )
                self.console.print(panel)
            else:
                print("\n  ✓ ALL VALIDATIONS PASSED")
        else:
            if RICH_AVAILABLE:
                error_text = "\n".join(f"• {e}" for e in errors)
                panel = Panel(
                    f"[red]{error_text}[/red]",
                    title=f"✗ {len(errors)} Validation Error(s)",
                    border_style="red",
                )
                self.console.print(panel)
            else:
                print(f"\n  ✗ {len(errors)} issue(s):")
                for e in errors:
                    print(f"    • {e}")

    def distribution_table(
        self,
        title: str,
        column: str,
        expected: Dict[str, int],
        actual: Dict[str, int],
    ):
        """Display distribution comparison table."""
        if not RICH_AVAILABLE:
            print(f"\n  {title}:")
            for val in expected:
                exp = expected[val]
                got = actual.get(val, 0)
                ok = "✓" if got == exp else "✗"
                print(f"    {ok} {val}: {got}/{exp}")
            return

        table = Table(
            title=title,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Value", style="cyan")
        table.add_column("Expected", justify="right")
        table.add_column("Actual", justify="right")
        table.add_column("Status", justify="center")

        for val in expected:
            exp = expected[val]
            got = actual.get(val, 0)
            status = "[green]✓[/green]" if got == exp else "[red]✗[/red]"
            got_style = "green" if got == exp else "red"
            table.add_row(str(val), str(exp), f"[{got_style}]{got}[/{got_style}]", status)

        self.console.print(table)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SKELETON DISPLAYS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def skeleton_summary(
        self,
        total_rows: int,
        themes: int,
        objects_per_theme: int,
        cycles: int,
        base_unit: int,
    ):
        """Display skeleton build summary."""
        data = {
            "Total Rows": f"{total_rows:,}",
            "Themes": str(themes),
            "Objects/Theme": str(objects_per_theme),
            "Total Objects": str(themes * objects_per_theme),
            "Cycles": str(cycles),
            "Base Unit": str(base_unit),
        }
        self.banner(
            title="📊 Skeleton Built",
            data=data,
            style="green",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FILE OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def file_saved(self, path: str, rows: int, cols: int, size_kb: float):
        """Display file saved notification."""
        if RICH_AVAILABLE:
            panel = Panel(
                f"[green]{path}[/green]\n"
                f"[dim]{rows:,} rows × {cols} columns ({size_kb:.1f} KB)[/dim]",
                title="💾 Dataset Saved",
                border_style="green",
            )
            self.console.print(panel)
        else:
            print(f"\n  ✓ Saved: {path}")
            print(f"    {rows:,} rows × {cols} cols ({size_kb:.1f} KB)")

    def file_loaded(self, path: str, rows: int, cols: int):
        """Display file loaded notification."""
        if RICH_AVAILABLE:
            self.console.print(
                f"  [blue]📂[/blue] Loaded: [cyan]{path}[/cyan] "
                f"[dim]({rows:,} rows × {cols} cols)[/dim]"
            )
        else:
            print(f"  📂 Loaded: {path} ({rows:,} rows × {cols} cols)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SPINNER / STATUS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def spinner(self, message: str):
        """Create a spinner context manager."""
        if RICH_AVAILABLE:
            return self.console.status(f"[bold cyan]{message}[/bold cyan]", spinner="dots")
        else:
            # Fake context manager for non-rich
            class FakeSpinner:
                def __enter__(self):
                    print(f"  {message}...", end="", flush=True)
                    return self
                def __exit__(self, *args):
                    print(" done")
            return FakeSpinner()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELP & TASKS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def show_tasks(self, tasks: Dict[str, Any]):
        """Display available tasks."""
        if not RICH_AVAILABLE:
            print("\nAvailable Tasks:")
            for name, task in tasks.items():
                print(f"  • {name}: {task.output_columns}")
            return

        table = Table(
            title="📋 Available Tasks",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Task Name", style="cyan", no_wrap=True)
        table.add_column("Output Columns", style="green")
        table.add_column("Primary", style="yellow")
        table.add_column("Min Length", justify="right")

        for name, task in tasks.items():
            table.add_row(
                name,
                ", ".join(task.output_columns),
                task.primary_column,
                str(task.min_length),
            )

        self.console.print()
        self.console.print(table)

    def show_help(self):
        """Display help information."""
        if not RICH_AVAILABLE:
            print("""
Dataset Generator — Production Pipeline

USAGE:
    python main.py <command> [args]

COMMANDS:
    run <task> [csv]    Run single task
    generate            Run full pipeline
    skeleton            Build skeleton only
    validate <csv>      Validate dataset
    status              Show progress
    clean               Remove progress files
    tasks               List available tasks
    help                Show this help
""")
            return

        help_text = """
[bold cyan]Dataset Generator[/bold cyan] — Production Pipeline

[bold]USAGE:[/bold]
    python main.py <command> [args]

[bold]COMMANDS:[/bold]
    [green]run[/green] <task> [csv]    Run single task (optionally on existing CSV)
    [green]generate[/green]            Run full pipeline (text → img_desc → cta → audience)
    [green]generate --fresh[/green]    Full pipeline, ignore saved progress
    [green]skeleton[/green]            Build skeleton only (no LLM calls)
    [green]validate[/green] <csv>      Validate CSV file
    [green]status[/green]              Show progress for all tasks
    [green]clean[/green]               Remove all progress files
    [green]tasks[/green]               List available tasks
    [green]help[/green]                Show this help

[bold]EXAMPLES:[/bold]
    python main.py run text                     # Generate text column
    python main.py run img_desc dataset.csv    # Add img_desc to existing CSV
    python main.py generate                    # Full pipeline
    python main.py status                      # Check progress
"""
        panel = Panel(
            Markdown(help_text),
            title="📖 Help",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    def show_status(self, progress_list: List[Dict]):
        """Display status for all tasks."""
        if not progress_list:
            self.warning("No saved progress found.")
            return

        if not RICH_AVAILABLE:
            for p in progress_list:
                task = p.get("task_name", "?")
                completed = p.get("completed_batches", 0)
                total = p.get("total_batches", 0)
                print(f"\n  {task}: {completed}/{total} batches")
            return

        self.console.print()
        
        for p in progress_list:
            task = p.get("task_name", "?")
            completed = p.get("completed_batches", 0)
            total = p.get("total_batches", 0)
            rows = p.get("completed_rows", 0)
            total_rows = p.get("total_rows", 0)
            pct = rows / total_rows * 100 if total_rows > 0 else 0

            # Progress bar
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            table = Table(
                title=f"📊 {task.upper()}",
                box=box.ROUNDED,
                show_header=False,
                border_style="cyan",
            )
            table.add_column("Key", style="cyan", width=15)
            table.add_column("Value", style="green", width=45)

            table.add_row("Progress", f"[yellow]{bar}[/yellow] {pct:.0f}%")
            table.add_row("Batches", f"{completed} / {total}")
            table.add_row("Rows", f"{rows:,} / {total_rows:,}")
            table.add_row("Model", p.get("model", "?"))
            table.add_row("Last Saved", p.get("last_saved", "?"))

            self.console.print(table)
            self.console.print()

        self.info(f"Run 'python main.py run <task>' to resume")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FINAL SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def final_summary(
        self,
        tasks_completed: List[str],
        total_time: float,
        total_rows: int,
        output_path: str,
        validation_passed: bool,
    ):
        """Display final pipeline summary."""
        speed = total_rows / total_time if total_time > 0 else 0

        if not RICH_AVAILABLE:
            print(f"\n{'═'*60}")
            print(f"  PIPELINE COMPLETE")
            print(f"{'═'*60}")
            print(f"  Tasks: {', '.join(tasks_completed)}")
            print(f"  Rows: {total_rows:,}")
            print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Speed: {speed:.1f} rows/sec")
            print(f"  Validation: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
            print(f"  Output: {output_path}")
            print(f"{'═'*60}\n")
            return

        # Build summary table
        table = Table(
            box=box.DOUBLE_EDGE,
            show_header=False,
            border_style="green" if validation_passed else "red",
            padding=(0, 2),
        )
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=40)

        table.add_row("Tasks", ", ".join(tasks_completed))
        table.add_row("Total Rows", f"{total_rows:,}")
        table.add_row("Total Time", f"{total_time:.1f}s ({total_time/60:.1f} min)")
        table.add_row("Speed", f"{speed:.1f} rows/sec")
        table.add_row(
            "Validation",
            "[green]✓ PASSED[/green]" if validation_passed else "[red]✗ FAILED[/red]"
        )
        table.add_row("Output", output_path)

        panel = Panel(
            table,
            title="🎉 Pipeline Complete" if validation_passed else "⚠ Pipeline Complete (with issues)",
            border_style="green" if validation_passed else "yellow",
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL DISPLAY INSTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

display = DisplayManager()