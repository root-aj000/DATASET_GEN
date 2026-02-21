# # agents/base_agent.py
# """
# BaseAgent — Generic 5-phase generation engine.
# Works with ANY TaskConfig — columns are fully dynamic.
# """

# import os
# import time
# from typing import List, Dict, Optional, Set

# from config import (
#     MAX_RETRIES,
#     RETRY_DELAY,
#     LOG_RESPONSES,
#     LOG_DIR,
#     TEMPERATURE,
# )
# from core.client import APIClient
# from core.parser import JSONParser
# from agents.task_config import TaskConfig


# class BaseAgent:
#     """
#     Universal agent that executes any TaskConfig.

#     5-Phase Generation:
#         Phase 1: Full batch API call
#         Phase 2: Retry gaps
#         Phase 3: Micro-batches (5 rows)
#         Phase 4: Single generation with variations
#         Phase 5: Rule-based fallback
#     """

#     def __init__(self, task: TaskConfig, verbose: bool = False):
#         self.task = task
#         self.verbose = verbose

#         # Shared client and parser
#         self.client = APIClient(verbose=verbose)
#         self.parser = JSONParser(verbose=verbose)

#         # Deduplication tracking
#         self._used_values: Set[str] = set()

#         # Stats
#         self._api_calls = 0

#         self._log(f"  Task      : {task.name}")
#         self._log(f"  Columns   : {task.output_columns}")
#         self._log(f"  Primary   : {task.primary_column}")
#         self._log(f"  Min length: {task.min_length}")

#     def _log(self, msg: str):
#         if self.verbose:
#             print(msg)

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  MAIN ENTRY POINT
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def generate(
#         self,
#         rows: List[Dict],
#         batch_label: str = "",
#     ) -> List[Dict]:
#         """
#         Generate values for all output columns.

#         Args:
#             rows: Input rows (skeleton data)
#             batch_label: Label for logging

#         Returns:
#             Rows with output columns filled
#         """
#         total = len(rows)
#         result = [row.copy() for row in rows]

#         # Initialize output columns
#         for r in result:
#             for col in self.task.output_columns:
#                 if col not in r:
#                     r[col] = ""

#         # Execute 5 phases
#         phases = [
#             (1, "Full batch", lambda: self._phase_full(rows, result, batch_label, total)),
#             (2, "Retry gaps", lambda: self._phase_retry(rows, result, batch_label, total)),
#             (3, "Micro-batches", lambda: self._phase_micro(rows, result, batch_label, total)),
#             (4, "Single generation", lambda: self._phase_single(rows, result, batch_label, total)),
#             (5, "Fallback", lambda: self._phase_fallback(rows, result, batch_label, total)),
#         ]

#         for phase_num, phase_name, phase_fn in phases:
#             phase_fn()
#             filled = self._count_filled(result)

#             if filled == total:
#                 self._log(f"      ✓ Phase {phase_num} ({phase_name}): {filled}/{total} — COMPLETE")
#                 return result
#             else:
#                 self._log(f"      Phase {phase_num} ({phase_name}): {filled}/{total}")

#         return result

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  PHASE IMPLEMENTATIONS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _phase_full(self, skel: List[Dict], result: List[Dict], label: str, total: int):
#         """Phase 1: Full batch API call."""
#         items = self._call_batch(skel, f"{label}_p1_full")
#         if items:
#             self._fill(result, items, list(range(total)))

#     def _phase_retry(self, skel: List[Dict], result: List[Dict], label: str, total: int):
#         """Phase 2: Retry gaps."""
#         gaps = self._find_gaps(result)
#         if not gaps:
#             return

#         self._log(f"        Retrying {len(gaps)} gaps")
#         gap_rows = [skel[i] for i in gaps]
#         items = self._call_batch(gap_rows, f"{label}_p2_retry")
#         if items:
#             self._fill(result, items, gaps)

#     def _phase_micro(self, skel: List[Dict], result: List[Dict], label: str, total: int):
#         """Phase 3: Micro-batches (5 rows at a time)."""
#         gaps = self._find_gaps(result)
#         if not gaps:
#             return

#         self._log(f"        Micro-batches for {len(gaps)} rows")
#         for chunk_idx in range(0, len(gaps), 5):
#             chunk = gaps[chunk_idx:chunk_idx + 5]
#             chunk_rows = [skel[i] for i in chunk]
#             items = self._call_batch(chunk_rows, f"{label}_p3_m{chunk_idx // 5}")
#             if items:
#                 self._fill(result, items, chunk)
#             time.sleep(1)

#     def _phase_single(self, skel: List[Dict], result: List[Dict], label: str, total: int):
#         """Phase 4: One-by-one generation with variations."""
#         gaps = self._find_gaps(result)
#         if not gaps:
#             return

#         self._log(f"        Single generation for {len(gaps)} rows")
#         for idx in gaps:
#             values = self._single_generate(skel[idx], f"{label}_p4_s{idx}")
#             if values and self._is_valid(values):
#                 self._apply_values(result[idx], values)
#                 self._log(f"          ✓ Row {idx}")
#             else:
#                 self._log(f"          ⚠ Row {idx}: needs Phase 5")
#             time.sleep(1)

#     def _phase_fallback(self, skel: List[Dict], result: List[Dict], label: str, total: int):
#         """Phase 5: Rule-based fallback construction."""
#         gaps = self._find_gaps(result)
#         if not gaps:
#             return

#         if not self.task.fallback_builder:
#             self._log(f"        ⚠ No fallback builder — {len(gaps)} rows unfilled")
#             return

#         self._log(f"        Constructing {len(gaps)} rows via fallback")
#         for idx in gaps:
#             values = self.task.fallback_builder(skel[idx])
#             if values:
#                 self._apply_values(result[idx], values)
#                 self._log(f"          ✓ Row {idx}: fallback")

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  API CALLS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _call_batch(self, rows: List[Dict], label: str) -> List[Dict]:
#         """Call API for batch with retries."""
#         if not self.task.batch_prompt_builder:
#             return []

#         prompt = self.task.batch_prompt_builder(rows)
#         max_retries = self.task.max_retries or MAX_RETRIES
#         retry_delay = self.task.retry_delay or RETRY_DELAY

#         for attempt in range(1, max_retries + 1):
#             self._log(f"        Attempt {attempt}/{max_retries} ({len(rows)} rows) [{label}]")

#             raw = self.client.call(
#                 prompt=prompt,
#                 system=self.task.system_prompt,
#                 temperature=self.task.temperature,
#                 log_label=f"{label}_a{attempt}" if LOG_RESPONSES else None,
#             )

#             if raw is None:
#                 time.sleep(retry_delay)
#                 continue

#             # Parse response
#             items = self.parser.parse_batch(
#                 raw=raw,
#                 primary_key=self.task.primary_column,
#                 all_keys=self.task.output_columns,
#             )

#             if items:
#                 self._api_calls += 1
#                 return items

#             self._log(f"        ✗ No items parsed")
#             time.sleep(retry_delay)

#         return []

#     def _single_generate(self, row: Dict, label: str) -> Optional[Dict[str, str]]:
#         """Generate single row with variations."""
#         if not self.task.single_prompt_builder:
#             return None

#         for variation in range(6):
#             temp_override = self.task.temperature or TEMPERATURE
#             temp = min(temp_override + (variation * 0.15), 1.0)

#             prompt = self.task.single_prompt_builder(row, variation)

#             raw = self.client.call(
#                 prompt=prompt,
#                 system=self.task.get_system_prompt_single(),
#                 temperature=temp,
#                 log_label=f"{label}_v{variation}" if LOG_RESPONSES else None,
#             )

#             if raw is None:
#                 time.sleep(2)
#                 continue

#             # Parse response
#             values = self.parser.parse_single(
#                 raw=raw,
#                 keys=self.task.output_columns,
#                 min_length=self.task.min_length,
#             )

#             # Clean values
#             cleaned = {k: JSONParser.clean_text(v) for k, v in values.items()}

#             if self._is_valid(cleaned):
#                 self._api_calls += 1
#                 return cleaned

#             time.sleep(1)

#         return None

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  FILL HELPERS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _fill(self, result: List[Dict], items: List[Dict], indices: List[int]):
#         """Fill result rows from parsed items, avoiding duplicates."""
#         used_item_indices = set()

#         for pos, row_idx in enumerate(indices):
#             # Skip if already filled
#             if self._is_row_filled(result[row_idx]):
#                 continue

#             # Try direct position mapping
#             if pos < len(items):
#                 values = self._extract_values(items[pos])
#                 if self._is_valid(values) and self._check_dedup(values):
#                     self._apply_values(result[row_idx], values)
#                     used_item_indices.add(pos)
#                     continue

#             # Try any unused item
#             for j in range(len(items)):
#                 if j in used_item_indices:
#                     continue

#                 values = self._extract_values(items[j])
#                 if self._is_valid(values) and self._check_dedup(values):
#                     self._apply_values(result[row_idx], values)
#                     used_item_indices.add(j)
#                     break

#     def _extract_values(self, item: Dict) -> Dict[str, str]:
#         """Extract output column values from parsed item."""
#         return {
#             col: JSONParser.clean_text(str(item.get(col, "")))
#             for col in self.task.output_columns
#         }

#     def _apply_values(self, row: Dict, values: Dict[str, str]):
#         """Apply values to row and track for deduplication."""
#         for col, val in values.items():
#             row[col] = val

#         # Track primary value for deduplication
#         if self.task.deduplicate:
#             primary_val = values.get(self.task.primary_column, "")
#             if primary_val:
#                 self._used_values.add(primary_val)

#     def _check_dedup(self, values: Dict[str, str]) -> bool:
#         """Check if primary value is unique (if deduplication enabled)."""
#         if not self.task.deduplicate:
#             return True

#         primary_val = values.get(self.task.primary_column, "")
#         return primary_val not in self._used_values

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  VALIDATION HELPERS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _is_valid(self, values: Dict[str, str]) -> bool:
#         """Check if values are valid (primary column meets min_length)."""
#         primary_val = values.get(self.task.primary_column, "")
#         return len(primary_val) >= self.task.min_length

#     def _is_row_filled(self, row: Dict) -> bool:
#         """Check if row's primary column is filled."""
#         val = row.get(self.task.primary_column, "")
#         return isinstance(val, str) and len(val) >= self.task.min_length

#     def _find_gaps(self, result: List[Dict]) -> List[int]:
#         """Find indices of unfilled rows."""
#         return [i for i, r in enumerate(result) if not self._is_row_filled(r)]

#     def _count_filled(self, result: List[Dict]) -> int:
#         """Count filled rows."""
#         return sum(1 for r in result if self._is_row_filled(r))

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  STATS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def get_stats(self) -> Dict:
#         """Get generation statistics."""
#         return {
#             "task": self.task.name,
#             "api_calls": self.client.api_calls,
#             "unique_values": len(self._used_values),
#         }

#     def load_existing_values(self, values: List[str]):
#         """Load existing values for deduplication (for resume)."""
#         for v in values:
#             if v and isinstance(v, str) and len(v) >= self.task.min_length:
#                 self._used_values.add(v)
#         self._log(f"  Loaded {len(self._used_values)} existing values for dedup")

# agents/base_agent.py
"""
BaseAgent — Generic 5-phase generation engine.
With Rich display integration.
"""

import os
import time
from typing import List, Dict, Optional, Set

from config import (
    MAX_RETRIES,
    RETRY_DELAY,
    LOG_RESPONSES,
    LOG_DIR,
    TEMPERATURE,
)
from core.client import APIClient
from core.parser import JSONParser
from agents.task_config import TaskConfig

# Try to import display
try:
    from pipeline.display import DisplayManager
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False


class BaseAgent:
    """
    Universal agent that executes any TaskConfig.

    5-Phase Generation:
        Phase 1: Full batch API call
        Phase 2: Retry gaps
        Phase 3: Micro-batches (5 rows)
        Phase 4: Single generation with variations
        Phase 5: Rule-based fallback
    """

    def __init__(self, task: TaskConfig, verbose: bool = False):
        self.task = task
        self.verbose = verbose

        # Shared client and parser
        self.client = APIClient(verbose=False)  # Client has own verbose
        self.parser = JSONParser(verbose=False)

        # Display manager
        self.display = DisplayManager(verbose=verbose) if DISPLAY_AVAILABLE else None

        # Deduplication tracking
        self._used_values: Set[str] = set()

        if self.verbose:
            self._log_init()

    def _log(self, msg: str):
        """Log message."""
        if self.verbose:
            if self.display:
                self.display.log(msg)
            else:
                print(f"    {msg}")

    def _log_init(self):
        """Log initialization."""
        if self.display:
            self.display.info(f"Agent initialized: {self.task.name}")
            self.display.log(f"Columns: {self.task.output_columns}")
            self.display.log(f"Primary: {self.task.primary_column}")
        else:
            print(f"  Agent: {self.task.name}")
            print(f"    Columns: {self.task.output_columns}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate(
        self,
        rows: List[Dict],
        batch_label: str = "",
    ) -> List[Dict]:
        """
        Generate values for all output columns.
        """
        total = len(rows)
        result = [row.copy() for row in rows]

        # Initialize output columns
        for r in result:
            for col in self.task.output_columns:
                if col not in r:
                    r[col] = ""

        # Execute 5 phases
        phases = [
            (1, "Full batch", lambda: self._phase_full(rows, result, batch_label, total)),
            (2, "Retry gaps", lambda: self._phase_retry(rows, result, batch_label, total)),
            (3, "Micro-batches", lambda: self._phase_micro(rows, result, batch_label, total)),
            (4, "Single", lambda: self._phase_single(rows, result, batch_label, total)),
            (5, "Fallback", lambda: self._phase_fallback(rows, result, batch_label, total)),
        ]

        for phase_num, phase_name, phase_fn in phases:
            phase_fn()
            filled = self._count_filled(result)

            if self.verbose and self.display:
                self.display.phase_status(phase_num, phase_name, filled, total, filled == total)
            elif self.verbose:
                status = "✓ COMPLETE" if filled == total else f"{filled}/{total}"
                print(f"      Phase {phase_num} ({phase_name}): {status}")

            if filled == total:
                return result

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PHASE IMPLEMENTATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _phase_full(self, skel: List[Dict], result: List[Dict], label: str, total: int):
        """Phase 1: Full batch API call."""
        items = self._call_batch(skel, f"{label}_p1")
        if items:
            self._fill(result, items, list(range(total)))

    def _phase_retry(self, skel: List[Dict], result: List[Dict], label: str, total: int):
        """Phase 2: Retry gaps."""
        gaps = self._find_gaps(result)
        if not gaps:
            return

        self._log(f"Retrying {len(gaps)} gaps")
        gap_rows = [skel[i] for i in gaps]
        items = self._call_batch(gap_rows, f"{label}_p2")
        if items:
            self._fill(result, items, gaps)

    def _phase_micro(self, skel: List[Dict], result: List[Dict], label: str, total: int):
        """Phase 3: Micro-batches (5 rows at a time)."""
        gaps = self._find_gaps(result)
        if not gaps:
            return

        self._log(f"Micro-batches for {len(gaps)} rows")
        for chunk_idx in range(0, len(gaps), 5):
            chunk = gaps[chunk_idx:chunk_idx + 5]
            chunk_rows = [skel[i] for i in chunk]
            items = self._call_batch(chunk_rows, f"{label}_p3m{chunk_idx//5}")
            if items:
                self._fill(result, items, chunk)
            time.sleep(1)

    def _phase_single(self, skel: List[Dict], result: List[Dict], label: str, total: int):
        """Phase 4: One-by-one generation with variations."""
        gaps = self._find_gaps(result)
        if not gaps:
            return

        self._log(f"Single generation for {len(gaps)} rows")
        for idx in gaps:
            values = self._single_generate(skel[idx], f"{label}_p4s{idx}")
            if values and self._is_valid(values):
                self._apply_values(result[idx], values)
            time.sleep(1)

    def _phase_fallback(self, skel: List[Dict], result: List[Dict], label: str, total: int):
        """Phase 5: Rule-based fallback construction."""
        gaps = self._find_gaps(result)
        if not gaps:
            return

        if not self.task.fallback_builder:
            self._log(f"No fallback — {len(gaps)} rows unfilled")
            return

        self._log(f"Fallback for {len(gaps)} rows")
        for idx in gaps:
            values = self.task.fallback_builder(skel[idx])
            if values:
                self._apply_values(result[idx], values)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  API CALLS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_batch(self, rows: List[Dict], label: str) -> List[Dict]:
        """Call API for batch with retries."""
        if not self.task.batch_prompt_builder:
            return []

        prompt = self.task.batch_prompt_builder(rows)
        max_retries = self.task.max_retries or MAX_RETRIES
        retry_delay = self.task.retry_delay or RETRY_DELAY

        for attempt in range(1, max_retries + 1):
            raw = self.client.call(
                prompt=prompt,
                system=self.task.system_prompt,
                temperature=self.task.temperature,
                log_label=f"{label}_a{attempt}" if LOG_RESPONSES else None,
            )

            if raw is None:
                time.sleep(retry_delay)
                continue

            items = self.parser.parse_batch(
                raw=raw,
                primary_key=self.task.primary_column,
                all_keys=self.task.output_columns,
            )

            if items:
                return items

            time.sleep(retry_delay)

        return []

    def _single_generate(self, row: Dict, label: str) -> Optional[Dict[str, str]]:
        """Generate single row with variations."""
        if not self.task.single_prompt_builder:
            return None

        for variation in range(6):
            temp_override = self.task.temperature or TEMPERATURE
            temp = min(temp_override + (variation * 0.15), 1.0)

            prompt = self.task.single_prompt_builder(row, variation)

            raw = self.client.call(
                prompt=prompt,
                system=self.task.get_system_prompt_single(),
                temperature=temp,
                log_label=f"{label}_v{variation}" if LOG_RESPONSES else None,
            )

            if raw is None:
                time.sleep(2)
                continue

            values = self.parser.parse_single(
                raw=raw,
                keys=self.task.output_columns,
                min_length=self.task.min_length,
            )

            cleaned = {k: JSONParser.clean_text(v) for k, v in values.items()}

            if self._is_valid(cleaned):
                return cleaned

            time.sleep(1)

        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FILL HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fill(self, result: List[Dict], items: List[Dict], indices: List[int]):
        """Fill result rows from parsed items."""
        used_item_indices = set()

        for pos, row_idx in enumerate(indices):
            if self._is_row_filled(result[row_idx]):
                continue

            # Try direct position mapping
            if pos < len(items):
                values = self._extract_values(items[pos])
                if self._is_valid(values) and self._check_dedup(values):
                    self._apply_values(result[row_idx], values)
                    used_item_indices.add(pos)
                    continue

            # Try any unused item
            for j in range(len(items)):
                if j in used_item_indices:
                    continue

                values = self._extract_values(items[j])
                if self._is_valid(values) and self._check_dedup(values):
                    self._apply_values(result[row_idx], values)
                    used_item_indices.add(j)
                    break

    def _extract_values(self, item: Dict) -> Dict[str, str]:
        """Extract output column values from parsed item."""
        return {
            col: JSONParser.clean_text(str(item.get(col, "")))
            for col in self.task.output_columns
        }

    def _apply_values(self, row: Dict, values: Dict[str, str]):
        """Apply values to row and track for deduplication."""
        for col, val in values.items():
            row[col] = val

        if self.task.deduplicate:
            primary_val = values.get(self.task.primary_column, "")
            if primary_val:
                self._used_values.add(primary_val)

    def _check_dedup(self, values: Dict[str, str]) -> bool:
        """Check if primary value is unique."""
        if not self.task.deduplicate:
            return True

        primary_val = values.get(self.task.primary_column, "")
        return primary_val not in self._used_values

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  VALIDATION HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _is_valid(self, values: Dict[str, str]) -> bool:
        """Check if values are valid."""
        primary_val = values.get(self.task.primary_column, "")
        return len(primary_val) >= self.task.min_length

    def _is_row_filled(self, row: Dict) -> bool:
        """Check if row's primary column is filled."""
        val = row.get(self.task.primary_column, "")
        return isinstance(val, str) and len(val) >= self.task.min_length

    def _find_gaps(self, result: List[Dict]) -> List[int]:
        """Find indices of unfilled rows."""
        return [i for i, r in enumerate(result) if not self._is_row_filled(r)]

    def _count_filled(self, result: List[Dict]) -> int:
        """Count filled rows."""
        return sum(1 for r in result if self._is_row_filled(r))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STATS & RESUME
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return {
            "task": self.task.name,
            "api_calls": self.client.api_calls,
            "unique_values": len(self._used_values),
        }

    def load_existing_values(self, values: List[str]):
        """Load existing values for deduplication (for resume)."""
        for v in values:
            if v and isinstance(v, str) and len(v) >= self.task.min_length:
                self._used_values.add(v)
        
        if self.verbose:
            self._log(f"Loaded {len(self._used_values)} existing values")