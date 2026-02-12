# # generator.py
# """
# Orchestrator: skeleton → LLM text → validate → save.
# """

# import os
# import time
# import pandas as pd
# from datetime import datetime
# from typing import List

# from config import (
#     TOTAL_ROWS, TEXT_BATCH_SIZE,
#     DELAY_BETWEEN_BATCHES, OUTPUT_DIR,
#     THEME_OBJECTS, USE_OPENAI_SDK,
#     MODEL_NAME, ACTIVE_PROFILE,
# )
# from skeleton import build_skeleton, verify_skeleton, print_skeleton_summary
# from text_generator import TextGenerator
# from validator import validate_dataset, print_full_report


# class DatasetGenerator:

#     def __init__(self):
#         self.text_gen = TextGenerator()
#         os.makedirs(OUTPUT_DIR, exist_ok=True)

#     def generate(self, seed: int = 42) -> pd.DataFrame:
#         """Full pipeline."""

#         self._banner()

#         # Step 1: Skeleton
#         print(f"\n{'━'*60}")
#         print(f"  STEP 1: Building Skeleton ({TOTAL_ROWS} rows)")
#         print(f"{'━'*60}")

#         skeleton = build_skeleton(seed=seed)
#         errs = verify_skeleton(skeleton)

#         if errs:
#             print(f"  ✗ SKELETON ERRORS:")
#             for e in errs:
#                 print(f"    • {e}")
#             return pd.DataFrame()

#         print(f"  ✓ Skeleton: {len(skeleton)} rows — ALL PERFECT")
#         print_skeleton_summary(skeleton)

#         # Step 2: Fill texts
#         print(f"\n{'━'*60}")
#         print(f"  STEP 2: Generating Texts via LLM")
#         print(f"{'━'*60}")

#         t0 = time.time()
#         filled = self._fill_all_texts(skeleton)
#         elapsed = time.time() - t0
#         print(f"\n  Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

#         # Step 3: Save
#         print(f"\n{'━'*60}")
#         print(f"  STEP 3: Save")
#         print(f"{'━'*60}")

#         filepath = self._save(filled)
#         return filled

#     def _fill_all_texts(self, skeleton: pd.DataFrame) -> pd.DataFrame:
#         """Fill text + keywords in sub-batches."""

#         all_rows = skeleton.to_dict("records")
#         total = len(all_rows)
#         n_batches = -(-total // TEXT_BATCH_SIZE)

#         print(f"  Rows: {total} | Batch: {TEXT_BATCH_SIZE} | "
#               f"Batches: {n_batches}")

#         filled: List[dict] = []

#         for b in range(n_batches):
#             start = b * TEXT_BATCH_SIZE
#             end = min(start + TEXT_BATCH_SIZE, total)
#             batch = all_rows[start:end]

#             pct = end / total * 100
#             print(f"\n    ┌─ Batch {b+1}/{n_batches} "
#                   f"(rows {start+1}–{end}, {pct:.0f}%)")

#             result = self.text_gen.generate_texts(
#                 batch, batch_label=f"batch{b+1:02d}"
#             )
#             filled.extend(result)

#             if (b + 1) % 5 == 0 or b == n_batches - 1:
#                 self._save_progress(filled, b + 1)

#             if b < n_batches - 1:
#                 time.sleep(DELAY_BETWEEN_BATCHES)

#         filled_df = pd.DataFrame(filled)
#         filled_df = filled_df[skeleton.columns]
#         return filled_df

#     def _save_progress(self, rows: List[dict], batch_num: int):
#         df = pd.DataFrame(rows)
#         path = os.path.join(
#             OUTPUT_DIR, f"progress_{len(df)}rows_b{batch_num:02d}.csv"
#         )
#         df.to_csv(path, index=False)
#         print(f"    💾 Progress: {path}")

#     def _save(self, df: pd.DataFrame, filename: str = None) -> str:
#         if filename is None:
#             ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"dataset_{len(df)}rows_{ts}.csv"
#         path = os.path.join(OUTPUT_DIR, filename)
#         df.to_csv(path, index=False)
#         size_kb = os.path.getsize(path) / 1024
#         print(f"\n  ✓ SAVED: {path}")
#         print(f"    {len(df)} rows × {len(df.columns)} cols "
#               f"({size_kb:.1f} KB)")
#         return path

#     def _banner(self):
#         total_objs = sum(len(v) for v in THEME_OBJECTS.values())
#         mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
#         print(f"\n{'#'*60}")
#         print(f"  DATASET GENERATOR v3")
#         print(f"{'#'*60}")
#         print(f"  API Mode     : {mode}")
#         print(f"  Model        : {MODEL_NAME}")
#         print(f"  Profile      : {ACTIVE_PROFILE}")
#         print(f"  Total rows   : {TOTAL_ROWS}")
#         print(f"  Themes       : {len(THEME_OBJECTS)}")
#         print(f"  Objects      : {total_objs}")
#         print(f"  Rows/object  : 15")
#         print(f"  Sentiments   : 3 × 180")
#         print(f"  Emotions     : 5 × 108")
#         print(f"  Distributions: GUARANTEED by code")
#         print(f"  LLM generates: text + keywords ONLY")
#         print(f"{'#'*60}")




# generator.py
"""
Orchestrator: skeleton → LLM text → validate → save.
SUPPORTS RESUME — picks up from where it stopped.
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


# Progress tracking file
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_progress.json")
PROGRESS_CSV = os.path.join(OUTPUT_DIR, "_progress_data.csv")


class DatasetGenerator:
    """
    Pipeline with RESUME support:
    1. Check for existing progress
    2. Build skeleton (or load from progress)
    3. Fill texts starting from last completed batch
    4. Save after every batch
    5. Final save + cleanup
    """

    def __init__(self):
        self.text_gen = TextGenerator()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        print(f"\n{'━'*60}")
        print(f"  STEP 1: Building Skeleton ({TOTAL_ROWS} rows)")
        print(f"{'━'*60}")

        skeleton = build_skeleton(seed=seed)
        errs = verify_skeleton(skeleton)

        if errs:
            print(f"  ✗ SKELETON ERRORS:")
            for e in errs:
                print(f"    • {e}")
            return pd.DataFrame()

        print(f"  ✓ Skeleton: {len(skeleton)} rows — ALL PERFECT")
        print_skeleton_summary(skeleton)

        # Save skeleton
        skeleton_path = os.path.join(OUTPUT_DIR, "_skeleton.csv")
        skeleton.to_csv(skeleton_path, index=False)
        print(f"  💾 Skeleton saved: {skeleton_path}")

        # Step 2: Fill texts
        print(f"\n{'━'*60}")
        print(f"  STEP 2: Generating Texts via LLM")
        print(f"{'━'*60}")

        t0 = time.time()
        filled = self._fill_all_texts(skeleton, start_batch=0)
        elapsed = time.time() - t0
        print(f"\n  Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Step 3: Save final
        print(f"\n{'━'*60}")
        print(f"  STEP 3: Save Final Dataset")
        print(f"{'━'*60}")

        filepath = self._save_final(filled)

        # Cleanup progress files
        self._cleanup_progress()

        return filled

    def _resume_generation(self, progress: dict, seed: int) -> pd.DataFrame:
        """Resume from saved progress."""

        completed_batches = progress["completed_batches"]
        total_batches = progress["total_batches"]
        completed_rows = progress["completed_rows"]

        print(f"\n{'━'*60}")
        print(f"  ♻ RESUMING FROM BATCH {completed_batches + 1}/{total_batches}")
        print(f"  Rows completed: {completed_rows}/{TOTAL_ROWS}")
        print(f"  Last saved: {progress.get('last_saved', 'unknown')}")
        print(f"{'━'*60}")

        # Load the progress CSV
        if not os.path.exists(PROGRESS_CSV):
            print(f"  ✗ Progress CSV not found: {PROGRESS_CSV}")
            print(f"  Starting fresh...")
            return self._fresh_generation(seed)

        df = pd.read_csv(PROGRESS_CSV)
        print(f"  Loaded {len(df)} rows from progress file")

        # Verify row count matches
        if len(df) != TOTAL_ROWS:
            print(f"  ✗ Progress CSV has {len(df)} rows, expected {TOTAL_ROWS}")
            print(f"  Starting fresh...")
            return self._fresh_generation(seed)

        # Reload used texts to avoid duplicates
        existing_texts = df["text"].astype(str).tolist()
        for text in existing_texts:
            if text and len(text) >= 8 and text != "":
                self.text_gen._used_texts.add(text)
        print(f"  Loaded {len(self.text_gen._used_texts)} existing texts for dedup")

        # Continue filling texts
        print(f"\n{'━'*60}")
        print(f"  CONTINUING TEXT GENERATION")
        print(f"{'━'*60}")

        t0 = time.time()
        filled = self._fill_all_texts(
            df,
            start_batch=completed_batches,
        )
        elapsed = time.time() - t0
        print(f"\n  Text generation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Save final
        print(f"\n{'━'*60}")
        print(f"  SAVING FINAL DATASET")
        print(f"{'━'*60}")

        filepath = self._save_final(filled)
        self._cleanup_progress()

        return filled

    def _fill_all_texts(
        self,
        skeleton: pd.DataFrame,
        start_batch: int = 0,
    ) -> pd.DataFrame:
        """
        Fill text + keywords in batches with progress saving.

        Args:
            skeleton: DataFrame with all rows
            start_batch: Batch index to start from (0-based)
        """
        all_rows = skeleton.to_dict("records")
        total = len(all_rows)
        n_batches = -(-total // TEXT_BATCH_SIZE)

        print(f"  Rows: {total} | Batch: {TEXT_BATCH_SIZE} | "
              f"Batches: {n_batches} | Starting from: {start_batch + 1}")

        # If resuming, rows before start_batch are already filled
        filled_rows = all_rows.copy()

        for b in range(start_batch, n_batches):
            start = b * TEXT_BATCH_SIZE
            end = min(start + TEXT_BATCH_SIZE, total)
            batch_rows = all_rows[start:end]

            # Check if this batch already has text
            batch_already_done = all(
                row.get("text") and len(str(row.get("text", ""))) >= 8
                for row in batch_rows
            )

            if batch_already_done:
                print(f"\n    ┌─ Batch {b+1}/{n_batches} — already complete, skipping")
                continue

            pct = end / total * 100
            print(f"\n    ┌─ Batch {b+1}/{n_batches} "
                  f"(rows {start+1}–{end}, {pct:.0f}%)")

            result = self.text_gen.generate_texts(
                batch_rows, batch_label=f"batch{b+1:02d}"
            )

            # Update filled_rows
            for i, row in enumerate(result):
                filled_rows[start + i] = row

            # Save progress after every batch
            self._save_progress(
                filled_rows=filled_rows,
                completed_batches=b + 1,
                total_batches=n_batches,
            )

            if b < n_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)

        # Rebuild DataFrame
        filled_df = pd.DataFrame(filled_rows)
        filled_df = filled_df[skeleton.columns]
        return filled_df

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

        # Count filled rows
        completed_rows = sum(
            1 for r in filled_rows
            if r.get("text") and len(str(r.get("text", ""))) >= 8
        )

        # Save progress metadata
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

        # Save data CSV
        df = pd.DataFrame(filled_rows)
        df.to_csv(PROGRESS_CSV, index=False)

        pct = completed_rows / TOTAL_ROWS * 100
        print(
            f"    💾 Progress saved: batch {completed_batches}/{total_batches} "
            f"| {completed_rows}/{TOTAL_ROWS} rows ({pct:.0f}%)"
        )

    def _load_progress(self) -> Optional[dict]:
        """Load saved progress if it exists."""

        if not os.path.exists(PROGRESS_FILE):
            return None

        try:
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)

            # Validate progress matches current config
            if progress.get("total_rows") != TOTAL_ROWS:
                print(
                    f"  ⚠ Saved progress is for {progress.get('total_rows')} rows, "
                    f"but current config is {TOTAL_ROWS} rows."
                )
                print(f"  Starting fresh...")
                return None

            if progress.get("base_unit") != BASE_UNIT:
                print(
                    f"  ⚠ Saved progress has base_unit={progress.get('base_unit')}, "
                    f"but current config has {BASE_UNIT}."
                )
                print(f"  Starting fresh...")
                return None

            completed = progress.get("completed_batches", 0)
            total = progress.get("total_batches", 0)

            if completed >= total:
                print(f"  ⚠ Progress shows all batches complete. Starting fresh...")
                return None

            # Check CSV exists
            if not os.path.exists(PROGRESS_CSV):
                print(f"  ⚠ Progress JSON exists but CSV missing. Starting fresh...")
                return None

            print(f"\n  ╔{'═'*50}╗")
            print(f"  ║  SAVED PROGRESS FOUND                          ║")
            print(f"  ╠{'═'*50}╣")

            batches_str = f"{completed}/{total}"
            rows_str = f"{progress.get('completed_rows', '?')}/{TOTAL_ROWS}"
            
            print(f"  ║  Batches : {batches_str}{' '*(38-len(batches_str))}║")
            print(f"  ║  Rows    : {rows_str}{' '*(38-len(rows_str))}║")
            print(f"  ║  Model   : {progress.get('model', '?')[:37]:<37}║")
            print(f"  ║  Saved   : {progress.get('last_saved', '?')[:37]:<37}║")



            # print(f"  ║  Batches : {completed}/{total}{' '*(38-len(f'{completed}/{total}'))}║")
            # batches_str = f"{completed}/{total}"
            # rows_str = f"{progress.get('completed_rows', '?')}/{TOTAL_ROWS}"
            # model_str = str(progress.get('model', '?'))[:37]
            # saved_str = str(progress.get('last_saved', '?'))[:37]

            # print(f"  ║  Batches : {batches_str:<38}║")
            # print(f"  ║  Rows    : {rows_str:<38}║")
            # print(f"  ║  Model   : {model_str:<37}║")
            # print(f"  ║  Saved   : {saved_str:<37}║")

            # print(f"  ║  Rows    : {progress.get('completed_rows', '?')}/{TOTAL_ROWS}{' '* (38-len(f"{progress.get('completed_rows', '?')}/{TOTAL_ROWS}"))}")
            # print(f"  ║  Model   : {progress.get('model', '?')[:37]:<37}║")
            # print(f"  ║  Saved   : {progress.get('last_saved', '?')[:37]:<37}║")
            print(f"  ╚{'═'*50}╝")

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

        except(json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ Corrupted progress file: {e}")
            return None

    def _cleanup_progress(self):
        """Remove progress files after successful completion."""
        for f in [PROGRESS_FILE, PROGRESS_CSV]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  🗑 Cleaned up: {f}")

        # Also clean skeleton
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
        print(f"\n  ✓ SAVED: {path}")
        print(f"    {len(df)} rows × {len(df.columns)} cols ({size_kb:.1f} KB)")
        return path

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BANNER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _banner(self):
        total_objs = sum(len(v) for v in THEME_OBJECTS.values())
        mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
        print(f"\n{'#'*60}")
        print(f"  DATASET GENERATOR v4 (with Resume)")
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