# #!/usr/bin/env python3
# # main.py

# import sys
# import os
# import time
# from datetime import datetime

# from config import OUTPUT_DIR, USE_OPENAI_SDK, MODEL_NAME, ACTIVE_PROFILE


# def run_generate():
#     """Full pipeline."""
#     from generator import DatasetGenerator
#     from validator import validate_dataset, print_full_report

#     print(f"\n{'*'*60}")
#     print(f"  Dataset Generator v3 — 540 Rows")
#     print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
#     print(f"  Mode: {mode} | Model: {MODEL_NAME} | Profile: {ACTIVE_PROFILE}")
#     print(f"{'*'*60}\n")

#     gen = DatasetGenerator()
#     t0 = time.time()
#     df = gen.generate(seed=42)
#     elapsed = time.time() - t0

#     if df.empty:
#         print("✗ Failed")
#         sys.exit(1)

#     # Validate
#     print(f"\n{'═'*60}")
#     print(f"  FINAL VALIDATION")
#     print(f"{'═'*60}")

#     ok, errs = validate_dataset(df)
#     if errs:
#         print(f"\n  ⚠ {len(errs)} issue(s):")
#         for e in errs:
#             print(f"    • {e}")
#     else:
#         print("  ✓ ALL VALIDATIONS PASSED — PERFECT DATASET")

#     print_full_report(df)

#     print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
#     print(f"\n{'*'*60}")
#     print(f"  DONE")
#     print(f"{'*'*60}\n")


# def run_skeleton():
#     """Skeleton only — no LLM."""
#     from skeleton import build_skeleton, verify_skeleton, print_skeleton_summary

#     print("Building skeleton …")
#     df = build_skeleton(seed=42)

#     errs = verify_skeleton(df)
#     if errs:
#         print(f"\n✗ {len(errs)} error(s):")
#         for e in errs:
#             print(f"  • {e}")
#     else:
#         print("✓ ALL DISTRIBUTIONS PERFECT")

#     print_skeleton_summary(df)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, "skeleton_540.csv")
#     df["text"] = "PLACEHOLDER"
#     df["keywords"] = "PLACEHOLDER"
#     df.to_csv(path, index=False)
#     print(f"Saved: {path}")


# def run_test():
#     """Test: skeleton + 15 rows of text."""
#     from skeleton import build_skeleton, verify_skeleton
#     from text_generator import TextGenerator

#     print("Building skeleton …")
#     df = build_skeleton(seed=42)
#     errs = verify_skeleton(df)
#     print(f"Skeleton: {len(df)} rows, {len(errs)} errors")

#     # Show Pizza breakdown
#     pizza = df[
#         (df["theme"] == "Food") &
#         (df["object_detected"] == "Pizza")
#     ]
#     print(f"\nPizza ({len(pizza)} rows):")
#     for _, r in pizza.iterrows():
#         print(f"  {r['emotion']:12s} {r['sentiment']:10s}")

#     # Generate text
#     print(f"\nGenerating text for 15 rows …")
#     tg = TextGenerator()
#     rows = df.head(15).to_dict("records")
#     result = tg.generate_texts(rows, batch_label="test")

#     print(f"\nResults:")
#     for r in result:
#         print(
#             f"  [{r['theme']:10s}|{r['object_detected']:10s}|"
#             f"{r['sentiment']:8s}|{r['emotion']:10s}]"
#         )
#         print(f"    → {r['text']}")

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     import pandas as pd
#     pd.DataFrame(result).to_csv(
#         os.path.join(OUTPUT_DIR, "test_15.csv"), index=False
#     )
#     print(f"\n✓ Saved test_15.csv")


# def run_validate(filepath: str):
#     """Validate existing CSV."""
#     import pandas as pd
#     from validator import validate_dataset, print_full_report

#     print(f"Validating: {filepath}")
#     df = pd.read_csv(filepath)
#     ok, errs = validate_dataset(df)
#     print_full_report(df)
#     if ok:
#         print("✓ VALID — PERFECT")
#     else:
#         print(f"✗ {len(errs)} issue(s):")
#         for e in errs:
#             print(f"  • {e}")


# def main():
#     cmds = {
#         "generate": run_generate,
#         "skeleton": run_skeleton,
#         "test":     run_test,
#     }

#     if len(sys.argv) < 2:
#         run_generate()
#         return

#     cmd = sys.argv[1].lower()
#     if cmd == "validate" and len(sys.argv) > 2:
#         run_validate(sys.argv[2])
#     elif cmd in cmds:
#         cmds[cmd]()
#     else:
#         print(
#             "Usage:\n"
#             "  python main.py              — full generation\n"
#             "  python main.py generate     — full generation\n"
#             "  python main.py skeleton     — skeleton only (no LLM)\n"
#             "  python main.py test         — test 15 rows\n"
#             "  python main.py validate F   — validate CSV file\n"
#         )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# main.py

import sys
import os
import time
from datetime import datetime

from config import OUTPUT_DIR, USE_OPENAI_SDK, MODEL_NAME, ACTIVE_PROFILE, TOTAL_ROWS


def run_generate(force_restart: bool = False):
    """Full pipeline with resume support."""
    from r_generator import DatasetGenerator
    from validator import validate_dataset, print_full_report

    print(f"\n{'*'*60}")
    print(f"  Dataset Generator v4 — {TOTAL_ROWS} Rows")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
    print(f"  Mode: {mode} | Model: {MODEL_NAME}")
    if force_restart:
        print(f"  ⚠ FRESH START (ignoring saved progress)")
    print(f"{'*'*60}\n")

    gen = DatasetGenerator()
    t0 = time.time()
    df = gen.generate(seed=42, force_restart=force_restart)
    elapsed = time.time() - t0

    if df.empty:
        print("✗ Failed")
        sys.exit(1)

    # Validate
    print(f"\n{'═'*60}")
    print(f"  FINAL VALIDATION")
    print(f"{'═'*60}")

    ok, errs = validate_dataset(df)
    if errs:
        print(f"\n  ⚠ {len(errs)} issue(s):")
        for e in errs:
            print(f"    • {e}")
    else:
        print("  ✓ ALL VALIDATIONS PASSED — PERFECT DATASET")

    print_full_report(df)

    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'*'*60}")
    print(f"  DONE")
    print(f"{'*'*60}\n")


def run_skeleton():
    """Skeleton only — no LLM."""
    from skeleton import build_skeleton, verify_skeleton, print_skeleton_summary

    print("Building skeleton …")
    df = build_skeleton(seed=42)

    errs = verify_skeleton(df)
    if errs:
        print(f"\n✗ {len(errs)} error(s):")
        for e in errs:
            print(f"  • {e}")
    else:
        print("✓ ALL DISTRIBUTIONS PERFECT")

    print_skeleton_summary(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"skeleton_{TOTAL_ROWS}.csv")
    df["text"] = "PLACEHOLDER"
    df["keywords"] = "PLACEHOLDER"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_test():
    """Test: skeleton + 15 rows of text."""
    from skeleton import build_skeleton, verify_skeleton
    from text_generator import TextGenerator

    print("Building skeleton …")
    df = build_skeleton(seed=42)
    errs = verify_skeleton(df)
    print(f"Skeleton: {len(df)} rows, {len(errs)} errors")

    pizza = df[
        (df["theme"] == "Food") &
        (df["object_detected"] == "Pizza")
    ]
    print(f"\nPizza ({len(pizza)} rows):")
    for _, r in pizza.iterrows():
        print(f"  {r['emotion']:12s} {r['sentiment']:10s}")

    print(f"\nGenerating text for 15 rows …")
    tg = TextGenerator()
    rows = df.head(15).to_dict("records")
    result = tg.generate_texts(rows, batch_label="test")

    print(f"\nResults:")
    for r in result:
        print(
            f"  [{r['theme']:10s}|{r['object_detected']:10s}|"
            f"{r['sentiment']:8s}|{r['emotion']:10s}]"
        )
        print(f"    → {r['text']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import pandas as pd
    pd.DataFrame(result).to_csv(
        os.path.join(OUTPUT_DIR, "test_15.csv"), index=False
    )
    print(f"\n✓ Saved test_15.csv")


def run_validate(filepath: str):
    """Validate existing CSV."""
    import pandas as pd
    from validator import validate_dataset, print_full_report

    print(f"Validating: {filepath}")
    df = pd.read_csv(filepath)
    ok, errs = validate_dataset(df)
    print_full_report(df)
    if ok:
        print("✓ VALID — PERFECT")
    else:
        print(f"✗ {len(errs)} issue(s):")
        for e in errs:
            print(f"  • {e}")


def run_status():
    """Show current progress status."""
    import json

    progress_file = os.path.join(OUTPUT_DIR, "_progress.json")
    progress_csv = os.path.join(OUTPUT_DIR, "_progress_data.csv")

    if not os.path.exists(progress_file):
        print("No saved progress found.")
        print(f"Run 'python main.py generate' to start.")
        return

    with open(progress_file, "r") as f:
        progress = json.load(f)

    completed = progress.get("completed_batches", 0)
    total = progress.get("total_batches", 0)
    rows_done = progress.get("completed_rows", 0)
    total_rows = progress.get("total_rows", 0)
    pct = rows_done / total_rows * 100 if total_rows > 0 else 0

    print(f"\n  ╔{'═'*50}╗")
    print(f"  ║  PROGRESS STATUS                                ║")
    print(f"  ╠{'═'*50}╣")
    print(f"  ║  Batches    : {completed}/{total} ({completed/total*100:.0f}%){' '*(33-len(f'{completed}/{total} ({completed/total*100:.0f}%)') )}║")
    print(f"  ║  Rows       : {rows_done}/{total_rows} ({pct:.0f}%){' '*(33-len(f'{rows_done}/{total_rows} ({pct:.0f}%)') )}║")
    print(f"  ║  Model      : {progress.get('model', '?')[:36]:<36}║")
    print(f"  ║  Profile    : {progress.get('profile', '?')[:36]:<36}║")
    print(f"  ║  Last saved : {progress.get('last_saved', '?')[:36]:<36}║")
    print(f"  ╚{'═'*50}╝")

    # Progress bar
    bar_width = 40
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  [{bar}] {pct:.1f}%")

    if os.path.exists(progress_csv):
        import pandas as pd
        df = pd.read_csv(progress_csv)
        texts_filled = sum(
            1 for _, r in df.iterrows()
            if r.get("text") and len(str(r.get("text", ""))) >= 8
        )
        print(f"\n  Texts generated: {texts_filled}/{len(df)}")
        print(f"  Texts remaining: {len(df) - texts_filled}")

    print(f"\n  Commands:")
    print(f"    python main.py generate   — resume from batch {completed + 1}")
    print(f"    python main.py fresh      — discard progress, start over")


def run_clean():
    """Remove all progress files."""
    import glob

    files = [
        os.path.join(OUTPUT_DIR, "_progress.json"),
        os.path.join(OUTPUT_DIR, "_progress_data.csv"),
        os.path.join(OUTPUT_DIR, "_skeleton.csv"),
    ]

    removed = 0
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed: {f}")
            removed += 1

    # Also remove old progress CSVs
    for f in glob.glob(os.path.join(OUTPUT_DIR, "progress_*rows*.csv")):
        os.remove(f)
        print(f"  Removed: {f}")
        removed += 1

    if removed == 0:
        print("  No progress files found.")
    else:
        print(f"\n  ✓ Cleaned {removed} file(s)")




# Add this function to main.py

def run_img_desc(filepath: str = None):
    """Run the Image Description Generator (auto-handles resume)."""
    from desc_generator import ImageDescGenerator
    
    # print(f"\n{'*'*60}")
    # print(f"  Image Query Generator")
    # print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # mode = "OpenAI SDK" if USE_OPENAI_SDK else "Raw requests"
    # print(f"  Mode: {mode} | Model: {MODEL_NAME}")
    # print(f"{'*'*60}\n")
    
    gen = ImageDescGenerator()
    t0 = time.time()
    df = gen.generate(filepath=filepath, force_restart=False)  # Always False, let it ask
    elapsed = time.time() - t0
    
    if df.empty:
        print("✗ Failed")
        sys.exit(1)
    
    # print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    # print(f"\n{'*'*60}")
    # print(f"  DONE")
    # print(f"{'*'*60}\n")


        

def main():
    cmds = {
        "generate": lambda: run_generate(force_restart=False),
        "resume":   lambda: run_generate(force_restart=False),
        "fresh":    lambda: run_generate(force_restart=True),
        "restart":  lambda: run_generate(force_restart=True),
        "skeleton": run_skeleton,
        "test":     run_test,
        "status":   run_status,
        "clean":    run_clean,
        "img_desc": lambda: run_img_desc(sys.argv[2] if len(sys.argv) > 2 else None),
        
    }

    if len(sys.argv) < 2:
        run_generate(force_restart=False)
        return

    cmd = sys.argv[1].lower()
    if cmd == "validate" and len(sys.argv) > 2:
        run_validate(sys.argv[2])
    elif cmd in cmds:
        cmds[cmd]()
    else:
        print(
            "Usage:\n"
            "  python main.py              — generate (auto-resume)\n"
            "  python main.py generate     — generate (auto-resume)\n"
            "  python main.py resume       — resume from last checkpoint\n"
            "  python main.py fresh        — discard progress, start over\n"
            "  python main.py restart      — same as fresh\n"
            "  python main.py status       — show progress\n"
            "  python main.py clean        — remove progress files\n"
            "  python main.py skeleton     — skeleton only (no LLM)\n"
            "  python main.py test         — test 15 rows\n"
            "  python main.py validate    — validate CSV file\n"
            "  python main.py img_desc  — generate image descriptions\n"
        )


if __name__ == "__main__":
    main()