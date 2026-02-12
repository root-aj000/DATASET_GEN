# validator.py
"""
Validates any scale dataset using auto-calculated EXPECTED from config.
"""

import pandas as pd
from collections import Counter
from typing import List, Tuple

from config import (
    THEME_OBJECTS, THEME_ORDER, EMOTION_ORDER, SENTIMENT_ORDER,
    THEME_NUM, SENTIMENT_NUM, EMOTION_NUM,
    AUDIENCE_NUM, SENTIMENT_TRUST_MAP,
    FORBIDDEN_COMBOS, PER_OBJECT_DISTRIBUTION,
    TOTAL_ROWS, NUM_CYCLES, ROWS_PER_OBJECT_PER_CYCLE,
    EXPECTED,
)


BRAND_NAMES = [
    "iphone", "samsung", "apple", "google", "pixel", "bmw",
    "mercedes", "tesla", "audi", "toyota", "honda", "ford",
    "nike", "adidas", "gucci", "prada", "rolex", "amazon",
    "microsoft", "facebook", "instagram", "twitter",
    "starbucks", "mcdonalds", "coca-cola", "pepsi",
    "netflix", "spotify", "uber", "airbnb", "sony",
    "dell", "hp", "lenovo", "canon", "nikon",
]

EXPECTED_COLUMNS = [
    "image_path", "text", "theme", "theme_num",
    "sentiment", "sentiment_num", "emotion", "emotion_num",
    "dominant_colour", "dominant_colour_num",
    "attention_score", "attention_score_num",
    "trust_safety", "trust_safety_num",
    "target_audience", "target_audience_num",
    "predicted_ctr", "predicted_ctr_num",
    "likelihood_shares", "likelihood_shares_num",
    "keywords", "monetary_mention",
    "call_to_action", "object_detected",
]


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Full validation for any scale. Returns (is_valid, errors)."""
    errors: List[str] = []

    # Columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, errors

    # Row count
    if len(df) != TOTAL_ROWS:
        errors.append(f"Rows: expected {TOTAL_ROWS}, got {len(df)}")

    # All distributions from EXPECTED dict
    for col, expected in EXPECTED.items():
        _chk(df, col, expected, errors)

    # Sentiment ↔ Trust
    trust_bad = sum(
        1 for _, r in df.iterrows()
        if SENTIMENT_TRUST_MAP.get(r["sentiment"]) != r["trust_safety"]
    )
    if trust_bad:
        errors.append(f"Sentiment↔Trust mismatches: {trust_bad}")

    # Forbidden combos
    forbidden_count = sum(
        1 for _, r in df.iterrows()
        if (r["emotion"], r["sentiment"]) in FORBIDDEN_COMBOS
    )
    if forbidden_count:
        errors.append(f"Forbidden Emotion×Sentiment: {forbidden_count}")

    # Per-object joint
    joint_bad = 0
    for theme in THEME_ORDER:
        for obj in THEME_OBJECTS[theme]:
            sub = df[(df["theme"]==theme) & (df["object_detected"]==obj)]
            for emotion, sentiment, base_count in PER_OBJECT_DISTRIBUTION:
                expected_joint = base_count * NUM_CYCLES
                actual = len(
                    sub[(sub["emotion"]==emotion) &
                        (sub["sentiment"]==sentiment)]
                )
                if actual != expected_joint:
                    joint_bad += 1
    if joint_bad:
        errors.append(f"Per-object joint errors: {joint_bad}")

    # Theme × Sentiment
    exp_ts = 30 * NUM_CYCLES
    ts_bad = sum(
        1
        for theme in THEME_ORDER
        for s in SENTIMENT_ORDER
        if len(df[(df["theme"]==theme) & (df["sentiment"]==s)]) != exp_ts
    )
    if ts_bad:
        errors.append(f"Theme×Sentiment errors: {ts_bad}")

    # Theme × Emotion
    exp_te = 18 * NUM_CYCLES
    te_bad = sum(
        1
        for theme in THEME_ORDER
        for e in EMOTION_ORDER
        if len(df[(df["theme"]==theme) & (df["emotion"]==e)]) != exp_te
    )
    if te_bad:
        errors.append(f"Theme×Emotion errors: {te_bad}")

    # Text quality
    texts = df["text"].astype(str).tolist()
    dupes = [t for t, c in Counter(texts).items() if c > 1 and len(t.strip()) > 5]
    if dupes:
        errors.append(f"Duplicate texts: {len(dupes)}")

    empty = sum(1 for t in texts if len(t.strip()) < 5)
    if empty:
        errors.append(f"Empty/tiny texts: {empty}")

    brand_hits = sum(
        1 for t in texts
        if any(b in t.lower() for b in BRAND_NAMES)
    )
    if brand_hits:
        errors.append(f"Texts with brand names: {brand_hits}")

    return len(errors) == 0, errors


def _chk(df, col, expected, errors):
    actual = Counter(df[col])
    for val, exp in expected.items():
        got = actual.get(val, 0)
        if got != exp:
            errors.append(f"{col}: '{val}' expected {exp}, got {got}")


def print_full_report(df: pd.DataFrame):
    """Print comprehensive report for any scale."""
    print(f"\n{'═'*70}")
    print(f"  FINAL DATASET REPORT  ({len(df)} rows, {NUM_CYCLES} cycles)")
    print(f"{'═'*70}")

    for title, col in [
        ("THEME", "theme"), ("SENTIMENT", "sentiment"),
        ("EMOTION", "emotion"), ("COLOUR", "dominant_colour"),
        ("TRUST", "trust_safety"), ("ATTENTION", "attention_score"),
        ("AUDIENCE", "target_audience"), ("CTR", "predicted_ctr"),
        ("SHARES", "likelihood_shares"),
    ]:
        expected = EXPECTED.get(col, {})
        print(f"\n  {title}:")
        actual = Counter(df[col])
        for val in sorted(expected.keys()):
            got = actual.get(val, 0)
            exp = expected[val]
            s = "✓" if got == exp else "✗"
            bar = "█" * max(1, got // max(1, TOTAL_ROWS // 50))
            print(f"    {s} {val:25s} {got:>6}/{exp:<6} {bar}")

    # Cross tabs
    exp_ts = 30 * NUM_CYCLES
    print(f"\n  THEME × SENTIMENT (expect {exp_ts} each):")
    print(f"    {'':15s} {'Pos':>8} {'Neg':>8} {'Neu':>8} {'Tot':>8}")
    for theme in THEME_ORDER:
        t = df[df["theme"]==theme]
        p = len(t[t["sentiment"]=="Positive"])
        n = len(t[t["sentiment"]=="Negative"])
        u = len(t[t["sentiment"]=="Neutral"])
        ok = "✓" if p==exp_ts and n==exp_ts and u==exp_ts else "✗"
        print(f"  {ok} {theme:15s} {p:>8} {n:>8} {u:>8} {p+n+u:>8}")

    exp_te = 18 * NUM_CYCLES
    print(f"\n  THEME × EMOTION (expect {exp_te} each):")
    hdr = f"    {'':15s}" + "".join(f" {e[:5]:>7}" for e in EMOTION_ORDER)
    print(hdr)
    for theme in THEME_ORDER:
        t = df[df["theme"]==theme]
        counts = [len(t[t["emotion"]==e]) for e in EMOTION_ORDER]
        ok = "✓" if all(c==exp_te for c in counts) else "✗"
        print(f"  {ok} {theme:15s}" + "".join(f" {c:>7}" for c in counts))

    # Objects
    exp_obj = ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
    print(f"\n  OBJECTS (each={exp_obj}):")
    for theme in THEME_ORDER:
        cnts = [len(df[df["object_detected"]==o]) for o in THEME_OBJECTS[theme]]
        ok = "✓" if all(c==exp_obj for c in cnts) else "✗"
        s = ", ".join(
            f"{o}={c}" for o, c in zip(THEME_OBJECTS[theme], cnts)
        )
        print(f"  {ok} {theme:12s}: {s}")

    # Text
    wc = df["text"].astype(str).str.split().str.len()
    dups = df["text"].duplicated().sum()
    print(f"\n  TEXT QUALITY:")
    print(f"    Words: min={wc.min()} avg={wc.mean():.1f} "
          f"med={wc.median():.0f} max={wc.max()}")
    print(f"    Duplicates: {dups}")
    print(f"    Empty: {(df['text'].astype(str).str.len() < 5).sum()}")
    print(f"{'═'*70}\n")