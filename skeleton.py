# skeleton.py
"""
Builds a PERFECTLY GROUPED skeleton of any size (multiple of 540).

GROUPING ORDER (top to bottom in the CSV):
  1. Theme      (Food → Fashion → Tech → Automotive → Travel → Finance)
  2. Object     (within each theme, objects in order)
  3. Emotion    (within each object: Joy → Anger → Trust → Excitement → Fear)
  4. Sentiment  (within each emotion: Positive → Negative → Neutral)
  5. Cycle      (within each cell: cycle 1, cycle 2, ... cycle N)

This means the CSV reads like a perfectly organized table:
  Food/Pizza/Joy/Positive    (cycle 1)
  Food/Pizza/Joy/Positive    (cycle 2)
  ...
  Food/Pizza/Joy/Neutral     (cycle 1)
  Food/Pizza/Joy/Neutral     (cycle 2)
  ...
  Food/Pizza/Anger/Negative  (cycle 1)
  ...
  Finance/Investment/Fear/Negative (cycle N)

All distributions are MATHEMATICALLY GUARANTEED.
"""

import random
import pandas as pd
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple

from config import (
    THEME_OBJECTS, THEME_ORDER, EMOTION_ORDER, SENTIMENT_ORDER,
    PER_OBJECT_DISTRIBUTION,
    SENTIMENT_TRUST_MAP, THEME_NUM, SENTIMENT_NUM,
    EMOTION_NUM, COLOUR_NUM, ATTENTION_NUM, TRUST_NUM,
    AUDIENCE_NUM, CTR_NUM, SHARES_NUM,
    COLOUR_DISTRIBUTION, TRIPLE_EACH, AUDIENCE_EACH,
    CTA_OPTIONS, MONETARY_OPTIONS,
    TOTAL_ROWS, BASE_UNIT, NUM_CYCLES,
    ROWS_PER_OBJECT_PER_CYCLE, OBJECTS_PER_THEME,
    NUM_THEMES, TOTAL_OBJECTS, FORBIDDEN_COMBOS, EXPECTED,
)


# ================================================================
#  Build the per-object block template (15 rows, sorted)
# ================================================================

def _build_object_block_template() -> List[Tuple[str, str]]:
    """
    Build the sorted template of (emotion, sentiment) pairs
    for a single object (15 rows).

    Sorted by: emotion_order → sentiment_order
    This ensures perfect grouping within each object.

    Returns:
        List of 15 (emotion, sentiment) tuples in sorted order.
    """
    # Create lookup for sort keys
    emo_rank = {e: i for i, e in enumerate(EMOTION_ORDER)}
    sent_rank = {s: i for i, s in enumerate(SENTIMENT_ORDER)}

    # Expand distribution into individual rows
    pairs: List[Tuple[str, str]] = []
    for emotion, sentiment, count in PER_OBJECT_DISTRIBUTION:
        for _ in range(count):
            pairs.append((emotion, sentiment))

    # Sort by emotion first, then sentiment
    pairs.sort(key=lambda p: (emo_rank[p[0]], sent_rank[p[1]]))

    assert len(pairs) == ROWS_PER_OBJECT_PER_CYCLE  # 15
    return pairs


# Pre-build the template once
_OBJECT_TEMPLATE = _build_object_block_template()


# ================================================================
#  Core row builder
# ================================================================

def _build_all_core_rows() -> List[Dict]:
    """
    Build ALL core rows for the entire dataset in PROPER GROUP ORDER.

    Order: Theme → Object → Emotion → Sentiment → Cycle

    For TOTAL_ROWS = 5400 (10 cycles):
      Each of the 15 (emotion,sentiment) slots gets 10 rows (one per cycle)
      Each object gets 15 × 10 = 150 rows
      Each theme gets 150 × 6 = 900 rows
      Total: 900 × 6 = 5400 rows

    Returns:
        List of TOTAL_ROWS row dicts in perfect order.
    """
    rows: List[Dict] = []

    for theme in THEME_ORDER:
        objects = THEME_OBJECTS[theme]

        for obj in objects:
            # For each (emotion, sentiment) pair in the template
            for emotion, sentiment in _OBJECT_TEMPLATE:
                # Repeat for each cycle
                for cycle in range(NUM_CYCLES):
                    rows.append({
                        "theme":           theme,
                        "object_detected": obj,
                        "emotion":         emotion,
                        "sentiment":       sentiment,
                        "trust_safety":    SENTIMENT_TRUST_MAP[sentiment],
                        "_cycle":          cycle,  # internal tracking
                    })

    assert len(rows) == TOTAL_ROWS, (
        f"Built {len(rows)} rows, expected {TOTAL_ROWS}"
    )
    return rows


# ================================================================
#  Auxiliary column pool builders
# ================================================================

def _build_pool(items: Dict[str, int]) -> List[str]:
    """Build a shuffled pool from {value: count} dict."""
    pool: List[str] = []
    for value, count in items.items():
        pool.extend([value] * count)
    assert len(pool) == TOTAL_ROWS, (
        f"Pool size = {len(pool)}, expected {TOTAL_ROWS}"
    )
    random.shuffle(pool)
    return pool


def _build_triple_pool() -> List[str]:
    """High/Medium/Low × TRIPLE_EACH, shuffled."""
    pool = (
        ["High"] * TRIPLE_EACH +
        ["Medium"] * TRIPLE_EACH +
        ["Low"] * TRIPLE_EACH
    )
    assert len(pool) == TOTAL_ROWS
    random.shuffle(pool)
    return pool


def _build_audience_pool() -> List[str]:
    """10 audiences × AUDIENCE_EACH, shuffled."""
    pool: List[str] = []
    for aud in AUDIENCE_NUM:
        pool.extend([aud] * AUDIENCE_EACH)
    assert len(pool) == TOTAL_ROWS
    random.shuffle(pool)
    return pool


# ================================================================
#  Main builder
# ================================================================

def build_skeleton(seed: int = 42) -> pd.DataFrame:
    """
    Build complete skeleton with PERFECT grouping and distributions.

    Grouping order in final CSV:
      Theme → Object → Emotion → Sentiment → Cycle

    Args:
        seed: Random seed for auxiliary columns (colours, attention, etc.)

    Returns:
        DataFrame with TOTAL_ROWS rows, all columns filled except text/keywords.
    """
    random.seed(seed)

    print(f"\n  Building skeleton:")
    print(f"    Total rows    : {TOTAL_ROWS}")
    print(f"    Base unit     : {BASE_UNIT}")
    print(f"    Cycles        : {NUM_CYCLES}")
    print(f"    Themes        : {NUM_THEMES}")
    print(f"    Objects/theme : {OBJECTS_PER_THEME}")
    print(f"    Total objects : {TOTAL_OBJECTS}")
    print(f"    Rows/object   : {ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES}")

    # 1. Core rows in PERFECT ORDER
    rows = _build_all_core_rows()
    # NOTE: NOT shuffled — order is the whole point

    # 2. Shuffled auxiliary pools
    colours   = _build_pool(COLOUR_DISTRIBUTION)
    attention = _build_triple_pool()
    ctr       = _build_triple_pool()
    shares    = _build_triple_pool()
    audience  = _build_audience_pool()

    # 3. Assign auxiliary columns + image paths
    for i, row in enumerate(rows):
        row["image_path"]        = f"sample_{i + 1:04d}.jpg"
        row["dominant_colour"]   = colours[i]
        row["attention_score"]   = attention[i]
        row["predicted_ctr"]     = ctr[i]
        row["likelihood_shares"] = shares[i]
        row["target_audience"]   = audience[i]

        # CTA
        if row["sentiment"] == "Positive":
            row["call_to_action"] = random.choice(CTA_OPTIONS[1:])
        elif row["sentiment"] == "Negative":
            row["call_to_action"] = random.choices(
                ["None"] + CTA_OPTIONS[1:],
                weights=[70] + [3] * (len(CTA_OPTIONS) - 1),
                k=1,
            )[0]
        else:
            row["call_to_action"] = random.choice(CTA_OPTIONS)

        row["monetary_mention"] = random.choice(MONETARY_OPTIONS)
        row["text"]     = ""
        row["keywords"] = ""

    # 4. DataFrame
    df = pd.DataFrame(rows)

    # 5. Remove internal tracking column
    if "_cycle" in df.columns:
        df = df.drop(columns=["_cycle"])

    # 6. Numeric codes
    num_mappings = {
        "theme":            THEME_NUM,
        "sentiment":        SENTIMENT_NUM,
        "emotion":          EMOTION_NUM,
        "dominant_colour":  COLOUR_NUM,
        "attention_score":  ATTENTION_NUM,
        "trust_safety":     TRUST_NUM,
        "target_audience":  AUDIENCE_NUM,
        "predicted_ctr":    CTR_NUM,
        "likelihood_shares": SHARES_NUM,
    }
    for col, mapping in num_mappings.items():
        df[f"{col}_num"] = df[col].map(mapping)

    # 7. Column order
    col_order = [
        "image_path", "text",
        "theme", "theme_num",
        "sentiment", "sentiment_num",
        "emotion", "emotion_num",
        "dominant_colour", "dominant_colour_num",
        "attention_score", "attention_score_num",
        "trust_safety", "trust_safety_num",
        "target_audience", "target_audience_num",
        "predicted_ctr", "predicted_ctr_num",
        "likelihood_shares", "likelihood_shares_num",
        "keywords", "monetary_mention",
        "call_to_action", "object_detected",
    ]
    df = df[col_order]

    return df


# ================================================================
#  Comprehensive Verification
# ================================================================

def verify_skeleton(df: pd.DataFrame) -> List[str]:
    """
    Verify ALL distribution requirements for any scale.
    Returns list of errors (empty = perfect).
    """
    errors: List[str] = []
    N = len(df)

    # ── Total rows ──
    if N != TOTAL_ROWS:
        errors.append(f"Total rows: expected {TOTAL_ROWS}, got {N}")

    # ── All marginal distributions ──
    for col, expected in EXPECTED.items():
        _chk(df, col, expected, errors)

    # ── Sentiment ↔ Trust (strict 1:1) ──
    trust_bad = 0
    for _, row in df.iterrows():
        if SENTIMENT_TRUST_MAP[row["sentiment"]] != row["trust_safety"]:
            trust_bad += 1
    if trust_bad:
        errors.append(f"Sentiment↔Trust mismatches: {trust_bad}")

    # ── Forbidden combos ──
    forbidden_count = 0
    for _, row in df.iterrows():
        if (row["emotion"], row["sentiment"]) in FORBIDDEN_COMBOS:
            forbidden_count += 1
    if forbidden_count:
        errors.append(f"Forbidden Emotion×Sentiment: {forbidden_count}")

    # ── Per-object checks ──
    expected_per_object = ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
    expected_sent_per_obj = 5 * NUM_CYCLES   # 5 per cycle
    expected_emo_per_obj = 3 * NUM_CYCLES    # 3 per cycle

    obj_errors = 0
    for theme in THEME_ORDER:
        for obj in THEME_OBJECTS[theme]:
            sub = df[
                (df["theme"] == theme) &
                (df["object_detected"] == obj)
            ]

            if len(sub) != expected_per_object:
                errors.append(
                    f"{theme}/{obj}: {len(sub)} rows "
                    f"(expected {expected_per_object})"
                )
                continue

            # Sentiment per object
            sc = Counter(sub["sentiment"])
            for s in SENTIMENT_ORDER:
                if sc.get(s, 0) != expected_sent_per_obj:
                    obj_errors += 1

            # Emotion per object
            ec = Counter(sub["emotion"])
            for e in EMOTION_ORDER:
                if ec.get(e, 0) != expected_emo_per_obj:
                    obj_errors += 1

            # Joint per object (scaled)
            for emotion, sentiment, base_count in PER_OBJECT_DISTRIBUTION:
                expected_joint = base_count * NUM_CYCLES
                actual = len(
                    sub[
                        (sub["emotion"] == emotion) &
                        (sub["sentiment"] == sentiment)
                    ]
                )
                if actual != expected_joint:
                    obj_errors += 1

    if obj_errors:
        errors.append(f"Per-object distribution errors: {obj_errors}")

    # ── Theme × Sentiment ──
    expected_theme_sent = 30 * NUM_CYCLES  # 30 per cycle
    ts_bad = 0
    for theme in THEME_ORDER:
        t_df = df[df["theme"] == theme]
        for s in SENTIMENT_ORDER:
            if len(t_df[t_df["sentiment"] == s]) != expected_theme_sent:
                ts_bad += 1
    if ts_bad:
        errors.append(f"Theme×Sentiment errors: {ts_bad}")

    # ── Theme × Emotion ──
    expected_theme_emo = 18 * NUM_CYCLES  # 18 per cycle
    te_bad = 0
    for theme in THEME_ORDER:
        t_df = df[df["theme"] == theme]
        for e in EMOTION_ORDER:
            if len(t_df[t_df["emotion"] == e]) != expected_theme_emo:
                te_bad += 1
    if te_bad:
        errors.append(f"Theme×Emotion errors: {te_bad}")

    # ── Grouping order check ──
    grouping_errors = _verify_grouping(df)
    errors.extend(grouping_errors)

    # ── Numeric codes ──
    code_errs = 0
    num_pairs = [
        ("theme",            "theme_num",            THEME_NUM),
        ("sentiment",        "sentiment_num",        SENTIMENT_NUM),
        ("emotion",          "emotion_num",          EMOTION_NUM),
        ("dominant_colour",  "dominant_colour_num",  COLOUR_NUM),
        ("attention_score",  "attention_score_num",  ATTENTION_NUM),
        ("trust_safety",     "trust_safety_num",     TRUST_NUM),
        ("target_audience",  "target_audience_num",  AUDIENCE_NUM),
        ("predicted_ctr",    "predicted_ctr_num",    CTR_NUM),
        ("likelihood_shares","likelihood_shares_num", SHARES_NUM),
    ]
    for txt, num, mapping in num_pairs:
        for _, row in df.iterrows():
            if mapping.get(row[txt]) != row[num]:
                code_errs += 1
                break  # one per column pair is enough
    if code_errs:
        errors.append(f"Numeric code issues in {code_errs} column pair(s)")

    return errors


def _verify_grouping(df: pd.DataFrame) -> List[str]:
    """
    Verify that rows are in proper group order:
    Theme → Object → Emotion → Sentiment
    """
    errors = []

    theme_rank = {t: i for i, t in enumerate(THEME_ORDER)}
    emo_rank = {e: i for i, e in enumerate(EMOTION_ORDER)}
    sent_rank = {s: i for i, s in enumerate(SENTIMENT_ORDER)}

    # Build object rank within theme
    obj_rank = {}
    for theme in THEME_ORDER:
        for i, obj in enumerate(THEME_OBJECTS[theme]):
            obj_rank[(theme, obj)] = i

    prev_key = None
    for idx, row in df.iterrows():
        t = row["theme"]
        o = row["object_detected"]
        e = row["emotion"]
        s = row["sentiment"]

        current_key = (
            theme_rank.get(t, 99),
            obj_rank.get((t, o), 99),
            emo_rank.get(e, 99),
            sent_rank.get(s, 99),
        )

        if prev_key is not None and current_key < prev_key:
            errors.append(
                f"Grouping broken at row {idx}: "
                f"{t}/{o}/{e}/{s} comes after previous group"
            )
            break  # one is enough to flag

        prev_key = current_key

    return errors


def _chk(df, col, expected, errors):
    """Check column distribution matches expected."""
    actual = Counter(df[col])
    for val, exp in expected.items():
        got = actual.get(val, 0)
        if got != exp:
            errors.append(f"{col}: '{val}' expected {exp}, got {got}")
    for val in actual:
        if val not in expected:
            errors.append(f"{col}: unexpected '{val}' × {actual[val]}")


# ================================================================
#  Pretty Print Summary
# ================================================================

def print_skeleton_summary(df: pd.DataFrame):
    """Print comprehensive grouped summary."""

    print(f"\n{'═'*70}")
    print(f"  SKELETON SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total rows    : {len(df)}")
    print(f"  Base unit     : {BASE_UNIT}")
    print(f"  Cycles        : {NUM_CYCLES}")
    print(f"  Rows/object   : {ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES}")

    # Marginal distributions
    for col_name, display_name in [
        ("theme", "THEME"),
        ("sentiment", "SENTIMENT"),
        ("emotion", "EMOTION"),
        ("dominant_colour", "COLOUR"),
        ("trust_safety", "TRUST SAFETY"),
        ("attention_score", "ATTENTION"),
        ("target_audience", "AUDIENCE"),
        ("predicted_ctr", "PREDICTED CTR"),
        ("likelihood_shares", "SHARES"),
    ]:
        expected = EXPECTED.get(col_name, {})
        print(f"\n  {display_name}:")
        actual = Counter(df[col_name])
        for val in (expected.keys() if expected else sorted(actual.keys())):
            got = actual.get(val, 0)
            exp = expected.get(val, "?")
            ok = "✓" if got == exp else "✗"
            bar = "█" * max(1, got // (TOTAL_ROWS // 50))
            print(f"    {ok} {val:25s} {got:>6}/{str(exp):<6} {bar}")

    # Objects
    print(f"\n  OBJECTS (each should have {ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES} rows):")
    for theme in THEME_ORDER:
        parts = []
        for obj in THEME_OBJECTS[theme]:
            cnt = len(df[df["object_detected"] == obj])
            expected_cnt = ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
            ok = "✓" if cnt == expected_cnt else "✗"
            parts.append(f"{obj}={cnt}")
        print(f"    {theme:12s}: {', '.join(parts)}")

    # Theme × Sentiment
    exp_ts = 30 * NUM_CYCLES
    print(f"\n  THEME × SENTIMENT (each cell should be {exp_ts}):")
    print(f"    {'':15s} {'Positive':>10} {'Negative':>10} {'Neutral':>10} {'Total':>8}")
    for theme in THEME_ORDER:
        t = df[df["theme"] == theme]
        p = len(t[t["sentiment"] == "Positive"])
        n = len(t[t["sentiment"] == "Negative"])
        u = len(t[t["sentiment"] == "Neutral"])
        ok = "✓" if p == exp_ts and n == exp_ts and u == exp_ts else "✗"
        print(f"  {ok} {theme:15s} {p:>10} {n:>10} {u:>10} {p+n+u:>8}")

    # Theme × Emotion
    exp_te = 18 * NUM_CYCLES
    print(f"\n  THEME × EMOTION (each cell should be {exp_te}):")
    hdr = f"    {'':15s}" + "".join(f" {e[:5]:>7}" for e in EMOTION_ORDER)
    print(hdr)
    for theme in THEME_ORDER:
        t = df[df["theme"] == theme]
        counts = [len(t[t["emotion"] == e]) for e in EMOTION_ORDER]
        ok = "✓" if all(c == exp_te for c in counts) else "✗"
        row_s = f"  {ok} {theme:15s}" + "".join(f" {c:>7}" for c in counts)
        print(row_s)

    # Sample grouping view
    print(f"\n  GROUPING PREVIEW (first 45 rows):")
    print(f"    {'Row':>5} {'Theme':>12} {'Object':>12} {'Emotion':>12} {'Sentiment':>10}")
    prev = ("", "", "")
    for i, (_, row) in enumerate(df.head(45).iterrows()):
        curr = (row["theme"], row["object_detected"], row["emotion"])
        marker = ""
        if curr[:1] != prev[:1]:
            marker = "  ── new theme ──"
        elif curr[:2] != prev[:2]:
            marker = "  ── new object ──"
        elif curr[:3] != prev[:3]:
            marker = "  ·"

        print(
            f"    {i+1:>5} {row['theme']:>12} {row['object_detected']:>12} "
            f"{row['emotion']:>12} {row['sentiment']:>10}{marker}"
        )
        prev = curr

    # Grouping at end of dataset
    print(f"\n  GROUPING PREVIEW (last 30 rows):")
    print(f"    {'Row':>5} {'Theme':>12} {'Object':>12} {'Emotion':>12} {'Sentiment':>10}")
    start_idx = len(df) - 30
    prev = ("", "", "")
    for i, (_, row) in enumerate(df.tail(30).iterrows()):
        curr = (row["theme"], row["object_detected"], row["emotion"])
        marker = ""
        if curr[:2] != prev[:2]:
            marker = "  ── new object ──"
        elif curr[:3] != prev[:3]:
            marker = "  ·"
        print(
            f"    {start_idx+i+1:>5} {row['theme']:>12} {row['object_detected']:>12} "
            f"{row['emotion']:>12} {row['sentiment']:>10}{marker}"
        )
        prev = curr

    print(f"\n{'═'*70}\n")


# ================================================================
#  Quick self-test
# ================================================================

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"  TOTAL_ROWS  = {TOTAL_ROWS}")
    print(f"  BASE_UNIT   = {BASE_UNIT}")
    print(f"  NUM_CYCLES  = {NUM_CYCLES}")

    print(f"\nObject template (15 rows, sorted):")
    for i, (emo, sent) in enumerate(_OBJECT_TEMPLATE, 1):
        print(f"  {i:>2}. {emo:12s} {sent:10s}")

    print(f"\nBuilding skeleton …")
    df = build_skeleton(seed=42)
    print(f"Shape: {df.shape}")

    print(f"\nRunning verification …")
    errs = verify_skeleton(df)
    if errs:
        print(f"\n✗ {len(errs)} ERROR(S):")
        for e in errs:
            print(f"  • {e}")
    else:
        print(f"\n✓ ALL {TOTAL_ROWS} ROWS VERIFIED — PERFECT DISTRIBUTIONS + GROUPING")

    print_skeleton_summary(df)

    # Save
    path = f"skeleton_{TOTAL_ROWS}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")