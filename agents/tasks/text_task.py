# agents/tasks/text_task.py
"""
Text generation task — fills "text" and "keywords" columns.
"""

from typing import List, Dict
from agents.task_config import TaskConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYSTEM PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are an expert ad-copy writer.
Write concise, emotionally resonant marketing text.
Output ONLY valid JSON arrays/objects.
Never use colons or apostrophes inside text strings."""

SYSTEM_PROMPT_SINGLE = """You are an expert ad-copy writer.
Write one concise, emotionally resonant marketing line.
Reply with ONLY a JSON object. Nothing else."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROMPT BUILDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_batch_prompt(rows: List[Dict]) -> str:
    """Build batch prompt for text generation."""
    lines = []
    for i, r in enumerate(rows):
        lines.append(
            f"{i+1}. Theme:{r.get('theme','')} | "
            f"Object:{r.get('object_detected','')} | "
            f"Emotion:{r.get('emotion','')} | "
            f"Sentiment:{r.get('sentiment','')}"
        )
    rows_block = "\n".join(lines)

    return f"""Generate {len(rows)} ad copy lines.

ROWS:
{rows_block}

OUTPUT FORMAT - JSON array with {len(rows)} objects:
[
  {{"id":1,"text":"Your ad copy here","keywords":"keyword1 keyword2 keyword3"}},
  {{"id":2,"text":"Another ad copy","keywords":"keyword1 keyword2"}}
]

RULES:
- Each "text" must be 8-15 words
- Each "keywords" must be 3-5 relevant words
- Match the emotion and sentiment exactly
- No brand names
- Output ONLY JSON array"""


def build_single_prompt(row: Dict, variation: int = 0) -> str:
    """Build single prompt with variation."""
    angles = [
        "focusing on benefits",
        "emphasizing urgency",
        "highlighting quality",
        "appealing to emotions",
        "using social proof",
        "creating curiosity",
    ]
    angle = angles[variation % len(angles)]

    return f"""Write ONE ad copy line {angle}.

Theme: {row.get('theme', '')}
Object: {row.get('object_detected', '')}
Emotion: {row.get('emotion', '')}
Sentiment: {row.get('sentiment', '')}

Reply with ONLY this JSON:
{{"text":"your ad copy here","keywords":"keyword1 keyword2 keyword3"}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FALLBACK BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_fallback(row: Dict) -> Dict[str, str]:
    """Rule-based fallback for Phase 5."""
    obj = row.get("object_detected", "product")
    theme = row.get("theme", "lifestyle")
    emotion = row.get("emotion", "Joy")
    sentiment = row.get("sentiment", "Positive")

    templates = {
        ("Joy", "Positive"): f"Experience pure happiness with our amazing {obj}",
        ("Joy", "Neutral"): f"Discover the {obj} that brings everyday joy",
        ("Anger", "Negative"): f"Tired of low quality? Our {obj} delivers",
        ("Anger", "Neutral"): f"No more compromises with our premium {obj}",
        ("Trust", "Positive"): f"Trusted by thousands - the reliable {obj}",
        ("Trust", "Neutral"): f"Quality you can depend on - our {obj}",
        ("Excitement", "Positive"): f"Get ready for the incredible {obj}",
        ("Excitement", "Neutral"): f"Something exciting awaits - discover {obj}",
        ("Fear", "Negative"): f"Do not miss out on this essential {obj}",
    }

    text = templates.get(
        (emotion, sentiment),
        f"Discover the perfect {obj} for your {theme} needs"
    )

    keywords = f"{obj} {theme} {emotion.lower()}"

    return {
        "text": text,
        "keywords": keywords,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEXT_TASK = TaskConfig(
    name="text",
    output_columns=["text", "keywords"],  # ← CHANGE COLUMN NAMES HERE
    primary_column="text",
    min_length=8,
    system_prompt=SYSTEM_PROMPT,
    system_prompt_single=SYSTEM_PROMPT_SINGLE,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
    deduplicate=False,
)