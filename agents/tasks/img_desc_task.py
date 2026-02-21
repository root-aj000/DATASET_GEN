# agents/tasks/img_desc_task.py
"""
Image description task — fills "img_desc" column.
"""

from typing import List, Dict
from agents.task_config import TaskConfig


SYSTEM_PROMPT = """You are a search query generator for image-text matching.
Generate precise Google Image search queries.
Output ONLY valid JSON.
Never use colons or apostrophes inside text strings."""

SYSTEM_PROMPT_SINGLE = """You are a search query generator.
Write one precise Google Image search query.
Reply with ONLY a JSON object."""


def build_batch_prompt(rows: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(rows):
        context = f"Ad Copy: {r.get('text', '')}" if r.get('text') else ""
        lines.append(
            f"{i+1}. {r.get('object_detected', '')} | "
            f"{r.get('theme', '')} | "
            f"{r.get('emotion', '')} | {context}"
        )
    rows_block = "\n".join(lines)

    return f"""Generate {len(rows)} Google Image search queries.

ROWS:
{rows_block}

OUTPUT FORMAT - JSON array with {len(rows)} objects:
[
  {{"id":1,"img_desc":"your search query here"}},
  {{"id":2,"img_desc":"another search query"}}
]

RULES:
- Each query 5-12 words
- Visual descriptors only
- Include "filetype:png" where appropriate
- Output ONLY JSON array"""


def build_single_prompt(row: Dict, variation: int = 0) -> str:
    angles = [
        "for a product catalog",
        "for social media advertising",
        "for a billboard campaign",
        "for email marketing",
        "for website hero image",
        "for print ads",
    ]
    angle = angles[variation % len(angles)]

    return f"""Create ONE Google Image search query {angle}.

Subject: {row.get('object_detected', '')}
Theme: {row.get('theme', '')}
Emotion: {row.get('emotion', '')}
Context: {row.get('text', '')}

Reply with ONLY:
{{"img_desc":"your search query here"}}"""


def build_fallback(row: Dict) -> Dict[str, str]:
    obj = row.get("object_detected", "object").lower()
    color = row.get("dominant_colour", "")
    theme = row.get("theme", "")
    emotion = row.get("emotion", "")

    parts = [f'"{obj}"']
    if color and color.lower() not in ['none', 'mixed']:
        parts.append(color.lower())
    if theme:
        parts.append(theme.lower())

    mood_map = {
        "Joy": "bright natural lighting",
        "Anger": "dramatic contrast",
        "Trust": "clean studio lighting",
        "Excitement": "vibrant dynamic",
        "Fear": "moody atmosphere"
    }
    if emotion in mood_map:
        parts.append(mood_map[emotion])

    parts.append("professional photography high resolution")

    return {"img_desc": " ".join(parts)}


IMG_DESC_TASK = TaskConfig(
    name="img_desc",
    output_columns=["img_desc"],  # Single column
    primary_column="img_desc",
    min_length=10,
    system_prompt=SYSTEM_PROMPT,
    system_prompt_single=SYSTEM_PROMPT_SINGLE,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
    deduplicate=False,
)