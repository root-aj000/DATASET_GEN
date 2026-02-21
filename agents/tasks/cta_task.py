# agents/tasks/cta_task.py
"""
CTA task — fills "monetary_mention" and "call_to_action" columns.
"""

from typing import List, Dict
from agents.task_config import TaskConfig


SYSTEM_PROMPT = """You are a marketing copy generator.
Create compelling monetary mentions and call-to-action phrases.
Output ONLY valid JSON."""

SYSTEM_PROMPT_SINGLE = """You are a marketing copy generator.
Write precise monetary mention and CTA.
Reply with ONLY a JSON object."""


def build_batch_prompt(rows: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(rows):
        context = f"Ad Copy: {r.get('text', '')}" if r.get('text') else ""
        lines.append(
            f"{i+1}. Object: {r.get('object_detected', '')} | "
            f"Theme: {r.get('theme', '')} | "
            f"Emotion: {r.get('emotion', '')} | {context}"
        )
    rows_block = "\n".join(lines)

    return f"""Generate {len(rows)} monetary mentions and CTAs.

ROWS:
{rows_block}

OUTPUT FORMAT - JSON array:
[
  {{"id":1,"monetary_mention":"Special offer 50% off","call_to_action":"Shop Now"}},
  {{"id":2,"monetary_mention":"Limited time discount","call_to_action":"Buy Today"}}
]

RULES:
- monetary_mention: 3-6 words, compelling offer
- call_to_action: 2-4 words, action-oriented
- Output ONLY JSON array"""


def build_single_prompt(row: Dict, variation: int = 0) -> str:
    angles = [
        "for an exclusive online offer",
        "for a seasonal promotion",
        "for a flash sale",
        "for a loyalty program",
        "for a product launch",
        "for clearance sale",
    ]
    angle = angles[variation % len(angles)]

    return f"""Create monetary mention and CTA {angle}.

Product: {row.get('object_detected', '')}
Theme: {row.get('theme', '')}
Emotion: {row.get('emotion', '')}

Reply with ONLY:
{{"monetary_mention":"your offer here","call_to_action":"your CTA here"}}"""


def build_fallback(row: Dict) -> Dict[str, str]:
    pass

CTA_TASK = TaskConfig(
    name="cta",
    output_columns=["monetary_mention", "call_to_action"],  # 2 columns
    primary_column="monetary_mention",
    min_length=3,
    system_prompt=SYSTEM_PROMPT,
    system_prompt_single=SYSTEM_PROMPT_SINGLE,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
    deduplicate=False,
)