# agents/tasks/audience_task.py
"""
Audience task — fills "target_audience" column.
"""

from typing import List, Dict
from agents.task_config import TaskConfig
SYSTEM_PROMPT = """You are a marketing expert.
Classify ads into COMMON audience categories.
Use only broad, repeatable audience groups.
Reply with ONLY a JSON object."""

SYSTEM_PROMPT_SINGLE = """You are a marketing expert.
Classify into common audience categories only.
Reply with ONLY a JSON object."""


# Predefined categories to constrain outputs

ALLOWED_AUDIENCES = """
CATEGORIES (use EXACTLY these names):

FOOD THEME:
- Pizza Lovers: pizza related ads
- Burger Lovers: burger related ads
- Coffee Lovers: coffee related ads
- Sushi Lovers: sushi related ads
- Salad Lovers: salad, healthy food ads
- Juice Lovers: juice, smoothie ads

FASHION THEME:
- Bag Enthusiasts: bags, purses ads
- Watch Enthusiasts: watches, timepieces ads
- Shoe Lovers: shoes, footwear ads
- Jacket Lovers: jackets, outerwear ads
- Dress Lovers: dresses, formal wear ads
- Scarf Lovers: scarves, accessories ads

TECH THEME:
- Phone Enthusiasts: smartphones, mobile ads
- Laptop Enthusiasts: laptops, computers ads
- Headphone Lovers: headphones, earbuds ads
- Camera Enthusiasts: cameras, photography ads
- Tablet Users: tablets, iPad ads
- Speaker Lovers: speakers, audio ads

AUTOMOTIVE THEME:
- Car Enthusiasts: cars, sedans ads
- Motorcycle Enthusiasts: motorcycles, bikes ads
- Truck Enthusiasts: trucks, pickups ads
- SUV Enthusiasts: SUVs, crossovers ads
- Scooter Riders: scooters, mopeds ads
- Helmet Buyers: helmets, safety gear ads

TRAVEL THEME:
- Hotel Seekers: hotels, accommodations ads
- Beach Lovers: beaches, coastal ads
- Resort Lovers: resorts, vacation ads
- Luggage Shoppers: luggage, bags ads
- Passport Holders: international travel ads
- Mountain Lovers: mountains, hiking ads

FINANCE THEME:
- Card Users: credit cards, debit cards ads
- Stock Traders: stocks, trading ads
- Wallet Shoppers: wallets, accessories ads
- Coin Collectors: coins, currency ads
- Bank Customers: banking, accounts ads
- Investment Seekers: investments, funds ads

HOME THEME:
- Sofa Shoppers: sofas, couches ads
- Lamp Buyers: lamps, lighting ads
- Blender Users: blenders, kitchen ads
- Rug Shoppers: rugs, carpets ads
- Pillow Lovers: pillows, bedding ads
- Clock Enthusiasts: clocks, decor ads

GAMING THEME:
- Console Gamers: gaming consoles ads
- Controller Users: controllers, gaming ads
- Headset Gamers: gaming headsets ads
- Monitor Enthusiasts: gaming monitors ads
- Gaming Chair Buyers: gaming chairs ads
- Keyboard Enthusiasts: gaming keyboards ads

EDUCATION THEME:
- Book Readers: books, reading ads
- Pen Collectors: pens, writing ads
- Backpack Shoppers: backpacks, school bags ads
- Notebook Users: notebooks, journals ads
- Calculator Users: calculators, math tools ads
- Globe Enthusiasts: globes, geography ads

DEFAULT:
- General: unclear or mixed themes
"""



def build_batch_prompt(rows: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(rows):
        context = f"Ad: {r.get('text', '')}" if r.get('text') else ""
        lines.append(
            f"{i+1}. Object: {r.get('object_detected', '')} | "
            f"Theme: {r.get('theme', '')} | {context}"
        )
    rows_block = "\n".join(lines)

    return f"""Classify {len(rows)} ads into common audience categories.

{ALLOWED_AUDIENCES}

ROWS:
{rows_block}

OUTPUT FORMAT:
[
  {{"id":1,"target_audience":"food lovers"}},
  {{"id":2,"target_audience":"travelers"}}
]

STRICT RULES:
- ONLY use categories from the allowed list above
- Pick the SINGLE best matching category
- 1-2 words only
- Reply with ONLY JSON array, no explanation

"""


def build_single_prompt(row: Dict) -> str:
    return f"""Classify this ad into ONE common audience category.

{ALLOWED_AUDIENCES}

Object: {row.get('object_detected', '')}
Theme: {row.get('theme', '')}
Ad Copy: {row.get('text', '')}

RULES:
- ONLY use a category from the list above
- Pick ONE best match
- 1-2 words only

Reply ONLY: {{"target_audience":"category here"}}"""



def build_fallback(row: Dict) -> Dict[str, str]:
    theme = row.get("theme", "").lower()

    theme_audiences = {
        "technology": "Tech-savvy professionals",
        "fashion": "Style-conscious millennials",
        "food": "Food enthusiasts",
        "health": "Health-conscious adults",
        "travel": "Adventure seekers",
        "finance": "Young professionals",
        "home": "Homeowners",
        "beauty": "Beauty enthusiasts",
        "sports": "Fitness enthusiasts",
        "automotive": "Car enthusiasts",
        "education": "Lifelong learners",
        "gaming": "Gaming enthusiasts",
    }

    audience = theme_audiences.get(theme, "General consumers")
    return {"target_audience": audience}


AUDIENCE_TASK = TaskConfig(
    name="audience",
    output_columns=["target_audience"],  # Single column
    primary_column="target_audience",
    min_length=3,
    system_prompt=SYSTEM_PROMPT,
    system_prompt_single=SYSTEM_PROMPT_SINGLE,
    batch_prompt_builder=build_batch_prompt,
    single_prompt_builder=build_single_prompt,
    fallback_builder=build_fallback,
    deduplicate=False,  # Audiences can repeat
)




