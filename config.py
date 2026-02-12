# config.py
"""
Central configuration file.
"""

# ============================================================
#  MASTER SWITCH
# ============================================================
USE_OPENAI_SDK = True
ACTIVE_PROFILE = "gml"  
STREAM_PREVIEW = True

# ============================================================
#  API KEY
# ============================================================
NVIDIA_API_KEY = 

# ============================================================
#  API ENDPOINTS
# ============================================================
BASE_URL = "https://integrate.api.nvidia.com/v1"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ============================================================
#  MODEL PROFILES
# ============================================================
MODEL_PROFILES = {
     "oss-120": {
        "model_name":        "openai/gpt-oss-120b",
        "temperature":       1,
        "top_p":             1,
        "max_tokens":        4000,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        # USE_OPENAI_SDK = True
    },

    "oss-20": {
        "model_name":        "openai/gpt-oss-20b",
        "temperature":       1,
        "top_p":             1,
        "max_tokens":        4000,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        # USE_OPENAI_SDK = True
    },
    "qwen": {
        "model_name":        "qwen/qwen2.5-coder-32b-instruct",
        "temperature":       0.2,
        "top_p":             0.7,
        "max_tokens":        8000,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            False,
        # USE_OPENAI_SDK = True

        
    },
    "llama": {
        "model_name":        "meta/llama-4-maverick-17b-128e-instruct",
        "temperature":       0.7,
        "top_p":             0.9,
        "max_tokens":        8000,
        "frequency_penalty": 0.10,
        "presence_penalty":  0.10,
        "stream":            False,
        # USE_OPENAI_SDK = False
        
    },
    "deepseek": {
        "model_name":        "deepseek-ai/deepseek-v3.2",
        "temperature":       1.0,        
        "top_p":             0.95,       
        "max_tokens":        8192,       
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,        
        "thinking":          True,   
        # USE_OPENAI_SDK = True     
    },
    "minimaxai": {
        "model_name":        "minimaxai/minimax-m2.1",
        "temperature":       1.0,         
        "top_p":             0.95,        
        "max_tokens":        8192,        
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,  
        # USE_OPENAI_SDK = True      
    },
     "gml": {
        "model_name":        "z-ai/glm4.7",
        "temperature":       1.0,         
        "top_p":             0.95,        
        "max_tokens":        8192,        
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,   
        # USE_OPENAI_SDK = True     
    },

}

_profile = MODEL_PROFILES[ACTIVE_PROFILE]
MODEL_NAME         = _profile["model_name"]
TEMPERATURE        = _profile["temperature"]
TOP_P              = _profile["top_p"]
MAX_TOKENS         = _profile["max_tokens"]
FREQUENCY_PENALTY  = _profile["frequency_penalty"]
PRESENCE_PENALTY   = _profile["presence_penalty"]
STREAM             = _profile["stream"]
THINKING           = _profile.get("thinking", False) 

# ============================================================
#         EDIT ZONE START — ONLY EDIT BELOW  
# ============================================================

# ── STEP 1: Define your themes and objects (6 objects each) ──
THEME_OBJECTS = {
    "Food":       ["Pizza", "Burger", "Coffee", "Sushi", "Salad", "Juice"],
    "Fashion":    ["Bag", "Watch", "Shoes", "Jacket", "Dress", "Scarf"],
    "Tech":       ["Phone", "Laptop", "Headphones", "Camera", "Tablet", "Speaker"],
    "Automotive": ["Car", "Motorcycle", "Truck", "SUV", "Scooter", "Helmet"],
    "Travel":     ["Hotel", "Beach", "Resort", "Luggage", "Passport", "Mountain"],
    "Finance":    ["Card", "Stocks", "Wallet", "Coins", "Bank", "Investment"],
    # ── ADD YOUR THEMES BELOW ──
    # "Health":     ["Vitamin", "Treadmill", "Yoga Mat", "Protein", "Scale", "Bottle"],
    # "Beauty":     ["Lipstick", "Serum", "Moisturizer", "Perfume", "Mask", "Brush"],
    # "Sports":     ["Football", "Racket", "Gloves", "Jersey", "Helmet", "Cleats"],
    "Home":       ["Sofa", "Lamp", "Blender", "Rug", "Pillow", "Clock"],
    "Gaming":     ["Console", "Controller", "Headset", "Monitor", "Chair", "Keyboard"],
    "Education":  ["Book", "Pen", "Backpack", "Notebook", "Calculator", "Globe"],
}

# ── STEP 2: List themes in order ──
THEME_ORDER = list(THEME_OBJECTS.keys())

# ── STEP 3: Numeric codes (auto-generated) ──
THEME_NUM = {theme: i for i, theme in enumerate(THEME_ORDER)}

# ── STEP 4: Set total rows (must be multiple of BASE_UNIT) ──
# BASE_UNIT is auto-calculated below. After first run you will see it printed.
# Common choices:
#   6 themes:  540, 1080, 2700, 5400, 10800
#   7 themes:  630, 1260, 3150, 6300, 12600
#   8 themes:  720, 1440, 3600, 7200, 14400
#   9 themes:  810, 1620, 4050, 8100, 16200
#   10 themes: 900, 1800, 4500, 9000, 18000
theme_obj = len(THEME_ORDER)
rows = 90 * theme_obj
cycle = 4
TOTAL_ROWS = rows * cycle

# ============================================================
#  ██████  EDIT ZONE END — DO NOT EDIT BELOW  ██████
# ============================================================

# ============================================================
#  AUTO-CALCULATED CONSTANTS
# ============================================================
EMOTION_ORDER = ["Joy", "Anger", "Trust", "Excitement", "Fear"]
SENTIMENT_ORDER = ["Positive", "Negative", "Neutral"]

PER_OBJECT_DISTRIBUTION = [
    ("Joy",        "Positive", 2),
    ("Joy",        "Neutral",  1),
    ("Anger",      "Negative", 2),
    ("Anger",      "Neutral",  1),
    ("Trust",      "Positive", 2),
    ("Trust",      "Neutral",  1),
    ("Excitement", "Positive", 1),
    ("Excitement", "Neutral",  2),
    ("Fear",       "Negative", 3),
]

SENTIMENT_TRUST_MAP = {
    "Positive": "Safe",
    "Negative": "Unsafe",
    "Neutral":  "Questionable",
}

# ── Dimensions ──
NUM_THEMES = len(THEME_OBJECTS)
OBJECTS_PER_THEME = len(list(THEME_OBJECTS.values())[0])
ROWS_PER_OBJECT_PER_CYCLE = sum(c for _, _, c in PER_OBJECT_DISTRIBUTION)
TOTAL_OBJECTS = NUM_THEMES * OBJECTS_PER_THEME
BASE_UNIT = NUM_THEMES * OBJECTS_PER_THEME * ROWS_PER_OBJECT_PER_CYCLE
NUM_CYCLES = TOTAL_ROWS // BASE_UNIT

# ── Validations ──
assert TOTAL_ROWS % BASE_UNIT == 0, (
    f"\n\n"
    f"  ✗ TOTAL_ROWS ({TOTAL_ROWS}) is NOT a multiple of BASE_UNIT ({BASE_UNIT}).\n"
    f"  Valid values for {NUM_THEMES} themes:\n"
    f"    " + ", ".join(str(i * BASE_UNIT) for i in [1, 2, 5, 10, 20, 50, 100]) + "\n"
)

for theme, objs in THEME_OBJECTS.items():
    assert len(objs) == OBJECTS_PER_THEME, (
        f"\n\n"
        f"  ✗ Theme '{theme}' has {len(objs)} objects, "
        f"but '{THEME_ORDER[0]}' has {OBJECTS_PER_THEME}.\n"
        f"  All themes must have the same number of objects.\n"
    )

assert len(THEME_ORDER) == NUM_THEMES, (
    f"THEME_ORDER has {len(THEME_ORDER)} entries but "
    f"THEME_OBJECTS has {NUM_THEMES} themes"
)

for theme in THEME_ORDER:
    assert theme in THEME_OBJECTS, (
        f"'{theme}' is in THEME_ORDER but not in THEME_OBJECTS"
    )

# ── Other numeric codes (fixed) ──
SENTIMENT_NUM = {"Positive": 0, "Negative": 1, "Neutral": 2}
EMOTION_NUM   = {"Joy": 0, "Anger": 1, "Trust": 2, "Excitement": 3, "Fear": 4}
COLOUR_NUM    = {
    "Red": 0, "Black": 1, "Blue": 2, "Green": 3,
    "White": 4, "Grey": 5, "Yellow": 6, "Brown": 7,
}
ATTENTION_NUM = {"High": 0, "Medium": 1, "Low": 2}
TRUST_NUM     = {"Safe": 0, "Unsafe": 1, "Questionable": 2}
AUDIENCE_NUM  = {
    "General": 0, "Food Lovers": 1, "Tech Enthusiasts": 2,
    "Fashionistas": 3, "Car Enthusiasts": 4, "Travelers": 5,
    "Investors": 6, "Young Adults": 7, "Families": 8,
    "Professionals": 9,
}
CTR_NUM    = {"High": 0, "Medium": 1, "Low": 2}
SHARES_NUM = {"High": 0, "Medium": 1, "Low": 2}

# ============================================================
#  AUTO-CALCULATED COLOUR DISTRIBUTION
#  Distributes 8 colours as evenly as possible into BASE_UNIT
# ============================================================
_NUM_COLOURS = len(COLOUR_NUM)  # 8
_colour_base = BASE_UNIT // _NUM_COLOURS
_colour_remainder = BASE_UNIT % _NUM_COLOURS
_colour_names = list(COLOUR_NUM.keys())

COLOUR_DISTRIBUTION_BASE = {}
for i, colour in enumerate(_colour_names):
    # First _colour_remainder colours get +1
    COLOUR_DISTRIBUTION_BASE[colour] = _colour_base + (1 if i < _colour_remainder else 0)

# Verify
_colour_sum = sum(COLOUR_DISTRIBUTION_BASE.values())
assert _colour_sum == BASE_UNIT, (
    f"Colour distribution sums to {_colour_sum}, expected {BASE_UNIT}"
)

# Scale to total
COLOUR_DISTRIBUTION = {
    colour: count * NUM_CYCLES
    for colour, count in COLOUR_DISTRIBUTION_BASE.items()
}

# ============================================================
#  AUTO-CALCULATED TRIPLE DISTRIBUTION (Attention/CTR/Shares)
#  BASE_UNIT must be divisible by 3
# ============================================================
assert BASE_UNIT % 3 == 0, (
    f"\n\n"
    f"  ✗ BASE_UNIT ({BASE_UNIT}) is not divisible by 3.\n"
    f"  This happens when themes×objects×15 is not divisible by 3.\n"
    f"  With 6 objects per theme and 15 rows per object,\n"
    f"  BASE_UNIT = themes × 90. Any number of themes works.\n"
)
TRIPLE_EACH_BASE = BASE_UNIT // 3
TRIPLE_EACH = TRIPLE_EACH_BASE * NUM_CYCLES

# ============================================================
#  AUTO-CALCULATED AUDIENCE DISTRIBUTION
#  BASE_UNIT must be divisible by 10
# ============================================================
assert BASE_UNIT % 10 == 0, (
    f"\n\n"
    f"  ✗ BASE_UNIT ({BASE_UNIT}) is not divisible by 10.\n"
    f"  This happens when themes×90 is not divisible by 10.\n"
    f"  Solution: use an even number of themes,\n"
    f"  or adjust OBJECTS_PER_THEME.\n"
)
AUDIENCE_EACH_BASE = BASE_UNIT // 10
AUDIENCE_EACH = AUDIENCE_EACH_BASE * NUM_CYCLES

# ============================================================
#  TEXT GENERATION
# ============================================================
TEXT_BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 2
MAX_RETRIES = 3
RETRY_DELAY = 3

# ============================================================
#  OUTPUT & LOGGING
# ============================================================
OUTPUT_DIR = "generated_datasets"
LOG_RESPONSES = True
LOG_DIR = "api_logs"

# ============================================================
#  CTA & MONETARY OPTIONS
# ============================================================
CTA_OPTIONS = [
    "None", "Order Now", "Book Now", "Shop Now", "Apply Now",
    "Sign Up", "Learn More", "Get Started", "Try Now",
    "Buy Now", "Subscribe",
]

MONETARY_OPTIONS = [
    "None", "None", "None", "None",
    "$9.99", "$14.99", "$19.99", "$29.99", "$49.99",
    "$99", "$199", "$299", "$499", "$899",
    "10% off", "15% off", "20% off", "25% off",
    "30% off", "50% off", "$5 off", "$10 off",
]

# ============================================================
#  FORBIDDEN COMBINATIONS
# ============================================================
FORBIDDEN_COMBOS = {
    ("Joy",        "Negative"),
    ("Anger",      "Positive"),
    ("Trust",      "Negative"),
    ("Excitement", "Negative"),
    ("Fear",       "Positive"),
}

# ============================================================
#  EXPECTED DISTRIBUTIONS (auto-calculated for validation)
# ============================================================
_ROWS_PER_THEME = OBJECTS_PER_THEME * ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
_SENT_PER_THEME = 5 * OBJECTS_PER_THEME * NUM_CYCLES
_SENT_GLOBAL = _SENT_PER_THEME * NUM_THEMES
_EMO_PER_THEME = 3 * OBJECTS_PER_THEME * NUM_CYCLES
_EMO_GLOBAL = _EMO_PER_THEME * NUM_THEMES

EXPECTED = {
    "theme":            {t: _ROWS_PER_THEME for t in THEME_NUM},
    "sentiment":        {s: _SENT_GLOBAL for s in SENTIMENT_NUM},
    "emotion":          {e: _EMO_GLOBAL for e in EMOTION_NUM},
    "dominant_colour":  COLOUR_DISTRIBUTION,
    "trust_safety":     {v: _SENT_GLOBAL for v in TRUST_NUM},
    "attention_score":  {v: TRIPLE_EACH for v in ATTENTION_NUM},
    "predicted_ctr":    {v: TRIPLE_EACH for v in CTR_NUM},
    "likelihood_shares":{v: TRIPLE_EACH for v in SHARES_NUM},
    "target_audience":  {a: AUDIENCE_EACH for a in AUDIENCE_NUM},
    "object_detected":  {
        obj: ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
        for objs in THEME_OBJECTS.values()
        for obj in objs
    },
}

# ============================================================
#  PRINT SUMMARY
# ============================================================
print(f"\n  ╔{'═'*50}╗")
print(f"  ║  CONFIG LOADED SUCCESSFULLY                    ║")
print(f"  ╠{'═'*50}╣")
print(f"  ║  Themes         : {NUM_THEMES:<30}║")
print(f"  ║  Objects/theme  : {OBJECTS_PER_THEME:<30}║")
print(f"  ║  Total objects  : {TOTAL_OBJECTS:<30}║")
print(f"  ║  Rows/object    : {ROWS_PER_OBJECT_PER_CYCLE:<30}║")
print(f"  ║  Base unit      : {BASE_UNIT:<30}║")
print(f"  ║  Cycles         : {NUM_CYCLES:<30}║")
print(f"  ║  TOTAL ROWS     : {TOTAL_ROWS:<30}║")
print(f"  ╠{'═'*50}╣")
print(f"  ║  Sentiments     : 3 × {_SENT_GLOBAL:<25}║")
print(f"  ║  Emotions       : 5 × {_EMO_GLOBAL:<25}║")
print(f"  ║  Colours        : 8 (auto-distributed)        ║")
print(f"  ║  Attention/CTR  : 3 × {TRIPLE_EACH:<25}║")
print(f"  ║  Audiences      : 10 × {AUDIENCE_EACH:<24}║")
print(f"  ╚{'═'*50}╝")
print(f"  Themes: {', '.join(THEME_ORDER)}")
print(f"  Colours: {COLOUR_DISTRIBUTION_BASE}")