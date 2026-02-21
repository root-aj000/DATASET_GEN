
# config.py
"""
Central configuration file.
Supports: Chat Completions API, Responses API, multiple thinking modes.
"""

# ============================================================
#  MASTER SWITCH
# ============================================================
USE_OPENAI_SDK = True
ACTIVE_PROFILE = "oss-120"  
STREAM_PREVIEW = True

# ============================================================
#  API KEY
# ============================================================
# NVIDIA_API_KEY = ""
NVIDIA_API_KEY="api-here"


# ============================================================
#  API ENDPOINTS
# ============================================================
BASE_URL = "https://integrate.api.nvidia.com/v1"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
RESPONSES_URL = "https://integrate.api.nvidia.com/v1"

# ============================================================
#  API TYPES
# ============================================================
# "chat"      = Chat Completions API (client.chat.completions.create)
# "responses" = Responses API (client.responses.create)
API_TYPE_CHAT = "chat"
API_TYPE_RESPONSES = "responses"

# ============================================================
#  THINKING MODE CONFIGURATIONS (per model type)
# ============================================================
THINKING_CONFIGS = {
    # Standard NVIDIA/DeepSeek style (Chat API)
    "standard": {
        "api_type": "chat",
        "extra_body": {"chat_template_kwargs": {"thinking": True}},
        "reasoning_field": "reasoning_content",
    },
    # GLM-style (Chat API)
    "glm": {
        "api_type": "chat",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
        "reasoning_field": "reasoning_content",
    },
    # Responses API with reasoning (like oss-120b)
    "responses": {
        "api_type": "responses",
        "extra_body": None,
        "reasoning_field": "reasoning_text",  # Different field in responses API
    },
    # No thinking
    "none": {
        "api_type": "chat",
        "extra_body": None,
        "reasoning_field": None,
    },
}

# ============================================================
#  MODEL PROFILES
# ============================================================
MODEL_PROFILES = {
    # ═══════════════════════════════════════════════════════
    # RESPONSES API MODELS
    # ═══════════════════════════════════════════════════════
    "oss-120": {
        "model_name":        "openai/gpt-oss-120b",
        "api_type":          "chat",  # Uses Responses API
        "temperature":       1,
        "top_p":             1,
        "max_tokens":        4096,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          False,  # Has reasoning output
        "thinking_style":    "responses",
    },

    "oss-20": {
        "model_name":        "openai/gpt-oss-20b",
        "api_type":          "responses",
        "temperature":       1,
        "top_p":             1,
        "max_tokens":        4096,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          True,
        "thinking_style":    "responses",
    },

    # ═══════════════════════════════════════════════════════
    # CHAT COMPLETIONS API MODELS
    # ═══════════════════════════════════════════════════════
    "qwen": {
        "model_name":        "qwen/qwen2.5-coder-32b-instruct",
        "api_type":          "chat",
        "temperature":       0.2,
        "top_p":             0.7,
        "max_tokens":        8000,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            False,
        "thinking":          False,
        "thinking_style":    "none",
    },

    "llama": {
        "model_name":        "meta/llama-4-maverick-17b-128e-instruct",
        "api_type":          "chat",
        "temperature":       0.7,
        "top_p":             0.9,
        "max_tokens":        8000,
        "frequency_penalty": 0.10,
        "presence_penalty":  0.10,
        "stream":            False,
        "thinking":          False,
        "thinking_style":    "none",
    },

    "deepseek": {
        "model_name":        "deepseek-ai/deepseek-v3.2",
        "api_type":          "chat",
        "temperature":       1.0,        
        "top_p":             0.95,       
        "max_tokens":        8192,       
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,        
        "thinking":          True,
        "thinking_style":    "standard",
    },

    "minimaxai": {
        "model_name":        "minimaxai/minimax-m2.1",
        "api_type":          "chat",
        "temperature":       1.0,         
        "top_p":             0.95,        
        "max_tokens":        8192,        
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          False,
        "thinking_style":    "none",
    },

    "glm4": {
        "model_name":        "z-ai/glm4.7",
        "api_type":          "chat",
        "temperature":       1.0,         
        "top_p":             0.95,        
        "max_tokens":        8192,        
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          True,
        "thinking_style":    "glm",
    },

    "glm5": {
        "model_name":        "z-ai/glm5",
        "api_type":          "chat",
        "temperature":       1.0,         
        "top_p":             1,        
        "max_tokens":        16384,        
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          True,
        "thinking_style":    "glm",
    },

    # ═══════════════════════════════════════════════════════
    # BOTH APIs SUPPORTED (example)
    # ═══════════════════════════════════════════════════════
    "dual-model": {
        "model_name":        "some/dual-model",
        "api_type":          "chat",  # Primary API
        "api_type_fallback": "responses",  # Fallback API
        "temperature":       1.0,
        "top_p":             0.95,
        "max_tokens":        8192,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "stream":            True,
        "thinking":          True,
        "thinking_style":    "standard",
    },
}

# ============================================================
#  EXTRACT ACTIVE PROFILE
# ============================================================
_profile = MODEL_PROFILES[ACTIVE_PROFILE]
MODEL_NAME         = _profile["model_name"]
API_TYPE           = _profile.get("api_type", "chat")  # Default to chat
API_TYPE_FALLBACK  = _profile.get("api_type_fallback", None)
TEMPERATURE        = _profile["temperature"]
TOP_P              = _profile["top_p"]
MAX_TOKENS         = _profile["max_tokens"]
FREQUENCY_PENALTY  = _profile.get("frequency_penalty", 0.0)
PRESENCE_PENALTY   = _profile.get("presence_penalty", 0.0)
STREAM             = _profile["stream"]
THINKING           = _profile.get("thinking", False)
THINKING_STYLE     = _profile.get("thinking_style", "none")

# Get thinking config
_thinking_cfg = THINKING_CONFIGS.get(THINKING_STYLE, THINKING_CONFIGS["none"])
THINKING_EXTRA_BODY = _thinking_cfg.get("extra_body")
THINKING_REASONING_FIELD = _thinking_cfg.get("reasoning_field")

# Override API type from thinking config if specified
if THINKING and _thinking_cfg.get("api_type"):
    API_TYPE = _thinking_cfg["api_type"]

# Custom thinking config override
if "thinking_config" in _profile:
    THINKING_EXTRA_BODY = _profile["thinking_config"]

# ============================================================
#  REST OF CONFIG (unchanged)
# ============================================================

THEME_OBJECTS = {
    "Food":       ["Pizza", "Burger", "Coffee", "Sushi", "Salad", "Juice"],
    "Fashion":    ["Bag", "Watch", "Shoes", "Jacket", "Dress", "Scarf"],
    "Tech":       ["Phone", "Laptop", "Headphones", "Camera", "Tablet", "Speaker"],
    "Automotive": ["Car", "Motorcycle", "Truck", "SUV", "Scooter", "Helmet"],
    "Travel":     ["Hotel", "Beach", "Resort", "Luggage", "Passport", "Mountain"],
    "Finance":    ["Card", "Stocks", "Wallet", "Coins", "Bank", "Investment"],
    "Home":       ["Sofa", "Lamp", "Blender", "Rug", "Pillow", "Clock"],
    "Gaming":     ["Console", "Controller", "Headset", "Monitor", "Chair", "Keyboard"],
    "Education":  ["Book", "Pen", "Backpack", "Notebook", "Calculator", "Globe"],
}

THEME_ORDER = list(THEME_OBJECTS.keys())
THEME_NUM = {theme: i for i, theme in enumerate(THEME_ORDER)}

theme_obj = len(THEME_ORDER)
rows = 90 * theme_obj
cycle = 4
TOTAL_ROWS = rows * cycle

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

NUM_THEMES = len(THEME_OBJECTS)
OBJECTS_PER_THEME = len(list(THEME_OBJECTS.values())[0])
ROWS_PER_OBJECT_PER_CYCLE = sum(c for _, _, c in PER_OBJECT_DISTRIBUTION)
TOTAL_OBJECTS = NUM_THEMES * OBJECTS_PER_THEME
BASE_UNIT = NUM_THEMES * OBJECTS_PER_THEME * ROWS_PER_OBJECT_PER_CYCLE
NUM_CYCLES = TOTAL_ROWS // BASE_UNIT

# Validations
assert TOTAL_ROWS % BASE_UNIT == 0
for theme, objs in THEME_OBJECTS.items():
    assert len(objs) == OBJECTS_PER_THEME
assert len(THEME_ORDER) == NUM_THEMES

# Numeric codes
SENTIMENT_NUM = {"Positive": 0, "Negative": 1, "Neutral": 2}
EMOTION_NUM   = {"Joy": 0, "Anger": 1, "Trust": 2, "Excitement": 3, "Fear": 4}
COLOUR_NUM    = {"Red": 0, "Black": 1, "Blue": 2, "Green": 3, "White": 4, "Grey": 5, "Yellow": 6, "Brown": 7}
ATTENTION_NUM = {"High": 0, "Medium": 1, "Low": 2}
TRUST_NUM     = {"Safe": 0, "Unsafe": 1, "Questionable": 2}
AUDIENCE_NUM  = {"General": 0, "Food Lovers": 1, "Tech Enthusiasts": 2, "Fashionistas": 3, "Car Enthusiasts": 4, "Travelers": 5, "Investors": 6, "Young Adults": 7, "Families": 8, "Professionals": 9}
CTR_NUM    = {"High": 0, "Medium": 1, "Low": 2}
SHARES_NUM = {"High": 0, "Medium": 1, "Low": 2}

# Distributions
_NUM_COLOURS = len(COLOUR_NUM)
_colour_base = BASE_UNIT // _NUM_COLOURS
_colour_remainder = BASE_UNIT % _NUM_COLOURS
_colour_names = list(COLOUR_NUM.keys())
COLOUR_DISTRIBUTION_BASE = {colour: _colour_base + (1 if i < _colour_remainder else 0) for i, colour in enumerate(_colour_names)}
COLOUR_DISTRIBUTION = {colour: count * NUM_CYCLES for colour, count in COLOUR_DISTRIBUTION_BASE.items()}

assert BASE_UNIT % 3 == 0
TRIPLE_EACH_BASE = BASE_UNIT // 3
TRIPLE_EACH = TRIPLE_EACH_BASE * NUM_CYCLES

assert BASE_UNIT % 10 == 0
AUDIENCE_EACH_BASE = BASE_UNIT // 10
AUDIENCE_EACH = AUDIENCE_EACH_BASE * NUM_CYCLES

# Text generation
TEXT_BATCH_SIZE = 30
DELAY_BETWEEN_BATCHES = 2
MAX_RETRIES = 3
RETRY_DELAY = 3

# Output & logging
OUTPUT_DIR = "generated_datasets"
LOG_RESPONSES = True
LOG_DIR = "api_logs"

# CTA & Monetary
CTA_OPTIONS = ["None", "Order Now", "Book Now", "Shop Now", "Apply Now", "Sign Up", "Learn More", "Get Started", "Try Now", "Buy Now", "Subscribe"]
MONETARY_OPTIONS = ["None", "None", "None", "None", "$9.99", "$14.99", "$19.99", "$29.99", "$49.99", "$99", "$199", "$299", "$499", "$899", "10% off", "15% off", "20% off", "25% off", "30% off", "50% off", "$5 off", "$10 off"]

# Forbidden combinations
FORBIDDEN_COMBOS = {("Joy", "Negative"), ("Anger", "Positive"), ("Trust", "Negative"), ("Excitement", "Negative"), ("Fear", "Positive")}

# Expected distributions
_ROWS_PER_THEME = OBJECTS_PER_THEME * ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES
_SENT_PER_THEME = 5 * OBJECTS_PER_THEME * NUM_CYCLES
_SENT_GLOBAL = _SENT_PER_THEME * NUM_THEMES
_EMO_PER_THEME = 3 * OBJECTS_PER_THEME * NUM_CYCLES
_EMO_GLOBAL = _EMO_PER_THEME * NUM_THEMES

EXPECTED = {
    "theme": {t: _ROWS_PER_THEME for t in THEME_NUM},
    "sentiment": {s: _SENT_GLOBAL for s in SENTIMENT_NUM},
    "emotion": {e: _EMO_GLOBAL for e in EMOTION_NUM},
    "dominant_colour": COLOUR_DISTRIBUTION,
    "trust_safety": {v: _SENT_GLOBAL for v in TRUST_NUM},
    "attention_score": {v: TRIPLE_EACH for v in ATTENTION_NUM},
    "predicted_ctr": {v: TRIPLE_EACH for v in CTR_NUM},
    "likelihood_shares": {v: TRIPLE_EACH for v in SHARES_NUM},
    "target_audience": {a: AUDIENCE_EACH for a in AUDIENCE_NUM},
    "object_detected": {obj: ROWS_PER_OBJECT_PER_CYCLE * NUM_CYCLES for objs in THEME_OBJECTS.values() for obj in objs},
}

# Print summary
print(f"\n  ╔{'═'*50}╗")
print(f"  ║  CONFIG LOADED SUCCESSFULLY                    ║")
print(f"  ╠{'═'*50}╣")
print(f"  ║  Model          : {MODEL_NAME[:30]:<30}║")
print(f"  ║  API Type       : {API_TYPE.upper():<30}║")
print(f"  ║  Thinking       : {'ON (' + THINKING_STYLE + ')' if THINKING else 'OFF':<30}║")
print(f"  ║  Streaming      : {'ON' if STREAM else 'OFF':<30}║")
print(f"  ╠{'═'*50}╣")
print(f"  ║  Themes         : {NUM_THEMES:<30}║")
print(f"  ║  Objects/theme  : {OBJECTS_PER_THEME:<30}║")
print(f"  ║  TOTAL ROWS     : {TOTAL_ROWS:<30}║")
print(f"  ╚{'═'*50}╝")