# # search_agent.py
# """
# Generates GOOGLE IMAGE SEARCH QUERIES (img_desc) using NVIDIA API.
# Based on text_generator.py architecture.
# NO FALLBACK TEMPLATES - but has rule-based construction for Phase 5.
# """

# import os
# import time
# import re
# from typing import Optional, List, Tuple
# import json

# from config import (
#     USE_OPENAI_SDK,
#     NVIDIA_API_KEY,
#     BASE_URL,
#     INVOKE_URL,
#     STREAM,
#     THINKING,
#     MODEL_NAME,
#     MAX_TOKENS,
#     TEMPERATURE,
#     TOP_P,
#     FREQUENCY_PENALTY,
#     PRESENCE_PENALTY,
#     MAX_RETRIES,
#     RETRY_DELAY,
#     LOG_RESPONSES,
#     LOG_DIR,
#     ACTIVE_PROFILE,
# )

# if USE_OPENAI_SDK:
#     from openai import OpenAI
# else:
#     import requests


# SYSTEM_INSTRUCTION = """You are a search query generator for image-text matching. Given user input, generate a query that matches the SUBJECT content in image descriptions.

# Dataset text field contains descriptions like:
# "Bite into a crispy crust loaded with toppings for pure joy"

# Your task:
# 1. Extract ONLY subject-related terms (objects, items, physical descriptions)
# 2. IGNORE promotional/emotional words (joy, happiness, amazing, ultimate, perfect)
# 3. IGNORE action phrases (experience, treat yourself, indulge, satisfy)
# 4. IGNORE advertising language

# Focus on:
# ✓ Physical objects (cheese, crust, toppings, slice)
# ✓ Descriptive attributes (crispy, hot, fresh, melty, cheesy)
# ✓ Subject context (pizza, food, meal)

# Ignore:
# ✗ Emotions (joy, happiness, smile, love)
# ✗ Promotional words (ultimate, amazing, perfect, pure)
# ✗
# """
# SYSTEM_INSTRUCTION_SINGLE = """You are a search query generator for image-text matching. Given user input, generate a query that matches the SUBJECT content in image descriptions.

# Dataset text field contains descriptions like:
# "Bite into a crispy crust loaded with toppings for pure joy"

# Your task:
# 1. Extract ONLY subject-related terms (objects, items, physical descriptions)
# 2. IGNORE promotional/emotional words (joy, happiness, amazing, ultimate, perfect)
# 3. IGNORE action phrases (experience, treat yourself, indulge, satisfy)
# 4. IGNORE advertising language

# Focus on:
# ✓ Physical objects (cheese, crust, toppings, slice)
# ✓ Descriptive attributes (crispy, hot, fresh, melty, cheesy)
# ✓ Subject context 

# Ignore:
# ✗ Emotions (joy, happiness, smile, love)
# ✗ Promotional words (ultimate, amazing, perfect, pure)
# ✗
# """

# class SearchAgent:
#     """Generates img_desc column. Matches text_generator.py architecture."""

#     def __init__(self):
#         if USE_OPENAI_SDK:
#             self.client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
#             print(f"  API Mode  : OpenAI SDK")
#         else:
#             self.client = None
#             self.headers = {
#                 "Authorization": f"Bearer {NVIDIA_API_KEY}",
#                 "Accept": ("text/event-stream" if STREAM else "application/json"),
#                 "Content-Type": "application/json",
#             }
#             print(f"  API Mode  : Raw requests")

#         print(f"  Model     : {MODEL_NAME}")
#         print(f"  Profile   : {ACTIVE_PROFILE}")
#         print(f"  Streaming : {'ON' if STREAM else 'OFF'}")
#         print(f"  Thinking  : {'ON' if THINKING else 'OFF'}")
#         print(f"  Task      : IMAGE SEARCH QUERIES")

#         self._used_queries: set = set()  # ← Dedup like text_generator
#         self._api_calls: int = 0

#         if LOG_RESPONSES:
#             os.makedirs(LOG_DIR, exist_ok=True)

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  API CALLS — IDENTICAL TO TEXT_GENERATOR
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _call_api(self, prompt: str, system: str = None, temperature: float = None) -> Optional[str]:
#         self._api_calls += 1
#         temp = temperature if temperature is not None else TEMPERATURE
#         sys_msg = system or SYSTEM_INSTRUCTION

#         if USE_OPENAI_SDK:
#             return self._call_sdk(prompt, sys_msg, temp)
#         else:
#             return self._call_req(prompt, sys_msg, temp)

#     def _call_sdk(self, prompt: str, system: str, temp: float) -> Optional[str]:
#         """OpenAI SDK call with streaming + thinking support."""
#         try:
#             kwargs = {
#                 "model": MODEL_NAME,
#                 "messages": [
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": prompt},
#                 ],
#                 "temperature": temp,
#                 "top_p": TOP_P,
#                 "max_tokens": MAX_TOKENS,
#                 "stream": STREAM,
#             }

#             if THINKING:
#                 kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

#             completion = self.client.chat.completions.create(**kwargs)

#             if STREAM:
#                 content_chunks = []
#                 reasoning_chunks = []
#                 prompt_tokens = 0
#                 completion_tokens = 0

#                 for chunk in completion:
#                     if not getattr(chunk, "choices", None):
#                         if hasattr(chunk, "usage") and chunk.usage:
#                             prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
#                             completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
#                         continue

#                     if not chunk.choices:
#                         continue

#                     delta = chunk.choices[0].delta

#                     # Skip reasoning content
#                     if THINKING:
#                         reasoning = getattr(delta, "reasoning_content", None)
#                         if reasoning:
#                             reasoning_chunks.append(reasoning)
#                             continue

#                     # Collect actual content
#                     if delta.content is not None:
#                         content_chunks.append(delta.content)

#                     # Check for usage
#                     if hasattr(chunk, "usage") and chunk.usage:
#                         prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
#                         completion_tokens = getattr(chunk.usage, "completion_tokens", 0)

#                 content = "".join(content_chunks)

#                 # Log token info
#                 if prompt_tokens or completion_tokens:
#                     total = prompt_tokens + completion_tokens
#                     print(f"        Tokens: p={prompt_tokens} c={completion_tokens} t={total}")

#                 if THINKING and reasoning_chunks:
#                     reasoning_text = "".join(reasoning_chunks)
#                     print(f"        Thinking: {len(reasoning_text)} chars (discarded)")

#                     if LOG_RESPONSES:
#                         log_path = os.path.join(LOG_DIR, f"thinking_{self._api_calls}.txt")
#                         with open(log_path, "w", encoding="utf-8") as f:
#                             f.write(f"=== REASONING ===\n{reasoning_text}\n\n")
#                             f.write(f"=== CONTENT ===\n{content}\n")

#                 return content

#             else:
#                 # Non-streaming
#                 content = completion.choices[0].message.content

#                 if hasattr(completion, "usage") and completion.usage:
#                     u = completion.usage
#                     print(f"        Tokens: p={u.prompt_tokens} c={u.completion_tokens} t={u.total_tokens}")

#                 return content

#         except Exception as e:
#             print(f"        ✗ API: {e}")
#             return None

#     def _call_req(self, prompt: str, system: str, temp: float) -> Optional[str]:
#         """Raw requests call with streaming support."""
#         payload = {
#             "model": MODEL_NAME,
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": prompt},
#             ],
#             "max_tokens": MAX_TOKENS,
#             "temperature": temp,
#             "top_p": TOP_P,
#             "frequency_penalty": FREQUENCY_PENALTY,
#             "presence_penalty": PRESENCE_PENALTY,
#             "stream": STREAM,
#         }

#         if THINKING:
#             payload["chat_template_kwargs"] = {"thinking": True}

#         try:
#             r = requests.post(INVOKE_URL, headers=self.headers, json=payload, timeout=300)
#             if r.status_code != 200:
#                 print(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
#                 return None

#             if STREAM:
#                 content_chunks = []
#                 for line in r.iter_lines():
#                     if not line:
#                         continue
#                     d = line.decode("utf-8")
#                     if d.startswith("data: "):
#                         d = d[6:]
#                     if d.strip() == "[DONE]":
#                         break
#                     try:
#                         obj = json.loads(d)
#                         choices = obj.get("choices", [])
#                         if not choices:
#                             continue
#                         delta = choices[0].get("delta", {})

#                         # Skip reasoning
#                         if THINKING and "reasoning_content" in delta:
#                             continue

#                         # Collect content
#                         content = delta.get("content")
#                         if content:
#                             content_chunks.append(content)
#                     except json.JSONDecodeError:
#                         continue

#                 return "".join(content_chunks)
#             else:
#                 data = r.json()
#                 u = data.get("usage", {})
#                 if u:
#                     print(f"        Tokens: p={u.get('prompt_tokens','?')} c={u.get('completion_tokens','?')} t={u.get('total_tokens','?')}")
#                 return data.get("choices", [{}])[0].get("message", {}).get("content", "")

#         except requests.exceptions.Timeout:
#             print("        ✗ Timeout")
#             return None
#         except Exception as e:
#             print(f"        ✗ {e}")
#             return None

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  PROMPT BUILDERS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _build_batch_prompt(self, rows: List[dict]) -> str:
#         """Build batch prompt for multiple queries."""
#         lines = []
#         for i, r in enumerate(rows):
#             context = f"Ad Copy: {r.get('text', '')}"
#             lines.append(
#                 f"{i+1}. {r['object_detected']} | {r['theme']} | "
#                 f"{r['emotion']} | {context}"
#             )
#         rows_block = "\n".join(lines)

#         return f"""Generate {len(rows)} precise Google Image search queries.
# Each query should find high-quality images for advertising.

# RULES:
# - NO articles (a, the) or conversational phrases
# - Visual descriptors only
# - Include: "filetype:png" where needed not always
# - 5-8 words per query
# - Be specific enough that Google Images returns the EXACT image in top results and keep it short

# ROWS:
# {rows_block}

# OUTPUT FORMAT:
# Return EXACTLY {len(rows)} search queries, one per line, in order.
# No numbering, no explanations, just the queries.

# QUERIES:"""
# # - Translate emotions to lighting (Joy = bright sunlight, Fear = dark moody)
#     def _build_single_prompt(self, row: dict, variation: int = 0) -> str:
#         """Build single query prompt with variations."""
#         obj = row["object_detected"]
#         theme = row["theme"]
#         emotion = row["emotion"]
#         context = f"Ad Copy: {row.get('text', '')}"

#         # Variation angles
#         angles = [
#             "for a product catalog",
#             "for social media advertising",
#             "for a billboard campaign",
#             "for an email marketing banner",
#             "for a website hero image",
#             "for print magazine ads",
#         ]
#         angle = angles[variation % len(angles)]

#         return f"""Create ONE precise Google Image search query {angle}.

# Subject: {obj}
# Theme: {theme}
# Emotion: {emotion}
# {context}

# Rules:
# - No articles (a/the) or conversational words
# - Visual descriptors only
# - Translate {emotion} to lighting/mood
# - Include: "filetype:png" where needed not always
# - 5-12 words total

# Reply with ONLY the search query string:"""

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  LINE PARSERS (like text_generator's JSON parsers)
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _parse_batch(self, raw: str, expected: int) -> List[str]:
#         """Parse batch output - tries 4 strategies."""
#         for name, fn in [
#             ("direct", self._p1_direct_lines),
#             ("clean", self._p2_clean_markdown),
#             ("numbered", self._p3_strip_numbers),
#             ("split", self._p4_aggressive_split),
#         ]:
#             lines = fn(raw, expected)
#             if lines and len(lines) >= expected * 0.5:  # Accept if >= 50% success
#                 print(f"        Parsed via {name}: {len(lines)} queries")
#                 return lines
#         return []

#     def _p1_direct_lines(self, raw: str, expected: int) -> List[str]:
#         """Strategy 1: Simple line splitting."""
#         lines = [line.strip() for line in raw.strip().split('\n') if line.strip()]
#         cleaned = []
#         for line in lines:
#             # Skip headers/footers
#             if any(word in line.lower() for word in ['output', 'queries', 'query', 'here are']):
#                 continue
#             # Basic cleaning
#             line = line.strip('"').strip("'").strip()
#             if len(line) > 10:  # Min query length
#                 cleaned.append(line)
#         return cleaned[:expected]

#     def _p2_clean_markdown(self, raw: str, expected: int) -> List[str]:
#         """Strategy 2: Remove markdown fences."""
#         text = raw.strip()
        
#         # Remove code blocks
#         if "```" in text:
#             parts = text.split("```")
#             # Take the part inside code blocks
#             for i, part in enumerate(parts):
#                 if i % 2 == 1:  # Inside code block
#                     text = part
#                     break
#             else:
#                 # No code block content, just remove markers
#                 text = text.replace("```", "")

#         lines = [line.strip() for line in text.split('\n') if line.strip()]
#         cleaned = []
#         for line in lines:
#             # Remove numbering
#             line = re.sub(r'^[\d\-\.\)]+\s*', '', line)
#             # Remove quotes
#             line = line.strip('"').strip("'")
#             # Skip meta lines
#             if any(word in line.lower() for word in ['output', 'query', 'here']):
#                 continue
#             if len(line) > 10:
#                 cleaned.append(line)
#         return cleaned[:expected]

#     def _p3_strip_numbers(self, raw: str, expected: int) -> List[str]:
#         """Strategy 3: Aggressively strip numbering."""
#         lines = raw.strip().split('\n')
#         cleaned = []
#         for line in lines:
#             # Remove various numbering formats
#             line = re.sub(r'^\s*\d+[\.\)]\s*', '', line)  # 1. or 1)
#             line = re.sub(r'^\s*-\s*', '', line)  # - bullet
#             line = re.sub(r'^\s*\*\s*', '', line)  # * bullet
#             line = line.strip('"').strip("'").strip()
            
#             # Must contain key search terms
#             if len(line) > 10 and any(word in line.lower() for word in ['photography', '4k', 'professional']):
#                 cleaned.append(line)
#         return cleaned[:expected]

#     def _p4_aggressive_split(self, raw: str, expected: int) -> List[str]:
#         """Strategy 4: Split on multiple delimiters."""
#         # Try splitting on newlines, then semicolons
#         text = raw.strip()
        
#         # Remove markdown
#         text = text.replace("```", "")
        
#         # Split on newlines first
#         parts = [p.strip() for p in text.split('\n') if p.strip()]
        
#         # If not enough, try other delimiters
#         if len(parts) < expected:
#             parts = [p.strip() for p in text.split(';') if p.strip()]
        
#         cleaned = []
#         for part in parts:
#             part = re.sub(r'^[\d\-\.\)\*]+\s*', '', part)
#             part = part.strip('"').strip("'")
            
#             # Filter junk
#             if (len(part) > 10 and 
#                 not any(word in part.lower() for word in ['output', 'here are', 'queries']) and
#                 any(char.isalpha() for char in part)):
#                 cleaned.append(part)
        
#         return cleaned[:expected]

#     def _parse_single_response(self, raw: str) -> str:
#         """Parse single query response."""
#         cleaned = raw.strip()
        
#         # Remove markdown
#         if "```" in cleaned:
#             parts = cleaned.split("```")
#             for i, part in enumerate(parts):
#                 if i % 2 == 1:  # Inside code block
#                     cleaned = part.strip()
#                     break
#             else:
#                 cleaned = cleaned.replace("```", "")
        
#         # Remove common prefixes
#         cleaned = re.sub(r'^(query|search|here|output)\s*[:=]\s*', '', cleaned, flags=re.IGNORECASE)
        
#         # Take first line if multi-line
#         lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
#         if lines:
#             cleaned = lines[0]
        
#         # Remove quotes
#         cleaned = cleaned.strip('"').strip("'").strip()
        
#         # Validate
#         if len(cleaned) > 10 and len(cleaned.split()) >= 5:
#             return cleaned
        
#         return ""

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  MAIN ENTRY + PHASES (IDENTICAL STRUCTURE TO TEXT_GENERATOR)
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def generate_queries(self, skeleton_rows: List[dict], batch_label: str = "") -> List[dict]:
#         """
#         Main entry point. Identical 5-phase structure to text_generator.
#         Returns rows with img_desc filled.
#         """
#         total = len(skeleton_rows)
#         result = [row.copy() for row in skeleton_rows]
        
#         # Initialize img_desc column
#         for r in result:
#             if "img_desc" not in r:
#                 r["img_desc"] = ""

#         # Execute 5 phases
#         for phase, method in [
#             (1, lambda: self._phase_full(skeleton_rows, result, batch_label, total)),
#             (2, lambda: self._phase_retry(skeleton_rows, result, batch_label, total)),
#             (3, lambda: self._phase_micro(skeleton_rows, result, batch_label, total)),
#             (4, lambda: self._phase_single(skeleton_rows, result, batch_label, total)),
#             (5, lambda: self._phase_construct(skeleton_rows, result, batch_label, total)),
#         ]:
#             method()
#             filled = self._n_filled(result)
#             if filled == total:
#                 print(f"      ✓ Phase {phase} complete: {filled}/{total}")
#                 return result

#         return result

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  PHASE IMPLEMENTATIONS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _phase_full(self, skel, result, label, total):
#         """Phase 1: Full batch API call."""
#         print(f"      Phase 1: Full batch ({total} rows)")
#         queries = self._call_batch(skel, f"{label}_full")
#         if queries:
#             self._fill(result, queries, list(range(total)))
#         print(f"      Phase 1: {self._n_filled(result)}/{total}")

#     def _phase_retry(self, skel, result, label, total):
#         """Phase 2: Retry gaps."""
#         gaps = self._gaps(result)
#         if not gaps:
#             return
#         print(f"      Phase 2: Retry {len(gaps)} gaps")
#         gap_rows = [skel[i] for i in gaps]
#         queries = self._call_batch(gap_rows, f"{label}_retry")
#         if queries:
#             self._fill(result, queries, gaps)
#         print(f"      Phase 2: {self._n_filled(result)}/{total}")

#     def _phase_micro(self, skel, result, label, total):
#         """Phase 3: Micro-batches (5 at a time)."""
#         gaps = self._gaps(result)
#         if not gaps:
#             return
#         print(f"      Phase 3: Micro-batches for {len(gaps)} rows")
#         for ci in range(0, len(gaps), 5):
#             chunk = gaps[ci:ci+5]
#             chunk_rows = [skel[i] for i in chunk]
#             queries = self._call_batch(chunk_rows, f"{label}_m{ci//5}")
#             if queries:
#                 self._fill(result, queries, chunk)
#             time.sleep(1)
#         print(f"      Phase 3: {self._n_filled(result)}/{total}")

#     def _phase_single(self, skel, result, label, total):
#         """Phase 4: One-by-one generation."""
#         gaps = self._gaps(result)
#         if not gaps:
#             return
#         print(f"      Phase 4: One-by-one for {len(gaps)} rows")
#         for idx in gaps:
#             query = self._single_generate(skel[idx], f"{label}_s{idx}")
#             if query and len(query) >= 10 and query not in self._used_queries:
#                 self._used_queries.add(query)
#                 result[idx]["img_desc"] = query
#                 print(f"        ✓ Row {idx}")
#             else:
#                 print(f"        ⚠ Row {idx}: needs Phase 5")
#             time.sleep(1)
#         print(f"      Phase 4: {self._n_filled(result)}/{total}")

#     def _phase_construct(self, skel, result, label, total):
#         """Phase 5: Rule-based construction (fallback)."""
#         gaps = self._gaps(result)
#         if not gaps:
#             return
#         print(f"      Phase 5: Construct {len(gaps)} queries")
#         for idx in gaps:
#             query = self._construct_query(skel[idx])
#             self._used_queries.add(query)
#             result[idx]["img_desc"] = query
#             print(f"        ✓ Row {idx}: constructed")

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  HELPER METHODS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _call_batch(self, rows: List[dict], label: str) -> List[str]:
#         """Call API for batch, with retries."""
#         prompt = self._build_batch_prompt(rows)
#         for att in range(1, MAX_RETRIES + 1):
#             print(f"        Attempt {att}/{MAX_RETRIES} ({len(rows)} rows) [{label}]")
#             raw = self._call_api(prompt)
#             if raw is None:
#                 time.sleep(RETRY_DELAY)
#                 continue
            
#             if LOG_RESPONSES:
#                 with open(os.path.join(LOG_DIR, f"{label}_a{att}.txt"), "w", encoding="utf-8") as f:
#                     f.write(raw)
            
#             queries = self._parse_batch(raw, len(rows))
#             if queries:
#                 return queries
            
#             print(f"        ✗ No queries parsed")
#             time.sleep(RETRY_DELAY)
        
#         return []

#     def _single_generate(self, row: dict, label: str) -> str:
#         """Generate single query with variations."""
#         for variation in range(6):
#             temp = min(TEMPERATURE + (variation * 0.15), 1.0)
#             prompt = self._build_single_prompt(row, variation=variation)
#             raw = self._call_api(prompt, system=SYSTEM_INSTRUCTION_SINGLE, temperature=temp)
            
#             if raw is None:
#                 time.sleep(2)
#                 continue
            
#             if LOG_RESPONSES:
#                 with open(os.path.join(LOG_DIR, f"{label}_v{variation}.txt"), "w", encoding="utf-8") as f:
#                     f.write(raw)
            
#             query = self._parse_single_response(raw)
#             query = self._clean_query(query)
            
#             if query and len(query) >= 10 and query not in self._used_queries:
#                 return query
            
#             time.sleep(1)
        
#         return ""

#     def _construct_query(self, row: dict) -> str:
#         """Rule-based query construction (Phase 5 fallback)."""
#         obj = row.get("object_detected", "object").lower()
#         color = row.get("dominant_colour", "")
#         theme = row.get("theme", "")
#         emotion = row.get("emotion", "")
        
#         # Base query
#         query_parts = [f'"{obj}"']
        
#         # Add color if meaningful
#         if color and color.lower() not in ['none', 'mixed', 'various']:
#             query_parts.append(color.lower())
        
#         # Add theme
#         if theme:
#             query_parts.append(theme.lower())
        
#         # Lighting/mood based on emotion
#         mood_map = {
#             "Joy": "bright natural lighting",
#             "Anger": "dramatic contrast",
#             "Trust": "clean studio lighting",
#             "Excitement": "vibrant dynamic",
#             "Fear": "moody dark atmosphere"
#         }
#         if emotion in mood_map:
#             query_parts.append(mood_map[emotion])
        
#         # Tech specs
#         query_parts.append("professional photography 4k high resolution sharp focus")
        
#         # Negative keywords
#         query_parts.append("-cartoon -clipart -vector -illustration -logo -text -watermark")
        
#         # File type
#         query_parts.append("filetype:png")
        
#         return " ".join(query_parts)

#     def _fill(self, result: List[dict], queries: List[str], indices: List[int]):
#         """Fill result rows with queries, avoiding duplicates."""
#         used = set()
        
#         for pos, idx in enumerate(indices):
#             # Skip if already filled
#             # if result[idx].get("img_desc") and len(result[idx]["img_desc"]) >= 10:
#             if result[idx].get("img_desc") and isinstance(result[idx]["img_desc"], str) and len(result[idx]["img_desc"]) >= 10:
#                 continue
            
#             # Try to use positional match first
#             if pos < len(queries):
#                 q = self._clean_query(queries[pos])
#                 if len(q) >= 10 and q not in self._used_queries:
#                     self._used_queries.add(q)
#                     result[idx]["img_desc"] = q
#                     used.add(pos)
#                     continue
            
#             # Try to find any unused query
#             for j in range(len(queries)):
#                 if j in used:
#                     continue
#                 q = self._clean_query(queries[j])
#                 if len(q) >= 10 and q not in self._used_queries:
#                     self._used_queries.add(q)
#                     result[idx]["img_desc"] = q
#                     used.add(j)
#                     break

#     def _gaps(self, result: List[dict]) -> List[int]:
#         gaps = []
#         for i, r in enumerate(result):
#             desc = r.get("img_desc")
#             if not desc or not isinstance(desc, str) or len(desc) < 10:
#                 gaps.append(i)
#         return gaps

#     # In _n_filled method
#     def _n_filled(self, result: List[dict]) -> int:
#         count = 0
#         for r in result:
#             desc = r.get("img_desc")
#             if desc and isinstance(desc, str) and len(desc) >= 10:
#                 count += 1
#         return count

#     def _clean_query(self, query: str) -> str:
#         """Clean a query string."""
#         q = query.strip().strip('"').strip("'")
#         q = q.replace('\\"', '"').replace("\\'", "'")
#         q = q.replace("\\n", " ").replace("\\t", " ")
#         q = re.sub(r'\s+', ' ', q).strip()
#         return q

#     def get_stats(self) -> dict:
#         """Return generation statistics."""
#         return {
#             "api_calls": self._api_calls,
#             "unique_queries": len(self._used_queries)
#         }


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# #  STANDALONE TEST
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# if __name__ == "__main__":
#     # Test with sample data
#     test_rows = [
#         {
#             "object_detected": "Coffee Cup",
#             "theme": "Cozy Morning",
#             "emotion": "Joy",
#             "dominant_colour": "Brown",
#             "text": "Start your day right with fresh coffee"
#         },
#         {
#             "object_detected": "Running Shoes",
#             "theme": "Fitness",
#             "emotion": "Excitement",
#             "dominant_colour": "Blue",
#             "text": "Push your limits every day"
#         },
#     ]

#     agent = SearchAgent()
#     result = agent.generate_queries(test_rows, "test")

#     print("\n\nResults:")
#     for r in result:
#         print(f"  {r['object_detected']}: {r.get('img_desc', 'N/A')}")
    
#     print(f"\nStats: {agent.get_stats()}")


# search_agent.py
"""
Generates GOOGLE IMAGE SEARCH QUERIES (img_desc) using NVIDIA API.
Based on text_generator.py architecture.
JSON response format - same as text_generator.py
"""

import os
import time
import re
from typing import Optional, List, Tuple
import json

from config import (
    USE_OPENAI_SDK,
    NVIDIA_API_KEY,
    BASE_URL,
    INVOKE_URL,
    STREAM,
    THINKING,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    MAX_RETRIES,
    RETRY_DELAY,
    LOG_RESPONSES,
    LOG_DIR,
    ACTIVE_PROFILE,
    TEXT_BATCH_SIZE
)
from utils import get_retry_session
if USE_OPENAI_SDK:
    from openai import OpenAI
else:
    import requests
BATCH_SIZE = TEXT_BATCH_SIZE  
try:
    from config import STREAM_PREVIEW
except ImportError:
    STREAM_PREVIEW = True

# Add after all imports, before SYSTEM_INSTRUCTION
VERBOSE = True

def _log(msg):
    """Conditional print — suppressed when VERBOSE is False."""
    if VERBOSE:
        print(msg)

SYSTEM_INSTRUCTION = """You are a search query generator for image-text matching. Given user input, generate a query that matches the SUBJECT content in image descriptions.

You always output valid JSON.
Never use colons or apostrophes inside text strings.
Use dashes instead of colons and full words instead of contractions.

Dataset text field contains descriptions like:
"Bite into a crispy crust loaded with toppings for pure joy"

Your task:
1. Extract ONLY subject-related terms (objects, items, physical descriptions)
2. IGNORE promotional/emotional words (joy, happiness, amazing, ultimate, perfect)
3. IGNORE action phrases (experience, treat yourself, indulge, satisfy)
4. IGNORE advertising language

Focus on:
✓ Physical objects (cheese, crust, toppings, slice)
✓ Descriptive attributes (crispy, hot, fresh, melty, cheesy)
✓ Subject context (pizza, food, meal)

Ignore:
✗ Emotions (joy, happiness, smile, love)
✗ Promotional words (ultimate, amazing, perfect, pure)
"""

SYSTEM_INSTRUCTION_SINGLE = """You are a search query generator for image-text matching.
You write one precise Google Image search query.
Reply with ONLY a JSON object. Nothing else.
Never use colons or apostrophes in the query text.
"""


class SearchAgent:
    """Generates img_desc column. Matches text_generator.py architecture with JSON output."""

    def __init__(self):
        if USE_OPENAI_SDK:
            self.client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
            _log(f"  API Mode  : OpenAI SDK")
        else:
            self.client = None
            self.headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": ("text/event-stream" if STREAM else "application/json"),
                "Content-Type": "application/json",
            }
            _log(f"  API Mode  : Raw requests")

        _log(f"  Model     : {MODEL_NAME}")
        _log(f"  Profile   : {ACTIVE_PROFILE}")
        _log(f"  Streaming : {'ON' if STREAM else 'OFF'}")
        _log(f"  Thinking  : {'ON' if THINKING else 'OFF'}")
        _log(f"  Task      : IMAGE SEARCH QUERIES (JSON)")

        self._used_queries: set = set()
        self._api_calls: int = 0
        self.session = get_retry_session()
        if LOG_RESPONSES:
            os.makedirs(LOG_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  API CALLS — IDENTICAL TO TEXT_GENERATOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_api(self, prompt: str, system: str = None, temperature: float = None) -> Optional[str]:
        self._api_calls += 1
        temp = temperature if temperature is not None else TEMPERATURE
        sys_msg = system or SYSTEM_INSTRUCTION

        if USE_OPENAI_SDK:
            return self._call_sdk(prompt, sys_msg, temp)
        else:
            return self._call_req(prompt, sys_msg, temp)

    # def _call_sdk(self, prompt: str, system: str, temp: float) -> Optional[str]:
    #     """OpenAI SDK call with streaming + thinking support."""
    #     try:
    #         kwargs = {
    #             "model": MODEL_NAME,
    #             "messages": [
    #                 {"role": "system", "content": system},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             "temperature": temp,
    #             "top_p": TOP_P,
    #             "max_tokens": MAX_TOKENS,
    #             "stream": STREAM,
    #         }

    #         if THINKING:
    #             kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

    #         completion = self.client.chat.completions.create(**kwargs)

    #         if STREAM:
    #             content_chunks = []
    #             reasoning_chunks = []
    #             prompt_tokens = 0
    #             completion_tokens = 0

    #             for chunk in completion:
    #                 if not getattr(chunk, "choices", None):
    #                     if hasattr(chunk, "usage") and chunk.usage:
    #                         prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
    #                         completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
    #                     continue

    #                 if not chunk.choices:
    #                     continue

    #                 delta = chunk.choices[0].delta

    #                 if THINKING:
    #                     reasoning = getattr(delta, "reasoning_content", None)
    #                     if reasoning:
    #                         reasoning_chunks.append(reasoning)
    #                         continue

    #                 if delta.content is not None:
    #                     content_chunks.append(delta.content)

    #                 if hasattr(chunk, "usage") and chunk.usage:
    #                     prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
    #                     completion_tokens = getattr(chunk.usage, "completion_tokens", 0)

    #             content = "".join(content_chunks)

    #             if prompt_tokens or completion_tokens:
    #                 total = prompt_tokens + completion_tokens
    #                 _log(f"        Tokens: p={prompt_tokens} c={completion_tokens} t={total}")

    #             if THINKING and reasoning_chunks:
    #                 reasoning_text = "".join(reasoning_chunks)
    #                 _log(f"        Thinking: {len(reasoning_text)} chars (discarded)")

    #                 if LOG_RESPONSES:
    #                     log_path = os.path.join(LOG_DIR, f"thinking_{self._api_calls}.txt")
    #                     with open(log_path, "w", encoding="utf-8") as f:
    #                         f.write(f"=== REASONING ===\n{reasoning_text}\n\n")
    #                         f.write(f"=== CONTENT ===\n{content}\n")

    #             return content

    #         else:
    #             content = completion.choices[0].message.content

    #             if hasattr(completion, "usage") and completion.usage:
    #                 u = completion.usage
    #                 _log(f"        Tokens: p={u.prompt_tokens} c={u.completion_tokens} t={u.total_tokens}")

    #             return content

    #     except Exception as e:
    #         _log(f"        ✗ API: {e}")
    #         return None

    # def _call_req(self, prompt: str, system: str, temp: float) -> Optional[str]:
    #     """Raw requests call with streaming support."""
    #     payload = {
    #         "model": MODEL_NAME,
    #         "messages": [
    #             {"role": "system", "content": system},
    #             {"role": "user", "content": prompt},
    #         ],
    #         "max_tokens": MAX_TOKENS,
    #         "temperature": temp,
    #         "top_p": TOP_P,
    #         "frequency_penalty": FREQUENCY_PENALTY,
    #         "presence_penalty": PRESENCE_PENALTY,
    #         "stream": STREAM,
    #     }

    #     if THINKING:
    #         payload["chat_template_kwargs"] = {"thinking": True}

    #     try:
    #         r = self.session.post(INVOKE_URL, headers=self.headers, json=payload, timeout=600)

    #         if r.status_code != 200:
    #             _log(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
    #             return None

    #         if STREAM:
    #             content_chunks = []
    #             for line in r.iter_lines():
    #                 if not line:
    #                     continue
    #                 d = line.decode("utf-8")
    #                 if d.startswith("data: "):
    #                     d = d[6:]
    #                 if d.strip() == "[DONE]":
    #                     break
    #                 try:
    #                     obj = json.loads(d)
    #                     choices = obj.get("choices", [])
    #                     if not choices:
    #                         continue
    #                     delta = choices[0].get("delta", {})

    #                     if THINKING and "reasoning_content" in delta:
    #                         continue

    #                     content = delta.get("content")
    #                     if content:
    #                         content_chunks.append(content)
    #                 except json.JSONDecodeError:
    #                     continue

    #             return "".join(content_chunks)
    #         else:
    #             data = r.json()
    #             u = data.get("usage", {})
    #             if u:
    #                 _log(f"        Tokens: p={u.get('prompt_tokens','?')} c={u.get('completion_tokens','?')} t={u.get('total_tokens','?')}")
    #             return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    #     except requests.exceptions.Timeout:
    #         _log(f"        ✗ Timeout")
    #         return None
    #     except Exception as e:
    #         _log(f"        ✗ {e}")
    #         return None
    def _call_sdk(self, prompt: str, system: str, temp: float) -> Optional[str]:
        """OpenAI SDK call with streaming + thinking support."""
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temp,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "stream": STREAM,
            }

            if THINKING:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

            completion = self.client.chat.completions.create(**kwargs)

            if STREAM:
                content_chunks = []
                reasoning_chunks = []
                prompt_tokens = 0
                completion_tokens = 0

                # ═══ STREAM PREVIEW: Show prefix ═══
                import sys
                sys.stdout.write("\n        📝 Generating: ")
                sys.stdout.flush()

                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        if hasattr(chunk, "usage") and chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                            completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
                        continue

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if THINKING:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_chunks.append(reasoning)
                            # ═══ STREAM PREVIEW: Show thinking dots ═══
                            sys.stdout.write(".")
                            sys.stdout.flush()
                            continue

                    if delta.content is not None:
                        content_chunks.append(delta.content)

                        # ═══ STREAM PREVIEW: Show live tokens ═══
                        sys.stdout.write(delta.content.replace("\n", " ").replace("\r", " "))
                        sys.stdout.flush()

                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", 0)

                # ═══ STREAM PREVIEW: Newline after stream ends ═══ 
                # print()
                sys.stdout.write("\n")

                content = "".join(content_chunks)

                if prompt_tokens or completion_tokens:
                    total = prompt_tokens + completion_tokens
                    _log(f"        Tokens: p={prompt_tokens} c={completion_tokens} t={total}")

                if THINKING and reasoning_chunks:
                    reasoning_text = "".join(reasoning_chunks)
                    _log(f"        Thinking: {len(reasoning_text)} chars (discarded)")

                    if LOG_RESPONSES:
                        log_path = os.path.join(LOG_DIR, f"thinking_{self._api_calls}.txt")
                        with open(log_path, "w", encoding="utf-8") as f:
                            f.write(f"=== REASONING ===\n{reasoning_text}\n\n")
                            f.write(f"=== CONTENT ===\n{content}\n")

                return content

            else:
                content = completion.choices[0].message.content

                if hasattr(completion, "usage") and completion.usage:
                    u = completion.usage
                    _log(f"        Tokens: p={u.prompt_tokens} c={u.completion_tokens} t={u.total_tokens}")

                return content

        except Exception as e:
            # print()  # ═══ STREAM PREVIEW: Clean line on error ═══
            sys.stdout.write("\n")
            _log(f"        ✗ API: {e}")
            return None


    def _call_req(self, prompt: str, system: str, temp: float) -> Optional[str]:
        """Raw requests call with streaming support."""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": temp,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "stream": STREAM,
        }

        if THINKING:
            payload["chat_template_kwargs"] = {"thinking": True}

        try:
            r = self.session.post(INVOKE_URL, headers=self.headers, json=payload, timeout=600)

            if r.status_code != 200:
                _log(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
                return None

            if STREAM:
                content_chunks = []

                # ═══ STREAM PREVIEW: Show prefix ═══
                import sys
                sys.stdout.write("\n        📝 Generating: ")
                sys.stdout.flush()

                for line in r.iter_lines():
                    if not line:
                        continue
                    d = line.decode("utf-8")
                    if d.startswith("data: "):
                        d = d[6:]
                    if d.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(d)
                        choices = obj.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        if THINKING and "reasoning_content" in delta:
                            # ═══ STREAM PREVIEW: Show thinking dots ═══
                            sys.stdout.write(".")
                            sys.stdout.flush()
                            continue

                        content = delta.get("content")
                        if content:
                            content_chunks.append(content)

                            # ═══ STREAM PREVIEW: Show live tokens ═══
                            sys.stdout.write(content.replace("\n", " ").replace("\r", " "))
                            sys.stdout.flush()

                    except json.JSONDecodeError:
                        continue

                # ═══ STREAM PREVIEW: Newline after stream ends ═══
                # print()
                sys.stdout.write("\n")

                return "".join(content_chunks)
            else:
                data = r.json()
                u = data.get("usage", {})
                if u:
                    _log(f"        Tokens: p={u.get('prompt_tokens','?')} c={u.get('completion_tokens','?')} t={u.get('total_tokens','?')}")
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except requests.exceptions.Timeout:
            # print()  # ═══ STREAM PREVIEW: Clean line on error ═══
            sys.stdout.write("\n")
            _log(f"        ✗ Timeout")
            return None
        except Exception as e:
            # print()  # ═══ STREAM PREVIEW: Clean line on error ═══4
            sys.stdout.write("\n")
            _log(f"        ✗ {e}")
            return None
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PROMPT BUILDERS - JSON FORMAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_batch_prompt(self, rows: List[dict]) -> str:
        """Build batch prompt for multiple queries - JSON output."""
        lines = []
        for i, r in enumerate(rows):
            context = f"Ad Copy: {r.get('text', '')}"
            lines.append(
                f"{i+1}. {r['object_detected']} | {r['theme']} | "
                f"{r['emotion']} | {context}"
            )
        rows_block = "\n".join(lines)

        return f"""Generate {len(rows)} precise Google Image search queries.
Each query should find high-quality images for advertising.

RULES:
- Visual descriptors only
- Include "filetype:png" where appropriate
- 5-8 words per query
- Be specific enough that Google Images returns the EXACT image in top results

ROWS:
{rows_block}

OUTPUT FORMAT - a JSON array with EXACTLY {len(rows)} objects:
[
{{"id":1,"img_desc":"your search query here"}},
{{"id":2,"img_desc":"another search query here"}}
]

CRITICAL RULES:
- Output ONLY the JSON array - no other text
- Exactly {len(rows)} objects in the array
- Every "img_desc" must be 5-8 words search query
- Do NOT wrap in markdown or code fences

Generate the JSON array of search queries now:"""

    def _build_single_prompt(self, row: dict, variation: int = 0) -> str:
        """Build single query prompt with variations - JSON output."""
        obj = row["object_detected"]
        theme = row["theme"]
        emotion = row["emotion"]
        context = f"Ad Copy: {row.get('text', '')}"

        angles = [
            "for a product catalog",
            "for social media advertising",
            "for a billboard campaign",
            "for an email marketing banner",
            "for a website hero image",
            "for print magazine ads",
        ]
        angle = angles[variation % len(angles)]

        return f"""Create ONE precise Google Image search query {angle}.

Subject: {obj}
Theme: {theme}
Emotion: {emotion}
{context}

Rules:
- No articles (a/the) or conversational words
- Visual descriptors only
- Translate {emotion} to lighting/mood
- Include "filetype:png" where appropriate
- 5-12 words total

Reply with ONLY this JSON and nothing else:
{{"img_desc":"your search query here","keywords":"word1 word2 word3"}}"""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  JSON PARSERS (5 strategies) - SAME AS TEXT_GENERATOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _parse_batch(self, raw: str) -> List[dict]:
        """Parse batch output - tries 5 JSON parsing strategies."""
        for name, fn in [
            ("direct", self._p1_direct),
            ("clean", self._p2_clean),
            ("fix", self._p3_fix),
            ("line", self._p4_line),
            ("regex", self._p5_regex),
        ]:
            items = fn(raw)
            if items:
                _log(f"        Parsed via {name}: {len(items)} items")
                return items
        return []

    def _p1_direct(self, raw: str) -> List[dict]:
        """Strategy 1: Direct JSON parse."""
        try:
            d = json.loads(raw.strip())
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "img_desc" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p2_clean(self, raw: str) -> List[dict]:
        """Strategy 2: Clean markdown then parse."""
        t = raw.strip()
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in t:
            for p in t.split("```"):
                if p.strip().startswith("["):
                    t = p.strip()
                    break
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1:
            return []
        try:
            d = json.loads(t[s:e+1])
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "img_desc" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p3_fix(self, raw: str) -> List[dict]:
        """Strategy 3: Fix common JSON errors."""
        t = raw.strip()
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1:
            return []
        t = t[s:e+1]
        t = re.sub(r',\s*}', '}', t)
        t = re.sub(r',\s*]', ']', t)
        t = re.sub(r'("img_desc"\s*:\s*")(.*?)("\s*,\s*"keywords")',
            lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3), t, flags=re.DOTALL)
        t = re.sub(r'("keywords"\s*:\s*")(.*?)("\s*})',
            lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3), t, flags=re.DOTALL)
        t = self._fix_colons(t)
        t = re.sub(r'[\x00-\x1f\x7f]', ' ', t)
        try:
            d = json.loads(t)
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "img_desc" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p4_line(self, raw: str) -> List[dict]:
        """Strategy 4: Parse individual JSON objects."""
        items = []
        for m in re.findall(r'\{[^{}]*?"img_desc"[^{}]*?\}', raw, re.DOTALL):
            try:
                o = json.loads(m)
                if "img_desc" in o and len(str(o["img_desc"])) > 8:
                    items.append(o)
                    continue
            except json.JSONDecodeError:
                pass
            f = re.sub(r',\s*}', '}', m)
            f = re.sub(r'("img_desc"\s*:\s*")(.*?)("\s*[,}])',
                lambda x: x.group(1) + x.group(2).replace('"', "'").replace(":", " -") + x.group(3), f, flags=re.DOTALL)
            try:
                o = json.loads(f)
                if "img_desc" in o and len(str(o["img_desc"])) > 8:
                    items.append(o)
            except json.JSONDecodeError:
                continue
        return items

    def _p5_regex(self, raw: str) -> List[dict]:
        """Strategy 5: Regex extraction."""
        items = []
        for chunk in re.split(r'(?=\{)', raw):
            tm = re.search(r'"img_desc"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            if not tm or len(tm.group(1).strip()) < 8:
                continue
            km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            im = re.search(r'"id"\s*:\s*(\d+)', chunk)
            items.append({
                "id": int(im.group(1)) if im else len(items) + 1,
                "img_desc": tm.group(1).strip(),
                "keywords": km.group(1).strip() if km else "",
            })
        return items

    def _parse_single_response(self, raw: str) -> Tuple[str, str]:
        """Parse single query response - returns (img_desc, keywords)."""
        cleaned = raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                s = p.strip()
                if s.startswith("{") or '"img_desc"' in s:
                    cleaned = s
                    break
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            obj_str = cleaned[start:end+1]
            obj_str = re.sub(r',\s*}', '}', obj_str)
            obj_str = self._fix_colons(obj_str)
            try:
                obj = json.loads(obj_str)
                t = str(obj.get("img_desc", "")).strip()
                k = str(obj.get("keywords", "")).strip()
                if len(t) >= 8:
                    return t, k
            except json.JSONDecodeError:
                pass
        tm = re.search(r'"img_desc"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        if tm and len(tm.group(1).strip()) >= 8:
            km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
            return tm.group(1).strip(), km.group(1).strip() if km else ""
        # Fallback to plain text
        plain = cleaned.strip('"').strip("'").strip()
        plain = re.sub(r'^(query|search|here|output)\s*[:=]\s*', '', plain, flags=re.IGNORECASE)
        plain = plain.strip('"').strip("'").strip()
        wc = len(plain.split())
        if 5 <= wc <= 15 and not any(c in plain for c in '{}[]'):
            _log(f"        (plain text)")
            return plain, ""
        return "", ""

    def _fix_colons(self, s: str) -> str:
        """Fix colons inside JSON string values."""
        result = []
        in_str = False
        in_val = False
        after_c = False
        esc = False
        for c in s:
            if esc:
                result.append(c)
                esc = False
                continue
            if c == '\\':
                result.append(c)
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                if in_str and after_c:
                    in_val = True
                    after_c = False
                elif not in_str:
                    in_val = False
                result.append(c)
                continue
            if not in_str and c == ':':
                after_c = True
                result.append(c)
                continue
            if not in_str:
                after_c = False
            if in_val and c == ':':
                result.append(' -')
                continue
            result.append(c)
        return ''.join(result)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN ENTRY + PHASES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate_queries(self, skeleton_rows: List[dict], batch_label: str = "") -> List[dict]:
        """
        Main entry point. Identical 5-phase structure to text_generator.
        Returns rows with img_desc and img_keywords filled.
        """
        total = len(skeleton_rows)
        result = [row.copy() for row in skeleton_rows]

        # Initialize columns
        for r in result:
            if "img_desc" not in r:
                r["img_desc"] = ""
            # if "img_keywords" not in r:
            #     r["img_keywords"] = ""
        # for batch_start in range(0, total, BATCH_SIZE):
        #     batch_end = min(batch_start + BATCH_SIZE, total)
        #     batch_indices = list(range(batch_start, batch_end))
        #     batch_rows = [skeleton_rows[i] for i in batch_indices]
        #     batch_num = batch_start // BATCH_SIZE + 1

        #     _log(f"\n    ━━━ Batch {batch_num} (rows {batch_start+1}-{batch_end}) ━━━")
        

        # Execute 5 phases
        for phase, method in [
            (1, lambda: self._phase_full(skeleton_rows, result, batch_label, total)),
            (2, lambda: self._phase_retry(skeleton_rows, result, batch_label, total)),
            (3, lambda: self._phase_micro(skeleton_rows, result, batch_label, total)),
            (4, lambda: self._phase_single(skeleton_rows, result, batch_label, total)),
            (5, lambda: self._phase_construct(skeleton_rows, result, batch_label, total)),
        ]:
            method()
            filled = self._n_filled(result)
            if filled == total:
                _log(f"      ✓ Phase {phase} complete: {filled}/{total}")
                return result

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PHASE IMPLEMENTATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _phase_full(self, skel, result, label, total):
        """Phase 1: Full batch API call."""
        _log(f"      Phase 1: Full batch ({total} rows)")
        items = self._call_batch(skel, f"{label}_full")
        if items:
            self._fill(result, items, list(range(total)))
        _log(f"      Phase 1: {self._n_filled(result)}/{total}")

    def _phase_retry(self, skel, result, label, total):
        """Phase 2: Retry gaps."""
        gaps = self._gaps(result)
        if not gaps:
            return
        _log(f"      Phase 2: Retry {len(gaps)} gaps")
        gap_rows = [skel[i] for i in gaps]
        items = self._call_batch(gap_rows, f"{label}_retry")
        if items:
            self._fill(result, items, gaps)
        _log(f"      Phase 2: {self._n_filled(result)}/{total}")

    def _phase_micro(self, skel, result, label, total):
        """Phase 3: Micro-batches (5 at a time)."""
        gaps = self._gaps(result)
        if not gaps:
            return
        _log(f"      Phase 3: Micro-batches for {len(gaps)} rows")
        for ci in range(0, len(gaps), 5):
            chunk = gaps[ci:ci+5]
            chunk_rows = [skel[i] for i in chunk]
            items = self._call_batch(chunk_rows, f"{label}_m{ci//5}")
            if items:
                self._fill(result, items, chunk)
            time.sleep(1)
        _log(f"      Phase 3: {self._n_filled(result)}/{total}")

    def _phase_single(self, skel, result, label, total):
        """Phase 4: One-by-one generation."""
        gaps = self._gaps(result)
        if not gaps:
            return
        _log(f"      Phase 4: One-by-one for {len(gaps)} rows")
        for idx in gaps:
            query, kw = self._single_generate(skel[idx], f"{label}_s{idx}")
            if query and len(query) >= 10 and query not in self._used_queries:
                self._used_queries.add(query)
                result[idx]["img_desc"] = query
                # result[idx]["img_keywords"] = kw
                _log(f"        ✓ Row {idx}")
            else:
                _log(f"        ⚠ Row {idx}: needs Phase 5")
            time.sleep(1)
        _log(f"      Phase 4: {self._n_filled(result)}/{total}")

    def _phase_construct(self, skel, result, label, total):
        """Phase 5: Rule-based construction (fallback)."""
        gaps = self._gaps(result)
        if not gaps:
            return
        _log(f"      Phase 5: Construct {len(gaps)} queries")
        for idx in gaps:
            query = self._construct_query(skel[idx])
            self._used_queries.add(query)
            result[idx]["img_desc"] = query
            # result[idx]["img_keywords"] = f"{skel[idx].get('object_detected', '')} {skel[idx].get('theme', '')} {skel[idx].get('emotion', '')}"
            _log(f"        ✓ Row {idx}: constructed")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELPER METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_batch(self, rows: List[dict], label: str) -> List[dict]:
        """Call API for batch, with retries."""
        prompt = self._build_batch_prompt(rows)
        for att in range(1, MAX_RETRIES + 1):
            _log(f"        Attempt {att}/{MAX_RETRIES} ({len(rows)} rows) [{label}]")
            raw = self._call_api(prompt)
            if raw is None:
                time.sleep(RETRY_DELAY)
                continue

            if LOG_RESPONSES:
                with open(os.path.join(LOG_DIR, f"{label}_a{att}.txt"), "w", encoding="utf-8") as f:
                    f.write(raw)

            items = self._parse_batch(raw)
            if items:
                return items

            _log(f"        ✗ No items parsed")
            time.sleep(RETRY_DELAY)

        return []

    def _single_generate(self, row: dict, label: str) -> Tuple[str, str]:
        """Generate single query with variations."""
        for variation in range(6):
            temp = min(TEMPERATURE + (variation * 0.15), 1.0)
            prompt = self._build_single_prompt(row, variation=variation)
            raw = self._call_api(prompt, system=SYSTEM_INSTRUCTION_SINGLE, temperature=temp)

            if raw is None:
                time.sleep(2)
                continue

            if LOG_RESPONSES:
                with open(os.path.join(LOG_DIR, f"{label}_v{variation}.txt"), "w", encoding="utf-8") as f:
                    f.write(raw)

            query, kw = self._parse_single_response(raw)
            query = self._clean_query(query)

            if query and len(query) >= 10 and query not in self._used_queries:
                return query, kw

            time.sleep(1)

        return "", ""

    def _construct_query(self, row: dict) -> str:
        """Rule-based query construction (Phase 5 fallback)."""
        obj = row.get("object_detected", "object").lower()
        color = row.get("dominant_colour", "")
        theme = row.get("theme", "")
        emotion = row.get("emotion", "")

        query_parts = [f'"{obj}"']

        if color and color.lower() not in ['none', 'mixed', 'various']:
            query_parts.append(color.lower())

        if theme:
            query_parts.append(theme.lower())

        mood_map = {
            "Joy": "bright natural lighting",
            "Anger": "dramatic contrast",
            "Trust": "clean studio lighting",
            "Excitement": "vibrant dynamic",
            "Fear": "moody dark atmosphere"
        }
        if emotion in mood_map:
            query_parts.append(mood_map[emotion])

        query_parts.append("professional photography 4k high resolution sharp focus")
        query_parts.append("-cartoon -clipart -vector -illustration -logo -text -watermark")
        query_parts.append("filetype:png")

        return " ".join(query_parts)

    def _fill(self, result: List[dict], items: List[dict], indices: List[int]):
        """Fill result rows with queries, avoiding duplicates."""
        used = set()

        for pos, idx in enumerate(indices):
            if result[idx].get("img_desc") and isinstance(result[idx]["img_desc"], str) and len(result[idx]["img_desc"]) >= 10:
                continue

            if pos < len(items):
                q = self._clean_query(str(items[pos].get("img_desc", "")))
                k = str(items[pos].get("keywords", "")).strip()
                if len(q) >= 10 and q not in self._used_queries:
                    self._used_queries.add(q)
                    result[idx]["img_desc"] = q
                    # result[idx]["img_keywords"] = k
                    used.add(pos)
                    continue

            for j in range(len(items)):
                if j in used:
                    continue
                q = self._clean_query(str(items[j].get("img_desc", "")))
                if len(q) >= 10 and q not in self._used_queries:
                    self._used_queries.add(q)
                    result[idx]["img_desc"] = q
                    # result[idx]["img_keywords"] = str(items[j].get("keywords", "")).strip()
                    used.add(j)
                    break

    def _gaps(self, result: List[dict]) -> List[int]:
        gaps = []
        for i, r in enumerate(result):
            desc = r.get("img_desc")
            if not desc or not isinstance(desc, str) or len(desc) < 10:
                gaps.append(i)
        return gaps

    def _n_filled(self, result: List[dict]) -> int:
        count = 0
        for r in result:
            desc = r.get("img_desc")
            if desc and isinstance(desc, str) and len(desc) >= 10:
                count += 1
        return count

    def _clean_query(self, query: str) -> str:
        """Clean a query string."""
        q = query.strip().strip('"').strip("'")
        q = q.replace('\\"', '"').replace("\\'", "'")
        q = q.replace("\\n", " ").replace("\\t", " ")
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def get_stats(self) -> dict:
        """Return generation statistics."""
        return {
            "api_calls": self._api_calls,
            "unique_queries": len(self._used_queries)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STANDALONE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    test_rows = [
        {
            "object_detected": "Coffee Cup",
            "theme": "Cozy Morning",
            "emotion": "Joy",
            "dominant_colour": "Brown",
            "text": "Start your day right with fresh coffee"
        },
        {
            "object_detected": "Running Shoes",
            "theme": "Fitness",
            "emotion": "Excitement",
            "dominant_colour": "Blue",
            "text": "Push your limits every day"
        },
    ]

    agent = SearchAgent()
    result = agent.generate_queries(test_rows, "test")

    _log("\n\nResults:")
    for r in result:
        _log(f"  {r['object_detected']}:")
        _log(f"    img_desc: {r.get('img_desc', 'N/A')}")
        # _log(f"    img_keywords: {r.get('img_keywords', 'N/A')}")

    _log(f"\nStats: {agent.get_stats()}")