# # text_generator.py
# """
# Generates AD COPY text + keywords using NVIDIA API.
# NO FALLBACK TEMPLATES. All text from LLM.
# ALL text is advertising copy — NOT reviews.
# """

# import os
# import json
# import time
# import re
# from typing import Optional, List, Tuple

# from config import (
#     USE_OPENAI_SDK,
#     NVIDIA_API_KEY,
#     BASE_URL,
#     INVOKE_URL,
#     STREAM,
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


# SYSTEM_PROMPT = (
#     "You are a professional advertising copywriter who writes short punchy ad copy. "
#     "You write like ads seen on Instagram Facebook and Google. "
#     "You always output valid JSON. "
#     "Never use colons or apostrophes inside text strings. "
#     "Use dashes instead of colons and full words instead of contractions."
# )

# SYSTEM_PROMPT_SINGLE = (
#     "You are an ad copywriter. "
#     "You write one short punchy advertising line. "
#     "Reply with ONLY a JSON object. Nothing else. "
#     "Never use colons or apostrophes in the ad text."
# )


# class TextGenerator:
#     """Generates AD COPY text + keywords. NO FALLBACK TEMPLATES."""

#     def __init__(self):
#         if USE_OPENAI_SDK:
#             self.client = OpenAI(
#                 base_url=BASE_URL,
#                 api_key=NVIDIA_API_KEY,
#             )
#             print(f"  API Mode : OpenAI SDK")
#         else:
#             self.client = None
#             self.headers = {
#                 "Authorization": f"Bearer {NVIDIA_API_KEY}",
#                 "Accept": (
#                     "text/event-stream" if STREAM
#                     else "application/json"
#                 ),
#                 "Content-Type": "application/json",
#             }
#             print(f"  API Mode : Raw requests")

#         print(f"  Model    : {MODEL_NAME}")
#         print(f"  Profile  : {ACTIVE_PROFILE}")
#         print(f"  Text Type: AD COPY ONLY")
#         print(f"  Fallback : DISABLED")

#         self._used_texts: set = set()
#         self._api_calls: int = 0

#         if LOG_RESPONSES:
#             os.makedirs(LOG_DIR, exist_ok=True)

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  API CALLS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _call_api(
#         self, prompt: str, system: str = None, temperature: float = None,
#     ) -> Optional[str]:
#         self._api_calls += 1
#         temp = temperature if temperature is not None else TEMPERATURE
#         sys_msg = system or SYSTEM_PROMPT

#         if USE_OPENAI_SDK:
#             return self._call_sdk(prompt, sys_msg, temp)
#         else:
#             return self._call_req(prompt, sys_msg, temp)

#     def _call_sdk(self, prompt, system, temp) -> Optional[str]:
#         try:
#             c = self.client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": system},
#                     {"role": "user",   "content": prompt},
#                 ],
#                 temperature=temp,
#                 top_p=TOP_P,
#                 max_tokens=MAX_TOKENS,
#                 stream=False,
#             )
#             content = c.choices[0].message.content
#             if hasattr(c, "usage") and c.usage:
#                 u = c.usage
#                 print(
#                     f"        Tokens: p={u.prompt_tokens} "
#                     f"c={u.completion_tokens} t={u.total_tokens}"
#                 )
#             return content
#         except Exception as e:
#             print(f"        ✗ API: {e}")
#             return None

#     def _call_req(self, prompt, system, temp) -> Optional[str]:
#         payload = {
#             "model": MODEL_NAME,
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user",   "content": prompt},
#             ],
#             "max_tokens":        MAX_TOKENS,
#             "temperature":       temp,
#             "top_p":             TOP_P,
#             "frequency_penalty": FREQUENCY_PENALTY,
#             "presence_penalty":  PRESENCE_PENALTY,
#             "stream":            STREAM,
#         }
#         try:
#             r = requests.post(
#                 INVOKE_URL, headers=self.headers,
#                 json=payload, timeout=300,
#             )
#             if r.status_code != 200:
#                 print(f"        ✗ HTTP {r.status_code}")
#                 return None
#             if STREAM:
#                 chunks = []
#                 for line in r.iter_lines():
#                     if not line:
#                         continue
#                     d = line.decode("utf-8")
#                     if d.startswith("data: "):
#                         d = d[6:]
#                     if d.strip() == "[DONE]":
#                         break
#                     try:
#                         o = json.loads(d)
#                         chunks.append(
#                             o.get("choices", [{}])[0]
#                              .get("delta", {})
#                              .get("content", "")
#                         )
#                     except json.JSONDecodeError:
#                         continue
#                 return "".join(chunks)
#             else:
#                 data = r.json()
#                 u = data.get("usage", {})
#                 if u:
#                     print(
#                         f"        Tokens: p={u.get('prompt_tokens','?')} "
#                         f"c={u.get('completion_tokens','?')} "
#                         f"t={u.get('total_tokens','?')}"
#                     )
#                 return (
#                     data.get("choices", [{}])[0]
#                         .get("message", {})
#                         .get("content", "")
#                 )
#         except Exception as e:
#             print(f"        ✗ {e}")
#             return None

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  BATCH PROMPT — AD COPY ONLY
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _build_batch_prompt(self, rows: List[dict]) -> str:
#         lines = []
#         for i, r in enumerate(rows):
#             lines.append(
#                 f"{i+1}. {r['theme']} | {r['object_detected']} | "
#                 f"{r['sentiment']} | {r['emotion']}"
#             )
#         rows_block = "\n".join(lines)

#         return f"""You are writing short advertising copy for social media ads and banners.
# Write one unique ad headline or tagline (10-18 words) for each row below.
# Each ad line must promote the Object and match the Sentiment and Emotion.
# Think Instagram ads - Facebook sponsored posts - Google display ads - billboard slogans.
# No brand names. No colons. No apostrophes. Use dashes instead.

# AD COPY EXAMPLES by Sentiment and Emotion:

# POSITIVE ADS:
# - Positive Joy: "Grab a hot fresh pizza tonight and taste pure happiness in every single bite"
# - Positive Trust: "Built to last and proven reliable - this laptop never lets you down ever"
# - Positive Excitement: "Get ready for the ride of your life with this incredible new sports car"

# NEGATIVE ADS (warning style / problem-solution / urgency):
# - Negative Anger: "Tired of flimsy bags that fall apart - demand better quality for your money"
# - Negative Fear: "Do not risk your safety with cheap helmets - protect what matters most today"

# NEUTRAL ADS (informational / comparison / factual):
# - Neutral Trust: "Standard performance you can count on - this watch does exactly what it should"
# - Neutral Joy: "A decent cup of coffee for your everyday morning routine - nothing fancy needed"
# - Neutral Anger: "Some minor issues but still gets the job done - decide for yourself today"
# - Neutral Excitement: "New features worth exploring - see what this tablet can do for you now"
# - Neutral Fear: "Know the facts before you buy - check all safety ratings on this scooter first"

# AD COPY STYLES TO MIX:
# - Headlines: "Upgrade Your Morning With Premium Fresh Juice Delivered To Your Door"
# - Taglines: "Life is better with the perfect pair of shoes on your feet"
# - Calls to action: "Do not miss out on this incredible new camera - order yours today"
# - Problem-solution: "Struggling with slow laptops - switch to lightning fast performance now"
# - Aspirational: "Drive the car you have always dreamed about - luxury within your reach"
# - Urgency: "Limited time offer on premium headphones - grab yours before they sell out"
# - Lifestyle: "Weekend getaways start with the perfect resort - book your escape today"

# ROWS TO WRITE ADS FOR:
# {rows_block}

# OUTPUT FORMAT - a JSON array with EXACTLY {len(rows)} objects:
# [
# {{"id":1,"text":"your unique ad copy here about the object","keywords":"word1 word2 word3"}},
# {{"id":2,"text":"another unique ad line matching sentiment and emotion","keywords":"word1 word2 word3"}}
# ]

# CRITICAL RULES:
# - Output ONLY the JSON array - no other text
# - Exactly {len(rows)} objects in the array
# - Every "text" must be 10-18 words of AD COPY not review text
# - Every "text" must promote or reference the object
# - Write like a professional ad copywriter not a reviewer
# - Every text must be completely unique
# - "keywords" should be 3-4 relevant words separated by spaces
# - Do NOT wrap in markdown or code fences

# Generate the JSON array of ad copy now:"""

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  SINGLE ROW PROMPT — AD COPY
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _build_single_prompt(self, row: dict, variation: int = 0) -> str:
#         obj = row["object_detected"]
#         sent = row["sentiment"]
#         emo = row["emotion"]
#         theme = row["theme"]

#         # Different ad angles for variation
#         angles = [
#             "as an Instagram sponsored post caption",
#             "as a Facebook ad headline",
#             "as a Google display ad tagline",
#             "as a billboard slogan for the highway",
#             "as a YouTube pre-roll ad opening line",
#             "as an email marketing subject line",
#             "as a print magazine advertisement headline",
#             "as a Twitter promoted post",
#             "as a TikTok ad caption",
#             "as a podcast sponsor read opening",
#             "as a shopping app push notification",
#             "as a banner ad on a news website",
#         ]
#         angle = angles[variation % len(angles)]

#         # Tone descriptions for ad copy
#         tone_map = {
#             ("Positive", "Joy"):
#                 "joyful and uplifting - make people smile and want the product",
#             ("Positive", "Trust"):
#                 "confident and reassuring - emphasize reliability and proven quality",
#             ("Positive", "Excitement"):
#                 "thrilling and energetic - create urgency and desire to try it",
#             ("Negative", "Anger"):
#                 "problem-focused - highlight frustration with alternatives and offer a better way",
#             ("Negative", "Fear"):
#                 "warning-style - highlight dangers of not acting or using inferior products",
#             ("Neutral", "Joy"):
#                 "pleasant and informational - mild positivity with factual appeal",
#             ("Neutral", "Anger"):
#                 "acknowledging common complaints - be honest about limitations",
#             ("Neutral", "Trust"):
#                 "factual and straightforward - state what the product does reliably",
#             ("Neutral", "Excitement"):
#                 "curious and intriguing - spark interest without overselling",
#             ("Neutral", "Fear"):
#                 "cautious and advisory - encourage informed decision making",
#         }
#         tone = tone_map.get(
#             (sent, emo),
#             f"{sent.lower()} tone conveying {emo.lower()}"
#         )

#         return f"""Write exactly one advertising headline or tagline (10-18 words) for a {obj} in the {theme} category.
# Write it {angle}.
# The tone should be {tone}.
# This is AD COPY not a review - write like a professional copywriter creating an ad.
# No brand names. No colons. No apostrophes.

# Reply with ONLY this JSON and nothing else:
# {{"text":"your ad copy sentence here","keywords":"word1 word2 word3"}}"""

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  JSON PARSERS (5 strategies)
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _parse_batch(self, raw: str) -> List[dict]:
#         for name, fn in [
#             ("direct",  self._p1_direct),
#             ("clean",   self._p2_clean),
#             ("fix",     self._p3_fix),
#             ("line",    self._p4_line),
#             ("regex",   self._p5_regex),
#         ]:
#             items = fn(raw)
#             if items:
#                 print(f"        Parsed via {name}: {len(items)} items")
#                 return items
#         return []

#     def _p1_direct(self, raw):
#         try:
#             d = json.loads(raw.strip())
#             if isinstance(d, list):
#                 return [x for x in d if isinstance(x, dict) and "text" in x]
#         except json.JSONDecodeError:
#             pass
#         return []

#     def _p2_clean(self, raw):
#         t = raw.strip()
#         if "```json" in t:
#             t = t.split("```json", 1)[1].split("```", 1)[0]
#         elif "```" in t:
#             for p in t.split("```"):
#                 if p.strip().startswith("["):
#                     t = p.strip()
#                     break
#         s, e = t.find("["), t.rfind("]")
#         if s == -1 or e == -1:
#             return []
#         try:
#             d = json.loads(t[s:e+1])
#             if isinstance(d, list):
#                 return [x for x in d if isinstance(x, dict) and "text" in x]
#         except json.JSONDecodeError:
#             pass
#         return []

#     def _p3_fix(self, raw):
#         t = raw.strip()
#         s, e = t.find("["), t.rfind("]")
#         if s == -1 or e == -1:
#             return []
#         t = t[s:e+1]
#         t = re.sub(r',\s*}', '}', t)
#         t = re.sub(r',\s*]', ']', t)
#         t = re.sub(
#             r'("text"\s*:\s*")(.*?)("\s*,\s*"keywords")',
#             lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
#             t, flags=re.DOTALL,
#         )
#         t = re.sub(
#             r'("keywords"\s*:\s*")(.*?)("\s*})',
#             lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
#             t, flags=re.DOTALL,
#         )
#         t = self._fix_colons(t)
#         t = re.sub(r'[\x00-\x1f\x7f]', ' ', t)
#         try:
#             d = json.loads(t)
#             if isinstance(d, list):
#                 return [x for x in d if isinstance(x, dict) and "text" in x]
#         except json.JSONDecodeError:
#             pass
#         return []

#     def _p4_line(self, raw):
#         items = []
#         for m in re.findall(r'\{[^{}]*?"text"[^{}]*?\}', raw, re.DOTALL):
#             try:
#                 o = json.loads(m)
#                 if "text" in o and len(str(o["text"])) > 8:
#                     items.append(o)
#                     continue
#             except json.JSONDecodeError:
#                 pass
#             f = re.sub(r',\s*}', '}', m)
#             f = re.sub(
#                 r'("text"\s*:\s*")(.*?)("\s*[,}])',
#                 lambda x: x.group(1) + x.group(2).replace('"',"'").replace(":"," -") + x.group(3),
#                 f, flags=re.DOTALL,
#             )
#             try:
#                 o = json.loads(f)
#                 if "text" in o and len(str(o["text"])) > 8:
#                     items.append(o)
#             except json.JSONDecodeError:
#                 continue
#         return items

#     def _p5_regex(self, raw):
#         items = []
#         for chunk in re.split(r'(?=\{)', raw):
#             tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
#             if not tm or len(tm.group(1).strip()) < 8:
#                 continue
#             km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
#             im = re.search(r'"id"\s*:\s*(\d+)', chunk)
#             items.append({
#                 "id": int(im.group(1)) if im else len(items)+1,
#                 "text": tm.group(1).strip(),
#                 "keywords": km.group(1).strip() if km else "",
#             })
#         return items

#     def _parse_single_response(self, raw: str) -> Tuple[str, str]:
#         cleaned = raw.strip()

#         if "```" in cleaned:
#             for p in cleaned.split("```"):
#                 s = p.strip()
#                 if s.startswith("{") or '"text"' in s:
#                     cleaned = s
#                     break

#         # Strategy A: JSON object
#         start = cleaned.find("{")
#         end = cleaned.rfind("}")
#         if start != -1 and end != -1:
#             obj_str = cleaned[start:end+1]
#             obj_str = re.sub(r',\s*}', '}', obj_str)
#             obj_str = self._fix_colons(obj_str)
#             try:
#                 obj = json.loads(obj_str)
#                 t = str(obj.get("text", "")).strip()
#                 k = str(obj.get("keywords", "")).strip()
#                 if len(t) >= 8:
#                     return t, k
#             except json.JSONDecodeError:
#                 pass

#         # Strategy B: Regex
#         tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
#         if tm and len(tm.group(1).strip()) >= 8:
#             km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
#             return tm.group(1).strip(), km.group(1).strip() if km else ""

#         # Strategy C: Plain text
#         plain = cleaned.strip('"').strip("'").strip()
#         plain = re.sub(
#             r'^(text|sentence|here|output|ad|copy)\s*[:=]\s*',
#             '', plain, flags=re.IGNORECASE
#         )
#         plain = plain.strip('"').strip("'").strip()
#         wc = len(plain.split())
#         if 8 <= wc <= 30 and not any(c in plain for c in '{}[]'):
#             print(f"        (plain text)")
#             return plain, ""

#         return "", ""

#     def _fix_colons(self, s):
#         result = []
#         in_str = False
#         in_val = False
#         after_c = False
#         esc = False
#         for c in s:
#             if esc:
#                 result.append(c); esc = False; continue
#             if c == '\\':
#                 result.append(c); esc = True; continue
#             if c == '"':
#                 in_str = not in_str
#                 if in_str and after_c:
#                     in_val = True; after_c = False
#                 elif not in_str:
#                     in_val = False
#                 result.append(c); continue
#             if not in_str and c == ':':
#                 after_c = True; result.append(c); continue
#             if not in_str:
#                 after_c = False
#             if in_val and c == ':':
#                 result.append(' -'); continue
#             result.append(c)
#         return ''.join(result)

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  MAIN ENTRY POINT
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def generate_texts(
#         self,
#         skeleton_rows: List[dict],
#         batch_label: str = "",
#     ) -> List[dict]:
#         """5-phase ad copy generation. No fallback templates."""
#         total = len(skeleton_rows)
#         result = [row.copy() for row in skeleton_rows]
#         for r in result:
#             r["text"] = ""
#             r["keywords"] = ""

#         # Phase 1: Full batch
#         print(f"      Phase 1: Full batch ({total} rows)")
#         items = self._call_batch(skeleton_rows, f"{batch_label}_full")
#         if items:
#             self._fill(result, items, list(range(total)))

#         filled = self._n_filled(result)
#         if filled == total:
#             print(f"      ✓ Phase 1 complete: {filled}/{total}")
#             return result
#         print(f"      Phase 1: {filled}/{total}")

#         # Phase 2: Retry gaps
#         gaps = self._gaps(result)
#         if gaps:
#             print(f"      Phase 2: Retry {len(gaps)} gaps")
#             gap_rows = [skeleton_rows[i] for i in gaps]
#             items = self._call_batch(gap_rows, f"{batch_label}_retry")
#             if items:
#                 self._fill(result, items, gaps)

#         filled = self._n_filled(result)
#         if filled == total:
#             print(f"      ✓ Phase 2 complete: {filled}/{total}")
#             return result
#         print(f"      Phase 2: {filled}/{total}")

#         # Phase 3: Micro-batches of 5
#         gaps = self._gaps(result)
#         if gaps:
#             print(f"      Phase 3: Micro-batches for {len(gaps)} rows")
#             for ci in range(0, len(gaps), 5):
#                 chunk = gaps[ci:ci+5]
#                 chunk_rows = [skeleton_rows[i] for i in chunk]
#                 items = self._call_batch(
#                     chunk_rows, f"{batch_label}_m{ci//5}"
#                 )
#                 if items:
#                     self._fill(result, items, chunk)
#                 time.sleep(1)

#         filled = self._n_filled(result)
#         if filled == total:
#             print(f"      ✓ Phase 3 complete: {filled}/{total}")
#             return result
#         print(f"      Phase 3: {filled}/{total}")

#         # Phase 4: One-by-one
#         gaps = self._gaps(result)
#         if gaps:
#             print(f"      Phase 4: One-by-one for {len(gaps)} rows")
#             for idx in gaps:
#                 row = skeleton_rows[idx]
#                 text, kw = self._single_generate(
#                     row, f"{batch_label}_s{idx}"
#                 )
#                 if text and len(text) >= 8 and text not in self._used_texts:
#                     self._used_texts.add(text)
#                     result[idx]["text"] = text
#                     result[idx]["keywords"] = kw
#                     print(f"        ✓ Row {idx}")
#                 else:
#                     print(f"        ⚠ Row {idx}: needs Phase 5")
#                 time.sleep(1)

#         filled = self._n_filled(result)
#         if filled == total:
#             print(f"      ✓ Phase 4 complete: {filled}/{total}")
#             return result
#         print(f"      Phase 4: {filled}/{total}")

#         # Phase 5: Construct ad copy from metadata
#         gaps = self._gaps(result)
#         if gaps:
#             print(f"      Phase 5: Construct {len(gaps)} ad lines")
#             for idx in gaps:
#                 text = self._construct_ad(skeleton_rows[idx])
#                 self._used_texts.add(text)
#                 result[idx]["text"] = text
#                 result[idx]["keywords"] = (
#                     f"{skeleton_rows[idx]['object_detected']} "
#                     f"{skeleton_rows[idx]['theme']} "
#                     f"{skeleton_rows[idx]['emotion']}"
#                 )
#                 print(f"        ✓ Row {idx}: constructed")

#         print(f"      ✓ FINAL: {self._n_filled(result)}/{total}")
#         return result

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  BATCH CALL
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _call_batch(self, rows, label) -> List[dict]:
#         prompt = self._build_batch_prompt(rows)
#         for att in range(1, MAX_RETRIES + 1):
#             print(f"        Attempt {att}/{MAX_RETRIES} ({len(rows)} rows) [{label}]")
#             raw = self._call_api(prompt)
#             if raw is None:
#                 time.sleep(RETRY_DELAY)
#                 continue
#             if LOG_RESPONSES:
#                 p = os.path.join(LOG_DIR, f"{label}_a{att}.txt")
#                 with open(p, "w", encoding="utf-8") as f:
#                     f.write(raw)
#             items = self._parse_batch(raw)
#             if items:
#                 return items
#             print(f"        ✗ No items parsed")
#             time.sleep(RETRY_DELAY)
#         return []

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  SINGLE ROW WITH VARIATION
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _single_generate(self, row, label) -> Tuple[str, str]:
#         for variation in range(6):
#             temp = min(TEMPERATURE + (variation * 0.15), 1.0)
#             prompt = self._build_single_prompt(row, variation=variation)
#             raw = self._call_api(
#                 prompt, system=SYSTEM_PROMPT_SINGLE, temperature=temp,
#             )
#             if raw is None:
#                 time.sleep(2)
#                 continue
#             if LOG_RESPONSES:
#                 p = os.path.join(LOG_DIR, f"{label}_v{variation}.txt")
#                 with open(p, "w", encoding="utf-8") as f:
#                     f.write(raw)
#             text, kw = self._parse_single_response(raw)
#             text = self._clean(text)
#             if text and len(text) >= 8 and text not in self._used_texts:
#                 return text, kw
#             time.sleep(1)
#         return "", ""

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  CONSTRUCT AD COPY (Phase 5 — unique per row)
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _construct_ad(self, row: dict) -> str:
#         """Build unique ad copy from metadata. Every row = different ad."""
#         obj = row.get("object_detected", "item").lower()
#         sent = row.get("sentiment", "Neutral")
#         emo = row.get("emotion", "Trust")
#         img = row.get("image_path", "0")

#         n = int(re.search(r'(\d+)', img).group(1)) if re.search(r'(\d+)', img) else 0

#         ad_templates = {
#             ("Positive", "Joy"): [
#                 f"Bring home this amazing {obj} and discover pure happiness in every moment",
#                 f"Treat yourself to this wonderful {obj} and let the joy begin today",
#                 f"Experience the delight of owning this perfect {obj} starting right now",
#                 f"Make every day brighter with this incredible {obj} in your life",
#                 f"Unwrap happiness with this fantastic {obj} that will light up your world",
#                 f"Add a spark of joy to your routine with this beautiful {obj} today",
#                 f"Find your happy place with this outstanding {obj} made just for you",
#                 f"Start smiling more with this delightful {obj} that brings real joy",
#                 f"Let this gorgeous {obj} fill your days with pure unmatched happiness",
#                 f"Choose happiness and choose this perfect {obj} for your lifestyle today",
#             ],
#             ("Positive", "Trust"): [
#                 f"Count on this proven {obj} for reliable performance day after day",
#                 f"Trust the quality that thousands already depend on with this {obj}",
#                 f"Built to last and made to impress - this {obj} earns your confidence",
#                 f"Reliable performance guaranteed with every single use of this {obj}",
#                 f"When dependability matters most choose this battle-tested {obj} today",
#                 f"Proven quality you can always count on with this outstanding {obj}",
#                 f"Stand behind a {obj} that stands behind you with rock solid quality",
#                 f"Dependable from day one this {obj} delivers what it promises always",
#                 f"Quality you can trust and performance you can rely on with this {obj}",
#                 f"Put your confidence in this {obj} that never lets its owners down",
#             ],
#             ("Positive", "Excitement"): [
#                 f"Get ready for something incredible - this {obj} will blow your mind",
#                 f"The wait is over - experience the thrill of this brand new {obj}",
#                 f"Unleash your excitement with this game-changing {obj} available now",
#                 f"Feel the rush of owning this extraordinary {obj} starting today",
#                 f"Do not miss this electrifying new {obj} that everyone is talking about",
#                 f"Prepare to be amazed by this revolutionary {obj} hitting shelves now",
#                 f"Your next adventure starts with this thrilling {obj} in your hands",
#                 f"Experience the excitement everyone is buzzing about with this new {obj}",
#                 f"Ignite your passion with this sensational {obj} that changes everything",
#                 f"The most exciting {obj} of the year is here and waiting for you",
#             ],
#             ("Negative", "Anger"): [
#                 f"Stop settling for less - upgrade from your frustrating old {obj} today",
#                 f"Tired of disappointing quality - demand the {obj} you actually deserve",
#                 f"Fed up with unreliable products - it is time for a better {obj} now",
#                 f"Do not waste another dollar on a {obj} that lets you down again",
#                 f"Had enough of subpar options - discover what a real {obj} should be",
#                 f"Frustrated with your current {obj} - make the switch you need today",
#                 f"Refuse to accept mediocre quality from your {obj} any longer now",
#                 f"Your old {obj} is holding you back - break free with something better",
#                 f"Say goodbye to frustration and hello to a {obj} that actually works",
#                 f"No more excuses for poor quality - get the {obj} that performs right",
#             ],
#             ("Negative", "Fear"): [
#                 f"Do not risk your safety with an unreliable {obj} - choose wisely today",
#                 f"Protect yourself from danger with a properly certified {obj} instead",
#                 f"Your safety is not worth gambling on - invest in a quality {obj} now",
#                 f"Before you buy any {obj} check the safety ratings that matter most",
#                 f"An unsafe {obj} is never worth the savings - protect what matters",
#                 f"Do not let a faulty {obj} put your wellbeing at risk ever again",
#                 f"Safety first - make sure your next {obj} meets every safety standard",
#                 f"Think twice before buying a cheap {obj} that could put you in danger",
#                 f"Your family deserves a {obj} that is certified safe and fully tested",
#                 f"Stop worrying about safety - get a {obj} you can actually trust today",
#             ],
#             ("Neutral", "Joy"): [
#                 f"A solid everyday {obj} that gets the job done with quiet satisfaction",
#                 f"Simple pleasures start with a decent {obj} for your daily routine",
#                 f"This {obj} brings a touch of comfort to your ordinary everyday life",
#                 f"Nothing fancy just a good {obj} that adds mild enjoyment to your day",
#             ],
#             ("Neutral", "Anger"): [
#                 f"Not perfect but still functional - this {obj} handles the basics well",
#                 f"Know the trade-offs before buying - this {obj} has pros and cons",
#                 f"Minor issues aside this {obj} still delivers acceptable performance overall",
#                 f"Be aware of the limitations but this {obj} works for most needs",
#             ],
#             ("Neutral", "Trust"): [
#                 f"Standard reliable performance from a {obj} that does what it should",
#                 f"No surprises just consistent delivery from this straightforward {obj}",
#                 f"What you see is what you get with this predictable dependable {obj}",
#                 f"Steady and reliable this {obj} meets expectations without any drama",
#             ],
#             ("Neutral", "Excitement"): [
#                 f"Discover some interesting features in this {obj} worth checking out",
#                 f"Curious about new options - this {obj} has a few tricks worth seeing",
#                 f"Something new on the market - explore what this {obj} has to offer",
#                 f"Intriguing possibilities await with this newly updated {obj} model today",
#             ],
#             ("Neutral", "Fear"): [
#                 f"Do your research before committing - know what this {obj} really offers",
#                 f"Make an informed choice - read the full specs on this {obj} first",
#                 f"Consider all factors carefully before adding this {obj} to your cart",
#                 f"Smart buyers check twice - review all details on this {obj} today",
#             ],
#         }

#         key = (sent, emo)
#         options = ad_templates.get(key, ad_templates[("Neutral", "Trust")])
#         return options[n % len(options)]

#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     #  HELPERS
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#     def _fill(self, result, items, indices):
#         used = set()
#         for pos, idx in enumerate(indices):
#             if result[idx].get("text") and len(result[idx]["text"]) >= 8:
#                 continue
#             if pos < len(items):
#                 t = self._clean(str(items[pos].get("text", "")))
#                 k = str(items[pos].get("keywords", "")).strip()
#                 if len(t) >= 8 and t not in self._used_texts:
#                     self._used_texts.add(t)
#                     result[idx]["text"] = t
#                     result[idx]["keywords"] = k
#                     used.add(pos)
#                     continue
#             for j in range(len(items)):
#                 if j in used:
#                     continue
#                 t = self._clean(str(items[j].get("text", "")))
#                 if len(t) >= 8 and t not in self._used_texts:
#                     self._used_texts.add(t)
#                     result[idx]["text"] = t
#                     result[idx]["keywords"] = str(
#                         items[j].get("keywords", "")
#                     ).strip()
#                     used.add(j)
#                     break

#     def _gaps(self, result):
#         return [i for i, r in enumerate(result)
#                 if not r.get("text") or len(r["text"]) < 8]

#     def _n_filled(self, result):
#         return sum(1 for r in result
#                    if r.get("text") and len(r["text"]) >= 8)

#     def _clean(self, text):
#         t = text.strip().strip('"').strip("'")
#         t = t.replace('\\"', '"').replace("\\'", "'")
#         t = t.replace("\\n", " ").replace("\\t", " ")
#         t = re.sub(r'\s+', ' ', t).strip()
#         return t

#     def get_stats(self):
#         return {
#             "api_calls": self._api_calls,
#             "unique_texts": len(self._used_texts),
#         }





# text_generator.py
"""
Generates AD COPY text + keywords using NVIDIA API.
Supports: Qwen, Llama, DeepSeek (with thinking mode + streaming).
NO FALLBACK TEMPLATES. All text from LLM.
"""

import os
import json
import time
import re
from typing import Optional, List, Tuple

from config import (
    USE_OPENAI_SDK,
    NVIDIA_API_KEY,
    BASE_URL,
    INVOKE_URL,
    STREAM,
    THINKING,              # ← NEW
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
)

if USE_OPENAI_SDK:
    from openai import OpenAI
else:
    import requests


SYSTEM_PROMPT = (
    "You are a professional advertising copywriter who writes short punchy ad copy. "
    "You write like ads seen on Instagram Facebook and Google. "
    "You always output valid JSON. "
    "Never use colons or apostrophes inside text strings. "
    "Use dashes instead of colons and full words instead of contractions."
)

SYSTEM_PROMPT_SINGLE = (
    "You are an ad copywriter. "
    "You write one short punchy advertising line. "
    "Reply with ONLY a JSON object. Nothing else. "
    "Never use colons or apostrophes in the ad text."
)


class TextGenerator:
    """Generates AD COPY text + keywords. Supports all models including DeepSeek."""

    def __init__(self):
        if USE_OPENAI_SDK:
            self.client = OpenAI(
                base_url=BASE_URL,
                api_key=NVIDIA_API_KEY,
            )
            print(f"  API Mode  : OpenAI SDK")
        else:
            self.client = None
            self.headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": (
                    "text/event-stream" if STREAM
                    else "application/json"
                ),
                "Content-Type": "application/json",
            }
            print(f"  API Mode  : Raw requests")

        print(f"  Model     : {MODEL_NAME}")
        print(f"  Profile   : {ACTIVE_PROFILE}")
        print(f"  Streaming : {'ON' if STREAM else 'OFF'}")
        print(f"  Thinking  : {'ON' if THINKING else 'OFF'}")
        print(f"  Text Type : AD COPY ONLY")
        print(f"  Fallback  : DISABLED")

        self._used_texts: set = set()
        self._api_calls: int = 0

        if LOG_RESPONSES:
            os.makedirs(LOG_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  API CALLS — HANDLES ALL MODELS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_api(
        self, prompt: str, system: str = None, temperature: float = None,
    ) -> Optional[str]:
        self._api_calls += 1
        temp = temperature if temperature is not None else TEMPERATURE
        sys_msg = system or SYSTEM_PROMPT

        if USE_OPENAI_SDK:
            return self._call_sdk(prompt, sys_msg, temp)
        else:
            return self._call_req(prompt, sys_msg, temp)

    def _call_sdk(self, prompt: str, system: str, temp: float) -> Optional[str]:
        """
        OpenAI SDK call. Handles three modes:
        1. Non-streaming (Qwen, Llama)
        2. Streaming without thinking (DeepSeek no-think)
        3. Streaming with thinking (DeepSeek thinking mode)
        """
        try:
            # Build request kwargs
            kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": temp,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "stream": STREAM,
            }

            # Add thinking mode for DeepSeek
            if THINKING:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"thinking": True}
                }

            completion = self.client.chat.completions.create(**kwargs)

            if STREAM:
                # ── Streaming mode (DeepSeek) ──
                content_chunks = []
                reasoning_chunks = []
                prompt_tokens = 0
                completion_tokens = 0

                for chunk in completion:
                    # Skip empty chunks
                    if not getattr(chunk, "choices", None):
                        # Check for usage in final chunk
                        if hasattr(chunk, "usage") and chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                            completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
                        continue

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Collect reasoning (thinking) content — discard it
                    if THINKING:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_chunks.append(reasoning)
                            continue

                    # Collect actual content
                    if delta.content is not None:
                        content_chunks.append(delta.content)

                    # Check for usage
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", 0)

                content = "".join(content_chunks)

                # Log token info
                if prompt_tokens or completion_tokens:
                    total = prompt_tokens + completion_tokens
                    print(
                        f"        Tokens: p={prompt_tokens} "
                        f"c={completion_tokens} t={total}"
                    )

                if THINKING and reasoning_chunks:
                    reasoning_text = "".join(reasoning_chunks)
                    print(
                        f"        Thinking: {len(reasoning_text)} chars "
                        f"(discarded)"
                    )

                    # Log reasoning for debugging
                    if LOG_RESPONSES:
                        log_path = os.path.join(
                            LOG_DIR,
                            f"thinking_{self._api_calls}.txt"
                        )
                        with open(log_path, "w", encoding="utf-8") as f:
                            f.write(f"=== REASONING ===\n{reasoning_text}\n\n")
                            f.write(f"=== CONTENT ===\n{content}\n")

                return content

            else:
                # ── Non-streaming mode (Qwen, Llama) ──
                content = completion.choices[0].message.content

                if hasattr(completion, "usage") and completion.usage:
                    u = completion.usage
                    print(
                        f"        Tokens: p={u.prompt_tokens} "
                        f"c={u.completion_tokens} t={u.total_tokens}"
                    )

                return content

        except Exception as e:
            print(f"        ✗ API: {e}")
            return None

    def _call_req(self, prompt: str, system: str, temp: float) -> Optional[str]:
        """Raw requests call. Handles streaming for DeepSeek."""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":        MAX_TOKENS,
            "temperature":       temp,
            "top_p":             TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty":  PRESENCE_PENALTY,
            "stream":            STREAM,
        }

        # Add thinking mode
        if THINKING:
            payload["chat_template_kwargs"] = {"thinking": True}

        try:
            r = requests.post(
                INVOKE_URL, headers=self.headers,
                json=payload, timeout=300,
            )
            if r.status_code != 200:
                print(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
                return None

            if STREAM:
                content_chunks = []
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

                        # Skip reasoning content
                        if THINKING and "reasoning_content" in delta:
                            continue

                        # Collect actual content
                        content = delta.get("content")
                        if content:
                            content_chunks.append(content)
                    except json.JSONDecodeError:
                        continue

                return "".join(content_chunks)
            else:
                data = r.json()
                u = data.get("usage", {})
                if u:
                    print(
                        f"        Tokens: p={u.get('prompt_tokens','?')} "
                        f"c={u.get('completion_tokens','?')} "
                        f"t={u.get('total_tokens','?')}"
                    )
                return (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )
        except requests.exceptions.Timeout:
            print("        ✗ Timeout")
            return None
        except Exception as e:
            print(f"        ✗ {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  EVERYTHING BELOW IS IDENTICAL — NO CHANGES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_batch_prompt(self, rows: List[dict]) -> str:
        lines = []
        for i, r in enumerate(rows):
            lines.append(
                f"{i+1}. {r['theme']} | {r['object_detected']} | "
                f"{r['sentiment']} | {r['emotion']}"
            )
        rows_block = "\n".join(lines)

        return f"""You are writing short advertising copy for social media ads and banners.
Write one unique ad headline or tagline (10-18 words) for each row below.
Each ad line must promote the Object and match the Sentiment and Emotion.
Think Instagram ads - Facebook sponsored posts - Google display ads - billboard slogans.
No brand names. No colons. No apostrophes. Use dashes instead.

AD COPY EXAMPLES by Sentiment and Emotion:

POSITIVE ADS:
- Positive Joy: "Grab a hot fresh pizza tonight and taste pure happiness in every single bite"
- Positive Trust: "Built to last and proven reliable - this laptop never lets you down ever"
- Positive Excitement: "Get ready for the ride of your life with this incredible new sports car"

NEGATIVE ADS (warning style / problem-solution / urgency):
- Negative Anger: "Tired of flimsy bags that fall apart - demand better quality for your money"
- Negative Fear: "Do not risk your safety with cheap helmets - protect what matters most today"

NEUTRAL ADS (informational / comparison / factual):
- Neutral Trust: "Standard performance you can count on - this watch does exactly what it should"
- Neutral Joy: "A decent cup of coffee for your everyday morning routine - nothing fancy needed"
- Neutral Anger: "Some minor issues but still gets the job done - decide for yourself today"
- Neutral Excitement: "New features worth exploring - see what this tablet can do for you now"
- Neutral Fear: "Know the facts before you buy - check all safety ratings on this scooter first"

AD COPY STYLES TO MIX:
- Headlines: "Upgrade Your Morning With Premium Fresh Juice Delivered To Your Door"
- Taglines: "Life is better with the perfect pair of shoes on your feet"
- Calls to action: "Do not miss out on this incredible new camera - order yours today"
- Problem-solution: "Struggling with slow laptops - switch to lightning fast performance now"
- Aspirational: "Drive the car you have always dreamed about - luxury within your reach"
- Urgency: "Limited time offer on premium headphones - grab yours before they sell out"
- Lifestyle: "Weekend getaways start with the perfect resort - book your escape today"

ROWS TO WRITE ADS FOR:
{rows_block}

OUTPUT FORMAT - a JSON array with EXACTLY {len(rows)} objects:
[
{{"id":1,"text":"your unique ad copy here about the object","keywords":"word1 word2 word3"}},
{{"id":2,"text":"another unique ad line matching sentiment and emotion","keywords":"word1 word2 word3"}}
]

CRITICAL RULES:
- Output ONLY the JSON array - no other text
- Exactly {len(rows)} objects in the array
- Every "text" must be 10-18 words of AD COPY not review text
- Every "text" must promote or reference the object
- Write like a professional ad copywriter not a reviewer
- Every text must be completely unique
- "keywords" should be 3-4 relevant words separated by spaces
- Do NOT wrap in markdown or code fences

Generate the JSON array of ad copy now:"""

    def _build_single_prompt(self, row: dict, variation: int = 0) -> str:
        obj = row["object_detected"]
        sent = row["sentiment"]
        emo = row["emotion"]
        theme = row["theme"]

        angles = [
            "as an Instagram sponsored post caption",
            "as a Facebook ad headline",
            "as a Google display ad tagline",
            "as a billboard slogan for the highway",
            "as a YouTube pre-roll ad opening line",
            "as an email marketing subject line",
            "as a print magazine advertisement headline",
            "as a Twitter promoted post",
            "as a TikTok ad caption",
            "as a podcast sponsor read opening",
            "as a shopping app push notification",
            "as a banner ad on a news website",
        ]
        angle = angles[variation % len(angles)]

        tone_map = {
            ("Positive", "Joy"):        "joyful and uplifting - make people smile and want the product",
            ("Positive", "Trust"):      "confident and reassuring - emphasize reliability and proven quality",
            ("Positive", "Excitement"): "thrilling and energetic - create urgency and desire to try it",
            ("Negative", "Anger"):      "problem-focused - highlight frustration with alternatives and offer a better way",
            ("Negative", "Fear"):       "warning-style - highlight dangers of not acting or using inferior products",
            ("Neutral", "Joy"):         "pleasant and informational - mild positivity with factual appeal",
            ("Neutral", "Anger"):       "acknowledging common complaints - be honest about limitations",
            ("Neutral", "Trust"):       "factual and straightforward - state what the product does reliably",
            ("Neutral", "Excitement"):  "curious and intriguing - spark interest without overselling",
            ("Neutral", "Fear"):        "cautious and advisory - encourage informed decision making",
        }
        tone = tone_map.get((sent, emo), f"{sent.lower()} tone conveying {emo.lower()}")

        return f"""Write exactly one advertising headline or tagline (10-18 words) for a {obj} in the {theme} category.
Write it {angle}.
The tone should be {tone}.
This is AD COPY not a review - write like a professional copywriter creating an ad.
No brand names. No colons. No apostrophes.

Reply with ONLY this JSON and nothing else:
{{"text":"your ad copy sentence here","keywords":"word1 word2 word3"}}"""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  JSON PARSERS (5 strategies) — UNCHANGED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _parse_batch(self, raw: str) -> List[dict]:
        for name, fn in [
            ("direct", self._p1_direct), ("clean", self._p2_clean),
            ("fix", self._p3_fix), ("line", self._p4_line),
            ("regex", self._p5_regex),
        ]:
            items = fn(raw)
            if items:
                print(f"        Parsed via {name}: {len(items)} items")
                return items
        return []

    def _p1_direct(self, raw):
        try:
            d = json.loads(raw.strip())
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "text" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p2_clean(self, raw):
        t = raw.strip()
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in t:
            for p in t.split("```"):
                if p.strip().startswith("["):
                    t = p.strip(); break
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1: return []
        try:
            d = json.loads(t[s:e+1])
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "text" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p3_fix(self, raw):
        t = raw.strip()
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1: return []
        t = t[s:e+1]
        t = re.sub(r',\s*}', '}', t)
        t = re.sub(r',\s*]', ']', t)
        t = re.sub(r'("text"\s*:\s*")(.*?)("\s*,\s*"keywords")',
            lambda m: m.group(1)+m.group(2).replace('"',"'")+m.group(3), t, flags=re.DOTALL)
        t = re.sub(r'("keywords"\s*:\s*")(.*?)("\s*})',
            lambda m: m.group(1)+m.group(2).replace('"',"'")+m.group(3), t, flags=re.DOTALL)
        t = self._fix_colons(t)
        t = re.sub(r'[\x00-\x1f\x7f]', ' ', t)
        try:
            d = json.loads(t)
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and "text" in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p4_line(self, raw):
        items = []
        for m in re.findall(r'\{[^{}]*?"text"[^{}]*?\}', raw, re.DOTALL):
            try:
                o = json.loads(m)
                if "text" in o and len(str(o["text"])) > 8: items.append(o); continue
            except json.JSONDecodeError: pass
            f = re.sub(r',\s*}', '}', m)
            f = re.sub(r'("text"\s*:\s*")(.*?)("\s*[,}])',
                lambda x: x.group(1)+x.group(2).replace('"',"'").replace(":"," -")+x.group(3), f, flags=re.DOTALL)
            try:
                o = json.loads(f)
                if "text" in o and len(str(o["text"])) > 8: items.append(o)
            except json.JSONDecodeError: continue
        return items

    def _p5_regex(self, raw):
        items = []
        for chunk in re.split(r'(?=\{)', raw):
            tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            if not tm or len(tm.group(1).strip()) < 8: continue
            km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            im = re.search(r'"id"\s*:\s*(\d+)', chunk)
            items.append({
                "id": int(im.group(1)) if im else len(items)+1,
                "text": tm.group(1).strip(),
                "keywords": km.group(1).strip() if km else "",
            })
        return items

    def _parse_single_response(self, raw: str) -> Tuple[str, str]:
        cleaned = raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                s = p.strip()
                if s.startswith("{") or '"text"' in s: cleaned = s; break
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            obj_str = cleaned[start:end+1]
            obj_str = re.sub(r',\s*}', '}', obj_str)
            obj_str = self._fix_colons(obj_str)
            try:
                obj = json.loads(obj_str)
                t = str(obj.get("text", "")).strip()
                k = str(obj.get("keywords", "")).strip()
                if len(t) >= 8: return t, k
            except json.JSONDecodeError: pass
        tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        if tm and len(tm.group(1).strip()) >= 8:
            km = re.search(r'"keywords"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
            return tm.group(1).strip(), km.group(1).strip() if km else ""
        plain = cleaned.strip('"').strip("'").strip()
        plain = re.sub(r'^(text|sentence|here|output|ad|copy)\s*[:=]\s*', '', plain, flags=re.IGNORECASE)
        plain = plain.strip('"').strip("'").strip()
        wc = len(plain.split())
        if 8 <= wc <= 30 and not any(c in plain for c in '{}[]'):
            print(f"        (plain text)"); return plain, ""
        return "", ""

    def _fix_colons(self, s):
        result = []; in_str = False; in_val = False; after_c = False; esc = False
        for c in s:
            if esc: result.append(c); esc = False; continue
            if c == '\\': result.append(c); esc = True; continue
            if c == '"':
                in_str = not in_str
                if in_str and after_c: in_val = True; after_c = False
                elif not in_str: in_val = False
                result.append(c); continue
            if not in_str and c == ':': after_c = True; result.append(c); continue
            if not in_str: after_c = False
            if in_val and c == ':': result.append(' -'); continue
            result.append(c)
        return ''.join(result)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN ENTRY + PHASES — UNCHANGED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate_texts(self, skeleton_rows: List[dict], batch_label: str = "") -> List[dict]:
        total = len(skeleton_rows)
        result = [row.copy() for row in skeleton_rows]
        for r in result: r["text"] = ""; r["keywords"] = ""

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
                print(f"      ✓ Phase {phase} complete: {filled}/{total}")
                return result
        return result

    def _phase_full(self, skel, result, label, total):
        print(f"      Phase 1: Full batch ({total} rows)")
        items = self._call_batch(skel, f"{label}_full")
        if items: self._fill(result, items, list(range(total)))
        print(f"      Phase 1: {self._n_filled(result)}/{total}")

    def _phase_retry(self, skel, result, label, total):
        gaps = self._gaps(result)
        if not gaps: return
        print(f"      Phase 2: Retry {len(gaps)} gaps")
        items = self._call_batch([skel[i] for i in gaps], f"{label}_retry")
        if items: self._fill(result, items, gaps)
        print(f"      Phase 2: {self._n_filled(result)}/{total}")

    def _phase_micro(self, skel, result, label, total):
        gaps = self._gaps(result)
        if not gaps: return
        print(f"      Phase 3: Micro-batches for {len(gaps)} rows")
        for ci in range(0, len(gaps), 5):
            chunk = gaps[ci:ci+5]
            items = self._call_batch([skel[i] for i in chunk], f"{label}_m{ci//5}")
            if items: self._fill(result, items, chunk)
            time.sleep(1)
        print(f"      Phase 3: {self._n_filled(result)}/{total}")

    def _phase_single(self, skel, result, label, total):
        gaps = self._gaps(result)
        if not gaps: return
        print(f"      Phase 4: One-by-one for {len(gaps)} rows")
        for idx in gaps:
            text, kw = self._single_generate(skel[idx], f"{label}_s{idx}")
            if text and len(text) >= 8 and text not in self._used_texts:
                self._used_texts.add(text)
                result[idx]["text"] = text; result[idx]["keywords"] = kw
                print(f"        ✓ Row {idx}")
            else:
                print(f"        ⚠ Row {idx}: needs Phase 5")
            time.sleep(1)
        print(f"      Phase 4: {self._n_filled(result)}/{total}")

    def _phase_construct(self, skel, result, label, total):
        gaps = self._gaps(result)
        if not gaps: return
        print(f"      Phase 5: Construct {len(gaps)} ad lines")
        for idx in gaps:
            text = self._construct_ad(skel[idx])
            self._used_texts.add(text)
            result[idx]["text"] = text
            result[idx]["keywords"] = f"{skel[idx]['object_detected']} {skel[idx]['theme']} {skel[idx]['emotion']}"
            print(f"        ✓ Row {idx}: constructed")

    def _call_batch(self, rows, label) -> List[dict]:
        prompt = self._build_batch_prompt(rows)
        for att in range(1, MAX_RETRIES + 1):
            print(f"        Attempt {att}/{MAX_RETRIES} ({len(rows)} rows) [{label}]")
            raw = self._call_api(prompt)
            if raw is None: time.sleep(RETRY_DELAY); continue
            if LOG_RESPONSES:
                with open(os.path.join(LOG_DIR, f"{label}_a{att}.txt"), "w", encoding="utf-8") as f: f.write(raw)
            items = self._parse_batch(raw)
            if items: return items
            print(f"        ✗ No items parsed"); time.sleep(RETRY_DELAY)
        return []

    def _single_generate(self, row, label) -> Tuple[str, str]:
        for variation in range(6):
            temp = min(TEMPERATURE + (variation * 0.15), 1.0)
            prompt = self._build_single_prompt(row, variation=variation)
            raw = self._call_api(prompt, system=SYSTEM_PROMPT_SINGLE, temperature=temp)
            if raw is None: time.sleep(2); continue
            if LOG_RESPONSES:
                with open(os.path.join(LOG_DIR, f"{label}_v{variation}.txt"), "w", encoding="utf-8") as f: f.write(raw)
            text, kw = self._parse_single_response(raw)
            text = self._clean(text)
            if text and len(text) >= 8 and text not in self._used_texts: return text, kw
            time.sleep(1)
        return "", ""

    def _construct_ad(self, row: dict) -> str:
        obj = row.get("object_detected", "item").lower()
        sent = row.get("sentiment", "Neutral")
        emo = row.get("emotion", "Trust")
        img = row.get("image_path", "0")
        n = int(re.search(r'(\d+)', img).group(1)) if re.search(r'(\d+)', img) else 0
        t = {
            ("Positive","Joy"): [f"Bring home this amazing {obj} and discover pure happiness in every moment",f"Treat yourself to this wonderful {obj} and let the joy begin today",f"Experience the delight of owning this perfect {obj} starting right now",f"Make every day brighter with this incredible {obj} in your life",f"Unwrap happiness with this fantastic {obj} that will light up your world",f"Add a spark of joy to your routine with this beautiful {obj} today",f"Find your happy place with this outstanding {obj} made just for you",f"Start smiling more with this delightful {obj} that brings real joy",f"Let this gorgeous {obj} fill your days with pure unmatched happiness",f"Choose happiness and choose this perfect {obj} for your lifestyle today"],
            ("Positive","Trust"): [f"Count on this proven {obj} for reliable performance day after day",f"Trust the quality that thousands already depend on with this {obj}",f"Built to last and made to impress - this {obj} earns your confidence",f"Reliable performance guaranteed with every single use of this {obj}",f"When dependability matters most choose this battle-tested {obj} today",f"Proven quality you can always count on with this outstanding {obj}",f"Stand behind a {obj} that stands behind you with rock solid quality",f"Dependable from day one this {obj} delivers what it promises always",f"Quality you can trust and performance you can rely on with this {obj}",f"Put your confidence in this {obj} that never lets its owners down"],
            ("Positive","Excitement"): [f"Get ready for something incredible - this {obj} will blow your mind",f"The wait is over - experience the thrill of this brand new {obj}",f"Unleash your excitement with this game-changing {obj} available now",f"Feel the rush of owning this extraordinary {obj} starting today",f"Do not miss this electrifying new {obj} that everyone is talking about",f"Prepare to be amazed by this revolutionary {obj} hitting shelves now",f"Your next adventure starts with this thrilling {obj} in your hands",f"Experience the excitement everyone is buzzing about with this new {obj}",f"Ignite your passion with this sensational {obj} that changes everything",f"The most exciting {obj} of the year is here and waiting for you"],
            ("Negative","Anger"): [f"Stop settling for less - upgrade from your frustrating old {obj} today",f"Tired of disappointing quality - demand the {obj} you actually deserve",f"Fed up with unreliable products - it is time for a better {obj} now",f"Do not waste another dollar on a {obj} that lets you down again",f"Had enough of subpar options - discover what a real {obj} should be",f"Frustrated with your current {obj} - make the switch you need today",f"Refuse to accept mediocre quality from your {obj} any longer now",f"Your old {obj} is holding you back - break free with something better",f"Say goodbye to frustration and hello to a {obj} that actually works",f"No more excuses for poor quality - get the {obj} that performs right"],
            ("Negative","Fear"): [f"Do not risk your safety with an unreliable {obj} - choose wisely today",f"Protect yourself from danger with a properly certified {obj} instead",f"Your safety is not worth gambling on - invest in a quality {obj} now",f"Before you buy any {obj} check the safety ratings that matter most",f"An unsafe {obj} is never worth the savings - protect what matters",f"Do not let a faulty {obj} put your wellbeing at risk ever again",f"Safety first - make sure your next {obj} meets every safety standard",f"Think twice before buying a cheap {obj} that could put you in danger",f"Your family deserves a {obj} that is certified safe and fully tested",f"Stop worrying about safety - get a {obj} you can actually trust today"],
            ("Neutral","Joy"): [f"A solid everyday {obj} that gets the job done with quiet satisfaction",f"Simple pleasures start with a decent {obj} for your daily routine",f"This {obj} brings a touch of comfort to your ordinary everyday life",f"Nothing fancy just a good {obj} that adds mild enjoyment to your day"],
            ("Neutral","Anger"): [f"Not perfect but still functional - this {obj} handles the basics well",f"Know the trade-offs before buying - this {obj} has pros and cons",f"Minor issues aside this {obj} still delivers acceptable performance overall",f"Be aware of the limitations but this {obj} works for most needs"],
            ("Neutral","Trust"): [f"Standard reliable performance from a {obj} that does what it should",f"No surprises just consistent delivery from this straightforward {obj}",f"What you see is what you get with this predictable dependable {obj}",f"Steady and reliable this {obj} meets expectations without any drama"],
            ("Neutral","Excitement"): [f"Discover some interesting features in this {obj} worth checking out",f"Curious about new options - this {obj} has a few tricks worth seeing",f"Something new on the market - explore what this {obj} has to offer",f"Intriguing possibilities await with this newly updated {obj} model today"],
            ("Neutral","Fear"): [f"Do your research before committing - know what this {obj} really offers",f"Make an informed choice - read the full specs on this {obj} first",f"Consider all factors carefully before adding this {obj} to your cart",f"Smart buyers check twice - review all details on this {obj} today"],
        }
        opts = t.get((sent, emo), t[("Neutral","Trust")])
        return opts[n % len(opts)]

    def _fill(self, result, items, indices):
        used = set()
        for pos, idx in enumerate(indices):
            if result[idx].get("text") and len(result[idx]["text"]) >= 8: continue
            if pos < len(items):
                t = self._clean(str(items[pos].get("text", "")))
                k = str(items[pos].get("keywords", "")).strip()
                if len(t) >= 8 and t not in self._used_texts:
                    self._used_texts.add(t); result[idx]["text"] = t; result[idx]["keywords"] = k; used.add(pos); continue
            for j in range(len(items)):
                if j in used: continue
                t = self._clean(str(items[j].get("text", "")))
                if len(t) >= 8 and t not in self._used_texts:
                    self._used_texts.add(t); result[idx]["text"] = t
                    result[idx]["keywords"] = str(items[j].get("keywords", "")).strip()
                    used.add(j); break

    def _gaps(self, result): return [i for i, r in enumerate(result) if not r.get("text") or len(r["text"]) < 8]
    def _n_filled(self, result): return sum(1 for r in result if r.get("text") and len(r["text"]) >= 8)
    def _clean(self, text):
        t = text.strip().strip('"').strip("'")
        t = t.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ").replace("\\t", " ")
        return re.sub(r'\s+', ' ', t).strip()
    def get_stats(self): return {"api_calls": self._api_calls, "unique_texts": len(self._used_texts)}