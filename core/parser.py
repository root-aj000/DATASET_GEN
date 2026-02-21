# core/parser.py
"""
JSON parser with 5 strategies.
Parameterized by output column names — works for ANY task.
"""

import re
import json
from typing import List, Dict, Optional, Tuple


class JSONParser:
    """
    Parses LLM JSON output using 5 fallback strategies.
    Works with any column names — passed at parse time.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BATCH PARSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def parse_batch(
        self,
        raw: str,
        primary_key: str,
        all_keys: List[str],
    ) -> List[Dict]:
        """
        Parse batch JSON output.

        Args:
            raw: Raw LLM response
            primary_key: The key that MUST exist (e.g., "text", "img_desc")
            all_keys: All expected keys (e.g., ["text", "keywords"])

        Returns:
            List of dicts with parsed values
        """
        for name, fn in [
            ("direct", lambda: self._p1_direct(raw, primary_key)),
            ("clean", lambda: self._p2_clean(raw, primary_key)),
            ("fix", lambda: self._p3_fix(raw, primary_key, all_keys)),
            ("line", lambda: self._p4_line(raw, primary_key, all_keys)),
            ("regex", lambda: self._p5_regex(raw, primary_key, all_keys)),
        ]:
            items = fn()
            if items:
                self._log(f"        Parsed via {name}: {len(items)} items")
                return items
        return []

    def _p1_direct(self, raw: str, primary_key: str) -> List[Dict]:
        """Strategy 1: Direct JSON parse."""
        try:
            d = json.loads(raw.strip())
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and primary_key in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p2_clean(self, raw: str, primary_key: str) -> List[Dict]:
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
                return [x for x in d if isinstance(x, dict) and primary_key in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p3_fix(self, raw: str, primary_key: str, all_keys: List[str]) -> List[Dict]:
        """Strategy 3: Fix common JSON errors."""
        t = raw.strip()
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1:
            return []

        t = t[s:e+1]
        t = re.sub(r',\s*}', '}', t)
        t = re.sub(r',\s*]', ']', t)

        # Fix quotes inside string values for each key
        for key in all_keys:
            pattern = rf'("{key}"\s*:\s*")(.*?)("\s*[,}}])'
            t = re.sub(
                pattern,
                lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
                t,
                flags=re.DOTALL
            )

        t = self._fix_colons(t)
        t = re.sub(r'[\x00-\x1f\x7f]', ' ', t)

        try:
            d = json.loads(t)
            if isinstance(d, list):
                return [x for x in d if isinstance(x, dict) and primary_key in x]
        except json.JSONDecodeError:
            pass
        return []

    def _p4_line(self, raw: str, primary_key: str, all_keys: List[str]) -> List[Dict]:
        """Strategy 4: Parse individual JSON objects."""
        items = []
        pattern = r'\{[^{}]*?"' + re.escape(primary_key) + r'"[^{}]*?\}'

        for m in re.findall(pattern, raw, re.DOTALL):
            try:
                o = json.loads(m)
                if primary_key in o and len(str(o[primary_key])) > 3:
                    items.append(o)
                    continue
            except json.JSONDecodeError:
                pass

            # Try fixing
            f = re.sub(r',\s*}', '}', m)
            for key in all_keys:
                f = re.sub(
                    rf'("{key}"\s*:\s*")(.*?)("\s*[,}}])',
                    lambda x: x.group(1) + x.group(2).replace('"', "'").replace(":", " -") + x.group(3),
                    f,
                    flags=re.DOTALL
                )

            try:
                o = json.loads(f)
                if primary_key in o and len(str(o[primary_key])) > 3:
                    items.append(o)
            except json.JSONDecodeError:
                continue

        return items

    def _p5_regex(self, raw: str, primary_key: str, all_keys: List[str]) -> List[Dict]:
        """Strategy 5: Regex extraction."""
        items = []

        for chunk in re.split(r'(?=\{)', raw):
            # Extract primary key
            pm = re.search(rf'"{primary_key}"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            if not pm or len(pm.group(1).strip()) < 3:
                continue

            item = {primary_key: pm.group(1).strip()}

            # Extract other keys
            for key in all_keys:
                if key == primary_key:
                    continue
                km = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
                if km:
                    item[key] = km.group(1).strip()

            # Extract id if present
            im = re.search(r'"id"\s*:\s*(\d+)', chunk)
            if im:
                item["id"] = int(im.group(1))
            else:
                item["id"] = len(items) + 1

            items.append(item)

        return items

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SINGLE PARSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def parse_single(
        self,
        raw: str,
        keys: List[str],
        min_length: int = 3,
    ) -> Dict[str, str]:
        """
        Parse single JSON object response.

        Args:
            raw: Raw LLM response
            keys: Expected keys (e.g., ["text", "keywords"])
            min_length: Minimum valid length for primary key

        Returns:
            Dict with extracted values (empty strings for missing)
        """
        result = {k: "" for k in keys}
        primary_key = keys[0]

        cleaned = raw.strip()

        # Clean markdown
        if "```" in cleaned:
            for p in cleaned.split("```"):
                s = p.strip()
                if s.startswith("{") or f'"{primary_key}"' in s:
                    cleaned = s
                    break

        # Try JSON parse
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:
            obj_str = cleaned[start:end+1]
            obj_str = re.sub(r',\s*}', '}', obj_str)
            obj_str = self._fix_colons(obj_str)

            try:
                obj = json.loads(obj_str)
                for key in keys:
                    val = str(obj.get(key, "")).strip()
                    if val:
                        result[key] = val

                if len(result.get(primary_key, "")) >= min_length:
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: regex extraction
        for key in keys:
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
            if m:
                result[key] = m.group(1).strip()

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text."""
        t = text.strip().strip('"').strip("'")
        t = t.replace('\\"', '"').replace("\\'", "'")
        t = t.replace("\\n", " ").replace("\\t", " ")
        t = re.sub(r'\s+', ' ', t).strip()
        return t