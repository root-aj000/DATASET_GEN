# # core/client.py
# """
# Unified API client for all agents.
# Supports: OpenAI SDK, Raw requests, Streaming, Thinking mode.
# """

# import os
# import json
# import time
# from typing import Optional

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
#     LOG_RESPONSES,
#     LOG_DIR,
# )
# from utils import get_retry_session

# if USE_OPENAI_SDK:
#     from openai import OpenAI


# class APIClient:
#     """
#     Single API client used by all agents.
#     Extracted from duplicated code in text_generator, search_agent, etc.
#     """

#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         self._api_calls = 0

#         if USE_OPENAI_SDK:
#             self.client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
#             self._mode = "OpenAI SDK"
#         else:
#             self.client = None
#             self.headers = {
#                 "Authorization": f"Bearer {NVIDIA_API_KEY}",
#                 "Accept": ("text/event-stream" if STREAM else "application/json"),
#                 "Content-Type": "application/json",
#             }
#             self._mode = "Raw requests"

#         self.session = get_retry_session()

#         if LOG_RESPONSES:
#             os.makedirs(LOG_DIR, exist_ok=True)

#         self._log(f"  API Mode  : {self._mode}")
#         self._log(f"  Model     : {MODEL_NAME}")
#         self._log(f"  Streaming : {'ON' if STREAM else 'OFF'}")
#         self._log(f"  Thinking  : {'ON' if THINKING else 'OFF'}")

#     def _log(self, msg: str):
#         if self.verbose:
#             print(msg)

#     @property
#     def api_calls(self) -> int:
#         return self._api_calls

#     def call(
#         self,
#         prompt: str,
#         system: str,
#         temperature: Optional[float] = None,
#         log_label: Optional[str] = None,
#     ) -> Optional[str]:
#         """
#         Make an API call.

#         Args:
#             prompt: User prompt
#             system: System instruction
#             temperature: Override temperature (default from config)
#             log_label: Label for logging file

#         Returns:
#             Response content or None on failure
#         """
#         self._api_calls += 1
#         temp = temperature if temperature is not None else TEMPERATURE

#         if USE_OPENAI_SDK:
#             content = self._call_sdk(prompt, system, temp)
#         else:
#             content = self._call_raw(prompt, system, temp)

#         # Log response if enabled
#         if LOG_RESPONSES and content and log_label:
#             log_path = os.path.join(LOG_DIR, f"{log_label}.txt")
#             with open(log_path, "w", encoding="utf-8") as f:
#                 f.write(content)

#         return content

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
#                 return self._handle_stream_sdk(completion)
#             else:
#                 content = completion.choices[0].message.content
#                 if hasattr(completion, "usage") and completion.usage:
#                     u = completion.usage
#                     self._log(f"        Tokens: p={u.prompt_tokens} c={u.completion_tokens} t={u.total_tokens}")
#                 return content

#         except Exception as e:
#             self._log(f"        ✗ API: {e}")
#             return None

#     def _handle_stream_sdk(self, completion) -> str:
#         """Handle streaming response from SDK."""
#         content_chunks = []
#         reasoning_chunks = []
#         prompt_tokens = 0
#         completion_tokens = 0

#         for chunk in completion:
#             if not getattr(chunk, "choices", None):
#                 if hasattr(chunk, "usage") and chunk.usage:
#                     prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
#                     completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
#                 continue

#             if not chunk.choices:
#                 continue

#             delta = chunk.choices[0].delta

#             # Handle thinking mode
#             if THINKING:
#                 reasoning = getattr(delta, "reasoning_content", None)
#                 if reasoning:
#                     reasoning_chunks.append(reasoning)
#                     continue

#             if delta.content is not None:
#                 content_chunks.append(delta.content)

#             if hasattr(chunk, "usage") and chunk.usage:
#                 prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
#                 completion_tokens = getattr(chunk.usage, "completion_tokens", 0)

#         content = "".join(content_chunks)

#         if prompt_tokens or completion_tokens:
#             total = prompt_tokens + completion_tokens
#             self._log(f"        Tokens: p={prompt_tokens} c={completion_tokens} t={total}")

#         if THINKING and reasoning_chunks:
#             reasoning_text = "".join(reasoning_chunks)
#             self._log(f"        Thinking: {len(reasoning_text)} chars (discarded)")

#         return content

#     def _call_raw(self, prompt: str, system: str, temp: float) -> Optional[str]:
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
#             r = self.session.post(INVOKE_URL, headers=self.headers, json=payload, timeout=600)

#             if r.status_code != 200:
#                 self._log(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
#                 return None

#             if STREAM:
#                 return self._handle_stream_raw(r)
#             else:
#                 data = r.json()
#                 u = data.get("usage", {})
#                 if u:
#                     self._log(f"        Tokens: p={u.get('prompt_tokens','?')} c={u.get('completion_tokens','?')} t={u.get('total_tokens','?')}")
#                 return data.get("choices", [{}])[0].get("message", {}).get("content", "")

#         except Exception as e:
#             self._log(f"        ✗ {e}")
#             return None

#     def _handle_stream_raw(self, response) -> str:
#         """Handle streaming response from raw request."""
#         content_chunks = []

#         for line in response.iter_lines():
#             if not line:
#                 continue
#             d = line.decode("utf-8")
#             if d.startswith("data: "):
#                 d = d[6:]
#             if d.strip() == "[DONE]":
#                 break
#             try:
#                 obj = json.loads(d)
#                 choices = obj.get("choices", [])
#                 if not choices:
#                     continue
#                 delta = choices[0].get("delta", {})

#                 if THINKING and "reasoning_content" in delta:
#                     continue

#                 content = delta.get("content")
#                 if content:
#                     content_chunks.append(content)
#             except json.JSONDecodeError:
#                 continue

#         return "".join(content_chunks)

# core/client.py
"""
Unified API client for all agents.
Supports:
  - Chat Completions API (client.chat.completions.create)
  - Responses API (client.responses.create)
  - Streaming
  - Model-specific Thinking/Reasoning modes
"""

import os
import json
import time
from typing import Optional, Dict, Any, List, Union

from config import (
    USE_OPENAI_SDK,
    NVIDIA_API_KEY,
    BASE_URL,
    INVOKE_URL,
    RESPONSES_URL,
    STREAM,
    THINKING,
    THINKING_EXTRA_BODY,
    THINKING_REASONING_FIELD,
    API_TYPE,
    API_TYPE_FALLBACK,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    LOG_RESPONSES,
    LOG_DIR,
    ACTIVE_PROFILE,
)
from utils import get_retry_session

if USE_OPENAI_SDK:
    from openai import OpenAI


class APIClient:
    """
    Unified API client supporting both Chat Completions and Responses API.
    Automatically handles model-specific thinking/reasoning modes.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._api_calls = 0
        self._reasoning_chars = 0

        if USE_OPENAI_SDK:
            self.client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
            self._mode = "OpenAI SDK"
        else:
            self.client = None
            self.headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": ("text/event-stream" if STREAM else "application/json"),
                "Content-Type": "application/json",
            }
            self._mode = "Raw requests"

        self.session = get_retry_session()

        if LOG_RESPONSES:
            os.makedirs(LOG_DIR, exist_ok=True)

        self._log(f"  API Mode   : {self._mode}")
        self._log(f"  API Type   : {API_TYPE.upper()}")
        self._log(f"  Model      : {MODEL_NAME}")
        self._log(f"  Profile    : {ACTIVE_PROFILE}")
        self._log(f"  Streaming  : {'ON' if STREAM else 'OFF'}")
        self._log(f"  Thinking   : {'ON' if THINKING else 'OFF'}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @property
    def api_calls(self) -> int:
        return self._api_calls

    @property
    def reasoning_chars(self) -> int:
        return self._reasoning_chars

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def call(
        self,
        prompt: str,
        system: str = None,
        temperature: Optional[float] = None,
        log_label: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        api_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Make an API call using the appropriate API type.

        Args:
            prompt: User prompt (or full input for Responses API)
            system: System instruction (used in Chat API, prepended in Responses API)
            temperature: Override temperature
            log_label: Label for logging file
            extra_body: Override extra_body for thinking config
            api_type: Override API type ("chat" or "responses")

        Returns:
            Response content or None on failure
        """
        self._api_calls += 1
        temp = temperature if temperature is not None else TEMPERATURE
        final_api_type = api_type or API_TYPE
        final_extra_body = extra_body if extra_body is not None else THINKING_EXTRA_BODY

        # Route to appropriate API
        if final_api_type == "responses":
            content = self._call_responses_api(prompt, system, temp, final_extra_body)
        else:
            content = self._call_chat_api(prompt, system, temp, final_extra_body)

        # Try fallback if primary fails
        if content is None and API_TYPE_FALLBACK:
            self._log(f"        ⟳ Trying fallback API: {API_TYPE_FALLBACK}")
            if API_TYPE_FALLBACK == "responses":
                content = self._call_responses_api(prompt, system, temp, final_extra_body)
            else:
                content = self._call_chat_api(prompt, system, temp, final_extra_body)

        # Log response
        if LOG_RESPONSES and content and log_label:
            log_path = os.path.join(LOG_DIR, f"{log_label}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CHAT COMPLETIONS API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_chat_api(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call using Chat Completions API."""
        if USE_OPENAI_SDK:
            return self._call_chat_sdk(prompt, system, temp, extra_body)
        else:
            return self._call_chat_raw(prompt, system, temp, extra_body)

    def _call_chat_sdk(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Chat Completions API via OpenAI SDK."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temp,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "stream": STREAM,
            }

            if extra_body:
                kwargs["extra_body"] = extra_body

            completion = self.client.chat.completions.create(**kwargs)

            if STREAM:
                return self._handle_chat_stream_sdk(completion)
            else:
                content = completion.choices[0].message.content
                self._log_usage(completion)
                return content

        except Exception as e:
            self._log(f"        ✗ Chat API: {e}")
            return None

    def _handle_chat_stream_sdk(self, completion) -> str:
        """Handle streaming from Chat Completions API (SDK)."""
        content_chunks = []
        reasoning_chunks = []

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                if hasattr(chunk, "usage") and chunk.usage:
                    self._log_usage_obj(chunk.usage)
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Filter reasoning content (various field names)
            if THINKING:
                for field in ["reasoning_content", "thinking_content", "reasoning"]:
                    reasoning = getattr(delta, field, None)
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                        continue

            if delta.content is not None:
                content_chunks.append(delta.content)

        content = "".join(content_chunks)
        self._log_reasoning(reasoning_chunks)

        return content

    def _call_chat_raw(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Chat Completions API via raw requests."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": temp,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "stream": STREAM,
        }

        if extra_body:
            for key, value in extra_body.items():
                payload[key] = value

        try:
            r = self.session.post(INVOKE_URL, headers=self.headers, json=payload, timeout=600)

            if r.status_code != 200:
                self._log(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
                return None

            if STREAM:
                return self._handle_chat_stream_raw(r)
            else:
                data = r.json()
                self._log_usage_dict(data.get("usage", {}))
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as e:
            self._log(f"        ✗ Chat API: {e}")
            return None

    def _handle_chat_stream_raw(self, response) -> str:
        """Handle streaming from Chat Completions API (raw)."""
        content_chunks = []
        reasoning_chunks = []

        for line in response.iter_lines():
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

                # Filter reasoning
                if THINKING:
                    for field in ["reasoning_content", "thinking_content", "reasoning"]:
                        if field in delta:
                            reasoning_chunks.append(delta[field])
                            continue

                content = delta.get("content")
                if content:
                    content_chunks.append(content)
            except json.JSONDecodeError:
                continue

        self._log_reasoning(reasoning_chunks)
        return "".join(content_chunks)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RESPONSES API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_responses_api(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call using Responses API."""
        if USE_OPENAI_SDK:
            return self._call_responses_sdk(prompt, system, temp, extra_body)
        else:
            return self._call_responses_raw(prompt, system, temp, extra_body)

    def _call_responses_sdk(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Responses API via OpenAI SDK."""
        try:
            # Combine system + prompt for Responses API
            full_input = prompt
            if system:
                full_input = f"{system}\n\n{prompt}"

            kwargs = {
                "model": MODEL_NAME,
                "input": full_input,
                "max_output_tokens": MAX_TOKENS,
                "temperature": temp,
                "top_p": TOP_P,
                "stream": STREAM,
            }

            if extra_body:
                kwargs.update(extra_body)

            response = self.client.responses.create(**kwargs)

            if STREAM:
                return self._handle_responses_stream_sdk(response)
            else:
                # Non-streaming responses API
                return self._extract_responses_content(response)

        except Exception as e:
            self._log(f"        ✗ Responses API: {e}")
            return None

    def _handle_responses_stream_sdk(self, response) -> str:
        """Handle streaming from Responses API (SDK)."""
        content_chunks = []
        reasoning_chunks = []
        reasoning_done = False

        for chunk in response:
            chunk_type = getattr(chunk, "type", None)

            # Handle reasoning/thinking chunks
            if chunk_type == "response.reasoning_text.delta":
                delta = getattr(chunk, "delta", "")
                if delta:
                    reasoning_chunks.append(delta)

            # Handle output text chunks
            elif chunk_type == "response.output_text.delta":
                if not reasoning_done and reasoning_chunks:
                    reasoning_done = True
                delta = getattr(chunk, "delta", "")
                if delta:
                    content_chunks.append(delta)

            # Handle other delta types (some models use different names)
            elif chunk_type and "delta" in chunk_type:
                delta = getattr(chunk, "delta", "")
                if "reasoning" in chunk_type.lower() or "thinking" in chunk_type.lower():
                    if delta:
                        reasoning_chunks.append(delta)
                elif "output" in chunk_type.lower() or "text" in chunk_type.lower():
                    if delta:
                        content_chunks.append(delta)

        content = "".join(content_chunks)
        self._log_reasoning(reasoning_chunks)

        return content

    def _extract_responses_content(self, response) -> str:
        """Extract content from non-streaming Responses API response."""
        # The response structure may vary - try common patterns
        if hasattr(response, "output_text"):
            return response.output_text
        if hasattr(response, "output"):
            if isinstance(response.output, str):
                return response.output
            if hasattr(response.output, "text"):
                return response.output.text
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "content"):
            return response.content

        # Try to get from choices-like structure
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "text"):
                return choice.text
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content

        self._log(f"        ⚠ Unknown response structure: {type(response)}")
        return str(response)

    def _call_responses_raw(
        self,
        prompt: str,
        system: str,
        temp: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Responses API via raw requests."""
        full_input = prompt
        if system:
            full_input = f"{system}\n\n{prompt}"

        payload = {
            "model": MODEL_NAME,
            "input": full_input,
            "max_output_tokens": MAX_TOKENS,
            "temperature": temp,
            "top_p": TOP_P,
            "stream": STREAM,
        }

        if extra_body:
            payload.update(extra_body)

        try:
            r = self.session.post(RESPONSES_URL, headers=self.headers, json=payload, timeout=600)

            if r.status_code != 200:
                self._log(f"        ✗ HTTP {r.status_code}: {r.text[:200]}")
                return None

            if STREAM:
                return self._handle_responses_stream_raw(r)
            else:
                data = r.json()
                return data.get("output_text", data.get("output", data.get("text", "")))

        except Exception as e:
            self._log(f"        ✗ Responses API: {e}")
            return None

    def _handle_responses_stream_raw(self, response) -> str:
        """Handle streaming from Responses API (raw)."""
        content_chunks = []
        reasoning_chunks = []

        for line in response.iter_lines():
            if not line:
                continue
            d = line.decode("utf-8")
            if d.startswith("data: "):
                d = d[6:]
            if d.strip() == "[DONE]":
                break
            try:
                obj = json.loads(d)
                chunk_type = obj.get("type", "")

                if "reasoning" in chunk_type.lower() or "thinking" in chunk_type.lower():
                    delta = obj.get("delta", "")
                    if delta:
                        reasoning_chunks.append(delta)
                elif "output" in chunk_type.lower() or "text" in chunk_type.lower():
                    delta = obj.get("delta", "")
                    if delta:
                        content_chunks.append(delta)

            except json.JSONDecodeError:
                continue

        self._log_reasoning(reasoning_chunks)
        return "".join(content_chunks)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LOGGING HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _log_usage(self, completion):
        """Log token usage from completion object."""
        if hasattr(completion, "usage") and completion.usage:
            self._log_usage_obj(completion.usage)

    def _log_usage_obj(self, usage):
        """Log token usage from usage object."""
        p = getattr(usage, "prompt_tokens", 0)
        c = getattr(usage, "completion_tokens", 0)
        t = getattr(usage, "total_tokens", p + c)
        self._log(f"        Tokens: p={p} c={c} t={t}")

    def _log_usage_dict(self, usage: dict):
        """Log token usage from usage dict."""
        if usage:
            self._log(f"        Tokens: p={usage.get('prompt_tokens','?')} c={usage.get('completion_tokens','?')} t={usage.get('total_tokens','?')}")

    def _log_reasoning(self, reasoning_chunks: List[str]):
        """Log reasoning/thinking content."""
        if reasoning_chunks:
            reasoning_text = "".join(reasoning_chunks)
            self._reasoning_chars += len(reasoning_text)
            self._log(f"        Thinking: {len(reasoning_text)} chars (discarded)")

            # Optionally save to file
            if LOG_RESPONSES:
                log_path = os.path.join(LOG_DIR, f"reasoning_{self._api_calls}.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(reasoning_text)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONVENIENCE METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def call_chat(
        self,
        prompt: str,
        system: str = None,
        temperature: Optional[float] = None,
        log_label: Optional[str] = None,
    ) -> Optional[str]:
        """Explicitly call Chat Completions API."""
        return self.call(prompt, system, temperature, log_label, api_type="chat")

    def call_responses(
        self,
        prompt: str,
        system: str = None,
        temperature: Optional[float] = None,
        log_label: Optional[str] = None,
    ) -> Optional[str]:
        """Explicitly call Responses API."""
        return self.call(prompt, system, temperature, log_label, api_type="responses")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "api_calls": self._api_calls,
            "reasoning_chars_discarded": self._reasoning_chars,
            "api_type": API_TYPE,
            "model": MODEL_NAME,
        }