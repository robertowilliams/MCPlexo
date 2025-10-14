# llm_client.py
import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from mcp_llm_bridge.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMResponse:
    """
    Thin wrapper around an OpenAI chat completion choice.message
    that normalizes tool-call detection and exposes message content.
    """

    def __init__(self, message: Any, finish_reason: Optional[str] = None) -> None:
        # The raw OpenAI message object
        self.message = message

        # Final text content (if any)
        self.content: str = getattr(message, "content", None) or ""

        # Tool calls (if any)
        self.tool_calls = getattr(message, "tool_calls", None)
        # Some SDK variants expose a dict-like structure; tolerate both
        if self.tool_calls is None and isinstance(message, dict):
            self.tool_calls = message.get("tool_calls")

        # Robust tool-call detection:
        # - Prefer the presence of tool_calls
        # - Also consider finish_reason=="tool_calls" as a hint
        self.is_tool_call: bool = bool(self.tool_calls) or (finish_reason == "tool_calls")


class LLMClient:
    """
    Small client over an OpenAI-compatible Chat Completions endpoint.
    Handles:
      - message list management
      - function/tool schema registration
      - adding tool results
      - non-blocking invoke (runs sync HTTP call off the event loop)
    """

    def __init__(self, config: LLMConfig) -> None:
        self.cfg = config

        # OpenAI client; works with OpenAI and OpenAI-compatible routers if base_url is set.
        self._client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,  # type: ignore[arg-type]
        )

        # Chat history, OpenAI format
        self._messages: List[Dict[str, Any]] = []

        # Function tools in OpenAI schema
        self._tools: Optional[List[Dict[str, Any]]] = None

    # -------------------------
    # Setup
    # -------------------------
    def set_system_prompt(self, text: str) -> None:
        """
        Prepend a system message to steer the assistant.
        Call before adding any user messages for best effect.
        """
        if not text:
            return
        # Ensure only one system message is at the front
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = text
        else:
            self._messages.insert(0, {"role": "system", "content": text})

    def set_tools(self, tools: List[Dict[str, Any]]) -> None:
        """
        Register OpenAI function tools.
        """
        self._tools = tools or []

    # -------------------------
    # Message management
    # -------------------------
    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_tool_result(self, *, tool_call_id: str, name: str, content: str) -> None:
        """
        Add a tool result to the conversation so the model can see it on the next turn.
        OpenAI expects role="tool" with the matching tool_call_id.
        """
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    # -------------------------
    # Invocation
    # -------------------------
    async def invoke(self) -> LLMResponse:
        """
        Invoke the chat completion in a non-blocking way.
        We use asyncio.to_thread to avoid blocking the event loop with the sync client.
        """
        model = self.cfg.model

        # Base request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._messages,
        }

        if self._tools:
            payload["tools"] = self._tools
            payload["tool_choice"] = "auto"

        # Optional tuning knobs
        if self.cfg.temperature is not None:
            payload["temperature"] = float(self.cfg.temperature)
        if self.cfg.top_p is not None:
            payload["top_p"] = float(self.cfg.top_p)
        if self.cfg.max_tokens is not None:
            payload["max_tokens"] = int(self.cfg.max_tokens)

        # Merge any future extras
        if self.cfg.extra:
            for k, v in self.cfg.extra.items():
                # Don't stomp on known keys
                if k not in payload:
                    payload[k] = v

        def _call_sync() -> Any:
            return self._client.chat.completions.create(**payload)

        try:
            completion = await asyncio.to_thread(_call_sync)
        except Exception as e:
            logger.exception("LLM invocation failed: %s", e)
            # Surface the error back as an assistant message so the REPL doesn't break
            err_msg = {"role": "assistant", "content": f"[llm_error] {type(e).__name__}: {e}"}
            self._messages.append(err_msg)
            return LLMResponse(err_msg, finish_reason="stop")

        # Extract the top choice
        choice = completion.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = choice.message

        # Append the assistant message to our history
        # (This could be a text reply OR a tool-call stub)
        serializable_msg = _serialize_openai_message(message)
        self._messages.append(serializable_msg)

        return LLMResponse(message, finish_reason=finish_reason)


# -------------------------
# Helpers
# -------------------------
def _serialize_openai_message(msg: Any) -> Dict[str, Any]:
    """
    Convert the SDK message object into a plain dict so it can live in our messages list.
    This helps ensure consistent behavior across SDK versions.
    """
    out: Dict[str, Any] = {"role": getattr(msg, "role", "assistant")}
    content = getattr(msg, "content", None)
    if content is not None:
        out["content"] = content

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        # Normalize tool_calls to plain dicts
        norm_calls = []
        for tc in tool_calls:
            norm_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                    },
                }
            )
        out["tool_calls"] = norm_calls

    return out
