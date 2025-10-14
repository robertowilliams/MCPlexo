# bridge.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from mcp import Tool
from mcp.types import TextContent
from mcp.client.session import ClientSession

from mcp_llm_bridge.config import BridgeConfig
from mcp_llm_bridge.mcp_client import MCPClient
from mcp_llm_bridge.llm_client import LLMClient
from mcp_llm_bridge.tools import DatabaseQueryTool, build_local_db_tool_spec

logger = logging.getLogger(__name__)


class MCPLLMBridge:
    """
    Orchestrates the LLM <-> MCP flow:
      - connects to MCP server(s)
      - syncs MCP tools and exposes them to the LLM as OpenAI function tools
      - adds a local SQLite query tool (query_database) and routes those calls locally
      - runs the tool-call loop until a final assistant answer is produced
    """

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.mcp_client: Optional[MCPClient] = None
        self.llm_client: Optional[LLMClient] = None

        # Tool name mapping: OpenAI-safe name -> actual MCP/local tool name
        self.tool_name_mapping: Dict[str, str] = {}

        # Local DB tool (executed in-process, not via MCP)
        db_path = self.config.db_path or "test.db"
        self.query_tool = DatabaseQueryTool(db_path)

        # Cached OpenAI tool list (function schemas)
        self._openai_tools: List[Dict[str, Any]] = []

        # System prompt (optionally enriched with DB schema)
        self._system_prompt: Optional[str] = self.config.system_prompt

    # -------------------------
    # Lifecycle
    # -------------------------
    async def initialize(self) -> None:
        """
        Connect to MCP, list tools, build OpenAI function schemas (including local tool),
        and prime the LLM with tools and system prompt.
        """
        # 1) Connect to MCP server
        self.mcp_client = MCPClient(self.config.mcp_server_params)
        await self.mcp_client.connect()

        # 2) List MCP tools
        mcp_tools = await self._safe_list_mcp_tools()

        # 3) Add our local SQLite tool spec (so the LLM can discover/use it)
        local_db_tool_spec = build_local_db_tool_spec()
        combined_tools: List[Any] = list(mcp_tools) + [local_db_tool_spec]

        # 4) Convert tool specs to OpenAI "function" schemas
        self._openai_tools = self._convert_tools_to_openai_functions(combined_tools)

        # 5) Build system prompt (append DB schema summary)
        system_prompt = self._build_system_prompt_with_schema()

        # 6) Init LLM client and set tools/system prompt
        self.llm_client = LLMClient(self.config.llm_config)
        self.llm_client.set_tools(self._openai_tools)
        if system_prompt:
            self.llm_client.set_system_prompt(system_prompt)

        logger.info("Bridge initialized: %d tool(s) exposed to the LLM.", len(self._openai_tools))

    async def shutdown(self) -> None:
        """
        Cleanup resources.
        """
        try:
            if self.mcp_client:
                await self.mcp_client.close()
        except Exception as e:
            logger.warning("Error during MCP shutdown: %s", e)

    # -------------------------
    # Public API
    # -------------------------
    async def process_user_input(self, user_text: str) -> str:
        """
        Send a user message to the LLM, handle tool-calls if any, and return the final assistant text.
        """
        if not self.llm_client:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")

        # Add user message and invoke the model
        self.llm_client.add_user_message(user_text)
        response = await self.llm_client.invoke()

        # Loop while the LLM wants to call tools
        while getattr(response, "is_tool_call", False):
            logger.debug("Tool-calls requested: %s", response.tool_calls)
            tool_outputs = await self._handle_tool_calls(response.tool_calls)

            # Feed tool outputs back to the model
            for tc_id, name, output in tool_outputs:
                self.llm_client.add_tool_result(tool_call_id=tc_id, name=name, content=output)

            # Invoke again for the final (or next) assistant message
            response = await self.llm_client.invoke()

        final_text = response.content or ""
        return final_text

    # -------------------------
    # Internals
    # -------------------------
    async def _safe_list_mcp_tools(self) -> List[Tool]:
        """
        List MCP tools; tolerate servers that return non-standard payloads.
        """
        assert self.mcp_client is not None
        try:
            tools = await self.mcp_client.list_tools()
            return tools or []
        except Exception as e:
            logger.warning("Failed to list MCP tools; continuing with none. Error: %s", e)
            return []

    def _build_system_prompt_with_schema(self) -> Optional[str]:
        """
        Append a SQLite schema snippet to the configured system prompt to steer the LLM
        toward correct SQL usage.
        """
        base = (self._system_prompt or "").strip()
        schema_snippet = self.query_tool.generate_schema_summary()

        if schema_snippet:
            block = (
                f"{base}\n\n"
                "You also have access to a local SQLite database via the `query_database` tool.\n"
                "Here is the current schema summary to help you write correct SQL:\n"
                "----\n"
                f"{schema_snippet}\n"
                "----"
            ).strip()
            return block
        return base or None

    def _convert_tools_to_openai_functions(self, tools_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert a mix of MCP Tool objects and dict-based specs into OpenAI 'function' tools.
        Populates self.tool_name_mapping with OpenAI-safe names -> actual tool names.
        """
        openai_tools: List[Dict[str, Any]] = []
        self.tool_name_mapping.clear()

        for tool in tools_list:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # MCP Tool object
                name = getattr(tool, "name")
                description = getattr(tool, "description") or ""
                schema = getattr(tool, "inputSchema", {"type": "object", "properties": {}, "required": []})
            elif isinstance(tool, dict) and "name" in tool and "description" in tool:
                # Dict-based tool spec (e.g., our local DB tool)
                name = tool["name"]
                description = tool.get("description", "")
                schema = tool.get("inputSchema", {"type": "object", "properties": {}, "required": []})
            else:
                continue

            openai_name = self._sanitize_tool_name(str(name))
            self.tool_name_mapping[openai_name] = str(name)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": description,
                        "parameters": schema,
                    },
                }
            )

        return openai_tools

    async def _handle_tool_calls(self, tool_calls: List[Any]) -> List[Tuple[str, str, str]]:
        """
        Execute tool-calls and return a list of (tool_call_id, name, output_text).
        - Routes 'query_database' to the local DatabaseQueryTool
        - Routes other tools to the connected MCP server
        """
        results: List[Tuple[str, str, str]] = []
        assert self.mcp_client is not None

        for tc in tool_calls or []:
            # Each tool_call is expected to have .id and .function.{name, arguments}
            try:
                openai_name = tc.function.name
                args_json = tc.function.arguments or "{}"
                arguments = json.loads(args_json)
            except Exception:
                # Be defensive: tolerate dict-like structures
                openai_name = tc.get("function", {}).get("name") if isinstance(tc, dict) else None
                args_json = (tc.get("function", {}).get("arguments") if isinstance(tc, dict) else None) or "{}"
                arguments = json.loads(args_json)

            actual_name = self.tool_name_mapping.get(openai_name, openai_name)

            # Route local DB tool
            if actual_name == "query_database":
                output_text = await self._run_local_db_tool(arguments)
                results.append((tc.id, openai_name, output_text))
                continue

            # Else, call MCP tool
            try:
                mcp_result = await self.mcp_client.call_tool(actual_name, arguments)
                output_text = self._normalize_mcp_result(mcp_result)
            except Exception as e:
                logger.exception("MCP tool '%s' call failed: %s", actual_name, e)
                output_text = f"[tool_error] {type(e).__name__}: {e}"

            results.append((tc.id, openai_name, output_text))

        return results

    async def _run_local_db_tool(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the local SQLite tool and serialize the output rows as JSON.
        Expected arguments: { "query": "...", "params": { ... } }
        """
        try:
            rows = await self.query_tool.execute(arguments)
            return json.dumps(rows, ensure_ascii=False)
        except Exception as e:
            logger.exception("Local DB tool error: %s", e)
            return f"[tool_error] {type(e).__name__}: {e}"

    @staticmethod
    def _normalize_mcp_result(result: Any) -> str:
        """
        Convert various MCP SDK return shapes into a plain text string.
        """
        if result is None:
            return ""
        # Raw string
        if isinstance(result, str):
            return result

        # Result with 'content' list of TextContent
        content = getattr(result, "content", None)
        if isinstance(content, list):
            # Join text content pieces
            texts = []
            for c in content:
                if isinstance(c, TextContent):
                    texts.append(c.text or "")
                else:
                    # Fall back to stringification for unknown content parts
                    texts.append(str(getattr(c, "text", c)))
            return "\n".join(t for t in texts if t)

        # Fallback
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            return str(result)

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        """
        OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$.
        Normalize arbitrary tool names conservatively.
        """
        safe = []
        for ch in name:
            if ch.isalnum() or ch in {"_", "-"}:
                safe.append(ch)
            else:
                safe.append("_")
        joined = "".join(safe)
        return joined[:64] if joined else "tool"


class BridgeManager:
    """
    Async context manager around MCPLLMBridge.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.bridge = MCPLLMBridge(config)

    async def __aenter__(self) -> "BridgeManager":
        await self.bridge.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.bridge.shutdown()

    # Public facade used by main.py
    async def process_message(self, user_text: str) -> str:
        return await self.bridge.process_user_input(user_text)
