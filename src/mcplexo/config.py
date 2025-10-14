# config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

from mcp import StdioServerParameters


@dataclass
class LLMConfig:
    """
    Configuration for the OpenAI-compatible LLM endpoint.
    """
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Optional tuning knobs (unused by default but handy to keep around)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None  # passthrough bag for future use


@dataclass
class BridgeConfig:
    """
    End-to-end configuration for the MCP <-> LLM bridge.
    """
    mcp_server_params: StdioServerParameters
    llm_config: LLMConfig

    # Optional system prompt to seed the assistant with behavior/instructions.
    system_prompt: Optional[str] = None

    # Optional SQLite DB path used by the local query tool. If provided,
    # this should match the DB used by the MCP SQLite server for consistency.
    db_path: Optional[str] = None
