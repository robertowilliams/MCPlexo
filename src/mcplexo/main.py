# main.py
import os
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from mcp import StdioServerParameters

# Fixed: import from the actual package/module name
from mcp_llm_bridge.config import BridgeConfig, LLMConfig
from mcp_llm_bridge.bridge import BridgeManager


def load_yaml_config() -> Dict[str, Any]:
    """
    Load YAML config from:
      1) MCPLEXO_CONFIG (absolute/relative path), or
      2) ./config.yaml in the current working directory, or
      3) ./config/config.yaml (fallback).
    """
    override = os.getenv("MCPLEXO_CONFIG")
    candidates = []

    if override:
        candidates.append(Path(override).expanduser().resolve())

    # Common local defaults
    candidates.append((Path.cwd() / "config.yaml").resolve())
    candidates.append((Path.cwd() / "config" / "config.yaml").resolve())

    for path in candidates:
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    raise FileNotFoundError(
        "No config YAML found. Set MCPLEXO_CONFIG or place config.yaml in the project root."
    )


def build_configs(yaml_conf: Dict[str, Any]) -> BridgeConfig:
    """
    Build LLM and Bridge configs from YAML + environment.
    """
    # Load env for API keys, etc.
    load_dotenv()

    # ----- LLM config -----
    llm_section = yaml_conf.get("llm", {}) or {}
    llm_model = llm_section.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o")
    llm_base_url = llm_section.get("base_url") or os.getenv("OPENAI_BASE_URL")
    llm_api_key = llm_section.get("api_key") or os.getenv("OPENAI_API_KEY")

    if not llm_api_key:
        print(
            "[WARN] No API key provided. Set OPENAI_API_KEY env var or llm.api_key in YAML.",
            file=sys.stderr,
        )

    llm_cfg = LLMConfig(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
    )

    # ----- MCP server (stdio) -----
    mcp_section = yaml_conf.get("mcp_server", {}) or {}
    command = mcp_section.get("command") or "uvx"
    args = mcp_section.get("args") or [
        "mcp-server-sqlite",
        "--db-path",
        "./test.db",
    ]
    env = mcp_section.get("env") or {}

    stdio_params = StdioServerParameters(command=command, args=args, env=env)

    # ----- Database path (to keep local tool and MCP server in sync) -----
    db_section = yaml_conf.get("database", {}) or {}
    db_path = db_section.get("path")

    # ----- System prompt (optional) -----
    system_prompt = yaml_conf.get("system_prompt")

    return BridgeConfig(
        mcp_server_params=stdio_params,
        llm_config=llm_cfg,
        system_prompt=system_prompt,
        db_path=db_path,  # <— NEW: pass through to local DB tool
    )


async def interactive_loop(manager: BridgeManager) -> None:
    """
    Simple REPL: read from stdin, send to the bridge, print the assistant response.
    Type 'exit' or Ctrl-D to quit.
    """
    print("MCP LLM Bridge ready. Type your message (or 'exit' to quit).")
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            print()
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        try:
            response = await manager.process_message(user)
            # Print final assistant content
            print(response or "")
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)


async def async_main() -> None:
    yaml_conf = load_yaml_config()
    bridge_conf = build_configs(yaml_conf)

    # Spin up the bridge (connects to MCP, syncs tools, etc.)
    async with BridgeManager(bridge_conf) as manager:
        await interactive_loop(manager)


def main() -> None:
    try:
        asyncio.run(async_main())
    except FileNotFoundError as e:
        print(f"[Config] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[Fatal] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
