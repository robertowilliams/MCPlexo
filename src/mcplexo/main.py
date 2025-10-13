import os
import asyncio
import logging
import colorlog
import yaml
from dotenv import load_dotenv
from pathlib import Path
from mcp import StdioServerParameters
from mcplexo.config import BridgeConfig, LLMConfig
from mcplexo.bridge import BridgeManager

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s: %(cyan)s%(name)s%(reset)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%'
))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helper functions to load configuration
# ---------------------------------------------------------------------------
def load_yaml_config():
    """Load non-sensitive configuration values from config.yaml."""
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

def load_env():
    """Load sensitive values like API keys from .env."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(env_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Load configurations
    load_env()
    yaml_conf = load_yaml_config()

    # Extract database path (or default to root/test.db)
    project_root = Path(__file__).resolve().parents[2]
    db_path = yaml_conf.get("database", {}).get("path", str(project_root / "test.db"))

    # Extract LLM configuration from YAML
    model_name = yaml_conf.get("llm", {}).get("model", "gpt-4o")
    base_url = yaml_conf.get("llm", {}).get("base_url", None)

    # Load API key securely from .env
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        logger.warning("No API key found in environment. Please set OPENAI_API_KEY in .env.")

    # Configure the bridge
    config = BridgeConfig(
        mcp_server_params=StdioServerParameters(
            command=yaml_conf.get("mcp_server", {}).get("command", "uvx"),
            args=yaml_conf.get("mcp_server", {}).get("args", ["mcp-server-sqlite", "--db-path", db_path]),
            env=None
        ),
        llm_config=LLMConfig(
            api_key=api_key,
            model=model_name,
            base_url=base_url
        ),
        system_prompt=yaml_conf.get("system_prompt", "You are a helpful assistant that can use tools to help answer questions.")
    )

    logger.info(f"Starting MCPlexo bridge with model: {config.llm_config.model}")
    logger.info(f"Database path: {db_path}")

    # Run bridge interaction loop
    async with BridgeManager(config) as bridge:
        while True:
            try:
                user_input = input("\nEnter your prompt (or 'quit' to exit): ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                response = await bridge.process_message(user_input)
                print(f"\nResponse: {response}")

            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error occurred: {e}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
