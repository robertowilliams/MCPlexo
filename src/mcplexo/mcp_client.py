# mcp_client.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from mcp import StdioServerParameters, Tool
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Thin async wrapper around an MCP stdio server.

    Lifecycle:
      - connect(): start the stdio transport and initialize a ClientSession
      - list_tools(): fetch available tools from the server
      - call_tool(name, arguments): invoke a tool by name with JSON-able args
      - close(): cleanly tear down the session and transport
    """

    def __init__(self, params: StdioServerParameters) -> None:
        self.params = params

        # Context managers we manually enter/exit to control lifetime outside a 'with' block
        self._stdio_cm = None
        self._session_cm = None

        # Active transport and session objects once connected
        self._read = None
        self._write = None
        self._session: Optional[ClientSession] = None

        # Track connection state
        self._connected: bool = False

    # -------------------------
    # Lifecycle
    # -------------------------
    async def connect(self, *, initialize_timeout: float = 30.0) -> None:
        """
        Start the stdio transport to the MCP server and initialize a ClientSession.
        """
        if self._connected:
            logger.debug("MCPClient.connect() called but already connected.")
            return

        logger.info("Starting MCP stdio server: %s %s", self.params.command, " ".join(self.params.args or []))

        # Enter stdio transport context
        self._stdio_cm = stdio_client(self.params)
        self._read, self._write = await self._stdio_cm.__aenter__()  # type: ignore[assignment]

        # Enter session context
        self._session_cm = ClientSession(self._read, self._write)
        self._session = await self._session_cm.__aenter__()  # type: ignore[assignment]

        # Initialize handshake with an optional timeout
        try:
            await asyncio.wait_for(self._session.initialize(), timeout=initialize_timeout)
        except asyncio.TimeoutError:
            await self._teardown_on_error()
            raise TimeoutError("Timed out initializing MCP ClientSession with the server.")
        except Exception as e:
            await self._teardown_on_error()
            raise RuntimeError(f"Failed to initialize MCP ClientSession: {e}") from e

        self._connected = True
        logger.info("MCP session initialized successfully.")

    async def close(self) -> None:
        """
        Cleanly close session and transport contexts.
        """
        if not self._connected:
            # Still attempt to exit contexts if they were partially opened
            await self._safe_exit_contexts()
            return

        await self._safe_exit_contexts()
        self._connected = False
        logger.info("MCP session closed.")

    # -------------------------
    # Public API
    # -------------------------
    async def list_tools(self) -> List[Tool]:
        """
        List tools from the MCP server. Returns an empty list on failure.
        """
        if not self._session:
            raise RuntimeError("MCPClient is not connected. Call connect() first.")

        try:
            tools = await self._session.list_tools()
            # Some servers may return None or a non-list; normalize
            return list(tools or [])
        except Exception as e:
            logger.exception("Error listing MCP tools: %s", e)
            return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool by name on the MCP server with JSON-able arguments.
        Returns the raw MCP result (often an object with .content).
        """
        if not self._session:
            raise RuntimeError("MCPClient is not connected. Call connect() first.")

        try:
            result = await self._session.call_tool(name, arguments or {})
            return result
        except Exception as e:
            logger.exception("Error calling MCP tool '%s': %s", name, e)
            raise

    # -------------------------
    # Internals
    # -------------------------
    async def _teardown_on_error(self) -> None:
        """
        Best-effort cleanup used when connect/initialize fails.
        """
        try:
            await self._safe_exit_contexts()
        finally:
            self._connected = False

    async def _safe_exit_contexts(self) -> None:
        """
        Exit session and stdio contexts in the right order, tolerating partial setup.
        """
        # Exit session first
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Ignoring session __aexit__ error: %s", e)
            finally:
                self._session_cm = None
                self._session = None

        # Then exit stdio transport
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Ignoring stdio __aexit__ error: %s", e)
            finally:
                self._stdio_cm = None
                self._read = None
                self._write = None
