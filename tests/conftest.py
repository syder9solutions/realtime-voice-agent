"""Make repo root importable without PYTHONPATH (pytest rootdir is not always on sys.path)."""

import asyncio
import socket
import sys
from pathlib import Path

import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import pipelines  # noqa: E402


@pytest.fixture()
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
async def live_server(free_port):
    """Real uvicorn server so WebSocket tests exercise the actual ASGI transport."""
    config = uvicorn.Config("src.api:app", host="127.0.0.1", port=free_port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    pipelines.clear()
    try:
        yield f"127.0.0.1:{free_port}"
    finally:
        server.should_exit = True
        await task
        pipelines.clear()
