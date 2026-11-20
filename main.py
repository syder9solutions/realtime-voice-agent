"""Entrypoint: serve the ASGI app, print the latency config, or run evals."""

from __future__ import annotations

import argparse
import uvicorn

from src.models import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Voice Agent")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve", "config", "eval"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.command == "config":
        config = PipelineConfig()
        print(config.model_dump_json(indent=2))
        return
    if args.command == "eval":
        import asyncio

        from evals.voice_eval import run_evals

        results = asyncio.run(run_evals())
        print(results.render())
        raise SystemExit(0 if results.passed else 1)

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
