#!/usr/bin/env python3
"""Organizer entry point: python /app/run_agent.py --tasks-dir /data/tasks."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys

from harness.llm import ALLOWED_MODELS
from harness.runtime import Harness


def main(argv=None):
    parser = argparse.ArgumentParser(description="Budget-aware openJiuwen coding agent")
    parser.add_argument("--tasks-dir", type=Path, default=Path(os.environ.get("TASKS", "/data/tasks")))
    parser.add_argument("--output-file", "--output", type=Path, default=None)
    parser.add_argument("--budget", type=float, default=None, help="Defaults to BUDGET environment variable or 100000")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("AGENT_CONCURRENCY", "3")))
    parser.add_argument("--time-limit", type=float, default=float(os.environ.get("TIME_LIMIT", "3450")))
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=os.environ.get("AGENT_MODEL", ALLOWED_MODELS[0]))
    parser.add_argument("--reasoning-model", choices=ALLOWED_MODELS, default=os.environ.get("REASONING_MODEL", "qwen3.8-27b"))
    parser.add_argument("--deterministic-only", action="store_true", help="No proxy calls; offline benchmark")
    parser.add_argument("--resume", action="store_true", help="Reuse previously persisted answers")
    parser.add_argument("--without-framework", action="store_true", help="Offline testing only; submission uses openJiuwen")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    if not args.tasks_dir.is_dir():
        parser.error("Task directory does not exist")
    try:
        budget = args.budget if args.budget is not None else float(os.environ.get("BUDGET", "100000"))
        if budget < 0 or not __import__("math").isfinite(budget):
            raise ValueError()
    except ValueError:
        parser.error("BUDGET must be a finite nonnegative number")
    output = args.output_file
    if output is None and os.environ.get("OUT"):
        output = Path(os.environ["OUT"])
        if output.is_dir() or not output.suffix:
            output = output / "solutions.json"
    options = dict(tasks_dir=args.tasks_dir, budget=budget,
                   base_url=os.environ.get("LLM_BASE_URL", ""), api_key=os.environ.get("LLM_API_KEY", ""),
                   output=output, concurrency=args.concurrency, time_limit=min(args.time_limit, 3500),
                   deterministic_only=args.deterministic_only, resume=args.resume,
                   model=args.model, reasoning_model=args.reasoning_model)
    if args.without_framework:
        result = Harness(**options).run()
    else:
        from harness.jiuwen import invoke_harness
        result = asyncio.run(invoke_harness(lambda _: Harness(**options).run(), {}))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
