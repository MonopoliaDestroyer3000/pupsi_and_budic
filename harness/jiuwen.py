"""Run the competition scheduler as a real openJiuwen custom agent.

The framework owns invocation, callbacks and session lifecycle; the synchronous
callback owns admission control, tools, persistence and the shared token ledger.
No framework model client is created, so every paid call uses the one budgeted
proxy client in the harness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


def _agent_class():
    # Configure before importing BaseAgent: imports themselves can emit logs.
    # The judging container mounts /app read-only, so file sinks are disabled.
    from openjiuwen.core.common.logging.log_config import configure_log_config

    configure_log_config(
        {
            "backend": "default",
            "level": "WARNING",
            "output": ["console"],
            "interface_output": ["console"],
            "performance_output": ["console"],
        }
    )
    from openjiuwen.core.single_agent import AgentCard, BaseAgent

    class BudgetAgent(BaseAgent):
        def __init__(self, callback: Callable[[dict[str, Any]], Any]):
            super().__init__(
                AgentCard(
                    id="budget-aware-harness",
                    name="BudgetAwareHarness",
                    description="Select, solve, verify and persist tasks within a shared token budget.",
                )
            )
            self._callback = callback

        def configure(self, config):
            self._config = config
            return self

        async def invoke(self, inputs, session=None):
            return await asyncio.to_thread(self._callback, inputs)

        async def stream(self, inputs, session=None, stream_modes=None):
            yield await self.invoke(inputs, session=session)

    return BudgetAgent


async def invoke_harness(
    callback: Callable[[dict[str, Any]], Any], inputs: dict[str, Any]
) -> Any:
    """Invoke a synchronous scheduler through openJiuwen's Runner and session."""
    BudgetAgent = _agent_class()
    from openjiuwen.core.runner import Runner

    agent = BudgetAgent(callback)
    return await Runner.run_agent(agent, inputs)
