"""One shared budget, including in-flight requests and uncached input tokens."""
from __future__ import annotations

import threading
import math
from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    cost: float
    output_limit: int


class Budget:
    def __init__(self, limit: float):
        if limit < 0 or not math.isfinite(limit):
            raise ValueError("BUDGET must be finite and nonnegative")
        self.limit = float(limit)
        self.spent = 0.0
        self.reserved = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.limit - self.spent - self.reserved)

    def reserve(self, input_estimate: int, output_limit: int, task_remaining: float) -> Reservation:
        with self._lock:
            available = min(self.limit - self.spent - self.reserved, task_remaining)
            input_cost = input_estimate / 10
            output_limit = min(output_limit, int(available - input_cost))
            if output_limit < 96:
                raise BudgetExhausted("Insufficient remaining budget for this request")
            reservation = Reservation(input_cost + output_limit, output_limit)
            self.reserved += reservation.cost
            return reservation

    def settle(self, reservation: Reservation, usage: dict | None) -> float:
        """Reasoning is normally INCLUDED in completion_tokens; never double count it.

        Providers with a separate total can expose extra reasoning: total minus input
        is an additional lower bound for output. Without usage, charge the reservation.
        """
        with self._lock:
            self.reserved = max(0.0, self.reserved - reservation.cost)
            self.calls += 1
            try:
                if not isinstance(usage, dict) or not usage:
                    raise ValueError("Missing usage")
                incoming = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                outgoing = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                reasoning = int(details.get("reasoning_tokens", 0) or 0)
                separate_reasoning = int(usage.get("reasoning_tokens", 0) or 0)
                total = int(usage.get("total_tokens", 0) or 0)
                if min(incoming, outgoing, reasoning, separate_reasoning, total) < 0:
                    raise ValueError("Negative usage")
                if not total and separate_reasoning and not reasoning:
                    outgoing += separate_reasoning
                outgoing = max(outgoing, reasoning, separate_reasoning, total - incoming)
                cost = incoming / 10 + outgoing
                # Missing input or output counters cannot imply a free request.
                if not any(k in usage for k in ("prompt_tokens", "input_tokens")) or not any(
                    k in usage for k in ("completion_tokens", "output_tokens", "total_tokens")
                ):
                    cost = max(cost, reservation.cost)
                self.input_tokens += incoming
                self.output_tokens += outgoing
            except (ValueError, TypeError, AttributeError, OverflowError):
                cost = reservation.cost
            self.spent += cost
            return cost

    def snapshot(self) -> dict:
        with self._lock:
            return {"limit": self.limit, "spent": round(self.spent, 3),
                    "reserved": round(self.reserved, 3), "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens, "calls": self.calls}
