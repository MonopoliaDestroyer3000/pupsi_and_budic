"""Online cost calibration and a quantized 0/1 knapsack with optional retries."""
from __future__ import annotations

import math
import threading
from collections import defaultdict


class Planner:
    PRIORS = {"math": (850, .86), "search": (1400, .80),
              "data_analysis": (1800, .83), "coding": (3700, .72)}

    def __init__(self):
        self.observations = defaultdict(list)
        self._lock = threading.Lock()

    def estimate(self, task) -> tuple[float, float]:
        cost, probability = self.PRIORS.get(task.kind, (2400, .55))
        if task.prompt_bytes:
            # Expected input, unlike the conservative transport reservation, is
            # calibrated at four UTF-8 bytes per token. Output and calls vary by route.
            cost = (task.prompt_bytes / 40 + task.expected_output) * task.expected_calls
        # Broad size signal; do not read labels or request a paid planning call.
        size_factor = min(2.0, 1 + task.file_bytes / 120000 + len(task.instruction) / 10000)
        if not task.prompt_bytes:
            cost *= size_factor
        with self._lock:
            samples = list(self.observations[task.kind])
        if samples:
            weight = min(.75, len(samples) / (len(samples) + 4))
            measured = sum(x[0] for x in samples) / len(samples)
            cost = cost * (1 - weight) + max(200, measured) * weight
            probability = (probability * 4 + sum(x[1] for x in samples)) / (4 + len(samples))
        if task.attempts:
            cost *= 1.45
            probability *= .55
        if task.missing_evidence:
            probability *= .35
        return cost, probability

    def observe(self, kind: str, cost: float, solved: bool) -> None:
        with self._lock:
            self.observations[kind].append((cost, int(solved)))

    def select(self, tasks: list, budget: float, slots: int = 1) -> list:
        if budget < 100 or not tasks:
            return []
        # At most 2000 budget bins; ceil item costs, floor capacity preserves feasibility.
        unit = max(25, math.ceil(budget / 2000))
        capacity = int(budget // unit)
        dp = [(0.0, ()) for _ in range(capacity + 1)]
        estimates = {}
        for i, task in enumerate(tasks):
            cost, p = self.estimate(task)
            estimates[i] = (cost, p)
            bins = max(1, math.ceil(cost / unit))
            if bins > capacity:
                continue
            value = max(0, task.reward) * p
            for b in range(capacity, bins - 1, -1):
                candidate = dp[b - bins][0] + value
                if candidate > dp[b][0]:
                    dp[b] = (candidate, dp[b - bins][1] + (i,))
        chosen = max(dp, key=lambda x: x[0])[1]
        if not chosen:
            # A small residual can still buy one concise attempt.
            feasible = [i for i in range(len(tasks)) if budget >= 300]
            chosen = tuple(sorted(feasible, key=lambda i: tasks[i].reward * estimates[i][1] / estimates[i][0], reverse=True)[:1])
        order = sorted(chosen, key=lambda i: tasks[i].reward * estimates[i][1] / estimates[i][0], reverse=True)
        return [tasks[i] for i in order[:slots]]
