"""Budget and proxy regression tests; no external model calls are made."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import threading
import unittest
from unittest.mock import patch
import urllib.error

from harness.budget import Budget, BudgetExhausted
from harness.llm import ALLOWED_MODELS, Proxy, ProxyError


class BudgetTests(unittest.TestCase):
    def test_finite_nonnegative_budget(self):
        for value in (-1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Budget(value)
        self.assertEqual(Budget(0).remaining, 0)
        with self.assertRaises(BudgetExhausted):
            Budget(0).reserve(100, 100, 1000)

    def test_inflight_reservations_are_shared_between_threads(self):
        budget = Budget(1000)
        barrier = threading.Barrier(16)

        def reserve():
            barrier.wait(timeout=5)
            try:
                return budget.reserve(100, 120, 1000)
            except BudgetExhausted:
                return None

        with ThreadPoolExecutor(max_workers=16) as workers:
            reservations = [r for r in workers.map(lambda _: reserve(), range(16)) if r is not None]
        self.assertGreater(len(reservations), 1)
        self.assertLess(len(reservations), 16)
        self.assertLessEqual(sum(r.cost for r in reservations), 1000)
        self.assertAlmostEqual(budget.reserved, sum(r.cost for r in reservations))
        for reservation in reservations:
            self.assertEqual(budget.settle(reservation, {"prompt_tokens": 100, "completion_tokens": 12}), 22)
        self.assertAlmostEqual(budget.reserved, 0)
        self.assertEqual(budget.spent, len(reservations) * 22)
        self.assertEqual(budget.remaining, 1000 - len(reservations) * 22)

    def test_uncached_input_and_included_reasoning_are_charged_once(self):
        budget = Budget(1000)
        reservation = budget.reserve(500, 400, 1000)
        usage = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                 "prompt_tokens_details": {"cached_tokens": 100},
                 "completion_tokens_details": {"reasoning_tokens": 30}}
        self.assertEqual(budget.settle(reservation, usage), 50)
        self.assertEqual(budget.snapshot()["input_tokens"], 100)
        self.assertEqual(budget.snapshot()["output_tokens"], 40)

    def test_reasoning_reported_as_separate_field_is_charged(self):
        for usage in (
            {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 15},
            {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 15, "total_tokens": 135},
        ):
            with self.subTest(usage=usage):
                budget = Budget(1000)
                reservation = budget.reserve(500, 400, 1000)
                self.assertEqual(budget.settle(reservation, usage), 45)

    def test_missing_partial_or_malformed_usage_cannot_be_free(self):
        for usage in (None, {}, [], "invalid", {"completion_tokens": 1},
                      {"prompt_tokens": "oops", "completion_tokens": 1},
                      {"prompt_tokens": 1, "completion_tokens": 1, "completion_tokens_details": [1]},
                      {"prompt_tokens": -100, "completion_tokens": 0}):
            with self.subTest(usage=usage):
                budget = Budget(1000)
                reservation = budget.reserve(100, 200, 1000)
                self.assertGreaterEqual(budget.settle(reservation, usage), reservation.cost)
                self.assertEqual(budget.reserved, 0)
                self.assertEqual(budget.calls, 1)

    def test_actual_provider_overrun_stops_further_requests(self):
        budget = Budget(300)
        reservation = budget.reserve(100, 200, 300)
        self.assertEqual(budget.settle(reservation, {"prompt_tokens": 5000, "completion_tokens": 200}), 700)
        self.assertEqual(budget.remaining, 0)
        with self.assertRaises(BudgetExhausted):
            budget.reserve(1, 100, 1000)


class ProxyTests(unittest.TestCase):
    def proxy(self):
        budget = Budget(10000)
        return Proxy("https://organizer.invalid/v1/", "private-test-key", budget), budget

    def call_with_result(self, result, proxy=None):
        proxy, budget = self.proxy() if proxy is None else (proxy, proxy.budget)
        raw = json.dumps(result).encode()
        with patch("harness.llm.urllib.request.build_opener") as build:
            build.return_value.open.return_value = io.BytesIO(raw)
            answer = proxy.complete([{"role": "user", "content": "hello"}], ALLOWED_MODELS[0], 300, 1000)
        return answer, budget, build

    def test_only_approved_model_and_configured_endpoint(self):
        proxy, budget = self.proxy()
        self.assertEqual(proxy.url, "https://organizer.invalid/v1/chat/completions")
        with self.assertRaises(ValueError):
            proxy.complete([], "unapproved-model", 100, 1000)
        self.assertEqual(budget.calls, 0)
        self.assertEqual(budget.reserved, 0)

    def test_success_settles_uncached_usage_and_auth_stays_in_header(self):
        result = {"choices": [{"message": {"content": "answer"}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        (content, cost), budget, build = self.call_with_result(result)
        self.assertEqual((content, cost), ("answer", 30))
        self.assertEqual(budget.spent, 30)
        self.assertEqual(budget.reserved, 0)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
        self.assertNotIn(b"private-test-key", request.data)

    def test_missing_usage_uses_conservative_reservation(self):
        (content, cost), budget, _ = self.call_with_result({"choices": [{"message": {"content": "answer"}}]})
        self.assertEqual(content, "answer")
        self.assertGreaterEqual(cost, 300)
        self.assertEqual(budget.spent, cost)
        self.assertEqual(budget.reserved, 0)

    def test_transport_failure_charged_and_error_hides_secret_response(self):
        proxy, budget = self.proxy()
        error = urllib.error.HTTPError(proxy.url, 429, "private-test-key", {}, io.BytesIO(b"private-test-key"))
        with patch("harness.llm.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaises(ProxyError) as raised:
                proxy.complete([], ALLOWED_MODELS[0], 300, 1000)
        self.assertEqual(str(raised.exception), "Proxy HTTP 429")
        self.assertGreater(raised.exception.cost, 0)
        self.assertEqual(budget.spent, raised.exception.cost)
        self.assertEqual(budget.reserved, 0)

    def test_malformed_json_response_charged(self):
        proxy, budget = self.proxy()
        with patch("harness.llm.urllib.request.build_opener") as build:
            build.return_value.open.return_value = io.BytesIO(b"not json")
            with self.assertRaises(ProxyError) as raised:
                proxy.complete([], ALLOWED_MODELS[0], 300, 1000)
        self.assertGreater(raised.exception.cost, 0)
        self.assertEqual(budget.spent, raised.exception.cost)
        self.assertEqual(budget.reserved, 0)

    def test_unusable_completion_shapes_still_settle_and_raise_proxy_error(self):
        for result in ([], {}, {"choices": []}, {"choices": [{"message": []}]}):
            with self.subTest(result=result):
                proxy, budget = self.proxy()
                with self.assertRaises(ProxyError) as raised:
                    self.call_with_result(result, proxy)
                self.assertGreater(raised.exception.cost, 0)
                self.assertEqual(budget.spent, raised.exception.cost)
                self.assertEqual(budget.reserved, 0)


if __name__ == "__main__":
    unittest.main()
