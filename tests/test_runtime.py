"""End-to-end persistence and model-loop tests using a scripted local proxy."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness.llm import ProxyError
from harness.runtime import Harness, discover, parse_action
import run_agent


LABEL = "DO-NOT-LEAK-EVALUATOR-LABEL-9f72"


class ScriptedProxy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages, model, max_tokens, task_remaining):
        self.messages.append(copy.deepcopy(messages))
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return json.dumps(action), 10.0


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        self.addCleanup(self.temp.cleanup)

    def task(self, name="task_1", kind="math", instruction="Find the missing value.", files=None):
        folder = self.tasks / name
        folder.mkdir()
        workspace = folder / "workspace"
        workspace.mkdir()
        metadata = {"task_id": name, "task_type": kind, "instruction": instruction, "reward": 7,
                    "expected": LABEL, "check": LABEL, "other_evaluator_metadata": {"secret": LABEL}}
        (folder / "task.json").write_text(json.dumps(metadata), encoding="utf-8")
        for path, content in (files or {}).items():
            target = workspace / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return folder, workspace

    def harness(self, actions=None, **options):
        harness = Harness(self.tasks, output=self.root / "solutions.json", **options)
        if actions is not None:
            harness.proxy = ScriptedProxy(actions)
        return harness

    def test_discover_excludes_labels_and_ignores_malformed_task(self):
        self.task()
        bad = self.tasks / "bad"
        bad.mkdir()
        (bad / "task.json").write_text('{"instruction":"x","reward":"NaN"}')
        tasks = discover(self.tasks)
        self.assertEqual(len(tasks), 1)
        self.assertNotIn(LABEL, repr(tasks[0]))
        self.assertFalse(hasattr(tasks[0], "expected"))

    def test_model_tool_loop_never_sees_labels_and_persists_answer_immediately(self):
        folder, workspace = self.task(files={"notes.txt": "The value is three."})
        harness = self.harness([
            {"tool": "read", "path": "notes.txt"},
            {"tool": "answer", "answer": "3"},
        ])
        solved, cost, evidence = harness.solve_llm(harness.tasks[0], 2000)
        self.assertTrue(solved, evidence)
        self.assertEqual(cost, 20)
        self.assertEqual((folder / "answer.txt").read_text(), "3\n")
        self.assertEqual((workspace / "answer.txt").read_text(), "3\n")
        report = json.loads((self.root / "solutions.json").read_text())
        self.assertEqual(report["results"]["task_1"]["answer"], "3")
        self.assertEqual(report["results"]["task_1"]["cost"], 20)
        self.assertNotIn(LABEL, json.dumps(harness.proxy.messages))
        self.assertNotIn("task.json", json.dumps(harness.proxy.messages))
        self.assertFalse(list(folder.glob(".answer-*")))

    def test_coding_answer_is_rejected_until_visible_tests_pass(self):
        folder, workspace = self.task(kind="coding", files={
            "mylib.py": "def custom(value):\n    return 0\n",
            "test_mylib.py": "import unittest\nimport mylib\nclass T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(mylib.custom(2), 4)\n",
        })
        harness = self.harness([{"tool": "answer", "answer": "DONE"}, {"tool": "skip", "reason": "cannot repair"}])
        solved, _, evidence = harness.solve_llm(harness.tasks[0], 2000)
        self.assertFalse(solved)
        self.assertIn("abstained", evidence)
        self.assertFalse((folder / "answer.txt").exists())
        self.assertFalse((workspace / "answer.txt").exists())
        self.assertNotIn("task_1", harness.results)

    def test_coding_write_then_real_test_passes_and_persists_done(self):
        folder, workspace = self.task(kind="coding", files={
            "mylib.py": "def custom(value):\n    return 0\n",
            "test_mylib.py": "import unittest\nimport mylib\nclass T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(mylib.custom(2), 4)\n",
        })
        before = (workspace / "test_mylib.py").read_bytes()
        harness = self.harness([
            {"tool": "write", "path": "mylib.py", "content": "def custom(value):\n    return value * 2\n"},
            {"tool": "test"},
        ])
        solved, _, evidence = harness.solve_llm(harness.tasks[0], 2000)
        self.assertTrue(solved, evidence)
        self.assertEqual((folder / "answer.txt").read_text(), "DONE\n")
        self.assertEqual((workspace / "test_mylib.py").read_bytes(), before)

    def test_failed_attempt_rolls_back_source_data_and_new_files(self):
        _, workspace = self.task(files={"data.csv": "value\n3\n", "mylib.py": "VALUE = 3\n"})
        originals = {p.name: p.read_bytes() for p in workspace.iterdir()}
        harness = self.harness([
            {"tool": "write", "path": "data.csv", "content": "value\n999\n"},
            {"tool": "write", "path": "mylib.py", "content": "VALUE = 999\n"},
            {"tool": "write", "path": "new.txt", "content": "wrong scratch data"},
            {"tool": "skip", "reason": "missing fact"},
        ])
        solved, _, _ = harness.solve_llm(harness.tasks[0], 2000)
        self.assertFalse(solved)
        self.assertEqual({p.name: p.read_bytes() for p in workspace.iterdir()}, originals)

    def test_failed_proxy_cost_is_returned_to_planner(self):
        self.task()
        harness = self.harness([ProxyError("Proxy HTTP 429", 123)])
        solved, cost, evidence = harness.solve_llm(harness.tasks[0], 2000)
        self.assertFalse(solved)
        self.assertEqual(cost, 123)
        self.assertEqual(evidence, "Proxy HTTP 429")

    def test_zero_budget_still_solves_free_task_and_abstains_on_unknown(self):
        folder, _ = self.task("free", instruction="A warehouse starts with 10 boxes. It receives 4 boxes. How many boxes does it have at the end?")
        unknown, _ = self.task("unknown", instruction="Find the undisclosed secret password.")
        harness = self.harness(budget=0, deterministic_only=True)
        result = harness.run()
        self.assertEqual((folder / "answer.txt").read_text().strip(), "14")
        self.assertFalse((unknown / "answer.txt").exists())
        self.assertEqual(result["budget"]["spent"], 0)
        self.assertEqual(result["budget"]["calls"], 0)
        self.assertEqual(result["submitted"], 1)

    def test_cli_uses_varied_budget_environment_and_override(self):
        for value in ("0", "1234.5", "100000"):
            with self.subTest(value=value), patch.dict(os.environ, {"BUDGET": value}), patch("run_agent.Harness") as constructor:
                constructor.return_value.run.return_value = {"ok": True}
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_agent.main(["--tasks-dir", str(self.tasks), "--without-framework"]), 0)
                self.assertEqual(constructor.call_args.kwargs["budget"], float(value))
        with patch.dict(os.environ, {"BUDGET": "0"}), patch("run_agent.Harness") as constructor:
            constructor.return_value.run.return_value = {}
            with contextlib.redirect_stdout(io.StringIO()):
                run_agent.main(["--tasks-dir", str(self.tasks), "--budget", "77", "--without-framework"])
            self.assertEqual(constructor.call_args.kwargs["budget"], 77)

    def test_cli_rejects_invalid_budget_environment(self):
        for value in ("nan", "inf", "-1", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"BUDGET": value}), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_agent.main(["--tasks-dir", str(self.tasks), "--without-framework"])
                self.assertEqual(error.exception.code, 2)

    def test_action_parser_accepts_json_fence_and_rejects_generated_code(self):
        self.assertEqual(parse_action('```json\n{"tool":"answer","answer":"3"}\n```')["answer"], "3")
        with self.assertRaises(ValueError):
            parse_action("import os; os.system('echo no')")


if __name__ == "__main__":
    unittest.main()
