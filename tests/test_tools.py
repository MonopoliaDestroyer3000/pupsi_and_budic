import json
from pathlib import Path
import tempfile
import unittest

from harness.tools import LocalTools, safe_path
from harness.runtime import Task
from harness.runtime import unavailable_evidence
from harness.planner import Planner


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.tools = LocalTools(self.root, max_seconds=2)

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_computation_and_imports(self):
        (self.root / "sales.csv").write_text("units,price\n2,3\n4,5\n")
        result = self.tools.execute({"tool": "python", "code": "import csv, statistics\nwith open('sales.csv') as f:\n print(sum(int(r['units']) for r in csv.DictReader(f)))"})
        self.assertIn("Exit 0\n6", result)

    def test_io_cannot_read_parent_labels(self):
        (self.root.parent / "task.json").write_text('{"expected":"SECRET_LABEL"}')
        result = self.tools.execute({"tool": "python", "code": "import io\nprint(io.open('../task.json').read())"})
        self.assertNotIn("SECRET_LABEL", result)
        self.assertNotIn("Exit 0", result)

    def test_computation_is_read_only(self):
        result = self.tools.execute({"tool": "python", "code": "import io\nio.open('bad.txt','w').write('changed')"})
        self.assertNotIn("Exit 0", result)
        self.assertFalse((self.root / "bad.txt").exists())

    def test_write_rollback_covers_data_and_new_files(self):
        (self.root / "data.csv").write_text("original")
        self.tools.execute({"tool": "write", "path": "data.csv", "content": "wrong"})
        self.tools.execute({"tool": "write", "path": "new.py", "content": "x=1"})
        self.tools.rollback()
        self.assertEqual((self.root / "data.csv").read_text(), "original")
        self.assertFalse((self.root / "new.py").exists())

    def test_test_directory_and_path_escape_protected(self):
        for path in ("../outside.txt", "tests/helper.py", "test_mylib.py", "answer.txt", "task.json"):
            result = self.tools.execute({"tool": "write", "path": path, "content": "x=1"})
            self.assertTrue(result.startswith("Tool error:"), (path, result))

    def test_infinite_computation_times_out(self):
        self.tools.max_seconds = .2
        self.assertIn("time limit", self.tools.execute({"tool": "python", "code": "while True: pass"}))


class PlannerTests(unittest.TestCase):
    def task(self, name, reward, kind="math"):
        return Task(name, kind, "question", reward, Path("."), Path("."))

    def test_knapsack_prefers_sum_of_two_cheaper_tasks(self):
        planner = Planner()
        tasks = [self.task("large", 10, "coding"), self.task("a", 6), self.task("b", 6)]
        picked = planner.select(tasks, 4000, 5)
        self.assertEqual({t.task_id for t in picked}, {"a", "b"})

    def test_budget_zero_and_concurrency_bound(self):
        planner = Planner()
        tasks = [self.task(str(i), i+1) for i in range(10)]
        self.assertEqual(planner.select(tasks, 0, 5), [])
        self.assertEqual(len(planner.select(tasks, 10000, 3)), 3)

    def test_missing_data_is_distinguished_from_unsupported_grammar(self):
        task = self.task("csv", 3, "data_analysis")
        task.evidence = "missing or unsafe workspace file"
        self.assertTrue(unavailable_evidence(task))
        task.evidence = "unsupported CSV aggregate"
        self.assertFalse(unavailable_evidence(task))


if __name__ == "__main__":
    unittest.main()
