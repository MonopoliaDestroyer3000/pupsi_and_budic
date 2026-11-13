from pathlib import Path
import os
import shutil
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from harness.coding import MAX_TEST_OUTPUT, run_tests, solve_coding


class CodingRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, content):
        (self.workspace / name).write_text(textwrap.dedent(content), encoding="utf-8")

    def test_all_visible_sample_coding_tasks(self):
        samples = Path(__file__).resolve().parents[1] / "task-dir"
        if not samples.is_dir():
            samples = Path(__file__).resolve().parent / "fixtures"
        for name in ("coding_0003", "coding_0007"):
            with self.subTest(name=name):
                folder = self.workspace / name
                shutil.copytree(samples / name / "workspace", folder)
                tests_before = (folder / "test_mylib.py").read_bytes()
                answer, evidence = solve_coding(folder, "Fix mylib.py", timeout=10)
                self.assertEqual(answer, "DONE", evidence)
                self.assertTrue(run_tests(folder)[0])
                self.assertEqual(tests_before, (folder / "test_mylib.py").read_bytes())
                self.assertFalse((folder / "answer.txt").exists())

    def test_failed_repair_restores_exact_original_bytes(self):
        original = b"def total(values):\r\n    return 0\r\n"
        (self.workspace / "mylib.py").write_bytes(original)
        self.write("test_mylib.py", """
            import unittest
            import mylib as m
            class Test(unittest.TestCase):
                def test_total(self):
                    self.assertEqual(m.total([1, 2]), 10)
        """)
        answer, evidence = solve_coding(self.workspace, "Fix mylib.py")
        self.assertIsNone(answer)
        self.assertIn("restored", evidence)
        self.assertEqual((self.workspace / "mylib.py").read_bytes(), original)

    def test_unknown_contract_is_not_guessed(self):
        self.write("mylib.py", "def frobnicate(x):\n    return 0\n")
        self.write("test_mylib.py", """
            import unittest
            from mylib import frobnicate
            class Test(unittest.TestCase):
                def test_custom(self):
                    self.assertEqual(frobnicate(3), 9)
        """)
        before = (self.workspace / "mylib.py").read_bytes()
        answer, _ = solve_coding(self.workspace, "Fix mylib.py")
        self.assertIsNone(answer)
        self.assertEqual(before, (self.workspace / "mylib.py").read_bytes())

    def test_direct_import_and_preserved_extra_function(self):
        self.write("helpers.py", """
            def count_even(values):
                return sum(1 for v in values if v % 2)

            def custom(value):
                return value + 10
        """)
        self.write("test_helpers.py", """
            import unittest
            from helpers import count_even as count
            class Test(unittest.TestCase):
                def test_count(self):
                    self.assertEqual(count([-2, -1, 0, 2, 3]), 3)
        """)
        answer, evidence = solve_coding(self.workspace, "Fix helpers.py")
        self.assertEqual(answer, "DONE", evidence)
        self.assertIn("return value + 10", (self.workspace / "helpers.py").read_text())

    def test_already_correct_and_zero_discovery(self):
        self.write("test_empty.py", "# no tests\n")
        self.assertFalse(run_tests(self.workspace)[0])
        self.write("test_empty.py", """
            import unittest
            class Test(unittest.TestCase):
                def test_ok(self):
                    self.assertTrue(True)
        """)
        self.assertEqual(solve_coding(self.workspace, "run tests")[0], "DONE")

    def test_contradictory_pure_assertions(self):
        self.write("mylib.py", "def total(values):\n    return sum(values)\n")
        self.write("test_mylib.py", """
            import unittest
            import mylib as m
            class Test(unittest.TestCase):
                def test_a(self):
                    self.assertEqual(m.total([1, 2]), 3)
                def test_b(self):
                    self.assertEqual(m.total([1, 2]), 4)
        """)
        answer, evidence = solve_coding(self.workspace, "Fix mylib.py")
        self.assertIsNone(answer)
        self.assertIn("contradictory literal assertions", evidence)

    def test_hanging_tests_are_bounded(self):
        self.write("test_loop.py", """
            import unittest
            class Test(unittest.TestCase):
                def test_hang(self):
                    while True:
                        pass
        """)
        passed, evidence = run_tests(self.workspace, timeout=0.2)
        self.assertFalse(passed)
        self.assertIn("timeout", evidence)

    def test_output_is_capped(self):
        self.write("test_noisy.py", """
            import unittest
            class Test(unittest.TestCase):
                def test_noise(self):
                    print('x' * 200000)
        """)
        passed, evidence = run_tests(self.workspace)
        self.assertFalse(passed)
        self.assertIn("output limit", evidence)
        self.assertLess(len(evidence), MAX_TEST_OUTPUT + 100)

    def test_subprocess_does_not_inherit_credentials(self):
        self.write("test_environment.py", """
            import os
            import unittest
            class Test(unittest.TestCase):
                def test_environment(self):
                    self.assertNotIn('LLM_API_KEY', os.environ)
                    self.assertNotIn('UNRELATED_PRIVATE_TOKEN', os.environ)
        """)
        with patch.dict(os.environ, {"LLM_API_KEY": "test-value", "UNRELATED_PRIVATE_TOKEN": "test-value"}):
            passed, evidence = run_tests(self.workspace)
        self.assertTrue(passed, evidence)


if __name__ == "__main__":
    unittest.main()
