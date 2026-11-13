#!/usr/bin/env python3
"""Evaluate the actual entrypoint on clean temporary copies, without label leaks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]


def _labels(source: Path) -> tuple[dict, dict]:
    oracle = source / ".oracle.json"
    if oracle.is_file():
        manifest = json.loads(oracle.read_text(encoding="utf-8"))
        return manifest["tasks"], {k: v for k, v in manifest.items() if k != "tasks"}
    labels = {}
    for file in sorted(source.glob("*/task.json")):
        task = json.loads(file.read_text(encoding="utf-8-sig"))
        if "expected" not in task:
            raise ValueError(f"No evaluator oracle or expected label for {file.parent.name}")
        labels[file.parent.name] = {"solvable": True, "expected": str(task["expected"]),
                                    "check": task.get("check", "exact_str"),
                                    "reward": task.get("reward", 1), "task_type": task.get("task_type", "unknown")}
    return labels, {"warning": "Public sample labels; hidden organizer tests are unavailable."}


def _ignore(folder, names):
    return [name for name in names if name in {"answer.txt", ".DS_Store", "__pycache__", ".oracle.json"}
            or (Path(folder) / name).is_symlink()]


def _copy_inputs(source: Path, target: Path) -> dict[str, str]:
    target.mkdir()
    hashes = {}
    for task_json in sorted(source.glob("*/task.json")):
        if task_json.is_symlink() or task_json.parent.is_symlink():
            raise ValueError("Task symlinks are not benchmark inputs")
        folder = target / task_json.parent.name
        shutil.copytree(task_json.parent, folder, ignore=_ignore)
        metadata = json.loads(task_json.read_text(encoding="utf-8-sig"))
        # The harness is actually run on a task file containing no evaluation
        # fields; labels are retained solely in this evaluator process.
        public = {key: metadata[key] for key in ("task_id", "task_type", "instruction", "reward") if key in metadata}
        (folder / "task.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
        for test in folder.rglob("test*.py"):
            hashes[str(test.relative_to(target))] = hashlib.sha256(test.read_bytes()).hexdigest()
    return hashes


def _tests(workspace: Path, timeout: float = 12) -> tuple[bool, str]:
    # Deliberately independent of harness.coding.run_tests.
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        process = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test*.py"],
                                 cwd=workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=timeout, text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "Independent evaluator test timeout"
    count = re.search(r"Ran\s+(\d+)\s+tests?\b", process.stdout)
    return process.returncode == 0 and count is not None and int(count[1]) > 0, process.stdout[-4000:]


def _grade(task: Path, label: dict, test_hashes: dict, tasks_root: Path, scratch: Path) -> tuple[bool, str, bool]:
    answer_file = task / "answer.txt"
    if not answer_file.is_file():
        return False, "No submitted answer", False
    answer = answer_file.read_text(encoding="utf-8").strip()
    if not label.get("solvable", True):
        return False, "Submitted an answer to an underdetermined/contradictory task", True
    check = label.get("check", "exact_str")
    if check == "unittest":
        if answer != "DONE":
            return False, "Coding answer must be DONE", True
        workspace = task / "workspace" if (task / "workspace").is_dir() else task
        original_tests = {key: digest for key, digest in test_hashes.items() if Path(key).parts[0] == task.name}
        current_tests = {str(test.relative_to(tasks_root)) for test in workspace.rglob("test*.py")}
        if current_tests != set(original_tests):
            return False, "Visible tests added or deleted", True
        for key, digest in original_tests.items():
            test = tasks_root / key
            if test.is_symlink() or hashlib.sha256(test.read_bytes()).hexdigest() != digest:
                return False, "Visible tests changed", True
        passed, evidence = _tests(workspace)
        if not passed:
            return False, "Independent visible tests failed: " + evidence, True
        hidden = label.get("hidden_test")
        if hidden:
            hidden_workspace = scratch / (task.name + "_hidden")
            hidden_workspace.mkdir()
            for source in workspace.glob("*.py"):
                if not source.name.startswith("test"):
                    shutil.copy2(source, hidden_workspace / source.name)
            (hidden_workspace / "test_hidden.py").write_text(hidden, encoding="utf-8")
            passed, evidence = _tests(hidden_workspace)
            return passed, "Independent hidden cases passed" if passed else "Independent hidden cases failed: " + evidence, True
        return True, "Independent public tests passed; organizer hidden tests unavailable", True
    expected = str(label["expected"]).strip()
    if check == "exact_int":
        try:
            valid = re.fullmatch(r"[+-]?\d+", answer) is not None and int(answer) == int(expected)
        except ValueError:
            valid = False
    else:
        valid = answer == expected
    return valid, "Exact expected answer matched" if valid else f"Expected {expected!r}, received {answer!r}", True


def benchmark(source: Path, budget: float = 100000, timeout: float = 300, use_framework: bool = False) -> dict:
    source = source.resolve()
    labels, provenance = _labels(source)
    if not labels:
        raise ValueError("Dataset contains no labeled tasks")
    with tempfile.TemporaryDirectory(prefix="jiuwen-benchmark-") as temp_name:
        scratch = Path(temp_name).resolve()
        target = scratch / "tasks"
        test_hashes = _copy_inputs(source, target)
        checkpoint = scratch / "checkpoint.json"
        command = [sys.executable, str(REPO / "run_agent.py"), "--tasks-dir", str(target),
                   "--output-file", str(checkpoint), "--deterministic-only", "--budget", str(budget),
                   "--time-limit", str(max(1, min(timeout - 5, 3450)))]
        if not use_framework:
            command.append("--without-framework")
        env = os.environ.copy()
        for key in ("LLM_BASE_URL", "LLM_API_KEY", "OUT", "TASKS"):
            env.pop(key, None)
        started = time.monotonic()
        timed_out = False
        try:
            process = subprocess.run(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=timeout, text=True, encoding="utf-8", errors="replace")
            exit_code, stderr = process.returncode, process.stderr[-12000:]
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            stderr = "Benchmark harness timeout; grading already persisted answers."
        elapsed = time.monotonic() - started
        rows = []
        for task_id, label in sorted(labels.items()):
            task = target / task_id
            correct, evidence, submitted = _grade(task, label, test_hashes, target, scratch)
            rows.append({"task_id": task_id, "task_type": label.get("task_type", "unknown"),
                         "solvable": label.get("solvable", True), "reward": label.get("reward", 1),
                         "submitted": submitted, "correct": correct, "evidence": evidence})
        checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else {}
        solvable = [row for row in rows if row["solvable"]]
        impossible = [row for row in rows if not row["solvable"]]
        summary = dict(total=len(rows), solvable=len(solvable), impossible=len(impossible),
                       submitted=sum(row["submitted"] for row in rows), correct=sum(row["correct"] for row in rows),
                       wrong_submissions=sum(row["submitted"] and not row["correct"] for row in rows),
                       impossible_abstained=sum(not row["submitted"] for row in impossible),
                       impossible_false_positive=sum(row["submitted"] for row in impossible),
                       reward_earned=sum(row["reward"] for row in rows if row["correct"]),
                       reward_available=sum(row["reward"] for row in solvable),
                       elapsed_seconds=round(elapsed, 3), harness_exit_code=exit_code, timed_out=timed_out,
                       budget=checkpoint_data.get("budget", {}))
        per_type = {}
        for kind in sorted({row["task_type"] for row in rows}):
            group = [row for row in rows if row["task_type"] == kind]
            per_type[kind] = {"total": len(group), "solvable": sum(row["solvable"] for row in group),
                              "correct": sum(row["correct"] for row in group),
                              "wrong_submissions": sum(row["submitted"] and not row["correct"] for row in group)}
        return {"schema": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "deterministic-only, actual entrypoint, " + ("openJiuwen" if use_framework else "without-framework"),
                "label_isolation": "Oracle and evaluation fields were removed from all harness task inputs.",
                "provenance": provenance, "summary": summary, "per_type": per_type,
                "harness_stderr": stderr, "tasks": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Generate a fresh temporary synthetic dataset")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=100000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--with-framework", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic == bool(args.tasks_dir):
        parser.error("Specify exactly one of --tasks-dir or --synthetic")
    if args.synthetic:
        sys.path.insert(0, str(REPO))
        from tools.generate_training import generate
        with tempfile.TemporaryDirectory(prefix="jiuwen-synthetic-") as temp_name:
            dataset = Path(temp_name) / "dataset"
            generate(dataset, args.count, args.seed)
            report = benchmark(dataset, args.budget, args.timeout, args.with_framework)
    else:
        report = benchmark(args.tasks_dir, args.budget, args.timeout, args.with_framework)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["harness_exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
