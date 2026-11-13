#!/usr/bin/env python3
"""Evaluate the actual entrypoint on clean temporary copies, without label leaks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


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


def benchmark(source: Path, budget: float = 100000, timeout: float = 300, use_framework: bool = False,
              container_image: str | None = None, *, live: bool = False, base_url: str | None = None,
              api_key: str | None = None, model: str = 'qwen3.6-35b-a3b-noreason') -> dict:
    if live and container_image:
        raise ValueError('Live mode runs locally; --docker-image is for isolated offline evaluation')
    if live:
        base_url = base_url or os.environ.get('LLM_BASE_URL')
        api_key = api_key or os.environ.get('LLM_API_KEY')
        if not base_url or not api_key:
            raise ValueError('Live mode requires a proxy URL and API key')
    source = source.resolve()
    labels, provenance = _labels(source)
    if not labels:
        raise ValueError("Dataset contains no labeled tasks")
    with tempfile.TemporaryDirectory(prefix="jiuwen-benchmark-") as temp_name:
        scratch = Path(temp_name).resolve()
        target = scratch / "tasks"
        test_hashes = _copy_inputs(source, target)
        duration = str(max(1, min(timeout - 5, 3450)))
        container_name = None
        if container_image:
            # The mount must remain writable after --cap-drop ALL, even when
            # the host and container use different UIDs. Only sanitized copies
            # in this temporary directory have their permissions changed.
            target.chmod(0o777)
            for path in target.rglob("*"):
                path.chmod(0o777 if path.is_dir() else 0o666)
            checkpoint = target / "checkpoint.json"
            container_name = "jiuwen-benchmark-" + uuid.uuid4().hex[:12]
            command = ["docker", "run", "--rm", "--name", container_name,
                       "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g",
                       "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                       "--cpus", "4", "--memory", "16g", "--pids-limit", "512",
                       "-e", "HOME=/tmp", "-e", f"BUDGET={budget}",
                       "-v", f"{target}:/data/tasks:rw", "--workdir", "/app",
                       "--entrypoint", "python", container_image,
                       "/app/run_agent.py", "--tasks-dir", "/data/tasks",
                       "--output-file", "/data/tasks/checkpoint.json",
                       "--deterministic-only", "--time-limit", duration]
        else:
            checkpoint = scratch / "checkpoint.json"
            command = [sys.executable, str(REPO / "run_agent.py"), "--tasks-dir", str(target),
                       "--output-file", str(checkpoint), "--deterministic-only", "--budget", str(budget),
                       "--time-limit", duration]
            if live:
                command.remove('--deterministic-only')
                command.extend(['--model',model,'--reasoning-model',model,'--concurrency','1'])
            elif not use_framework:
                command.append("--without-framework")
        env = os.environ.copy()
        for key in ("LLM_BASE_URL", "LLM_API_KEY", "OUT", "TASKS"):
            env.pop(key, None)
        if live:
            env['LLM_BASE_URL'], env['LLM_API_KEY'] = base_url, api_key
        started = time.monotonic()
        timed_out = False
        try:
            process = subprocess.run(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=timeout, text=True, encoding="utf-8", errors="replace")
            exit_code, stderr = process.returncode, process.stderr[-12000:]
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
            stderr = "Benchmark harness timeout; grading already persisted answers."
            if container_name:
                # Killing the Docker CLI alone does not stop its container.
                try:
                    subprocess.run(["docker", "rm", "-f", container_name],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    pass
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
        mode = (f"deterministic-only, Docker image {container_image}, openJiuwen" if container_image else
                "deterministic-only, actual entrypoint, " + ("openJiuwen" if use_framework else "without-framework"))
        report = {"schema": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": f'live proxy, openJiuwen, {model}' if live else mode,
                "label_isolation": "Oracle and evaluation fields were removed from all harness task inputs.",
                "provenance": provenance, "summary": summary, "per_type": per_type,
                "harness_stderr": stderr, "tasks": rows}
        if container_image:
            report["container_image"] = container_image
        if live:
            report['events'] = checkpoint_data.get('events', [])
            report['model'] = model
            # Defense in depth for unexpected downstream exception text.
            def redact(value):
                if isinstance(value,str): return value.replace(api_key,'[REDACTED]')
                if isinstance(value,list): return [redact(item) for item in value]
                if isinstance(value,dict): return {k:redact(v) for k,v in value.items()}
                return value
            report = redact(report)
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Generate a fresh temporary synthetic dataset")
    parser.add_argument("--hard", action="store_true", help="Independent harder fixtures including hidden coding contracts")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--with-framework", action="store_true")
    parser.add_argument("--docker-image", help="Run a built image with judge restrictions and independently grade answers")
    parser.add_argument('--live', action='store_true', help='Use the actual proxy and openJiuwen; prompts for the key if absent')
    parser.add_argument('--base-url', default=os.environ.get('LLM_BASE_URL','https://api.neuraldeep.ru/v1'))
    parser.add_argument('--model', choices=('qwen3.8-27b-noreason','qwen3.6-35b-a3b-noreason','qwen3.8-27b','qwen3.6-35b-a3b'), default='qwen3.6-35b-a3b-noreason')
    args = parser.parse_args(argv)
    if sum((args.synthetic, args.hard, bool(args.tasks_dir))) != 1:
        parser.error("Specify exactly one of --tasks-dir, --synthetic or --hard")
    if args.live and args.docker_image:
        parser.error('--live cannot be combined with --docker-image')
    budget = args.budget if args.budget is not None else (3000 if args.live else 100000)
    import math
    if not math.isfinite(budget) or budget < 0: parser.error('Budget must be finite and nonnegative')
    options = {}
    if args.live:
        key = os.environ.get('LLM_API_KEY') or getpass.getpass('API key (hidden input): ').strip()
        if not key or any(ord(c)<33 or ord(c)>126 for c in key):
            parser.error('Key contains invalid characters; copy only the API key')
        options = dict(live=True,base_url=args.base_url,api_key=key,model=args.model)
        print(f'LIVE: budget {budget}, model {args.model}; the API key is not saved.',flush=True)
    if args.synthetic or args.hard:
        sys.path.insert(0, str(REPO))
        if args.hard:
            from tools.hard_cases import generate
        else:
            from tools.generate_training import generate
        with tempfile.TemporaryDirectory(prefix="jiuwen-synthetic-") as temp_name:
            dataset = Path(temp_name) / "dataset"
            generate(dataset, args.count, args.seed)
            report = benchmark(dataset, budget, args.timeout, args.with_framework, args.docker_image, **options)
    else:
        report = benchmark(args.tasks_dir, budget, args.timeout, args.with_framework, args.docker_image, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.live:
        print('Report saved:',args.output.resolve())
        for event in [e for e in report.get('events',[]) if e.get('status')=='attempt_failed'][:3]:
            print('Attempt:',event.get('task_id'),event.get('evidence'))
    return 0 if report["summary"]["harness_exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
