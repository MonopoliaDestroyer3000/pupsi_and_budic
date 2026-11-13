"""Conservative, zero-token repairs for small Python utility exercises.

The visible tests are executed, never rewritten.  A repair is retained only if
the complete suite passes; otherwise every changed source file is restored.
No evaluator metadata is used by this module.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading


MAX_TEST_OUTPUT = 64 * 1024
MAX_SOURCE_BYTES = 128 * 1024


def _inside(workspace: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
        return not path.is_symlink()
    except (OSError, ValueError):
        return False


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate the complete test process group on the Linux judging image."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def run_tests(workspace: Path, timeout: float = 8.0) -> tuple[bool, str]:
    """Run local unittest tests with a deadline and bounded captured output.

    A zero-test discovery is not evidence that a repair works.  Test files must
    be ordinary files inside the selected workspace.  Linux process groups make
    the timeout cover subprocesses created by a hanging test as well.
    """
    workspace = Path(workspace).resolve()
    if timeout <= 0:
        return False, "test deadline exceeded"
    tests = [p for p in workspace.glob("test*.py") if _inside(workspace, p)]
    if not tests:
        return False, "no local test*.py files; verification unavailable"
    # Only platform/runtime essentials are inherited. Task code must never get
    # proxy credentials, cloud tokens or the harness's private configuration.
    essentials = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "TZ"}
    env = {key: value for key, value in os.environ.items() if key.upper() in essentials}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    options = {"start_new_session": True} if os.name == "posix" else {}
    try:
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test*.py", "-v"],
            cwd=workspace, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options,
        )
    except (OSError, ValueError) as exc:
        return False, f"cannot start tests: {type(exc).__name__}: {exc}"
    output = bytearray()
    capped = threading.Event()

    def drain() -> None:
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                remaining = MAX_TEST_OUTPUT - len(output)
                output.extend(chunk[:max(remaining, 0)])
                if len(chunk) > remaining:
                    capped.set()
                    _stop_process(process)
                    break
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process(process)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    # Descendants must not retain a pipe indefinitely after the runner exits.
    if os.name == "posix":
        _stop_process(process)
    reader.join(timeout=1)
    if process.stdout is not None:
        process.stdout.close()
    message = bytes(output).decode("utf-8", errors="replace")
    if timed_out:
        return False, message + "\n[test timeout]"
    if capped.is_set():
        return False, message + "\n[test output limit exceeded]"
    count = re.search(r"Ran\s+(\d+)\s+tests?\b", message)
    passed = process.returncode == 0 and count is not None and int(count.group(1)) > 0
    return passed, message if message else f"test process exited {process.returncode} without verification"


# Each recipe expresses a standard utility contract. It is considered only for
# a small pure top-level function named by an actually failing test.
_RECIPES: dict[str, tuple[int, str]] = {}


def _recipe(names: str, argc: int, body: str) -> None:
    for name in names.split():
        _RECIPES[name] = argc, body


_recipe("total sum_list sum_numbers", 1, "return sum(A)")
_recipe("maximum find_max max_value", 1, "return max(A)")
_recipe("minimum find_min min_value", 1, "return min(A)")
_recipe("average mean", 1, "return sum(A) / len(A)")
_recipe("first_n", 2, "return A[:B]")
_recipe("last_n", 2, "return A[-B:] if B else A[:0]")
_recipe("count_even", 1, "return sum(1 for item in A if item % 2 == 0)")
_recipe("count_odd", 1, "return sum(1 for item in A if item % 2 != 0)")
_recipe("reverse_words", 1, "return ' '.join(reversed(A.split()))")
_recipe("reverse_string", 1, "return A[::-1]")
_recipe("is_palindrome", 1, "return A == A[::-1]")
_recipe("deduplicate unique_preserve_order", 1, "return list(dict.fromkeys(A))")
_recipe("flatten", 1, "return [item for group in A for item in group]")
_recipe("clamp", 3, "return max(B, min(C, A))")
_recipe("factorial", 1, "if A < 0:\n    raise ValueError('factorial is undefined for negative numbers')\nresult = 1\nfor item in range(2, A + 1):\n    result *= item\nreturn result")
_recipe("is_prime", 1, "if A < 2:\n    return False\ndivisor = 2\nwhile divisor * divisor <= A:\n    if A % divisor == 0:\n        return False\n    divisor += 1\nreturn True")
_recipe("gcd", 2, "while B:\n    A, B = B, A % B\nreturn abs(A)")


def _read_tree(path: Path) -> tuple[str, ast.Module] | None:
    try:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            return None
        source = path.read_text(encoding="utf-8-sig")
        return source, ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return None


def _test_references(workspace: Path, failures: set[str] | None = None) -> dict[Path, set[str]]:
    """Map failing test calls to the local modules they import."""
    references: dict[Path, set[str]] = {}
    for path in sorted(workspace.glob("test*.py")):
        if not _inside(workspace, path):
            continue
        parsed = _read_tree(path)
        if parsed is None:
            continue
        _, tree = parsed
        modules: dict[str, Path] = {}
        direct: dict[str, tuple[Path, str]] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = workspace / (alias.name + ".py")
                    if candidate.is_file() and _inside(workspace, candidate):
                        modules[alias.asname or alias.name] = candidate
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                candidate = workspace / (node.module + ".py")
                if candidate.is_file() and _inside(workspace, candidate):
                    for alias in node.names:
                        if alias.name != "*":
                            direct[alias.asname or alias.name] = candidate, alias.name
        for test in ast.walk(tree):
            if not isinstance(test, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not test.name.startswith("test") or (failures is not None and test.name not in failures):
                continue
            for call in ast.walk(test):
                if not isinstance(call, ast.Call):
                    continue
                target = call.func
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in modules:
                    references.setdefault(modules[target.value.id], set()).add(target.attr)
                elif isinstance(target, ast.Name) and target.id in direct:
                    module, function = direct[target.id]
                    references.setdefault(module, set()).add(function)
    return references


def _literal_conflict(workspace: Path) -> str | None:
    """Find incompatible literal assertions about a known pure utility call.

    This is a conservative diagnostic, not a general impossibility proof. Calls
    with variables, state-changing statements, mocks or setup are ignored.
    """
    seen: dict[str, object] = {}
    for path in sorted(workspace.glob("test*.py")):
        if not _inside(workspace, path):
            continue
        parsed = _read_tree(path)
        if parsed is None:
            continue
        _, tree = parsed
        # Setup/mocking can deliberately make equal-looking calls differ.
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ("setUp", "setUpClass") for n in ast.walk(tree)):
            continue
        for test in ast.walk(tree):
            if not isinstance(test, ast.FunctionDef) or not test.name.startswith("test"):
                continue
            expressions = [s for s in test.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
            if any(not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Call) for s in expressions):
                continue
            for statement in expressions:
                assertion = statement.value
                if not isinstance(assertion.func, ast.Attribute) or assertion.func.attr != "assertEqual" or len(assertion.args) != 2:
                    continue
                call, expected = assertion.args
                if not isinstance(call, ast.Call) or call.keywords:
                    continue
                name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ""
                if name not in _RECIPES:
                    continue
                try:
                    for argument in call.args:
                        ast.literal_eval(argument)
                    value = ast.literal_eval(expected)
                except (ValueError, TypeError, SyntaxError):
                    continue
                key = str(path) + ast.dump(call, include_attributes=False)
                if key in seen and seen[key] != value:
                    return f"contradictory literal assertions for {ast.unparse(call)}"
                seen[key] = value
    return None


def _replacement(function: ast.FunctionDef) -> str | None:
    recipe = _RECIPES.get(function.name)
    if recipe is None or function.decorator_list or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        return None
    argc, body = recipe
    args = function.args.posonlyargs + function.args.args
    if len(args) != argc or (function.end_lineno or 0) - function.lineno > 70:
        return None
    nodes = list(ast.walk(function))
    if len(nodes) > 250 or any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.With, ast.AsyncWith, ast.ClassDef)) for n in nodes):
        return None
    if sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes) != 1:
        return None
    # AST substitution avoids replacing letters within literals or identifiers.
    replacements = dict(zip(("A", "B", "C"), (a.arg for a in args)))

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            return ast.copy_location(ast.Name(id=replacements.get(node.id, node.id), ctx=node.ctx), node)

    replacement = copy.deepcopy(function)
    replacement.body = Rename().visit(ast.parse(body)).body
    if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
        replacement.body.insert(0, copy.deepcopy(function.body[0]))
    return ast.unparse(ast.fix_missing_locations(replacement))


def _apply_recipes(source: str, tree: ast.Module, targets: set[str]) -> tuple[str, list[str]]:
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str, str]] = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in targets:
            continue
        replacement = _replacement(function)
        if replacement is not None:
            edits.append((function.lineno - 1, function.end_lineno or function.lineno, replacement + "\n", function.name))
    for start, end, replacement, _ in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines), [edit[3] for edit in edits]


def solve_coding(workspace: Path, instruction: str, timeout: float = 8.0) -> tuple[str | None, str]:
    """Repair recognized utility bugs and prove the result with local tests.

    ``timeout`` is the per-test-run limit. There are at most two test runs and
    only imported local source files can be edited. The caller writes answer.txt.
    """
    workspace = Path(workspace).resolve()
    passed, baseline = run_tests(workspace, timeout)
    if passed:
        return "DONE", "visible tests already pass\n" + baseline[-6000:]
    if "[test timeout]" in baseline or "[test output limit exceeded]" in baseline:
        return None, baseline[-6000:]
    conflict = _literal_conflict(workspace)
    if conflict:
        return None, "infeasible candidate: " + conflict + "\n" + baseline[-6000:]
    failures = set(re.findall(r"(?:FAIL|ERROR):\s+(test\w*)\s+\(", baseline))
    if not failures:
        return None, "no recognized failing test methods\n" + baseline[-6000:]
    references = _test_references(workspace, failures)
    originals: dict[Path, bytes] = {}
    repaired: list[str] = []
    verified = False
    try:
        for path, targets in references.items():
            parsed = _read_tree(path)
            if parsed is None or path.name.startswith("test") or not _inside(workspace, path):
                continue
            source, tree = parsed
            candidate, functions = _apply_recipes(source, tree, targets)
            if not functions or candidate == source:
                continue
            originals[path] = path.read_bytes()
            path.write_text(candidate, encoding="utf-8")
            repaired.extend(f"{path.name}:{name}" for name in functions)
        if not originals:
            return None, "no conservative utility repair applies\n" + baseline[-6000:]
        verified, result = run_tests(workspace, timeout)
        if verified:
            return "DONE", "repaired " + ", ".join(repaired) + "\n" + result[-6000:]
        return None, "candidate repair failed; restored original sources\n" + result[-6000:]
    except (OSError, ValueError, SyntaxError) as exc:
        return None, f"repair unavailable: {type(exc).__name__}: {exc}"
    finally:
        if not verified:
            for path, original in originals.items():
                try:
                    path.write_bytes(original)
                except OSError:
                    # Preserve the original error for the caller; the process
                    # may have reached a filesystem resource/permission limit.
                    pass
