"""Small local tool surface. Every model-visible path is task-relative."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".answer-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_path(root: Path, relative: str, writable: bool = False) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path must stay inside this task workspace")
    if any(part.startswith(".") for part in path.relative_to(root.resolve()).parts):
        raise ValueError("Hidden files are not available")
    if path.name in {"task.json", "answer.txt"}:
        raise ValueError("Task metadata and submission files are managed by the harness")
    if writable and (any("test" in p.lower() for p in path.relative_to(root.resolve()).parts) or path.suffix not in {".py", ".txt", ".json", ".csv"}):
        raise ValueError("Cannot modify tests or unsupported file types")
    return path


def inventory(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        if len(result) >= 400:
            break
        if path.is_file() and not path.is_symlink():
            try:
                safe_path(root, str(path.relative_to(root)))
            except ValueError:
                continue
            if "__pycache__" not in path.parts:
                result.append(path)
    return sorted(result)


# This runner restricts model-generated computation; it is not a general shell.
_RUNNER = r'''
import ast, builtins, json, os, pathlib, sys
payload = json.load(sys.stdin)
root = pathlib.Path(payload['root']).resolve()
try:
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
except ImportError:
    pass
code = payload['code']
allowed = {'math','csv','json','re','statistics','collections','decimal','fractions',
           'itertools','functools','datetime','heapq','bisect','operator','string','io'}
tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
        if any(n.split('.')[0] not in allowed for n in names):
            raise ValueError('Unsupported import; use file tools for inspection')
    if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
        raise ValueError('Private attributes are unavailable')
    if isinstance(node, ast.Name) and node.id in {'eval','exec','compile','globals','locals','getattr','setattr','delattr','vars','__import__','breakpoint','help','input'}:
        raise ValueError('Unsupported operation')
real_open = builtins.open
def guarded_open(file, mode='r', *args, **kwargs):
    if not isinstance(file,(str,bytes,os.PathLike)):
        raise ValueError('Expected a task-relative filename')
    path = pathlib.Path(file).resolve()
    if not path.is_relative_to(root) or any(p.startswith('.') for p in path.relative_to(root).parts):
        raise ValueError('File must stay inside this workspace')
    if path.name in {'task.json','answer.txt'}:
        raise ValueError('Use the answer action to submit')
    if any(c in mode for c in 'wax+'):
        raise ValueError('Python tool is read-only; use write for code edits')
    return real_open(path,mode,*args,**kwargs)
builtins.open = guarded_open
def audit(event,args):
    if event == 'open':
        file, mode, flags = args
        if not isinstance(file,(str,bytes,os.PathLike)):
            raise PermissionError('Only task files are available')
        path = pathlib.Path(file).resolve()
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            raise PermissionError('Computation tool cannot write files')
        if path.is_relative_to(root):
            if path.name in {'task.json','answer.txt'} or any(p.startswith('.') for p in path.relative_to(root).parts):
                raise PermissionError('Metadata unavailable')
        elif not path.is_relative_to(pathlib.Path(sys.base_prefix).resolve()):
            raise PermissionError('File is outside the workspace')
    if event.startswith(('socket.','subprocess.','ctypes.')) or event in {'os.system','os.exec','os.spawn','os.fork','os.remove','os.rename','os.rmdir','os.mkdir'}:
        raise PermissionError('Operation blocked in computation tool')
sys.addaudithook(audit)
namespace = {'__builtins__': builtins.__dict__}
exec(compile(tree,'<task-computation>','exec'),namespace)
'''


class LocalTools:
    def __init__(self, root: Path, max_seconds: float = 10):
        self.root = root.resolve()
        self.max_seconds = max_seconds
        self.written = False
        self.originals: dict[Path, bytes | None] = {}

    def rollback(self):
        for path, original in self.originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)

    def execute(self, action: dict) -> str:
        try:
            tool = action.get("tool")
            if tool == "list":
                return "\n".join(str(p.relative_to(self.root)).replace("\\", "/") for p in inventory(self.root))[:12000]
            if tool == "read":
                path = safe_path(self.root, str(action["path"]))
                if path.stat().st_size > 1_000_000:
                    return "File too large: use Python CSV aggregation or grep"
                text = path.read_text(encoding="utf-8", errors="replace")
                start = max(0, int(action.get("offset", 0)))
                return text[start:start + min(12000, max(1, int(action.get("limit", 10000))))]
            if tool == "grep":
                needle = str(action["text"]).casefold()
                if not needle:
                    return "Empty search is not allowed"
                found = []
                for path in inventory(self.root):
                    if path.stat().st_size > 250000:
                        continue
                    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if needle in line.casefold():
                            found.append(f"{path.relative_to(self.root)}:{number}: {line[:300]}")
                            if len(found) >= 50:
                                return "\n".join(found)[:12000]
                return "\n".join(found)[:12000] or "No matches"
            if tool == "write":
                path = safe_path(self.root, str(action["path"]), writable=True)
                content = str(action["content"])
                if len(content) > 40000:
                    return "Write exceeds 40000 characters"
                if path.suffix == ".py":
                    import ast
                    ast.parse(content)
                if path not in self.originals:
                    self.originals[path] = path.read_bytes() if path.exists() else None
                atomic_write(path, content)
                self.written = True
                return "Saved " + str(path.relative_to(self.root))
            if tool == "python":
                code = str(action["code"])
                if len(code) > 10000:
                    return "Code exceeds 10000 characters"
                env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH") if k in os.environ}
                env["PYTHONIOENCODING"] = "utf-8"
                # File-backed output bounds memory even for accidentally verbose code.
                with tempfile.TemporaryFile() as output:
                    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", _RUNNER],
                                               cwd=self.root, env=env, stdin=subprocess.PIPE,
                                               stdout=output, stderr=subprocess.STDOUT)
                    try:
                        process.communicate(json.dumps({"root": str(self.root), "code": code}).encode(), timeout=self.max_seconds)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                        return "Computation exceeded time limit"
                    output.seek(0)
                    result = output.read(12000).decode("utf-8", errors="replace")
                return f"Exit {process.returncode}\n{result}"
            if tool == "test":
                from .coding import run_tests
                passed, evidence = run_tests(self.root, timeout=self.max_seconds)
                return ("PASS\n" if passed else "FAIL\n") + evidence[-10000:]
            return "Unknown tool. Choose list/read/grep/python/write/test/answer/skip."
        except Exception as exc:
            return f"Tool error: {type(exc).__name__}: {str(exc)[:400]}"

    def context(self, instruction: str, kind: str) -> str:
        paths = inventory(self.root)
        listing = "Files: " + ", ".join(str(p.relative_to(self.root)).replace("\\", "/") for p in paths[:80])
        snippets = []
        if kind == "coding":
            selected = [p for p in paths if p.suffix == ".py"][:6]
        elif kind == "search":
            words = set(re.findall(r"[A-Za-z]{3,}", instruction.lower())) - {"the", "file", "directory", "current", "answer", "final", "only", "which", "year", "born", "tools", "with", "write"}
            selected = [p for p in paths if any(w in p.stem.lower() for w in words)][:4]
        elif kind == "data_analysis":
            selected = [p for p in paths if p.suffix in {".csv", ".tsv"}][:2]
        else:
            selected = []
        for path in selected:
            limit = 700 if path.suffix in {".csv", ".tsv"} else 5000
            if path.stat().st_size < 500000:
                snippets.append(f"--- {path.relative_to(self.root)} ---\n" + path.read_text(encoding="utf-8", errors="replace")[:limit])
        return (listing + "\n" + "\n".join(snippets))[:16000]
