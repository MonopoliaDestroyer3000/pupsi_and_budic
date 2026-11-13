#!/usr/bin/env python3
"""Generate reproducible synthetic exercises and a separate evaluator oracle.

The oracle is not task input. tools/benchmark.py removes it and every evaluation
field before it invokes the harness. These templates are development coverage,
not an estimate of the organizer's hidden task distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Callable


TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer "
        "to the file answer.txt in the current directory.")
KINDS = ("math", "search", "data_analysis", "coding")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _math(rng: random.Random, workspace: Path, impossible: bool) -> tuple[str, dict]:
    stock = rng.randint(500, 5000)
    clauses = [f"A warehouse starts with {stock} boxes."]
    for _ in range(rng.randint(2, 8)):
        operation = rng.choice(("ship", "deliver", "split", "double", "receive"))
        if operation == "ship" and stock < 100:
            operation = "receive"
        if operation == "ship":
            n, days = rng.randint(1, 20), rng.randint(1, 5)
            clauses.append(f"It ships {n} boxes per day for {days} days.")
            stock -= n * days
        elif operation == "deliver":
            n, boxes = rng.randint(1, 7), rng.randint(10, 150)
            clauses.append(f"It receives {n} deliveries of {boxes} boxes each.")
            stock += n * boxes
        elif operation == "split":
            n = rng.randint(2, 5)
            clauses.append(f"Then the stock is split equally into {n} warehouses and it keeps one share (rounded down).")
            stock //= n
        elif operation == "double":
            clauses.append("Then the stock is doubled by merging with a sister warehouse.")
            stock *= 2
        else:
            n = rng.randint(5, 180)
            clauses.append(f"It receives {n} boxes.")
            stock += n
    if impossible:
        variant = rng.randrange(2)
        if variant == 0:
            clauses[0] = "A warehouse starts with an unknown number of boxes."
        else:
            clauses.append("Some boxes are then lost, but the number lost is not recorded.")
        oracle = {"solvable": False, "reason": "Initial stock or an operation magnitude is not given."}
    else:
        oracle = {"solvable": True, "expected": str(stock), "check": "exact_int"}
    return " ".join(clauses) + " How many boxes does it have at the end?" + TAIL, oracle


def _search(rng: random.Random, workspace: Path, impossible: bool) -> tuple[str, dict]:
    people = [f"{first} {last}" for first, last in
              (("Aria", "Novak"), ("Boris", "Lee"), ("Chen", "Smith"), ("Dina", "Patel"),
               ("Emil", "Wang"), ("Farid", "Sato"), ("Grace", "Muller"), ("Hiro", "Diaz"))]
    companies = ["Aster Labs", "Beacon Systems", "Coral Data", "Dune Works", "Ember Optics"]
    cities = ["Osaka", "Leeds", "Helsinki", "Seoul", "Bristol", "Tallinn", "Novosibirsk"]
    born = {person: str(rng.randint(1960, 2005)) for person in people}
    employer = {person: rng.choice(companies) for person in people}
    mentor = {person: rng.choice([p for p in people if p != person]) for person in people}
    headquarters = {company: rng.choice(cities) for company in companies}
    target = rng.choice(people)
    variant = rng.randrange(4)
    if variant == 0:
        question = f"In which year was {target} born?"
        answer = born[target]
    elif variant == 1:
        question = f"In which city is the headquarters of {target}'s employer?"
        answer = headquarters[employer[target]]
    elif variant == 2:
        question = f"In which year was {target}'s mentor born?"
        answer = born[mentor[target]]
    else:
        question = f"In which city is the headquarters of {target}'s mentor's employer?"
        answer = headquarters[employer[mentor[target]]]
    for index, person in enumerate(people):
        if impossible and person == target:
            continue
        _write(workspace / "kb" / f"person_{index}.md",
               f"# {person}\n\nBorn: {born[person]}\nEmployer: {employer[person]}\nMentor: {mentor[person]}\n")
    for index, company in enumerate(companies):
        _write(workspace / "kb" / f"company_{index}.md",
               f"# {company}\n\nHeadquarters: {headquarters[company]}\nFounded: {rng.randint(1920, 2020)}\n")
    instruction = "The directory kb/ contains a small knowledge base of markdown files. " + question + TAIL
    return instruction, ({"solvable": False, "reason": "The referenced person's record is absent."} if impossible else
                         {"solvable": True, "expected": answer, "check": "exact_str"})


def _csv(rng: random.Random, workspace: Path, impossible: bool) -> tuple[str, dict]:
    products = ("widget", "gadget", "sprocket", "gizmo")
    regions = ("north", "south", "east", "west")
    rows = []
    for index in range(rng.choice((20, 50, 100, 300, 800))):
        rows.append(dict(order_id=index + 1, region=rng.choice(regions), product=rng.choice(products),
                         month=rng.randint(1, 12), units=rng.randint(1, 60), price=rng.randint(2, 120)))
    product, region = rng.choice(products), rng.choice(regions)
    lower = rng.randint(1, 10)
    upper = rng.randint(lower, 12)
    include_region = rng.choice((True, False))
    selected = [r for r in rows if r["product"] == product and lower <= r["month"] <= upper
                and (not include_region or r["region"] == region)]
    condition = f"of product '{product}'"
    if include_region:
        condition += f" in region '{region}'"
    condition += f" in months {lower} through {upper} inclusive"
    variant = rng.randrange(3)
    if variant == 0:
        question = f"How many units {condition} were sold?"
        answer = sum(row["units"] for row in selected)
    elif variant == 1:
        question = f"What is the total revenue {condition}?"
        answer = sum(row["units"] * row["price"] for row in selected)
    else:
        question = f"How many orders {condition} were placed?"
        answer = len(selected)
    workspace.mkdir(parents=True, exist_ok=True)
    if not impossible:
        with (workspace / "sales.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    instruction = (f"The file sales.csv has columns order_id,region,product,month,units,price "
                   f"({len(rows)} rows). {question}" + TAIL)
    return instruction, ({"solvable": False, "reason": "The requested sales.csv data is absent."} if impossible else
                         {"solvable": True, "expected": str(answer), "check": "exact_int"})


# Reference functions and hidden cases belong to the dataset generator/evaluator,
# not to the agent. Visible cases and hidden cases are deliberately different.
_CODE: list[tuple[str, str, Callable, list[tuple], list[tuple]]] = [
    ("total", "def total(xs):\n    return -sum(xs)\n", sum, [([1, 2, 3],)], [([],), ([-8, 2, 5],), ([0, 0],)]),
    ("maximum", "def maximum(xs):\n    return min(xs)\n", max, [([3, 9, 2],)], [([-8, -3, -6],), ([4],), ([9, 9, 2],)]),
    ("minimum", "def minimum(xs):\n    return max(xs)\n", min, [([3, 9, 2],)], [([-8, -3, -6],), ([4],), ([9, 9, 2],)]),
    ("average", "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n", lambda x: sum(x) / len(x), [([2, 4, 6],)], [([-8, 2],), ([0],), ([1, 3, 5, 7],)]),
    ("first_n", "def first_n(xs, n):\n    return xs[:n + 1]\n", lambda x, n: x[:n], [([5, 6, 7, 8], 3)], [([1, 2], 0), ([3, 1], 8), ([], 2)]),
    ("last_n", "def last_n(xs, n):\n    return xs[:n]\n", lambda x, n: x[-n:] if n else x[:0], [([5, 6, 7, 8], 2)], [([1, 2], 0), ([3, 1], 8), ([], 2)]),
    ("count_even", "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n", lambda x: sum(i % 2 == 0 for i in x), [([1, 2, 4, 6, 7],)], [([],), ([-2, -1, 0, 3, 4],), ([1, 3, 5],)]),
    ("count_odd", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n", lambda x: sum(i % 2 != 0 for i in x), [([1, 2, 4, 6, 7],)], [([],), ([-2, -1, 0, 3, 4],), ([1, 3, 5],)]),
    ("reverse_words", "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n", lambda s: " ".join(reversed(s.split())), [("c a b",)], [("one two three",), ("  a   b  ",), ("",)]),
    ("reverse_string", "def reverse_string(s):\n    return s\n", lambda s: s[::-1], [("abc",)], [("",), ("a b",), ("abcd",)]),
    ("is_palindrome", "def is_palindrome(s):\n    return True\n", lambda s: s == s[::-1], [("abc",)], [("",), ("abba",), ("abc",)]),
    ("deduplicate", "def deduplicate(xs):\n    return xs\n", lambda x: list(dict.fromkeys(x)), [([3, 1, 3, 2, 1],)], [([],), ([4, 4, 4],), (["b", "a", "b"],)]),
    ("flatten", "def flatten(xs):\n    return xs\n", lambda x: [y for a in x for y in a], [([[1, 2], [3]],)], [([],), ([[], [1], []],), ([["a"], ["b", "c"]],)]),
    ("clamp", "def clamp(value, lower, upper):\n    return value\n", lambda x, lo, hi: max(lo, min(hi, x)), [(20, 0, 10)], [(-5, -2, 3), (1, -2, 3), (3, 3, 3)]),
    ("gcd", "def gcd(a, b):\n    return a + b\n", __import__("math").gcd, [(12, 18)], [(0, 7), (-12, 18), (7, 0)]),
]


def _test_source(specs, hidden: bool = False) -> str:
    lines = ["import unittest", "import mylib as m", "", "class T(unittest.TestCase):"]
    for name, _, reference, visible, extra in specs:
        lines.append(f"    def test_{name}(self):")
        for args in extra if hidden else visible:
            call = ", ".join(repr(value) for value in args)
            lines.append(f"        self.assertEqual(m.{name}({call}), {reference(*args)!r})")
    lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def _coding(rng: random.Random, workspace: Path, impossible: bool) -> tuple[str, dict]:
    specs = rng.sample(_CODE, rng.randint(2, 6))
    _write(workspace / "mylib.py", "\n\n".join(spec[1] for spec in specs))
    tests = _test_source(specs)
    if impossible:
        # Both assertions concern the same pure call with the same literal
        # arguments; obeying the stated contract cannot satisfy both.
        specs = [_CODE[0]]
        _write(workspace / "mylib.py", specs[0][1])
        tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                 "    def test_total(self):\n"
                 "        self.assertEqual(m.total([1, 2, 3]), 6)\n"
                 "        self.assertEqual(m.total([1, 2, 3]), 7)\n\n"
                 "if __name__ == '__main__':\n    unittest.main()\n")
    _write(workspace / "test_mylib.py", tests)
    instruction = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
                   "(do not edit the tests) until all tests pass. Functions must remain deterministic and pure. "
                   "Work only inside the current directory. When you are done, write ONLY the word DONE "
                   "to the file answer.txt in the current directory.")
    return instruction, ({"solvable": False, "reason": "Unchanged tests contradict the required pure-function contract."} if impossible else
                         {"solvable": True, "expected": "tests_pass", "check": "unittest",
                          "hidden_test": _test_source(specs, hidden=True)})


def generate(output: Path, count: int = 100, seed: int = 20261113, impossible_fraction: float = .2) -> dict:
    if count < 1 or not 0 <= impossible_fraction <= 1:
        raise ValueError("count must be positive and impossible_fraction in [0, 1]")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing datasets are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Choose labels independently of rewards, then spread impossible tasks
    # across task families before shuffling to avoid ID/order hints.
    impossible_count = round(count * impossible_fraction)
    plan = [(KINDS[index % 4], index < impossible_count) for index in range(count)]
    rng.shuffle(plan)
    factories = {"math": _math, "search": _search, "data_analysis": _csv, "coding": _coding}
    oracle = {}
    for index, (kind, impossible) in enumerate(plan):
        task_id = f"{kind}_{index:05d}"
        folder = output / task_id
        workspace = folder / "workspace"
        workspace.mkdir(parents=True)
        instruction, label = factories[kind](rng, workspace, impossible)
        reward = rng.randint(1, 5)
        task = dict(task_id=task_id, task_type=kind, instruction=instruction, reward=reward)
        _write(folder / "task.json", json.dumps(task, ensure_ascii=False, indent=2) + "\n")
        oracle[task_id] = {**label, "task_type": kind, "reward": reward}
    manifest = {"schema": 1, "seed": seed, "count": count, "impossible_count": impossible_count,
                "warning": "Synthetic template coverage only; this is not the organizer's hidden distribution.",
                "tasks": oracle}
    _write(output / ".oracle.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--impossible-fraction", type=float, default=.2)
    args = parser.parse_args(argv)
    result = generate(args.output, args.count, args.seed, args.impossible_fraction)
    print(json.dumps({"output": str(args.output.resolve()), "count": result["count"],
                      "impossible": result["impossible_count"], "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
