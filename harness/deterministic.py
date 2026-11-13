"""Conservative zero-token solvers for the public task families.

These functions consume only the instruction and workspace. They never read
task.json (in particular its evaluation-only ``expected`` field).
"""
from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


class Abstain(ValueError):
    """The supported grammar cannot justify one unambiguous answer."""


def _child(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise Abstain("missing or unsafe workspace file")
    if path.stat().st_size > 8_000_000:
        raise Abstain("workspace file exceeds deterministic read limit")
    return path


def _question(instruction: str) -> str:
    questions = re.findall(
        r"\b((?:How many|What is|What are|What year|In which|In what|Where is|Who is)[^?]*\?)",
        instruction,
        re.I,
    )
    if len(questions) != 1:
        raise Abstain("expected exactly one supported question")
    return re.sub(r"\s+", " ", questions[0]).replace("\u2019", "'").strip()


def _math(instruction: str, workspace: Path) -> str:
    question = _question(instruction)
    if not re.fullmatch(
        r"How many boxes (?:does (?:it|the warehouse) have|are (?:left|there)) (?:at the end|now)\?",
        question, re.I,
    ):
        raise Abstain("unsupported math question")
    story = instruction[:instruction.index(question)] if question in instruction else instruction.split("?")[0]
    clauses = [re.sub(r"\s+", " ", x).strip() for x in story.split(".") if x.strip()]
    if not clauses:
        raise Abstain("missing warehouse story")
    first = re.fullmatch(r"(?:A|The) warehouse starts with (\d+) boxes", clauses.pop(0), re.I)
    if not first:
        raise Abstain("unsupported starting stock")
    stock = int(first[1])
    for raw in clauses:
        clause = re.sub(r"^Then\s+", "", raw, flags=re.I)
        found = False
        # Each match consumes the *whole* clause. Unknown side effects therefore
        # cannot be accidentally ignored while solving the recognized pieces.
        patterns = [
            (r"(?:It|The warehouse) ships (\d+) boxes per day for (\d+) days", "ship_days"),
            (r"(?:It|The warehouse) receives (\d+) deliveries of (\d+) boxes each", "deliveries"),
            (r"(?:It|The warehouse) receives (\d+) boxes", "add"),
            (r"(?:It|The warehouse) ships (\d+) boxes", "subtract"),
            (r"the stock is split equally into (\d+) warehouses and it keeps one share \(rounded down\)", "divide"),
            (r"the stock is doubled by merging with a sister warehouse", "double"),
            (r"the stock is (?:increased|decreased) by (\d+) boxes", "adjust"),
            (r"the stock is multiplied by (\d+)", "multiply"),
        ]
        for pattern, operation in patterns:
            match = re.fullmatch(pattern, clause, re.I)
            if not match:
                continue
            values = [int(x) for x in match.groups()]
            if operation == "ship_days":
                stock -= values[0] * values[1]
            elif operation == "deliveries":
                stock += values[0] * values[1]
            elif operation == "add":
                stock += values[0]
            elif operation == "subtract":
                stock -= values[0]
            elif operation == "divide":
                if values[0] == 0:
                    raise Abstain("division by zero")
                stock //= values[0]
            elif operation == "double":
                stock *= 2
            elif operation == "adjust":
                stock += values[0] * (-1 if "decreased" in clause.lower() else 1)
            elif operation == "multiply":
                stock *= values[0]
            found = True
            break
        if not found:
            raise Abstain("unsupported warehouse operation")
    return str(stock)


def _decimal(value: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite():
        raise Abstain("non-finite numeric data")
    return result


def _format_number(value: Decimal | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _csv(instruction: str, workspace: Path) -> str:
    question = _question(instruction).rstrip("?")
    mentioned = re.findall(r"\b(?:file|dataset)\s+[`'\"]?([^\s`'\"]+\.csv)", instruction, re.I)
    if len(set(mentioned)) != 1:
        raise Abstain("missing or ambiguous CSV filename")
    path = _child(workspace, mentioned[0])
    # Match the aggregate first, then consume every filtering clause below.
    aggregate_patterns = [
        (r"How many units\b", "sum", "units"),
        (r"How many (?:orders|rows|records)\b", "count", None),
        (r"What is the (?:total (?:number of )?units|sum of units)\b", "sum", "units"),
        (r"What is the (?:total revenue|total sales revenue)\b", "sum", "revenue"),
        (r"What is the (?:average|mean) (units|price)\b", "mean", "captured"),
        (r"What is the (?:maximum|highest) (units|price)\b", "max", "captured"),
        (r"What is the (?:minimum|lowest) (units|price)\b", "min", "captured"),
    ]
    for pattern, aggregate, column in aggregate_patterns:
        match = re.match(pattern, question, re.I)
        if match:
            if column == "captured":
                column = match[1].lower()
            rest = question[match.end():]
            break
    else:
        raise Abstain("unsupported CSV aggregate")

    filters: dict[str, str | tuple[int, int]] = {}
    # Quoted category values preserve exact data case and may contain spaces.
    category_pattern = r"\b(product|region)\s+(?:is\s+|=\s*)?(['\"])([^'\"]+)\2"
    def category(match: re.Match) -> str:
        key, value = match[1].lower(), match[3]
        if key in filters:
            raise Abstain("duplicate or contradictory category condition")
        filters[key] = value
        return " "
    rest = re.sub(category_pattern, category, rest, flags=re.I)
    # Also accept unquoted single-word regions, as commonly used in SQL-like
    # questions; arbitrary prose is still rejected by the final coverage check.
    def region(match: re.Match) -> str:
        if "region" in filters:
            raise Abstain("duplicate region condition")
        filters["region"] = match[1]
        return " "
    rest = re.sub(r"\bregion\s+(?:is\s+|=\s*)?([\w-]+)\b", region, rest, flags=re.I)

    month_patterns = [
        r"\bmonths?\s+(\d+)\s+(?:through|to|-)\s*(\d+)(?:\s+inclusive)?\b",
        r"\bmonths?\s+between\s+(\d+)\s+and\s+(\d+)(?:\s+inclusive)?\b",
        r"\bmonth\s+(\d+)\b",
    ]
    def month(match: re.Match) -> str:
        if "month" in filters:
            raise Abstain("duplicate month condition")
        lo = int(match[1])
        hi = int(match[2]) if match.lastindex == 2 else lo
        if not 1 <= lo <= hi <= 12:
            raise Abstain("invalid month range")
        filters["month"] = (lo, hi)
        return " "
    for pattern in month_patterns:
        rest = re.sub(pattern, month, rest, flags=re.I)
    rest = re.sub(r"\b(?:were sold|was sold|are sold|were placed|are there|were there)\b", " ", rest, flags=re.I)
    rest = re.sub(r"\b(?:of|for|in|the|and|with)\b", " ", rest, flags=re.I)
    if rest.strip(" ,\t\r\n"):
        raise Abstain("unrecognized CSV query condition")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise Abstain("duplicate CSV columns")
        required = set(filters)
        required.update({"units", "price"} if column == "revenue" else ({column} if column else set()))
        if not required.issubset(columns):
            raise Abstain("required CSV columns missing")
        values: list[Decimal] = []
        count = 0
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise Abstain("malformed CSV row")
            selected = True
            for key, condition in filters.items():
                if key == "month":
                    value = int(row[key])
                    selected &= condition[0] <= value <= condition[1]
                else:
                    selected &= row[key] == condition
            if not selected:
                continue
            count += 1
            if column == "revenue":
                values.append(_decimal(row["units"]) * _decimal(row["price"]))
            elif column:
                values.append(_decimal(row[column]))
    if aggregate == "count":
        return str(count)
    if aggregate == "sum":
        return _format_number(sum(values, Decimal(0)))
    if not values:
        raise Abstain("undefined aggregate over empty result")
    if aggregate == "mean":
        # Nonterminating means require an unspecified rounding convention.
        exact = sum((Fraction(value) for value in values), Fraction(0)) / len(values)
        denominator = exact.denominator
        for factor in (2, 5):
            while denominator % factor == 0:
                denominator //= factor
        if denominator != 1:
            raise Abstain("mean requires an unspecified rounding convention")
        result = Decimal(exact.numerator) / Decimal(exact.denominator)
        return _format_number(result)
    return _format_number(max(values) if aggregate == "max" else min(values))


def _knowledge_base(workspace: Path) -> dict[str, dict[str, str]]:
    root = (workspace / "kb").resolve()
    if not root.is_relative_to(workspace.resolve()) or not root.is_dir():
        raise Abstain("missing or unsafe knowledge base")
    files = sorted(root.glob("*.md"))
    if not files or len(files) > 2000:
        raise Abstain("empty or oversized knowledge base")
    entities: dict[str, dict[str, str]] = {}
    total = 0
    for file in files:
        path = _child(workspace, str(file.relative_to(workspace.resolve())))
        total += path.stat().st_size
        if total > 8_000_000:
            raise Abstain("oversized knowledge base")
        text = path.read_text(encoding="utf-8-sig")
        headings = re.findall(r"^#\s+(.+?)\s*$", text, re.M)
        if len(headings) != 1:
            raise Abstain("ambiguous knowledge-base heading")
        name = headings[0].casefold()
        fields: dict[str, str] = {"name": headings[0]}
        for field, value in re.findall(r"^([A-Za-z ]+):\s*(.*?)\s*$", text, re.M):
            field = field.casefold().strip()
            if field in fields and fields[field] != value:
                raise Abstain("contradictory knowledge-base field")
            fields[field] = value
        if name in entities and entities[name] != fields:
            raise Abstain("contradictory knowledge-base entities")
        entities[name] = fields
    return entities


def _search(instruction: str, workspace: Path) -> str:
    question = _question(instruction)
    patterns = [
        (r"(?:In which year|In what year|What year) was (.+?) born\?", "born"),
        (r"(?:In which year|In what year|What year) was (.+?) founded\?", "founded"),
        (r"In which city is the headquarters of (.+?)\?", "headquarters"),
        (r"In which city is (.+?) headquartered\?", "headquarters"),
        (r"Where is (.+?)'s headquarters\?", "headquarters"),
        (r"What is (.+?)'s employer\?", "employer"),
        (r"(?:Who is|What is the name of) (.+?)'s mentor\?", "mentor"),
    ]
    for pattern, field in patterns:
        match = re.fullmatch(pattern, question, re.I)
        if match:
            reference = match[1]
            break
    else:
        raise Abstain("unsupported knowledge-base question")
    entities = _knowledge_base(workspace)
    # Resolve explicit possessive hops only. If a direct name contains an
    # apostrophe it is looked up before treating it as a chain of relations.
    if reference.casefold() in entities:
        entity = entities[reference.casefold()]
    else:
        parts = re.split(r"'s\s+", reference)
        if len(parts) > 9 or parts[0].casefold() not in entities:
            raise Abstain("referenced entity missing")
        entity = entities[parts.pop(0).casefold()]
        for relation in parts:
            relation = relation.lower().strip()
            if relation not in {"employer", "mentor"}:
                raise Abstain("unsupported knowledge-base relation")
            target = entity.get(relation, "").casefold()
            if target not in entities:
                raise Abstain("relation points to missing entity")
            entity = entities[target]
    answer = entity.get(field, "").strip()
    if not answer or answer.casefold() in {"unknown", "n/a", "none", "?", "redacted"}:
        raise Abstain("requested knowledge-base field unavailable")
    if field in {"born", "founded"} and not re.fullmatch(r"\d{4}", answer):
        raise Abstain("ambiguous year field")
    return answer


def solve(task_type: str, instruction: str, workspace: Path) -> tuple[str | None, str]:
    """Return a justified answer or ``(None, reason)`` for the LLM fallback."""
    solvers = {"math": _math, "data_analysis": _csv, "search": _search}
    if task_type not in solvers:
        return None, "no deterministic solver for this task type"
    try:
        return solvers[task_type](instruction, Path(workspace)), "deterministic grammar and workspace evidence"
    except (Abstain, OSError, UnicodeError, ValueError, InvalidOperation, OverflowError) as exc:
        return None, str(exc)
