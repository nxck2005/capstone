"""Flag parameter-valued numeric literals in experiment source (SR-1).

The checker walks the scopes declared by ``params.config.literal_lint_scope``,
excluding ``params.config.literal_lint_excluded_paths``. A numeric literal is a
finding when it equals any numeric leaf in the generated parameter tree unless:

* the value is in ``params.config.literal_lint_exempt_values``; or
* its source line carries ``params.config.literal_lint_annotation`` followed by
  a non-empty reason.

Usage:
    python tools/check_literals.py
    python tools/check_literals.py -v
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import get, load_params  # noqa: E402


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class CheckResult:
    findings: tuple[Finding, ...]
    scanned: tuple[Path, ...]
    annotations: int


def _numeric_leaves(
    value: Any, path: str = "params"
) -> list[tuple[int | float, str]]:
    leaves: list[tuple[int | float, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            leaves.extend(_numeric_leaves(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(_numeric_leaves(item, f"{path}.{index}"))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        leaves.append((value, path))
    return leaves


class _LiteralVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[tuple[int, int | float]] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, int | float) and not isinstance(value, bool):
            self.values.append((node.lineno, value))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int | float)
            and not isinstance(node.operand.value, bool)
        ):
            self.values.append((node.lineno, -node.operand.value))
            return
        self.generic_visit(node)


def _is_excluded(path: Path, exclusions: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(excluded) for excluded in exclusions)


def _annotation_reasons(source: str, annotation: str) -> dict[int, str]:
    """Reason text keyed by line, considering comments rather than strings."""
    reasons: dict[int, str] = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and annotation in token.string:
                reasons[token.start[0]] = token.string.split(annotation, maxsplit=1)[1].strip()
    except tokenize.TokenError:
        # ast.parse below reports the authoritative syntax finding.
        pass
    return reasons


def check() -> CheckResult:
    params = load_params()
    exempt = set(get("config.literal_lint_exempt_values"))
    annotation = get("config.literal_lint_annotation")
    scopes = tuple((REPO / item).resolve() for item in get("config.literal_lint_scope"))
    exclusions = tuple(
        (REPO / item).resolve() for item in get("config.literal_lint_excluded_paths")
    )

    watched: dict[int | float, list[str]] = defaultdict(list)
    for value, path in _numeric_leaves(params):
        if value not in exempt:
            watched[value].append(path)

    findings: list[Finding] = []
    scanned: list[Path] = []
    valid_annotations = 0
    for scope in scopes:
        if not scope.exists():
            continue
        for path in sorted(scope.rglob("*.py")):
            resolved_path = path.resolve()
            if _is_excluded(resolved_path, exclusions):
                continue
            relative = resolved_path.relative_to(REPO)
            scanned.append(relative)
            source = path.read_text()
            reasons = _annotation_reasons(source, annotation)
            for line_number, reason in reasons.items():
                if reason:
                    valid_annotations += 1
                else:
                    findings.append(
                        Finding(
                            relative,
                            line_number,
                            f"{annotation} requires a non-empty reason",
                        )
                    )

            try:
                tree = ast.parse(source, filename=str(relative))
            except SyntaxError as exc:
                findings.append(
                    Finding(relative, exc.lineno or 1, f"cannot parse source: {exc.msg}")
                )
                continue
            visitor = _LiteralVisitor()
            visitor.visit(tree)
            for line_number, value in visitor.values:
                paths = watched.get(value)
                if not paths or reasons.get(line_number):
                    continue
                findings.append(
                    Finding(
                        relative,
                        line_number,
                        f"literal {value!r} matches {', '.join(sorted(paths))}",
                    )
                )

    return CheckResult(tuple(findings), tuple(scanned), valid_annotations)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="list scanned files")
    args = parser.parse_args()
    result = check()
    if args.verbose:
        for path in result.scanned:
            print(f"  scanned: {path}")
    if result.findings:
        print(f"{len(result.findings)} literal configuration finding(s):")
        for finding in result.findings:
            print(f"  - {finding.render()}")
        print(
            f"scanned {len(result.scanned)} Python files; "
            f"{result.annotations} reasoned literal-ok annotations"
        )
        return 1
    print(
        f"ok: scanned {len(result.scanned)} Python files; "
        f"0 findings; {result.annotations} reasoned literal-ok annotations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
