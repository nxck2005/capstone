#!/usr/bin/env python3
"""Generate the derived views of spec/SPEC.md.

SPEC.md is the single source of truth. This script parses it and emits:

  spec/params.generated.yaml  - machine-readable parameters, consumed by training code
  spec/DATASHEET.md           - terse parameter tables
  spec/concerns/*.md          - requirements grouped by concern

It also validates the spec itself: requirement IDs, parameter citations, and the
arithmetic relating image dimensions to channel-symbol budgets.

Usage:
    python tools/gen_spec_views.py            # write the generated files
    python tools/gen_spec_views.py --check    # validate only; non-zero exit on drift
"""

from __future__ import annotations

import argparse
import re
import sys
from fractions import Fraction
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "spec" / "SPEC.md"
SPEC_REL = "SPEC.md"

BANNER = "<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->"

# Requirement prefix -> (concern file stem, human title)
CONCERNS = {
    "SR": ("system", "System"),
    "BR": ("baseline", "Baseline"),
    "ER": ("experiments", "Experiments"),
    "DR": ("demo", "Demo"),
    "HR": ("hardware", "Hardware (Tier 2/3)"),
    "PR": ("programme", "Programme deliverables"),
    "OPT": ("roadmap", "Roadmap"),
    "FW": ("roadmap", "Roadmap"),
    "DEC": ("roadmap", "Roadmap"),
    "G": ("roadmap", "Roadmap"),
}
# Prefixes whose requirements must state how they are verified.
VERIFIABLE = {"SR", "BR", "ER", "DR", "HR", "PR"}

REQ_RE = re.compile(r"^- \*\*([A-Z]+)-(\d+)\*\* — (.+)$")
# A retired ID keeps its number reserved so live IDs are never renumbered (SPEC.md §0).
TOMBSTONE_RE = re.compile(r"^- ~~\*\*([A-Z]+)-(\d+)\*\*~~ — (.+)$")
CITE_RE = re.compile(r"`(params\.[A-Za-z0-9_.]+)`")
VERIFY_RE = re.compile(r"\*\(verify: .+\)\*$")


class SpecError(Exception):
    pass


# --------------------------------------------------------------------------- parse


def parse_params(text: str) -> dict:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("```") and "params" in line:
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("```"):
                    return yaml.safe_load("\n".join(lines[i + 1 : j]))
            raise SpecError("params block is never closed")
    raise SpecError("no ```yaml params block found in SPEC.md")


def parse_requirements(text: str) -> list[dict]:
    reqs = []
    for lineno, line in enumerate(text.splitlines(), 1):
        m = REQ_RE.match(line)
        if not m:
            continue
        prefix, num, body = m.group(1), int(m.group(2)), m.group(3)
        if prefix not in CONCERNS:
            raise SpecError(f"line {lineno}: unknown requirement prefix {prefix!r}")
        reqs.append(
            {
                "id": f"{prefix}-{num}",
                "prefix": prefix,
                "num": num,
                "body": body,
                "line": lineno,
                "cites": sorted(set(CITE_RE.findall(line))),
            }
        )
    if not reqs:
        raise SpecError("no requirement lines found")
    return reqs


def parse_tombstones(text: str) -> list[dict]:
    """Retired requirement IDs. They reserve their number but carry no obligation."""
    stones = []
    for lineno, line in enumerate(text.splitlines(), 1):
        m = TOMBSTONE_RE.match(line)
        if not m:
            continue
        prefix, num, body = m.group(1), int(m.group(2)), m.group(3)
        if prefix not in CONCERNS:
            raise SpecError(f"line {lineno}: unknown requirement prefix {prefix!r}")
        stones.append({"id": f"{prefix}-{num}", "prefix": prefix, "num": num, "body": body})
    return stones


# ---------------------------------------------------------------------- validation


def flatten(node, prefix="params") -> list[tuple[str, object]]:
    if isinstance(node, dict):
        out = []
        for key, value in node.items():
            out.extend(flatten(value, f"{prefix}.{key}"))
        return out
    return [(prefix, node)]


def resolve(params: dict, path: str) -> object:
    node = params
    for part in path.split(".")[1:]:  # drop leading "params"
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def validate(params: dict, reqs: list[dict], stones: list[dict]) -> list[str]:
    errors = []

    # 1. IDs unique; live plus retired IDs contiguous from 1 within each prefix.
    #    Retirement reserves a number so a live ID is never reused or renumbered.
    seen: dict[str, list[int]] = {}
    for req in reqs:
        seen.setdefault(req["prefix"], []).append(req["num"])
    retired: dict[str, list[int]] = {}
    for stone in stones:
        retired.setdefault(stone["prefix"], []).append(stone["num"])
    for prefix in sorted(set(seen) | set(retired)):
        live, dead = seen.get(prefix, []), retired.get(prefix, [])
        nums = live + dead
        dupes = {n for n in nums if nums.count(n) > 1}
        if dupes:
            errors.append(f"{prefix}: duplicate IDs {sorted(dupes)} (live and/or retired)")
        expected = list(range(1, len(nums) + 1))
        if sorted(nums) != expected:
            errors.append(
                f"{prefix}: live + retired numbering must run 1..{len(nums)}, got {sorted(nums)}"
            )

    # 2. Verifiable requirements state a verification method.
    for req in reqs:
        if req["prefix"] in VERIFIABLE and not VERIFY_RE.search(req["body"]):
            errors.append(f"{req['id']} (line {req['line']}): missing *(verify: ...)* clause")

    # 3. Every citation resolves.
    cited: set[str] = set()
    for req in reqs:
        for path in req["cites"]:
            if path.endswith(".yaml"):  # a filename, not a citation
                continue
            try:
                resolve(params, path)
            except KeyError:
                errors.append(f"{req['id']}: cites undefined parameter `{path}`")
            else:
                cited.add(path)

    # 4. Every top-level parameter section is cited by at least one requirement.
    for section in params:
        if not any(c == f"params.{section}" or c.startswith(f"params.{section}.") for c in cited):
            errors.append(f"params.{section}: defined but cited by no requirement")

    errors.extend(validate_arithmetic(params))
    return errors


def validate_arithmetic(params: dict) -> list[str]:
    """The numbers that a reader would otherwise have to trust."""
    errors = []
    datasets = params.get("datasets", {})
    bandwidth = params.get("bandwidth", {})
    ratios = bandwidth.get("ratios", {})

    for name, spec in datasets.items():
        if not isinstance(spec, dict):  # a scalar policy key, not a dataset entry
            continue
        h, w, c = spec["image_size"]
        if spec["n"] != h * w * c:
            errors.append(f"datasets.{name}.n = {spec['n']} but image_size gives {h * w * c}")

    if bandwidth.get("core_ratio") not in ratios:
        errors.append(f"bandwidth.core_ratio {bandwidth.get('core_ratio')!r} is not a key of ratios")

    for name, budgets in bandwidth.get("k_symbols", {}).items():
        if name not in datasets:
            errors.append(f"bandwidth.k_symbols.{name}: no such dataset")
            continue
        n = datasets[name]["n"]
        if set(budgets) != set(ratios):
            errors.append(f"bandwidth.k_symbols.{name}: ratio keys differ from bandwidth.ratios")
            continue
        for ratio_key, k in budgets.items():
            expected = Fraction(ratios[ratio_key]) * n
            if expected.denominator != 1:
                errors.append(
                    f"bandwidth.k_symbols.{name}.{ratio_key}: {ratios[ratio_key]} of n={n} "
                    f"is not a whole number of symbols"
                )
            elif int(expected) != k:
                errors.append(
                    f"bandwidth.k_symbols.{name}.{ratio_key} = {k}, expected {int(expected)} "
                    f"({ratios[ratio_key]} of n={n})"
                )
    return errors


# ------------------------------------------------------------------------ rendering


def fmt(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_params_yaml(params: dict) -> str:
    header = (
        "# GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT\n"
        "# Edit the ```yaml params block in spec/SPEC.md and regenerate.\n"
    )
    return header + yaml.safe_dump(params, sort_keys=False, default_flow_style=False)


def render_datasheet(params: dict, reqs: list[dict]) -> str:
    leaves = flatten(params)
    citations = [(req["id"], c) for req in reqs for c in req["cites"]]

    def cited_by(path: str) -> str:
        ids = [
            rid
            for rid, c in citations
            if c == path or path.startswith(c + ".") or c.startswith(path + ".")
        ]
        return ", ".join(sorted(set(ids), key=sort_id)) or "-"

    out = [
        BANNER,
        "",
        "# Datasheet",
        "",
        f"Every committed parameter, flattened. Normative source: [`{SPEC_REL}`]({SPEC_REL}) §4.",
        "",
    ]
    for section in params:
        out += [
            f"## {section}",
            "",
            "| Parameter | Value | Cited by |",
            "| --- | --- | --- |",
        ]
        for path, value in leaves:
            if path == f"params.{section}" or path.startswith(f"params.{section}."):
                key = path[len("params.") :]
                out.append(f"| `{key}` | {fmt(value)} | {cited_by(path)} |")
        out.append("")
    return "\n".join(out)


def sort_id(rid: str) -> tuple[str, int]:
    prefix, num = rid.split("-")
    return prefix, int(num)


def render_concern(stem: str, title: str, params: dict, reqs: list[dict], stones: list[dict]) -> str:
    mine = [r for r in reqs if CONCERNS[r["prefix"]][0] == stem]
    mine_dead = [s for s in stones if CONCERNS[s["prefix"]][0] == stem]
    out = [
        BANNER,
        "",
        f"# {title}",
        "",
        f"Requirements extracted from [`{SPEC_REL}`](../{SPEC_REL}). "
        "This view is for focused reading and review; the spec text is normative.",
        "",
    ]
    for prefix in sorted({r["prefix"] for r in mine}):
        group = sorted((r for r in mine if r["prefix"] == prefix), key=lambda r: r["num"])
        out += [f"## {prefix}", ""]
        out += [f"- **{r['id']}** — {r['body']}" for r in group]
        out.append("")

    if mine_dead:
        out += ["## Retired", ""]
        out += [
            f"- ~~**{s['id']}**~~ — {s['body']}"
            for s in sorted(mine_dead, key=lambda s: (s["prefix"], s["num"]))
        ]
        out.append("")

    cited = sorted({c for r in mine for c in r["cites"] if not c.endswith(".yaml")})
    if cited:
        out += ["## Parameters referenced here", "", "| Parameter | Value |", "| --- | --- |"]
        for path in cited:
            value = resolve(params, path)
            rendered = fmt(value) if not isinstance(value, dict) else "*(see datasheet)*"
            out.append(f"| `{path[len('params.'):]}` | {rendered} |")
        out.append("")
    return "\n".join(out)


def build(params: dict, reqs: list[dict], stones: list[dict]) -> dict[Path, str]:
    files = {
        REPO / "spec" / "params.generated.yaml": render_params_yaml(params),
        REPO / "spec" / "DATASHEET.md": render_datasheet(params, reqs),
    }
    for stem, title in sorted(set(CONCERNS.values())):
        files[REPO / "spec" / "concerns" / f"{stem}.md"] = render_concern(
            stem, title, params, reqs, stones
        )
    return files


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate without writing")
    args = ap.parse_args()

    text = SPEC.read_text(encoding="utf-8")
    try:
        params = parse_params(text)
        reqs = parse_requirements(text)
        stones = parse_tombstones(text)
    except SpecError as exc:
        print(f"SPEC.md: {exc}", file=sys.stderr)
        return 1

    errors = validate(params, reqs, stones)
    if errors:
        print(f"{len(errors)} spec validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    files = build(params, reqs, stones)

    if args.check:
        drifted = [
            path
            for path, content in files.items()
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        if drifted:
            print("generated files are stale; run tools/gen_spec_views.py:", file=sys.stderr)
            for path in drifted:
                print(f"  - {path.relative_to(REPO)}", file=sys.stderr)
            return 1
        print(
            f"ok: {len(reqs)} requirements ({len(stones)} retired), "
            f"{len(files)} generated files up to date"
        )
        return 0

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    print(f"ok: {len(reqs)} requirements validated ({len(stones)} retired)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
