"""Load the generated parameter file (SR-1).

`spec/params.generated.yaml` is generated from `spec/SPEC.md` by
`tools/gen_spec_views.py` and is the single source of truth for every
experiment-affecting constant. Editing the generated file directly is a mistake
that `--check` catches; edit `SPEC.md` and regenerate.

Kept dependency-light on purpose: PyYAML only, no torch import, so anything in
the project can read a parameter without pulling in the runtime stack.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "spec" / "params.generated.yaml"


@functools.cache
def load_params() -> dict[str, Any]:
    """The parsed parameter tree. Cached: the file does not change within a run."""
    return yaml.safe_load(PARAMS_PATH.read_text())


def get(path: str) -> Any:
    """Resolve a dotted parameter path, e.g. `get("environment.lock_file")`.

    Accepts the `params.` prefix used when citing a parameter in `SPEC.md`, so a
    citation can be pasted straight into code without editing.

    Raises KeyError naming the full path rather than the last component -- a bare
    `KeyError: 'lock_file'` from six levels down is not a useful error.
    """
    node: Any = load_params()
    for part in path.removeprefix("params.").split("."):
        try:
            node = node[part]
        except (KeyError, TypeError):
            raise KeyError(f"no such parameter: {path}") from None
    return node
