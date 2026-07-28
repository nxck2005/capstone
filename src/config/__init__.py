"""Configuration plumbing (SR-1).

Code reads `spec/params.generated.yaml`; it never parses `SPEC.md` and never
hard-codes an experiment-affecting constant.

This package is deliberately small in W1 batch 1 -- just the params loader, which
`src/env.py` needs. Batch 2 grows it into the run-config layer that derives a
`config_hash` (SR-13) and adds the literal lint (`tools/check_literals.py`).
"""

from .params import PARAMS_PATH, REPO_ROOT, load_params

__all__ = ["PARAMS_PATH", "REPO_ROOT", "load_params"]
