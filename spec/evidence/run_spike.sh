#!/usr/bin/env bash
# W0 LDPC spike — resumable runner. Safe to re-run any number of times.
#
# Everything lives outside /tmp deliberately: the original run used the session
# scratchpad, which may not survive a reboot. Wheels already downloaded sit in
# ~/.cache/pip and ARE persistent, so a rebuild after a wipe costs disk time
# rather than bandwidth — which matters on a slow link.
#
#   ./run_spike.sh          install if needed, then run the spike
#   ./run_spike.sh install  install only, stop before running
#   ./run_spike.sh run      run only, assume the venv is ready
#
set -uo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
# Override to keep the venv out of the repository when running from spec/evidence/.
VENV="${SPIKE_VENV:-$D/venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
LOG="$D/install.log"
CU_INDEX="https://download.pytorch.org/whl/cu130"
MODE="${1:-all}"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

have_torch_cuda() {
    [ -x "$PY" ] && "$PY" - <<'EOF' 2>/dev/null
import sys
try:
    import torch
    sys.exit(0 if torch.version.cuda else 1)
except Exception:
    sys.exit(1)
EOF
}

have_sionna() {
    [ -x "$PY" ] && "$PY" -c "import sionna.phy" >/dev/null 2>&1
}

if [ "$MODE" != "run" ]; then
    step "venv"
    [ -x "$PY" ] || python3 -m venv "$VENV"
    "$PY" --version

    step "torch (CUDA build — a bare 'pip install torch' gives the CPU build)"
    if have_torch_cuda; then
        echo "already present: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
    else
        # Resumable: completed wheels come from ~/.cache/pip without touching the
        # network. Only a wheel interrupted mid-download is re-fetched.
        "$PIP" install --index-url "$CU_INDEX" torch 2>&1 | tee -a "$LOG" | tail -3
        if ! have_torch_cuda; then
            echo "!! torch present but CPU-only, or install incomplete."
            echo "!! Fallback rung 1: pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1'"
            echo "!! Fallback rung 2: uv python install 3.13 && rebuild this venv on 3.13"
            exit 1
        fi
    fi

    step "sionna (no-rt: PHY only, skips the ray tracer build)"
    if have_sionna; then
        echo "already present"
    else
        "$PIP" install sionna-no-rt pyyaml 2>&1 | tee -a "$LOG" | tail -3 \
            || { echo "!! sionna-no-rt failed; fallback is the full 'sionna' package"; exit 1; }
    fi
fi

[ "$MODE" = "install" ] && { echo; echo "install done — run './run_spike.sh run' next"; exit 0; }

step "spike"
"$PY" "$D/spike_ldpc.py"
rc=$?
echo
echo "record: $D/g9_spike_record.json"
exit $rc
