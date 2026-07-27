#!/usr/bin/env python3
"""TS 38.212 packetisation conformance check (BR-10, AM-49, AM-58).

Pure arithmetic: no GPU, no Sionna, no network. Runs anywhere the project venv
runs, in under a second, and emits the per-configuration record BR-10's verify
clause requires -- A, TB CRC type, base graph, lifting size, code-block count,
every E_r, filler per block and in total, and effective code rate.

This is deliberately *separate* from `spike_ldpc.py`, which is the archived W0
run and stays as the record of what was measured then (AM-24). What changed
since is the packetisation contract, not the measurement.

**Rewritten by AM-58.** The first version (AM-49) reported zero failures while
its own rows violated four things it claimed to enforce:

  (a) `payload_solver` promises the largest *byte-aligned* A; it maximised over
      every integer A, leaving 92 of 215 rows unaligned;
  (b) TS 38.212 5.2.2 defines K' = B'/C by exact division -- NR gets that for
      free from the 38.214 TBS quantisation, which this project does not use --
      and `segment()` silently applied `ceil`, so 21 rows had C*K' > B' and the
      surplus bits were accounted nowhere;
  (c) on BG2 the encoded systematic length is K = 10*Z regardless of which K_b
      selected the lifting size, and filler was computed as K_b*Z - K', which
      undercounts 47 of 103 BG2 rows;
  (d) the base-graph minimum rate was checked strictly (`>`) and against E[0],
      the *smallest* E_r -- i.e. against the highest per-block rate. The floor
      binds on the *worst* block, and Sionna accepts equality: its guard is
      `if bg == "bg1" and r < 1/3: raise`, so rate exactly 1/3 is legal.

Fixing all four keeps 215 configurations feasible and every headline-dataset
configuration feasible; it moves source capacity in 18 rows by -8 to +1 byte,
and reduces the reported minimum-rate clamps from six to three.

Usage:
    python spec/evidence/check_packetisation.py            # summary
    python spec/evidence/check_packetisation.py --json OUT # full record
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
PARAMS = REPO / "spec" / "params.generated.yaml"

MOD_BITS = {"bpsk": 1, "qpsk": 2, "qam16": 4}

# Datasets whose every ladder ratio is a proof obligation. `live` marks today's
# three provisional selections, but ER-3 may select any rung at G-8, so the
# obligation cannot be scoped to them (AM-58).
HEADLINE_DATASETS = ("imagenette160", "stl10")

# TS 38.212 Table 5.3.2-1. Z = a * 2^j, capped at 384.
LIFTING_SET = sorted(
    {a * 2**j for a in (2, 3, 5, 7, 9, 11, 13, 15) for j in range(8) if a * 2**j <= 384}
)

# Sionna 2.0.1 LDPC5GEncoder refuses a per-codeword rate above this.
MAX_CODERATE = 0.95


def tb_crc_bits(A: int, tb: dict) -> int:
    """TS 38.212 7.2.1: CRC24A above the threshold, CRC16 at or below it."""
    return tb["large_bits"] if A > tb["threshold_payload_bits"] else tb["small_bits"]


def base_graph(A: int, R: float) -> int:
    """TS 38.212 7.2.2, from the TRANSPORT BLOCK size and rate.

    The distinction that matters: this is decided once, before segmentation, and
    then every code block uses it. Deriving it after segmentation from a per-code-
    block rate reproduces the same arithmetic at the wrong granularity, and can
    put a short final block on a different graph from the rest of its own
    transport block.
    """
    if A <= 292 or (A <= 3824 and R <= 0.67) or R <= 0.25:
        return 2
    return 1


def kb_for(bg: int, B: int) -> int:
    """Systematic-column count used to SELECT the lifting size (TS 38.212 5.2.2).

    Note this is *not* the encoded systematic length: see `systematic_length`.
    Conflating the two is defect (c) above.
    """
    if bg == 1:
        return 22
    if B > 640:
        return 10
    if B > 560:
        return 9
    if B > 192:
        return 8
    return 6


def systematic_length(bg: int, Z: int) -> int:
    """TS 38.212 5.2.2: K = 22*Z on BG1 and K = 10*Z on BG2, always."""
    return 22 * Z if bg == 1 else 10 * Z


def lifting_size(bg: int, B: int, K_prime: int) -> int | None:
    """Smallest Z in the standard set with K_b*Z >= K'."""
    kb = kb_for(bg, B)
    return next((z for z in LIFTING_SET if kb * z >= K_prime), None)


def segment(B: int, Kcb: int, L_cb: int) -> tuple[int, int] | None:
    """TS 38.212 5.2.2 code-block segmentation.

    Returns (C, K') or None when B' is not divisible by C. The standard defines
    K' = B'/C as an exact division and its own procedure assumes integrality;
    NR guarantees it upstream in TS 38.214's TBS quantisation, which this project
    does not use because it supplies arbitrary source sizes. So the solver must
    choose an A for which the division is exact rather than silently rounding.
    """
    if B <= Kcb:
        C, B_prime = 1, B
    else:
        C = math.ceil(B / (Kcb - L_cb))
        B_prime = B + C * L_cb
    if B_prime % C:
        return None
    return C, B_prime // C


def rate_match_E(G: int, C: int, Q_m: int, N_L: int = 1) -> list[int]:
    """TS 38.212 5.4.2.1. E_r must be a multiple of N_L*Q_m, so floor(G/C) is wrong."""
    unit = N_L * Q_m
    gamma = (G // unit) % C
    lo = unit * ((G // unit) // C)
    hi = unit * math.ceil((G // unit) / C)
    return [lo if r < C - gamma else hi for r in range(C)]


def packetise(A: int, G: int, Q_m: int, R: float, bl: dict) -> dict | None:
    """Complete packetisation of one byte-aligned payload A, or None if illegal."""
    if A <= 0 or A % 8:
        return None
    L_tb = tb_crc_bits(A, bl["tb_crc"])
    B = A + L_tb
    bg = base_graph(A, R)
    Kcb, L_cb = bl["code_block_max_bits"][f"bg{bg}"], bl["cb_crc_bits"]
    seg = segment(B, Kcb, L_cb)
    if seg is None:
        return None
    C, K_prime = seg
    Z = lifting_size(bg, B, K_prime)
    if Z is None:
        return None
    K = systematic_length(bg, Z)
    if K < K_prime:
        return None
    E = rate_match_E(G, C, Q_m)
    return {
        "A": A,
        "source_bytes": A // 8,
        "tb_crc_type": "crc24a" if L_tb == 24 else "crc16",
        "tb_crc_bits": L_tb,
        "B": B,
        "base_graph": bg,
        "code_block_max_bits": Kcb,
        "num_codeblocks": C,
        "cb_crc_bits_total": C * L_cb if C > 1 else 0,
        "B_prime": B + C * L_cb if C > 1 else B,
        "K_prime": K_prime,
        "K_b_for_lifting": kb_for(bg, B),
        "lifting_size": Z,
        "K": K,
        "filler_bits_per_block": K - K_prime,
        "filler_bits_total": (K - K_prime) * C,
        "E": E,
        "E_sum": sum(E),
        # Deliberately unrounded: the floor comparison below is decided at the
        # twelfth decimal place, and rounding to nine put exactly-1/3 rates
        # *below* a 1/3 floor -- which is how three spurious clamps appeared.
        "min_block_code_rate": K_prime / max(E),
        "max_block_code_rate": K_prime / min(E),
        "effective_code_rate": round(K_prime / max(E), 6),
    }


def floor_for(bg: int, bl: dict) -> float:
    return bl["ldpc_bg1_min_coderate"] if bg == 1 else bl["bg2_min_coderate"]


def meets_floor(r: dict, bl: dict) -> bool:
    """Every code block must clear its base graph's minimum rate.

    Checked on the WORST block -- the largest E_r, hence the lowest K'/E_r -- and
    with equality accepted, because Sionna's guard is a strict `r < floor` raise
    and BG1's mother code sits at exactly 1/3 (22Z systematic over 66Z
    transmitted). Checking `>` against E[0] tested the best block against the
    wrong predicate and manufactured three clamps the library never needed.
    """
    floor = floor_for(r["base_graph"], bl)
    return r["min_block_code_rate"] >= floor * (1 - 1e-9)


def solve(G: int, Q_m: int, R: float, bl: dict) -> dict:
    """`params.baseline.payload_solver`, made unique.

    Step 1 -- nominal: the largest byte-aligned A whose complete packetisation is
    legal and fits within floor(G * R_nominal).
    Step 2 -- named clamp, only if step 1's candidate misses a per-block rate
    floor: the SMALLEST larger byte-aligned A that satisfies every floor. This
    raises the realized rate above nominal, which is why both are recorded.
    """
    B_nominal = math.floor(G * R)
    ceiling = None
    for A in range((B_nominal // 8) * 8, 0, -8):
        if A + tb_crc_bits(A, bl["tb_crc"]) > B_nominal:
            continue
        r = packetise(A, G, Q_m, R, bl)
        if r:
            ceiling = r
            break
    if ceiling is None:
        return {"feasible": False, "reason": "no_legal_byte_aligned_A_within_nominal_budget",
                "B_nominal": B_nominal}
    if meets_floor(ceiling, bl):
        return {"feasible": True, "clamped": False, "B_nominal": B_nominal,
                "nominal_rate": round(R, 9), **ceiling}
    A = ceiling["A"] + 8
    for _ in range(8192):
        r = packetise(A, G, Q_m, R, bl)
        if r and meets_floor(r, bl):
            return {"feasible": True, "clamped": True, "B_nominal": B_nominal,
                    "nominal_rate": round(R, 9),
                    "clamp_reason": f"bg{r['base_graph']}_min_coderate", **r}
        A += 8
    return {"feasible": False, "reason": "min_coderate_clamp_did_not_converge",
            "B_nominal": B_nominal}


def er9_feasible(P: dict, A_bits: int) -> list[tuple[int, int]]:
    """(width, bits) pairs ER-9 can fit into a payload of A_bits.

    Budgeted against the RAW fixed-width stream plus the framing selector bit,
    not against the entropy-coded length. A range coder can expand on
    unfavourable data, so a control sized on the compressed length is not
    guaranteed to fit; sizing on `min(range-coded, raw)` with a counted selector
    bit makes the budget a proven bound rather than an expectation (AM-58).

    This is also the check whose absence let ER-9 ship infeasible: with the
    transmitted feature dimension pinned to the learned system's, D was 2k reals
    against a budget of Q_m*R/2 bits per real -- 0.167 at BPSK rate 1/3, against
    a 2-bit floor.
    """
    control = P["digital_semantic_control"]
    selector = control.get("framing_selector_bits", 1)
    return [
        (dim, bits)
        for dim in control["transmit_dim_grid"]
        for bits in control["quantiser_bits_grid"]
        if dim * bits + selector <= A_bits
    ]


def invariants(r: dict, G: int, Q_m: int, bl: dict) -> list[str]:
    """The assertions AM-58 requires over the whole grid."""
    bad = []
    if r["A"] % 8:
        bad.append(f"A={r['A']} is not byte-aligned")
    if r["A"] != 8 * r["source_bytes"]:
        bad.append("source accounting does not reconstruct A")
    if r["B_prime"] % r["num_codeblocks"]:
        bad.append(f"B'={r['B_prime']} is not divisible by C={r['num_codeblocks']}")
    if r["num_codeblocks"] * r["K_prime"] != r["B_prime"]:
        bad.append("C*K' != B'")
    if r["B_prime"] != r["A"] + r["tb_crc_bits"] + r["cb_crc_bits_total"]:
        bad.append("B' does not reconcile with A + TB CRC + per-block CRCs")
    expected_K = systematic_length(r["base_graph"], r["lifting_size"])
    if r["K"] != expected_K:
        bad.append(f"K={r['K']} != {'22' if r['base_graph'] == 1 else '10'}*Z")
    if r["filler_bits_total"] != (r["K"] - r["K_prime"]) * r["num_codeblocks"]:
        bad.append("filler total != C * (K - K')")
    if r["filler_bits_per_block"] < 0:
        bad.append("negative filler")
    if r["E_sum"] != G:
        bad.append(f"E sums to {r['E_sum']}, not G={G}")
    if any(e % Q_m for e in r["E"]):
        bad.append(f"an E_r is not a multiple of Q_m={Q_m}")
    if len(r["E"]) != r["num_codeblocks"]:
        bad.append("E has the wrong length")
    if not meets_floor(r, bl):
        bad.append(f"min block rate {r['min_block_code_rate']} below "
                   f"{floor_for(r['base_graph'], bl)}")
    if r["max_block_code_rate"] > MAX_CODERATE:
        bad.append(f"max block rate {r['max_block_code_rate']} above {MAX_CODERATE}")
    return bad


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the full per-configuration record")
    args = ap.parse_args()

    P = yaml.safe_load(PARAMS.read_text())
    bw, bl = P["bandwidth"], P["baseline"]
    live_ratios = {bw["crossover_ratio"], bw["efficiency_ratio"], bw["low_ratio_operating_point"]}

    rows, failures = [], []
    for ds, ratios in bw["k_symbols"].items():
        for rk, k in ratios.items():
            for mod in bl["modulations"]:
                Q_m = MOD_BITS[mod]
                G = k * Q_m
                for rs in bl["ldpc_rates"]:
                    num, den = (int(v) for v in rs.split("/"))
                    r = solve(G, Q_m, num / den, bl)
                    tag = f"{ds}/{rk}/{mod}/{rs}"
                    obligation = ds in HEADLINE_DATASETS
                    row = {"tag": tag, "dataset": ds, "ratio": rk, "modulation": mod,
                           "nominal_rate_str": rs, "k": k, "G": G,
                           "obligation": obligation, "live": obligation and rk in live_ratios,
                           **r}
                    if r["feasible"]:
                        row["er9_options"] = er9_feasible(P, r["A"])
                    rows.append(row)

                    if not r["feasible"]:
                        if obligation:
                            failures.append(f"{tag}: INFEASIBLE ({r['reason']})")
                        continue
                    for msg in invariants(r, G, Q_m, bl):
                        failures.append(f"{tag}: {msg}")
                    if obligation and not row["er9_options"]:
                        failures.append(f"{tag}: ER-9 has no feasible (width, bits) pair")

    obl = [r for r in rows if r["obligation"]]
    live = [r for r in rows if r["live"]]
    feas = [r for r in rows if r["feasible"]]
    clamped = [r for r in feas if r["clamped"]]
    print(f"configurations     : {len(rows)}  ({len(obl)} proof-obligation, {len(live)} live)")
    print(f"feasible           : {len(feas)}  ({sum(1 for r in obl if r['feasible'])} obligation)")
    print(f"byte-aligned A     : {sum(1 for r in feas if r['A'] % 8 == 0)}/{len(feas)}")
    print(f"B' divisible by C  : {sum(1 for r in feas if r['B_prime'] % r['num_codeblocks'] == 0)}"
          f"/{len(feas)}")
    print(f"zero-slack E sum   : {sum(1 for r in feas if r['E_sum'] == r['G'])}/{len(feas)}")
    print(f"CRC16 selected     : {sum(1 for r in feas if r['tb_crc_bits'] == 16)}"
          f"  ({sum(1 for r in live if r.get('tb_crc_bits') == 16)} live)")
    print(f"base graph 2       : {sum(1 for r in feas if r['base_graph'] == 2)}"
          f"  ({sum(1 for r in live if r.get('base_graph') == 2)} live)")
    print(f"min-rate clamped   : {len(clamped)}  -> {[r['tag'] for r in clamped]}")
    print(f"failures           : {len(failures)}")
    for f in failures:
        print(f"   - {f}")

    if args.json:
        args.json.write_text(json.dumps({
            "source": "spec/params.generated.yaml",
            "standard": bl["ldpc_standard"],
            "version": bl["ldpc_standard_version"],
            "status": "current",
            "generated_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "params_sha256": _sha256(PARAMS),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "python_version": sys.version.split()[0],
            "configurations": rows,
            "failures": failures,
        }, indent=2))
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
