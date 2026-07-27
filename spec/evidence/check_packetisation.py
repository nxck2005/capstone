#!/usr/bin/env python3
"""TS 38.212 packetisation conformance check (BR-10, AM-49).

Pure arithmetic: no GPU, no Sionna, no network. Runs anywhere the project venv
runs, in under a second, and emits the per-configuration record BR-10's verify
clause requires -- A, TB CRC type, base graph, lifting size, code-block count,
every E_r, filler bits and effective code rate.

This is deliberately *separate* from `spike_ldpc.py`, which is the archived W0
run and stays as the record of what was measured then (AM-24). What changed
since is the packetisation contract, not the measurement:

  (a) the transport-block CRC is conditional on payload size, not always 24;
  (b) the maximum code-block size depends on the base graph, not always 8448;
  (c) the base graph is chosen once per transport block from (A, R) *before*
      segmentation, not per code block from the post-segmentation rate.

Usage:
    python spec/evidence/check_packetisation.py            # summary
    python spec/evidence/check_packetisation.py --json OUT # full record
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
PARAMS = REPO / "spec" / "params.generated.yaml"

MOD_BITS = {"bpsk": 1, "qpsk": 2, "qam16": 4}

# TS 38.212 Table 5.3.2-1. Z = a * 2^j, capped at 384.
LIFTING_SET = sorted(
    {a * 2**j for a in (2, 3, 5, 7, 9, 11, 13, 15) for j in range(8) if a * 2**j <= 384}
)


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
    """Systematic-column count used to pick the lifting size (TS 38.212 5.2.2)."""
    if bg == 1:
        return 22
    if B > 640:
        return 10
    if B > 560:
        return 9
    if B > 192:
        return 8
    return 6


def lifting_size(bg: int, B: int, K_r: int) -> int | None:
    """Smallest Z in the standard set with Kb*Z >= K_r."""
    kb = kb_for(bg, B)
    return next((z for z in LIFTING_SET if kb * z >= K_r), None)


def segment(B: int, Kcb: int, L_cb: int) -> tuple[int, int]:
    """TS 38.212 5.2.2 code-block segmentation. Unchanged from the W0 spike."""
    if B <= Kcb:
        return 1, B
    C = math.ceil(B / (Kcb - L_cb))
    return C, math.ceil((B + C * L_cb) / C)


def rate_match_E(G: int, C: int, Q_m: int, N_L: int = 1) -> list[int]:
    """TS 38.212 5.4.2.1. E_r must be a multiple of N_L*Q_m, so floor(G/C) is wrong."""
    unit = N_L * Q_m
    gamma = (G // unit) % C
    lo = unit * ((G // unit) // C)
    hi = unit * math.ceil((G // unit) / C)
    return [lo if r < C - gamma else hi for r in range(C)]


def analyse(G: int, Q_m: int, num: int, den: int, bl: dict) -> dict:
    """Full packetisation for one configuration.

    Mirrors `params.baseline.payload_solver`: the largest payload A whose complete
    packetisation fits exactly within G. AM-24's minimum-coderate clamp is preserved
    and made base-graph aware -- BG1 supports only coderate above 1/3, BG2 down to
    1/5, and which floor applies now depends on a selection made before segmentation
    rather than inferred after it.
    """
    tb, R = bl["tb_crc"], num / den
    kcb, cb_l = bl["code_block_max_bits"], bl["cb_crc_bits"]
    floors = {1: bl["ldpc_bg1_min_coderate"], 2: bl["bg2_min_coderate"]}

    B_nominal = math.floor(G * R)
    A = max((a for a in range(B_nominal + 1) if a + tb_crc_bits(a, tb) <= B_nominal), default=0)
    if A <= 0 or A // 8 < 1:
        return {"feasible": False, "reason": "budget_below_tb_crc_plus_one_byte",
                "B_nominal": B_nominal, "A": A}

    clamped = False
    for _ in range(64):
        L = tb_crc_bits(A, tb)
        B = A + L
        bg = base_graph(A, R)
        C, K_r = segment(B, kcb[f"bg{bg}"], cb_l)
        E = rate_match_E(G, C, Q_m)
        if K_r / E[0] > floors[bg]:
            Z = lifting_size(bg, B, K_r)
            return {
                "feasible": True, "A": A, "tb_crc_type": "crc24a" if L == 24 else "crc16",
                "tb_crc_bits": L, "B": B, "base_graph": bg, "code_block_max_bits": kcb[f"bg{bg}"],
                "lifting_size": Z, "num_codeblocks": C, "K_r": K_r, "E": E, "E_sum": sum(E),
                "filler_bits": (kb_for(bg, B) * Z - K_r) if Z else None,
                "padding_bits": A - 8 * (A // 8), "source_bytes": A // 8,
                "effective_code_rate": round(K_r / E[0], 6), "clamped": clamped,
            }
        A += 1
        clamped = True
    return {"feasible": False, "reason": "min_coderate_clamp_did_not_converge"}


def er9_feasible(P: dict, ds: str, A_bits: int) -> list[tuple[int, int]]:
    """(width, bits) pairs ER-9 can actually fit into a payload of A_bits.

    This is the check whose absence let ER-9 ship infeasible: with the transmitted
    feature dimension pinned to the learned system's, D was 2k reals against a
    budget of Q_m*R/2 bits per real -- 0.167 at BPSK rate 1/3, against a 2-bit
    floor. Entropy coding is ignored here, so the answer is conservative: it can
    only make a listed pair cheaper, never an unlisted one affordable.
    """
    control = P["digital_semantic_control"]
    return [
        (dim, bits)
        for dim in control["transmit_dim_grid"]
        for bits in control["quantiser_bits_grid"]
        if dim * bits <= A_bits
    ]


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
                    r = analyse(G, Q_m, num, den, bl)
                    tag = f"{ds}/{rk}/{mod}/{rs}"
                    live = ds in ("imagenette160", "stl10") and rk in live_ratios
                    if r["feasible"]:
                        r["er9_options"] = er9_feasible(P, ds, r["A"])
                    rows.append({"tag": tag, "dataset": ds, "ratio": rk, "modulation": mod,
                                 "nominal_rate": rs, "k": k, "G": G, "live": live, **r})
                    if not r["feasible"]:
                        if live:
                            failures.append(f"{tag}: INFEASIBLE ({r['reason']})")
                        continue
                    if r["E_sum"] != G:
                        failures.append(f"{tag}: E sums to {r['E_sum']}, not G={G}")
                    if any(e % Q_m for e in r["E"]):
                        failures.append(f"{tag}: an E_r is not a multiple of Q_m={Q_m}")
                    if r["lifting_size"] is None:
                        failures.append(f"{tag}: no lifting size admits K_r={r['K_r']}")
                    if live and not r["er9_options"]:
                        failures.append(f"{tag}: ER-9 has no feasible (width, bits) pair")

    live = [r for r in rows if r["live"]]
    feas = [r for r in rows if r["feasible"]]
    print(f"configurations   : {len(rows)}  ({len(live)} live)")
    print(f"feasible         : {len(feas)}  ({sum(1 for r in live if r['feasible'])} live)")
    print(f"zero-slack E sum : {sum(1 for r in feas if r['E_sum'] == r['G'])}/{len(feas)}")
    print(f"CRC16 selected   : {sum(1 for r in feas if r['tb_crc_bits'] == 16)}"
          f"  ({sum(1 for r in live if r.get('tb_crc_bits') == 16)} live)")
    print(f"base graph 2     : {sum(1 for r in feas if r['base_graph'] == 2)}"
          f"  ({sum(1 for r in live if r.get('base_graph') == 2)} live)")
    print(f"min-rate clamped : {sum(1 for r in feas if r['clamped'])}"
          f"  -> {[r['tag'] for r in feas if r['clamped']]}")
    print(f"failures         : {len(failures)}")
    for f in failures:
        print(f"   - {f}")

    if args.json:
        args.json.write_text(json.dumps(
            {"source": "spec/params.generated.yaml", "standard": bl["ldpc_standard"],
             "version": bl["ldpc_standard_version"], "configurations": rows,
             "failures": failures}, indent=2))
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
