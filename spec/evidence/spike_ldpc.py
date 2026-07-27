#!/usr/bin/env python3
"""W0 LDPC spike — the last open item in G-9.

Throwaway. Nothing here lands in the repo; the findings get transcribed into
spec/SPEC.md (DEC-10, G-9, §16) as amendment AM-23.

Every constant comes from spec/params.generated.yaml, never from this file, so
each number in the record traces back to the spec. That is the discipline SR-1
will require of real code, and it means a spec edit cannot silently invalidate
the record.

Checks, in order of how much they hurt if wrong:
  1  install / CUDA actually present (the CPU-build trap)
  2  exact-n encoding + TS 38.212 §5.4.2.1 rate matching   <- BR-3 rests on this
  3  cn_update spelling accepted by the library
  4  encode -> QPSK -> AWGN -> decode sanity at every rate
  5  batched decode throughput at ldpc_max_iters
  6  projected ER-1 wall clock, one ratio and two
  7  smallest workable payload
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent


def _find_params() -> Path:
    """Locate spec/params.generated.yaml without hard-coding anyone's home dir.

    Walks up from this file, so it works both from the throwaway spike directory
    and from spec/evidence/ inside the repository. $CAPSTONE_PARAMS overrides.
    """
    import os
    if env := os.environ.get("CAPSTONE_PARAMS"):
        return Path(env)
    for base in [OUT, *OUT.parents]:
        for cand in (base / "params.generated.yaml",
                     base / "spec" / "params.generated.yaml"):
            if cand.is_file():
                return cand
    raise SystemExit("cannot find spec/params.generated.yaml; set $CAPSTONE_PARAMS")


PARAMS = _find_params()

results: dict = {"checks": {}}
failures: list[str] = []


def record(name: str, passed: bool, **detail):
    results["checks"][name] = {"pass": passed, **detail}
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}")
    for k, v in detail.items():
        print(f"        {k}: {v}")
    if not passed:
        failures.append(name)


# --------------------------------------------------------------- TS 38.212

def segment(B_bits: int, Kcb: int, L_cb: int) -> tuple[int, int]:
    """TS 38.212 §5.2.2 code-block segmentation.

    Returns (C, K_r): number of code blocks and per-block payload including the
    per-block CRC. Sionna does not provide this; DEC-10 says it is ours, which
    is exactly why the spike must prove the arithmetic lines up with what the
    encoder will accept.
    """
    if B_bits <= Kcb:
        return 1, B_bits
    C = math.ceil(B_bits / (Kcb - L_cb))
    B_prime = B_bits + C * L_cb
    return C, math.ceil(B_prime / C)


def base_graph(B_bits: int, R: float) -> int:
    """TS 38.212 §7.2.2 base-graph selection. Normative, so we do not override it.

    Sionna picks the graph internally; we reproduce the rule here only so the
    record can say *why* a configuration landed on BG1, which is what makes the
    rate-1/3 boundary below explicable rather than mysterious.
    """
    if B_bits <= 292 or (B_bits <= 3824 and R <= 0.67) or R <= 0.25:
        return 2
    return 1


def info_bits(G, num, den, Kcb, L_cb, Q_m):
    """Info-bit budget for a nominal rate, with the BG1 minimum-rate clamp.

    `floor(G*R)` is the natural budget, but at nominal rate 1/3 the *per-code-
    block* rate K_r/E_r can land a hair below 1/3 -- and BG1 supports only
    coderate > 1/3, since going lower needs repetition coding, which Sionna does
    not implement. Three of the 180 configurations hit it, at R = 0.333281
    against a limit of 0.333333. One is a live headline config, so it is not a
    corner case.

    The constraint binds on what the encoder is actually handed, after
    segmentation, so it has to be evaluated there -- checking B/G instead flags
    configurations that segmentation's own ceil() already rescued. Returns
    (B, C, K_r, E, clamped); the clamp moves the budget by one bit (0.02%).
    """
    B = math.floor(G * num / den)
    clamped = False
    for _ in range(64):                     # terminates in 1 step in practice
        C, K_r = segment(B, Kcb, L_cb)
        E = rate_match_E(G, C, Q_m)
        if base_graph(K_r, K_r / E[0]) != 1 or K_r / E[0] > 1 / 3:
            return B, C, K_r, E, clamped
        B += 1
        clamped = True
    raise RuntimeError(f"BG1 min-rate clamp did not converge for G={G} {num}/{den}")


def rate_match_E(G: int, C: int, Q_m: int, N_L: int = 1) -> list[int]:
    """TS 38.212 §5.4.2.1 per-code-block rate-matching output lengths.

    The subtlety worth having in a test: E_r must be a multiple of Q_m*N_L, so a
    naive floor(G/C) split is wrong. For G=51200, C=6, Q_m=2 the correct answer
    is two blocks at 8532 and four at 8534 -- both even -- summing to exactly G.
    floor(G/C) would give 8533, which is odd and would not be a valid E_r.
    """
    unit = N_L * Q_m
    gamma = (G // unit) % C
    lo = unit * ((G // unit) // C)
    hi = unit * math.ceil((G // unit) / C)
    return [lo if r < C - gamma else hi for r in range(C)]


def main() -> int:
    params = yaml.safe_load(PARAMS.read_text())
    bl = params["baseline"]
    bw = params["bandwidth"]
    ev = params["evaluation"]
    ds = params["datasets"]

    # Repo-relative when possible: this record is committed evidence, and an
    # absolute path pins it to one machine's home directory.
    try:
        results["params_source"] = str(PARAMS.resolve().relative_to(
            Path(__file__).resolve().parents[2]))
    except ValueError:
        results["params_source"] = PARAMS.name
    results["spec_values"] = {
        "ldpc_rates": bl["ldpc_rates"],
        "ldpc_max_iters": bl["ldpc_max_iters"],
        "code_block_max_bits": bl["code_block_max_bits"],
        "cb_crc_bits": bl["cb_crc_bits"],
        "tb_crc_bits": bl["tb_crc_bits"],
        "ldpc_decoder": bl["ldpc_decoder"],
        "ldpc_impl_version": bl["ldpc_impl_version"],
    }

    # ------------------------------------------------ 1. install / CUDA
    try:
        import torch

        cuda_ok = torch.cuda.is_available() and torch.version.cuda is not None
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        free_gb = total_gb = None
        if cuda_ok:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb, total_gb = round(free_b / 2**30, 2), round(total_b / 2**30, 2)
        import sionna
        import sionna.phy  # noqa: F401

        record(
            "1_install_cuda",
            cuda_ok,
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            sionna=getattr(sionna, "__version__", "unknown"),
            gpu=gpu,
            vram_total_gb=total_gb,
            vram_free_gb=free_gb,
            note="a plain 'pip install torch' yields the CPU build; cu130 index required",
        )
        if not cuda_ok:
            return 1
    except Exception as exc:
        record("1_install_cuda", False, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder

    dev = torch.device("cuda")

    # ---------------------------- 2. exact-n encoding + rate matching
    # The load-bearing check. Not "does the encoder accept some n", but "does the
    # full segmentation + per-block rate-matching arithmetic land on the channel
    # budget with zero slack", which is what BR-3's equal-channel-uses claim means.
    # The whole grid, not a sample: every dataset x ratio x modulation x rate the
    # spec admits. 180 configurations costs seconds and turns "we checked the
    # headline case" into "no configuration in the spec is unrealizable".
    mod_bits = {"bpsk": 1, "qpsk": 2, "qam16": 4}
    live_ratios = {bw["crossover_ratio"], bw["efficiency_ratio"],
                   bw["low_ratio_operating_point"]}
    cases = []
    for dataset, ratios in bw["k_symbols"].items():
        for ratio_key, k_sym in ratios.items():
            for mod in bl["modulations"]:
                Q_m = mod_bits[mod]
                G = k_sym * Q_m
                for rate_s in bl["ldpc_rates"]:
                    num, den = (int(v) for v in rate_s.split("/"))
                    # TB CRC already inside the info budget per budget_rule
                    B, C, K_r, E, clamped = info_bits(
                        G, num, den, bl["code_block_max_bits"],
                        bl["cb_crc_bits"], Q_m)
                    cases.append(
                        dict(dataset=dataset, ratio=ratio_key, mod=mod, Q_m=Q_m,
                             rate=rate_s, k_sym=k_sym, G=G, info=B, clamped=clamped,
                             C=C, K_r=K_r, E=E, E_sum=sum(E),
                             bg=base_graph(K_r, K_r / E[0]),
                             live=(dataset in ("imagenette160", "stl10")
                                   and ratio_key in live_ratios))
                    )

    exact_fail = []
    tested = 0
    for c in cases:
        tag = f"{c['dataset']}/{c['ratio']}/{c['mod']}/{c['rate']}"
        if sum(c["E"]) != c["G"]:
            exact_fail.append(f"{tag}: E sums to {sum(c['E'])} not G={c['G']}")
            continue
        for E_r in sorted(set(c["E"])):
            if E_r % (c["Q_m"]) != 0:
                exact_fail.append(f"{tag}: E_r={E_r} not a multiple of Q_m={c['Q_m']}")
                continue
            try:
                enc = LDPC5GEncoder(k=c["K_r"], n=E_r).to(dev)
                u = torch.randint(0, 2, (2, c["K_r"]), device=dev, dtype=torch.float32)
                out = enc(u)
                tested += 1
                if out.shape[-1] != E_r:
                    exact_fail.append(
                        f"{tag}: k={c['K_r']} n={E_r} -> emitted {out.shape[-1]}")
            except Exception as exc:
                exact_fail.append(f"{tag}: k={c['K_r']} n={E_r} raised "
                                  f"{type(exc).__name__}: {exc}")

    clamped = [f"{c['dataset']}/{c['ratio']}/{c['mod']}/{c['rate']} "
               f"k={c['K_r']} n={c['E'][0]} R={c['K_r'] / c['E'][0]:.6f}"
               for c in cases if c["clamped"]]

    # The specific arithmetic derived independently in the plan, asserted by value
    # so a library change cannot silently pass this.
    core_case = next(c for c in cases
                     if c["dataset"] == "imagenette160"
                     and c["ratio"] == bw["crossover_ratio"] and c["mod"] == "qpsk"
                     and c["rate"] == "5/6")
    canonical_ok = (core_case["C"] == 6
                    and sorted(set(core_case["E"])) == [8532, 8534]
                    and core_case["E"].count(8532) == 2
                    and core_case["E_sum"] == 51200)

    record(
        "2_exact_n_and_rate_matching",
        not exact_fail and canonical_ok,
        configs_tested=len(cases),
        live_configs=sum(1 for c in cases if c["live"]),
        encoder_calls=tested,
        base_graph_split={f"BG{g}": sum(1 for c in cases if c["bg"] == g)
                          for g in (1, 2)},
        canonical_core_case=dict(C=core_case["C"], K_r=core_case["K_r"],
                                 E=core_case["E"], E_sum=core_case["E_sum"]),
        canonical_matches_hand_derivation=canonical_ok,
        bg1_min_rate_clamps=clamped,
        failures=exact_fail[:10],
        note="BR-3 depends entirely on E summing to G with zero slack; the clamps "
             "are configs where floor(G*R) fell below BG1's coderate>1/3 limit",
    )

    # --------------------------------------------- 3. cn_update spelling
    accepted, rejected = [], {}
    probe_enc = LDPC5GEncoder(k=core_case["K_r"], n=core_case["E"][0]).to(dev)
    for cand in ("offset-minsum", "offset_min_sum", "minsum", "boxplus-phi"):
        try:
            LDPC5GDecoder(encoder=probe_enc, num_iter=2, cn_update=cand).to(dev)
            accepted.append(cand)
        except Exception as exc:
            rejected[cand] = f"{type(exc).__name__}: {exc}"[:120]
    spec_spelling = bl["ldpc_decoder"]
    record(
        "3_cn_update",
        "offset-minsum" in accepted or "offset_min_sum" in accepted,
        accepted=accepted,
        rejected=rejected,
        spec_records=spec_spelling,
        adapter_note="BR-14's seam is where spec spelling maps to library spelling",
    )

    # ------------------------------------ 4. encode -> QPSK -> AWGN -> decode
    cn = "offset-minsum" if "offset-minsum" in accepted else (
        "offset_min_sum" if "offset_min_sum" in accepted else "minsum")
    chain = {}
    chain_ok = True
    for rate_s in bl["ldpc_rates"]:
        num, den = rate_s.split("/")
        n_cw = core_case["E"][0]
        k_cw = int(n_cw * int(num) / int(den))
        try:
            enc = LDPC5GEncoder(k=k_cw, n=n_cw).to(dev)
            dec = LDPC5GDecoder(encoder=enc, num_iter=bl["ldpc_max_iters"],
                                cn_update=cn, hard_out=True).to(dev)
            u = torch.randint(0, 2, (64, k_cw), device=dev, dtype=torch.float32)
            c_bits = enc(u)
            x = 1.0 - 2.0 * c_bits              # BPSK-per-bit proxy for QPSK I/Q
            row = {}
            # A single low-SNR probe cannot serve every rate: at -4 dB Es/N0 the
            # rate-1/3 code sits at Eb/N0 = +0.8 dB, comfortably above its own
            # threshold, so it decodes cleanly and a "must fail at -4 dB" test
            # fails the *code* for being good. Sweep instead and require a
            # waterfall: clean at the top, broken at the bottom, monotone between.
            for esno_db in (12.0, 4.0, 0.0, -4.0, -8.0, -12.0):
                sigma = math.sqrt(0.5 * 10 ** (-esno_db / 10))
                y = x + sigma * torch.randn_like(x)
                # Sionna's LLR convention is log(p(x=1)/p(x=0)), so with the
                # mapping x = 1-2c the sign is NEGATIVE. Getting this backwards
                # is silent: the decoder still runs and still returns k bits, it
                # just returns garbage at every SNR. Measured BER 0.77 vs 0.00.
                llr = -2.0 * y / (sigma ** 2)
                u_hat = dec(llr)
                row[f"{esno_db:+.0f}dB"] = round((u_hat != u).float().mean().item(), 6)
            chain[rate_s] = row
            bers = list(row.values())
            if not (bers[0] == 0.0 and bers[-1] > 0.01
                    and all(a <= b + 1e-9 for a, b in zip(bers, bers[1:]))):
                chain_ok = False
        except Exception as exc:
            chain[rate_s] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            chain_ok = False
    record("4_chain_sanity", chain_ok, cn_update=cn, ber_by_rate=chain,
           note="sanity only; BR-2 does real validation at W3 against golden vectors")

    # ------------------------------------------- 5. decode throughput
    n_cw, k_cw = core_case["E"][0], core_case["K_r"]
    enc = LDPC5GEncoder(k=k_cw, n=n_cw).to(dev)
    dec = LDPC5GDecoder(encoder=enc, num_iter=bl["ldpc_max_iters"],
                        cn_update=cn, hard_out=True).to(dev)
    sigma = math.sqrt(0.5 * 10 ** (-6.0 / 10))
    best = {"cb_per_s": 0.0}
    sweep = []
    for bs in (32, 64, 128, 256, 512, 1024):
        try:
            u = torch.randint(0, 2, (bs, k_cw), device=dev, dtype=torch.float32)
            x = 1.0 - 2.0 * enc(u)
            llr = -2.0 * (x + sigma * torch.randn_like(x)) / (sigma ** 2)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):                       # warm-up
                dec(llr)
            torch.cuda.synchronize()
            # The decoder has no early termination (no such flag in the 2.0.1
            # signature), so timing is data-independent and reps only fight
            # launch-overhead noise. Small batches need more of them.
            reps = max(5, 512 // bs)
            t0 = time.perf_counter()
            for _ in range(reps):
                dec(llr)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            cbps = bs * reps / dt
            peak = torch.cuda.max_memory_allocated() / 2**30
            sweep.append(dict(batch=bs, cb_per_s=round(cbps, 1), reps=reps,
                              ms_per_call=round(1000 * dt / reps, 2),
                              peak_vram_gb=round(peak, 2)))
            if cbps > best["cb_per_s"] and peak < params["compute"]["vram_budget_gb"]:
                best = dict(batch=bs, cb_per_s=round(cbps, 1),
                            peak_vram_gb=round(peak, 2))
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            sweep.append(dict(batch=bs, error="OOM"))
            torch.cuda.empty_cache()
            break
        except Exception as exc:
            sweep.append(dict(batch=bs, error=f"{type(exc).__name__}: {exc}"[:100]))
            break
    record("5_throughput", best["cb_per_s"] > 0, best=best, sweep=sweep,
           num_iter=bl["ldpc_max_iters"])

    # -------------------------------- 6. ER-1 wall clock, one ratio and two
    n_snr = len(params["channel"]["test_snr_grid_db"])
    n_test = ds["imagenette160"]["test_images"]
    n_cells = len(ev["train_seeds"])          # zipped, not crossed (AM-17)

    # A single C is the wrong basis. DEC-16 makes modulation adaptive, so the
    # high-SNR points run 16-QAM, which packs more info bits into the same
    # channel uses and therefore needs *more* code blocks per image. Quoting only
    # the QPSK 5/6 figure would understate the run. Bound it instead.
    headline = [c for c in cases if c["dataset"] == "imagenette160"
                and c["ratio"] == bw["crossover_ratio"]]
    C_vals = sorted(c["C"] for c in headline)
    C_core, C_max = core_case["C"], max(C_vals)
    proj = {}
    if best["cb_per_s"] > 0:
        for label, C_b in (("core_qpsk_5_6", C_core), ("worst_case_qam16", C_max)):
            for arms, tag in ((1, "classical_only"), (2, "classical_plus_er9"),
                              (4, "two_ratios")):
                total = arms * n_snr * n_test * C_b * n_cells
                proj[f"{label}/{tag}"] = dict(
                    code_block_decodes=total,
                    hours=round(total / best["cb_per_s"] / 3600, 2))
    record("6_er1_projection", bool(proj),
           basis=dict(snr_points=n_snr, test_images=n_test, seed_cells=n_cells,
                      cb_per_image_core=C_core, cb_per_image_max=C_max,
                      cb_per_image_range_at_headline_ratio=[C_vals[0], C_vals[-1]],
                      seed_pairing=ev["seed_pairing"]),
           projections=proj,
           note="AM-18/AM-20: the two_ratios number is what G-8 uses; take the "
                "worst_case_qam16 row as the planning figure, since adaptive "
                "modulation will sit there at high SNR")

    # ------------------------------------------ 7. smallest workable payload
    smallest, probe = None, {}
    for k_try in (2048, 1024, 512, 256, 128, 64, 32, 16, 8):
        n_try = k_try * 3          # rate ~1/3, the most robust configured rate
        try:
            e = LDPC5GEncoder(k=k_try, n=n_try).to(dev)
            o = e(torch.randint(0, 2, (2, k_try), device=dev, dtype=torch.float32))
            ok = o.shape[-1] == n_try
            probe[k_try] = "ok" if ok else f"emitted {o.shape[-1]}"
            if ok:
                smallest = k_try
        except Exception as exc:
            probe[k_try] = f"{type(exc).__name__}: {exc}"[:90]
    record("7_smallest_payload", smallest is not None,
           smallest_k_bits=smallest, probe=probe,
           note="bounds feasibility at r_1_24 and STL-10's low ratios")

    # ------------------------------------------------------------- emit
    results["all_passed"] = not failures
    results["failed_checks"] = failures
    (OUT / "g9_spike_record.json").write_text(json.dumps(results, indent=2))
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
