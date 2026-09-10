# ER-9 v3 local-profile custody and supersession

Inspection date: 2026-09-10

The repository began at clean `5be9769a560407250c7ce3826e2af3e785949510`, equal
to `origin/main`. The owner runtime was inspected from the actual ignored files,
not from a prior summary. The detailed machine-readable custody record is
[`results/learned/er9/incidents/v3_local_profile_owner_supersession.json`](../results/learned/er9/incidents/v3_local_profile_owner_supersession.json).

The preserved namespaces are `checkpoints/er9`, `checkpoints/er9_successor`,
and `checkpoints/er9_successor_v2`. The first two are retained predecessor
attempts. The v3 local-profile runtime is `checkpoints/er9_successor_v2`,
authenticated against `local_4060_cu130`, `cuda:0`, NVIDIA GeForce RTX 4060
Laptop GPU UUID `GPU-607a5795-c53b-eab2-8c04-71164b173a32`, with profile-binding
SHA-256
`ff3552f595c4d4f090d673b77bce783629d30d54b154c15b586605ddc0c89c57` and
ordered-file inventory SHA-256
`c440a17a71916ad0697e44bcd92468e6b25beefc703157679467353ee7ee5813`.

The v3 runtime contains two candidate identities. `D64_b2` has an authenticated
consecutive epoch prefix 0–99 (100 epochs), a terminal `run_completion.json`,
and 26,500 optimizer opportunities, 26,485 applied steps and 15 GradScaler
skips. `D128_b2` has an authenticated consecutive prefix 0–66 (67 epochs), no
terminal completion or selected-checkpoint record, and 17,755 opportunities,
17,743 applied steps and 12 skips. Every epoch satisfies the arithmetic
`applied_steps + grad_scaler_skips = opportunities`; every record and sidecar
reports `test_access=0`; no staging/incomplete publication artifact exists.

There is no `results/learned/er9/stage1_evaluations/`, no
`er9_stage1_selection.json`, no `stage2_evaluations/`, and no
`er9_stage2_selection.json`. The runtime's training validation records use the
pre-real-chain training path and do not constitute accepted ER-9 real-chain
candidate evaluations. On the authenticated actual state, accepted real-chain
candidate evaluations are zero and Stage-1 selections are zero. This conclusion
is stated only after the file-level audit above; it was not assumed from the
previous audit.

The owner decision is additive supersession: these local-profile attempts and
all their checkpoints remain preserved, but are scientifically ineligible for
Pascal v4, cannot initialize or be promoted into v4, and will not be resumed,
rewritten, normalized, or deleted. The new v4 campaign must use fresh
initialization and a single prospectively frozen qualified Pascal GPU.
