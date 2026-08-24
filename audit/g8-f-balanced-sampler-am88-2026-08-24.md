# AM-88 — balanced G8_F training sampler — 2026-08-24

## Verdict and boundary

**AM-88 GREEN — BALANCED G8_F TRAINING SAMPLER FROZEN; PRIOR G8 EVIDENCE
PRESERVED; F0 STILL REQUIRES SEPARATE OWNER AUTHORIZATION.**

The repair was discovered and decided after immutable G8_E pass one and AM-87,
but before F0. At the starting main SHA `2bf9be9321289aa7237b42b30b85fbbf0d21cea3`,
G8_E E7 and pass one each authenticated protected counters with one pass one and
zero G8_F execution, training, pass two and test access; E7 said F0 authorization
was absent. The only G8_F results were AM-87's three metadata/compatibility
objects. No G8_F runtime, corpus object, authorization, optimizer state or
pass-two object existed, and no matching process was running.

This amendment is protocol/source/evidence/test work only. It decoded no image,
invoked no JPEG 2000 codec or classifier, performed no optimizer step, started
no Pascal worker and touched no test payload.

## What AM-87 still means

AM-87 remains immutable historical evidence and the complete legitimate support
definition. Its plan
`g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148`
(file SHA-256
`733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c`)
contains exactly 120 sorted `g8fquality-` identities and all 8,469 Imagenette
training stable IDs. AM-88 independently re-projects the frozen candidate and
measurement authorities and obtains that exact quality order. No quality was
added, removed, reweighted or selected from validation feasibility.

AM-87's `1,016,280 = 120 × 8,469` Cartesian attempt rule is retained as truthful
history but is superseded for future execution. It was a brute-force sufficient
construction, not a scientific requirement of BR-12.

## Frozen sampler

- version: `g8_f_balanced_sampler_v1`;
- seed: `am88-g8f-balanced-sampler-20260824-v1`;
- algorithm: `sha256_keyed_stable_id_order_global_quality_permutation_class_chunks_cyclic_v1`;
- variants per training image: 6;
- nominal attempts: `8,469 × 6 = 50,814`;
- reduction from AM-87 multiplicity: exactly 20×.

For each class, training stable IDs are ordered by SHA-256 over the frozen seed,
class domain and stable ID. The 120 AM-87 quality IDs receive one SHA-256-keyed
seed permutation. Classes are traversed by ascending label; each class's ordered
images consume contiguous six-position blocks from one continuously advancing
cyclic quality sequence. This is simpler and more balanced than separate class
permutations: every image gets six distinct qualities, each class's attempts are
distributed at floor/ceiling counts, and concatenating the class chunks gives
the same floor/ceiling guarantee globally.

Assignment reads only training stable ID, class label, AM-87 quality order,
seed and sampler version. It cannot accept pass-one expected accuracy, score,
rank or margin, selected PHY details, E4/validation feasibility or artifact
performance, future F1 outcomes, pass two, learned results, test results or
runtime order.

## Balance and omission semantics

Global quality counts are 423 or 424 (range 1, the arithmetic minimum for
50,814 attempts over 120 qualities). Per-class attempt totals and quality ranges:

| class | attempts | min/quality | max/quality | range |
|---:|---:|---:|---:|---:|
| 0 | 5,178 | 43 | 44 | 1 |
| 1 | 5,130 | 42 | 43 | 1 |
| 2 | 5,358 | 44 | 45 | 1 |
| 3 | 4,548 | 37 | 38 | 1 |
| 4 | 5,046 | 42 | 43 | 1 |
| 5 | 5,136 | 42 | 43 | 1 |
| 6 | 5,166 | 43 | 44 | 1 |
| 7 | 4,986 | 41 | 42 | 1 |
| 8 | 5,106 | 42 | 43 | 1 |
| 9 | 5,160 | 43 | 43 | 0 |

No pair repeats; every support quality and every training ID participates;
validation/test participation is zero. A typed image-level codec infeasibility
records the exact assigned pair as omitted and never resamples or substitutes.
Unexpected codec/decoder failure, runtime exception, foreign/corrupt identity
or unverified artifact remains HOLD.

## Compact evidence

The tracked plan is `results/baseline/g8_f/am88_sampler_plan.json`. It stores the
120 support IDs, exact training/class bindings, all 120 global counts and all
10×120 class-quality counts, but not 50,814 full pair objects. Generator and an
independently implemented verifier reproduce the sequence from frozen metadata.

- ordered-pair SHA-256:
  `c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229`;
- pair-set SHA-256:
  `255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e`.

The plan ID and file SHA are emitted by
`tools/verify_g8_f_sampler_plan.py`; they are content-derived and must be read
from the final committed bytes rather than copied from an intermediate build.

## Compute and storage consequence

No F1 timing benchmark was run. The storage arithmetic mechanically reuses the
AM-87/G8_E planning basis of 115,296 bytes per materialized pair:

- validation-incidence planning estimate: 44,039 objects, 5,077,520,544 bytes
  (about 5.08 GB / 4.73 GiB);
- hard maximum under the assignment count: 50,814 objects, 5,858,650,944 bytes
  (about 5.86 GB / 5.46 GiB);
- maximum plus the existing 25% safety factor: 7,323,313,680 bytes
  (about 7.32 GB / 6.82 GiB).

The only timing basis remains AM-87's conservative 59,799-second Git
publication window for 120,000 G8_E physical jobs. It includes unknown start
delay and closeout overhead and is **not measured F1 throughput**. Linear
planning extrapolation gives about 21,946 seconds (6.10 h) at validation
incidence and 25,322 seconds (7.03 h) at the 50,814-attempt maximum. These are
planning extrapolations, not hard duration bounds or a benchmark.

## Prior-work impact

AM-88 changes no request/result/state, BLER, G8_D contract, E2/E3/E4, pass-one,
E6/E7 or G1 evidence byte. Generated specification parameters changed only on
AM-88 G8_F sampler leaves. Additive exact-byte compatibility records chain the
AM-87 post-campaign verifier state to AM-88 and reject unrelated drift.

| work | recomputation |
|---|---|
| G8_C | NONE |
| G8_D | NONE |
| G8_E E2/E3/E4 | NONE |
| pass one | NONE |
| G1 training | NONE |
| test | access remains 0 |

## Owner audit and next gate

Run, from a clean checkout at the final main SHA:

```bash
.venv/bin/python tools/gen_g8_f_sampler_plan.py --check
.venv/bin/python tools/verify_g8_f_sampler_plan.py
.venv/bin/python -m pytest tests/test_g8_f_sampler_plan.py -q
.venv/bin/python tools/verify_g8_e_complete.py
```

Audit the emitted plan ID, file SHA, pair digests, class/global counts and all
zero boundaries. Only after accepting those exact committed bytes may the owner
issue a **separate, narrow F0 authorization**. This amendment does not issue or
simulate that authorization.
