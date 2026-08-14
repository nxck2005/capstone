# Pascal worker adoption audit — second-agent thoughts — 2026-08-14

> **Status: discussion memo only.** This note preserves a second technical
> position for debate at the next cold start. It authorizes no installation,
> delegation, campaign execution, evidence transfer, contract supersession,
> specification amendment, training, validation decoding or test access.
> `spec/SPEC.md`, authenticated campaign state and `instructions/RESUME.md`
> remain authoritative.

## Question

If the project owner wants to execute G8_C on the dedicated worker containing
a GeForce GTX 1080 Ti and TITAN Xp, what would be required to make that
scientifically and operationally defensible?

## Short answer

It is technically possible, but **not as an ordinary resume of the live
campaign**. Using the Pascal GPUs requires a CUDA 12.x/Pascal-capable PyTorch
environment, while the registered G8 runner contract records
`torch 2.13.0+cu130` and Torch CUDA 13.0. The current runtime guard would not
protect this boundary: it verifies the exact NumPy and Sionna versions but
accepts any PyTorch build for which `torch.version.cuda` is non-null.

There are two defensible migration shapes:

1. preserve the already accepted cu130 results and admit a separately
   authenticated cu126 suffix; or
2. preserve the existing campaign as immutable history and execute a clean
   replacement G8_C campaign entirely under one qualified cu126 environment.

If moving G8_C to the Pascal GPUs is a firm requirement, this audit recommends
the second option because it produces a simpler scientific lineage. If schedule
cost makes a complete recampaign unacceptable, the first option is possible but
needs considerably stronger environment attribution and merge verification.

Until one design is explicitly approved and implemented, the operational
answer remains: **do not run the live G8_C suffix on the Pascal worker and do
not import cu126 work-unit results into the current production tree.**

## Confirmed local facts

User-supplied PCI enumeration shows:

- `3b:00.0`: NVIDIA GP102 GeForce GTX 1080 Ti;
- `d8:00.0`: NVIDIA GP102 TITAN Xp; and
- `5e:00.0`: Intel NVMe controller.

The GPUs are Pascal compute-capability 6.1 devices. Their advertised memory is
11 GB and 12 GB respectively. The NVMe controller proves an NVMe device is
present, but not its mounted capacity, free space, filesystem or health.

CUDA Toolkit 13 removed offline compilation and library support for Pascal.
The worker therefore needs a CUDA 12.x-compatible PyTorch lane to use its GPUs.
It should normally run one complete process/job per GPU rather than pretending
the cards form a single 23 GB device.

## Repository evidence behind the current prohibition

The registered artifact
[`results/baseline/g8/bler_runner_contract.json`](../results/baseline/g8/bler_runner_contract.json)
records:

- `torch_version`: `2.13.0+cu130`; and
- `torch_cuda_version`: `13.0`.

The live dependency authentication in
[`src/baseline/g8_bler_runner.py`](../src/baseline/g8_bler_runner.py) does the
following:

- checks the exact frozen NumPy version;
- checks the exact configured Sionna version; and
- checks only that `torch.version.cuda` is not `None`.

It returns the observed Torch and Torch-CUDA versions as a dependency binding,
but the live authorization path does not compare those observed values with
the versions inside the registered runner contract. Consequently a cu126 build
could pass local authentication while contradicting the contract. That is a
fail-open gap for this proposed migration, not permission to exploit it.

The epoch-2 characterization manifest declares device and batch size to be
provenance-only. That makes heterogeneous hardware plausible in principle, but
does not erase the separate registered software-environment binding. Device
flexibility and dependency-version flexibility are different claims.

The current C2 handoff in [`instructions/RESUME.md`](../instructions/RESUME.md)
also requires an exact inspect/reconcile/marker/push-parity sequence before the
registered v2 coordinator resumes. A command copied to another clone is not an
authenticated multi-host protocol.

## Orchestration gaps that must be fixed first

The v2 coordinator's automatic topology is not sufficient for these two GPUs:

- it resolves `shard_count` to the number of visible CUDA devices;
- it reports the same number as `workers`;
- but it starts only one child process for the selected work-unit batch; and
- it passes the generic device string `cuda`, without binding a child to an
  explicit `cuda:0` or `cuda:1` device.

The LDPC adapter likewise receives the generic device selection. Therefore the
current coordinator must not be described as a verified two-GPU scheduler.
Running two ad-hoc shells with locally invented shard arguments would bypass
the frozen orchestration and marker protocol.

The state machinery also relies on local `fcntl.flock`. Two repository clones,
an NFS mount or manual copying do not automatically provide the same
crash-safety and single-writer guarantees. A remote worker needs an
authenticated export/import protocol, or the canonical repository must itself
execute and publish every state transition.

## Option A — mixed-runtime continuation

Preserve all currently accepted cu130 results and execute only the remaining
suffix under a qualified cu126 environment.

Minimum work before the first Pascal-produced scientific unit:

1. Freeze the exact Pascal Python/Torch/torchvision/CUDA/Sionna/NumPy tuple and
   generate a separate hashed lockfile.
2. Run an isolated, explicitly non-scientific parity campaign across cu130 and
   cu126, covering representative base graphs, lifting sizes, rates,
   modulations and waterfall SNRs. Compare final decoded decisions and BLER
   counts, not merely tensor shapes or successful kernel launches.
3. Decide and record what parity result is sufficient. Floating-point decoder
   differences near a decision boundary can change discrete bit/block errors;
   “same seed” alone does not prove equivalent evidence.
4. Amend the normative environment and reproducibility rules if the second
   lane will contribute scientific evidence.
5. Make dependency authentication compare observed Torch and CUDA versions to
   the selected registered environment profile, failing closed.
6. Supersede the runner contract through a new, authenticated contract rather
   than editing the registered artifact.
7. Add a new characterization source epoch at the exact accepted boundary.
   Preserve epoch 1 and epoch 2 bytes and explicitly attribute the already
   accepted ordinal ranges to them; attribute only the new suffix to the new
   epoch.
8. Correct the coordinator to bind one process and global shard to each
   physical GPU, with explicit device indices/UUIDs and tested failure cases.
9. Define a single-writer authenticated evidence export/import mechanism. Raw
   requests, results and state files remain immutable, and only the canonical
   repository reconciles campaign state.
10. Extend the merge/table verifier to prove complete coverage and exact
    environment/source-epoch attribution, with no duplicate unit accepted
    across hosts or environments.
11. Record, push and verify a new execution marker before scientific work.

Advantages:

- preserves the value of the already accepted G8_C work;
- moves only the remaining suffix; and
- can exploit both Pascal GPUs after the new coordinator is qualified.

Disadvantages:

- the final BLER table mixes two software/hardware execution profiles;
- parity must be argued for the nonlinear iterative decoder and discrete error
  counts, not only for deterministic random inputs;
- epoch attribution and review explanation become more complicated; and
- the existing campaign manifest binds exact specification/generated-parameter
  bytes, so a normative environment amendment cannot be treated as unrelated.

## Option B — clean cu126 recampaign

Preserve the present G8_C campaign and its accepted work-unit bytes as
historical/superseded evidence. Open a new authenticated campaign lineage whose
entire G8_C grid runs under one frozen cu126 environment.

This is not permission to delete, reset or overwrite the current production
tree. The replacement needs an explicit new campaign identity and
supersession/abandonment record. All contracts, source bindings, seeds,
work-unit identities and merge rules must either be regenerated for that new
lineage or proven deliberately reusable without claiming the old campaign is
the new one.

Minimum work includes the environment lock and qualification, amendment,
fail-closed dependency contract, corrected two-GPU coordinator, authenticated
remote evidence handling, new pre-data artifacts/state and complete rerun of
all required BLER work units.

Advantages:

- one environment and one execution story for the complete final table;
- simpler merge attribution and examiner-facing provenance;
- parity with the old cu130 results can be diagnostic rather than the legal
  basis for mixing evidence; and
- both Pascal GPUs can be used consistently from the first accepted unit.

Disadvantages:

- all currently accepted G8_C progress must be repeated;
- opening a replacement lineage may require substantial contract and verifier
  work; and
- the migration may cost more engineering time than finishing C2 on the RTX
  4060 path.

## Option C — CPU execution on the dedicated worker

Installing the exact cu130 lock and explicitly running the worker CPU could
avoid Pascal kernel support. This does not achieve the stated goal of using the
GPUs, is likely much slower, and the current authoritative restart command uses
`--device auto`, not an approved explicit CPU command. It is therefore not a
free operational shortcut. If considered, its command, topology, performance
and marker implications must be reviewed separately.

## Exact cu126 tuple remains a debate item

The companion first audit,
[`pascal-worker-adoption-audit-2026-08-14.md`](pascal-worker-adoption-audit-2026-08-14.md),
records a conservative tested tuple of Python 3.12,
`torch==2.9.1+cu126` and `torchvision==0.24.1+cu126`, and states that
`torch==2.13.0+cu126` was not established.

This second analysis observed entries for Linux CPython 3.14
`torch==2.13.0+cu126` on the official PyTorch CUDA 12.6 wheel index. That index
observation alone is not enough to select the production tuple: availability,
architecture contents, Sionna compatibility and successful execution on both
GP102 cards must all be established locally. The cold-start debate should
resolve this discrepancy from downloaded wheel metadata and an isolated
on-node qualification, not by silently choosing either memo's tuple.

Whichever tuple wins must be frozen as a new named environment profile. It
must not replace or mutate the existing cu130 lock.

## Recommendation for the next cold start

Before resuming more cu130 G8_C work, decide whether moving G8_C itself is a
firm requirement or merely an optimization idea.

- If it is only about finishing sooner, first estimate the remaining RTX 4060
  duration against the engineering cost of migration. Finishing the existing
  campaign is likely the lower-risk choice.
- If G8_C must execute on Pascal and a clean scientific story is the priority,
  choose the clean cu126 recampaign.
- If preserving current progress is essential, choose mixed-runtime
  continuation only after accepting its larger parity and provenance burden.
- If the worker is mainly desired for later neural training, finish G8_C on
  cu130 and revisit the worker at W5 under a separately amended environment.

Questions to settle explicitly:

1. Is Pascal execution a requirement for G8_C or merely desired acceleration?
2. How much wall-clock time remains on the current RTX path?
3. Is repeating all accepted G8_C work acceptable?
4. What exact cu126 package tuple works on both cards?
5. Must the two GPUs run concurrently, or is one-at-a-time acceptable?
6. Will the canonical repository execute remotely, or will evidence cross a
   host boundary?
7. What parity result would justify a mixed-runtime final BLER table?
8. Is the schedule gain worth the amendment, contract, coordinator and
   verifier work?

No recommendation in this memo changes the current phase cursor. Until the
debate produces an explicit decision, no Pascal worker is authorized and the
existing G8_C resume protocol remains unchanged.
