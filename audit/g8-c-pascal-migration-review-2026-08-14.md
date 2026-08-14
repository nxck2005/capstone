# G8_C Pascal migration review brief — 2026-08-14

> **Status: discussion and independent-review memo only.** This document records
> the project owner's desire to move G8_C characterization off the primary RTX
> 4060 laptop and onto the dedicated Pascal worker. It authorizes no package
> installation, specification amendment, contract supersession, campaign
> transition, evidence transfer, worker dispatch, characterization, validation
> decoding, selection, inference, training or test access. Until a migration
> design is explicitly approved and implemented, `instructions/RESUME.md`, the
> authenticated campaign state and the registered cu130 contracts remain the
> operational authority.

## Review request

Please independently review whether G8_C should move to the dedicated machine
containing a GeForce GTX 1080 Ti and TITAN Xp, and whether the final evidence
should use:

1. a **mixed-runtime continuation** that preserves the first 748 accepted cu130
   work units and executes only the suffix under a new cu126 epoch; or
2. a **clean cu126 recampaign** that preserves the existing campaign as
   immutable history but executes all 3,213 required work units again under one
   newly authenticated Pascal environment.

The current recommendation is the clean recampaign. That is a proposal for
review, not an adopted decision.

The owner's motivation is operational rather than scientific: the primary PC
is needed for productive work and should not remain occupied by a long-running
characterization campaign. The migration must solve that problem without
weakening the experiment, silently changing the environment, mixing
unauthenticated results, or rewriting existing evidence.

## Authoritative current state

The live scientific cursor remains `G8_C/characterization_open` with C0 and the
additive C1C correction complete and C2 paused at a user-requested durable
checkpoint.

Read-only verification performed on 2026-08-14 reported:

- campaign state SHA-256:
  `4a285ee7746f197a96c23230a0aac945c581c4ec8d40bc98bb3fa86b46f68ddd`;
- epoch-1 characterization source manifest SHA-256:
  `a917f839f945232e85852d6d27f02de4b5dc272adc72b1966a95e9b5e62a014e`;
- epoch-2 characterization source manifest SHA-256:
  `b654e5d6ffa585882c872a7ef6965b33ea486365f449ad10a632d3e0d0367660`;
- 748 of 3,213 required work units accepted;
- 2,465 work units remaining;
- 2,249 tracked work-unit evidence files;
- ordinal 748, `bler-3d67593f9deb3cfaab668644`, has a request-only
  attempt 1 and next legally proposes attempt 2;
- `in_progress_work_unit_id = null`;
- validation-decoding, inference, training and test-access counters are all
  zero;
- no characterization process and no tmux session are running; and
- local HEAD, local `origin/main` and remote `main` all resolved to
  `bc0aa5970c141f6994a8ce4b619927e3eda09e5c` at the audit point.

The dedicated read-only command passed:

```text
.venv/bin/python tools/verify_g8_evidence_readonly.py
read-only G8 evidence verification PASS: {"completed_count": 748,
"remaining_count": 2465, "test_split_access": 0,
"tracked_work_unit_files": 2249}
```

The exact current cu130 restart sequence remains the inspect, reconcile,
marker, push-parity and v2-coordinator sequence in
[`instructions/RESUME.md`](../instructions/RESUME.md). It must not be executed
on the Pascal worker.

## Hardware facts and unknowns

User-supplied PCI enumeration identifies:

- NVIDIA GeForce GTX 1080 Ti, 11 GB, Pascal GP102, compute capability 6.1;
- NVIDIA TITAN Xp, 12 GB, Pascal GP102, compute capability 6.1; and
- an Intel NVMe controller.

PCI enumeration does **not** establish the worker's operating system, NVIDIA
driver, CUDA initialization, mounted NVMe capacity, free space, filesystem,
health, remote-access path, thermal stability or ability to execute the
project's Sionna kernels. Those remain qualification items.

CUDA Toolkit 13 removed offline compilation and library support for Maxwell,
Pascal and Volta. CUDA 12.x remains the applicable toolkit family for Pascal.
Newer NVIDIA drivers can execute applications built against older CUDA
runtimes through backward compatibility.

Primary external sources:

- [CUDA Toolkit 13.0 release notes — deprecated architectures](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures)
- [NVIDIA CUDA minor-version and backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Official PyTorch cu126 Torch wheel index](https://download.pytorch.org/whl/cu126/torch/)
- [Official PyTorch cu126 TorchVision wheel index](https://download.pytorch.org/whl/cu126/torchvision/)
- [Official PyTorch previous-version commands](https://pytorch.org/get-started/previous-versions/)
- [Sionna 2.0.1 installation requirements](https://nvlabs.github.io/sionna/installation.html)

## Candidate Pascal environment

The preferred first qualification candidate is now the smallest possible
software delta from the registered primary environment:

- Python `3.14.6`;
- `torch==2.13.0+cu126`;
- `torchvision==0.28.0+cu126`;
- `numpy==2.5.1`; and
- `sionna-no-rt==2.0.1`.

The official cu126 indexes currently list Linux x86-64 CPython 3.14 wheels for
Torch 2.13.0 and TorchVision 0.28.0. This resolves the earlier question of wheel
*availability*, but not Pascal execution compatibility. The worker must still
prove that the selected Torch wheel can execute real compute-capability-6.1
kernels and the project's Sionna LDPC path on both cards.

If the exact-version cu126 candidate fails on Pascal, the previously researched
fallback is:

- Python 3.12;
- `torch==2.9.1+cu126`;
- `torchvision==0.24.1+cu126`;
- `numpy==2.5.1`; and
- `sionna-no-rt==2.0.1`.

No tuple is adopted until on-node qualification and a separately generated,
hashed lock succeed. The current cu130 lock remains the primary lane and must
not be replaced.

## Why the current command cannot simply move

### CUDA and runner-contract mismatch

The registered runner contract records:

- `torch_version = 2.13.0+cu130`; and
- `torch_cuda_version = 13.0`.

The epoch-2 characterization source manifest binds that exact runner-contract
artifact and SHA-256. Running under cu126 would therefore contradict the
registered environment even if every scientific parameter and request byte
were unchanged.

The current runtime dependency authentication is also insufficient for a
migration. `AuthenticatedRunnerContext._authenticate_dependencies()` checks
the exact NumPy and Sionna versions but only requires
`torch.version.cuda is not None`; it reports the observed Torch versions but
does not compare them with the registered runner contract. A cu126 build could
therefore pass the local guard while contradicting the contract. This is a
fail-open gap for the proposed migration, not permission to exploit it.

Relevant repository sources:

- [`results/baseline/g8/bler_runner_contract.json`](../results/baseline/g8/bler_runner_contract.json)
- [`src/baseline/g8_bler_runner.py`](../src/baseline/g8_bler_runner.py)
- [`results/baseline/g8/bler_characterization_source_manifest_v2.json`](../results/baseline/g8/bler_characterization_source_manifest_v2.json)

### The current coordinator is not a two-GPU scheduler

The v2 coordinator:

- counts visible GPUs;
- resolves automatic shard count to that count;
- reports the count as `workers`;
- starts only one child process for the selected batch; and
- passes generic `cuda` rather than an explicit device index.

`SionnaLDPCAdapter` maps generic `cuda` to `cuda:0`. Therefore, on the Pascal
worker the current command could report two workers and a two-way shard plan
while actually executing only shard 0 on GPU 0.

The adapter already accepts explicit `cuda:0` and `cuda:1` strings, so it
should not need modification. Avoiding changes under `src/baseline/ldpc/`
preserves the adjudicated G-2 runtime path.

Relevant sources:

- [`tools/run_g8_bler_characterization_v2.py`](../tools/run_g8_bler_characterization_v2.py)
- [`src/baseline/ldpc/adapter.py`](../src/baseline/ldpc/adapter.py)

### Local locks are not a multi-host transfer protocol

The resume/state implementation uses local `fcntl.flock` and atomic local
filesystem publication. It must not be extended to multiple concurrent clones
through NFS or manual copying. The simplest defensible operating model is a
single designated writer: custody of G8_C moves to one Pascal-worker clone for
the duration, that clone uses local NVMe storage, and batch evidence returns
through authenticated commits and pushes. The primary PC must not write G8_C
concurrently.

## Option A — mixed-runtime continuation

Preserve accepted ordinals 0–747 in the final table and execute ordinals
748–3212 under cu126.

### What this preserves

- All 748 accepted results remain final evidence.
- Only 2,465 work units, or 12,325,000 trials, remain to execute.
- Existing request-only history remains immutable.

### Required engineering

1. Add a named cu126 environment profile and separately hashed lock through an
   append-only specification amendment.
2. Add fail-closed exact environment authentication.
3. Create a superseding runner/environment contract without modifying the
   registered cu130 artifact.
4. Implement and verify a true two-GPU coordinator with explicit GPU indices
   and UUIDs.
5. Add a third characterization source/environment epoch beginning exactly at
   ordinal 748, attempt 2.
6. Preserve epoch-1 and epoch-2 artifacts byte-for-byte while recording that
   epoch 2's previously declared future range stops at the actual accepted
   boundary, ordinal 747.
7. Extend merge and table verification to require exact ordinal-to-environment
   attribution with no gaps, overlaps or duplicate acceptance.
8. Define and preregister a cross-environment parity criterion before looking
   at parity results.
9. Run a stratified, explicitly non-scientific cu130-versus-cu126 parity probe
   covering base graphs, lifting sizes, rates, modulations and waterfall SNRs.
10. Establish single-writer custody and authenticated evidence return.

### Main scientific cost

The final BLER table would mix two PyTorch/CUDA execution profiles. The
physical-layer algorithm and deterministic seeds may be identical, but small
floating-point differences in an iterative decoder can change discrete bit or
block decisions near a boundary. The project would therefore need an explicit
equivalence argument rather than relying on "same seed" or matching tensor
shapes.

### Repository complication

The immutable epoch-2 manifest currently assigns accepted authority ordinals
179–3212 to epoch 2. It cannot be edited after data. An additive epoch-3
manifest and verifier must explicitly supersede only epoch 2's unused future
range while preserving its accepted bytes and attribution through ordinal
747.

## Option B — clean cu126 recampaign

Preserve the current cu130 campaign and all 748 accepted results as immutable
historical/superseded evidence. Open a new campaign lineage and execute all
3,213 required work units under one cu126 environment.

### What is repeated

- all 3,213 G8_C work units;
- 16,065,000 full-strength trials;
- the first 748 work units, representing 3,740,000 repeated trials and 23.3%
  of the currently required grid;
- campaign opening, source/environment registration and bounded smoke for the
  new lineage.

### What is not repeated

- G-1 reference-classifier training or adjudication;
- G-2 conformance measurements;
- G-7 DJSCC feasibility profiling;
- W4 bounded baseline integration;
- dataset fetching, manifests or provenance;
- the validation-only transparency-bitrate probe; or
- First Review evidence.

G8_C C3–C7 have not started, so they are delayed rather than redone.

### Required engineering

1. Record an explicit abandonment/supersession decision without deleting or
   editing the current evidence.
2. Add the cu126 environment profile, amendment and separate hashed lock.
3. Create a new campaign identity, campaign state and production runtime root.
4. Bind the unchanged required structural grid deliberately, or regenerate its
   registration under the new campaign while proving that the physical grid is
   unchanged.
5. Create a fail-closed cu126 runner/environment contract.
6. Implement and verify the two-GPU coordinator.
7. Run non-scientific smoke and diagnostic cu130/cu126 parity probes.
8. Execute all 3,213 units under the new lineage.
9. Run the ordinary merge, complete-coverage proof, BLER-table freeze and
   G8_D handoff against the new campaign only.

### Main scientific benefit

The final table has one software environment and one execution story. Cross-
environment parity remains a diagnostic safety check rather than the legal
basis for combining final evidence.

## Recommendation

Use the **clean cu126 recampaign** if the project owner is willing to repeat the
first 748 units.

Reasons:

- only 23.3% of the grid is currently accepted;
- both Pascal GPUs may recover some or all of the repeated wall time;
- the owner values freeing the primary PC more than preserving already-spent
  compute;
- one environment is substantially easier to verify and explain to reviewers;
- no mixed-runtime BLER-table equivalence claim is required; and
- the current cu130 campaign remains a complete, authenticated history rather
  than being erased.

The recommendation may change if qualification shows that one or both Pascal
GPUs are unstable, much slower than expected, unable to run the chosen wheel,
or unable to return artifacts reliably.

## Specification and historical-evidence impact

Adopting cu126 for scientific work requires the next available append-only
amendment after rechecking the live AM sequence. Likely affected items include:

- DEC-4, to recognize an authenticated auxiliary compute environment;
- SR-11, to require profiling and limits per environment/device;
- SR-12, to freeze deterministic and cross-environment reproduction rules;
- SR-21, to support named environment profiles and separate hashed locks;
- `params.compute` and `params.environment` profile structure;
- `params.config.fingerprint_schema_version` if resolved run identity changes;
  and
- run/evidence metadata fields for environment profile, lock SHA, device UUID,
  compute capability and observed dependency versions.

Both `params.compute` and `params.environment` are currently included in the
configuration fingerprint. Adding a profile will therefore change new
fingerprints. `tools/verify_w4_baseline_integration.py` presently compares
archived parameter-root snapshots with current roots, so an additive
future/off-measurement-path environment profile may make valid historical W4
evidence fail. The amendment and implementation need a narrow, explicit
historical-compatibility rule. W4 evidence must not be regenerated or rerun
merely to accommodate the new profile.

The current G8 campaign manifest also binds exact bytes of `spec/SPEC.md` and
`spec/params.generated.yaml`. A spec amendment during G8 cannot be treated as
unrelated documentation. The chosen design must explicitly preserve and
supersede the old normative-source binding rather than allowing the existing
manifest generator or verifier to silently reinterpret it.

## Proposed two-GPU execution contract

The replacement coordinator should, at minimum:

- resolve and record both device UUIDs and indices;
- assign one process to `cuda:0` and one to `cuda:1`;
- use global shard count 2 with disjoint shard indices 0 and 1;
- verify the canonical modulo-shard assignment before execution;
- select no work unit in both workers;
- retain an at-most-128-unit durable launch marker initially, split across the
  two workers, unless a new contract deliberately authorizes a different
  ceiling;
- rebuild the authenticated resume plan before each unit;
- fail the batch if either child exits without a durable summary;
- stop safely if a GPU disappears or enumeration changes;
- reconcile only from authenticated files after both workers exit;
- record exact per-worker device/environment metadata; and
- preserve request bytes, trial counts, seed derivation and all physical-layer
  parameters.

Mutation and failure tests should cover:

- both workers bound to GPU 0;
- reversed GPU enumeration;
- duplicate shard indices;
- an omitted shard;
- overlapping work-unit lists;
- one child crashing before request publication;
- one child crashing after request publication;
- one child crashing after result publication;
- CUDA unavailable on one card;
- lock hash or dependency mismatch;
- cu130 launched under a cu126 contract and vice versa;
- evidence copied without the registered custody/return record; and
- concurrent primary-PC and Pascal-worker writers.

## Qualification-only phase before scientific work

The first Pascal activity must be explicitly non-scientific and must contribute
zero G8 coverage.

For each GPU independently:

1. Record OS, kernel, Python, driver, Torch, Torch CUDA, TorchVision, NumPy and
   Sionna versions.
2. Record device name, UUID, index, compute capability and usable VRAM.
3. Verify `torch.cuda.is_available()` and execute a real tensor kernel.
4. Verify that Torch reports or successfully executes an appropriate Pascal
   architecture path.
5. Execute representative Sionna LDPC encode/decode operations.
6. Apply and read back deterministic backend settings.
7. Repeat identical seeded operations and compare outputs.
8. Run checkpoint/resume and process-hard-exit drills.
9. Measure safe batch size, throughput, peak memory and sustained temperature.
10. Verify NVMe mount, capacity, free space, filesystem and health.
11. Verify SSH/remote-control recovery after disconnect.
12. Prove the authenticated commit/push artifact-return procedure.

For a clean recampaign, the cu130/cu126 parity probe is diagnostic: a material
disagreement holds the migration for investigation but does not justify mixing
old results into the new table. For a mixed continuation, parity becomes a
load-bearing acceptance gate and must have a preregistered quantitative rule.

## Proposed custody and artifact-return procedure

1. Reconcile and commit the current dirty primary worktree before any campaign
   action.
2. Push the reviewed migration contracts and qualification artifacts.
3. Fetch or clone the exact approved commit on the Pascal worker.
4. Record and push a custody-transfer marker naming the designated single
   writer, campaign, environment, runtime root and permitted batch.
5. Execute against local NVMe; do not use NFS or a shared live state tree.
6. Reconcile locally with the authenticated tools.
7. Commit canonical request/result/state evidence and the ledger checkpoint.
8. Push and verify remote parity before the next marker.
9. Do not run or reconcile G8_C from the primary PC while Pascal custody is
   active.
10. Return custody through another explicit pushed marker before the primary
    PC performs any G8 operation.

## Schedule considerations

The present First Review window is 18–22 August 2026. G8_C completion is not a
First Review prerequisite. The review package should not be weakened or delayed
to manufacture more G8 results.

Migration engineering may take longer than simply finishing the suffix on the
RTX 4060. That is not necessarily a reason to reject it: the owner's goal is to
free the primary computer and establish a useful future training worker, not
only to minimize the next batch's wall-clock completion time. The decision
should nevertheless compare:

- measured one-GPU and two-GPU Pascal throughput;
- engineering time for amendments, locks, contracts and tests;
- whether both GPUs may run unattended and thermally stable;
- the value of preserving 748 results versus one clean final lineage; and
- whether the Pascal node will remain useful for later W5/W7/W8 training.

## Questions for the independent reviewer

1. Is a clean recampaign scientifically preferable given that 23.3% of C2 is
   complete, or is the mixed-runtime table defensible enough to preserve it?
2. Can a new environment profile be added without changing the campaign ID,
   or would that conflict with the current self-hashed campaign manifest and
   seed derivation?
3. Is ordinal-range source/environment attribution sufficient, or must every
   result bind the observed environment profile directly?
4. Should the result or state schema change to carry an environment-contract
   ID, lock SHA and GPU UUID, or is an immutable batch/epoch manifest enough?
5. What parity criterion is defensible for an iterative floating-point LDPC
   decoder across cu130/Ada and cu126/Pascal?
6. Does the exact-version `2.13.0+cu126` candidate execute Pascal kernels on
   both cards, and does the wheel expose the required architecture support?
7. Should a launch marker authorize 128 units total or 128 per GPU?
8. Is Git commit/push custody sufficient for authenticated artifact return, or
   is a dedicated signed export/import manifest required?
9. Can the existing runner source remain the algorithm contract with a new
   outer environment contract, or should an additive runner module and contract
   fully supersede it?
10. What is the narrowest historical-compatibility rule that preserves W4
    evidence after adding the Pascal environment profile?
11. Which parts of C0/C1/B4-B6 must be re-executed as contract qualification,
    versus merely re-verified against unchanged sources?
12. What additional failure modes arise from two non-identical Pascal cards
    with 11 GB and 12 GB of VRAM?

## Decision gate

No installation or scientific execution should begin until the owner chooses
between:

- **clean cu126 recampaign** — current recommendation;
- **mixed-runtime continuation**; or
- **no migration**.

After that choice, the next artifact should be a durable migration instruction
and qualification plan. The plan must finish its non-scientific qualification,
amendment, lock, contract, coordinator and verification gates before creating
the first Pascal-produced G8 result.

## Related local discussion and authority

- [`audit/pascal-worker-adoption-audit-2026-08-14.md`](pascal-worker-adoption-audit-2026-08-14.md)
- [`audit/pascal-worker-adoption-audit-2026-08-14-SECOND-AGENT-THOUGHTS.md`](pascal-worker-adoption-audit-2026-08-14-SECOND-AGENT-THOUGHTS.md)
- [`instructions/RESUME.md`](../instructions/RESUME.md)
- [`NEXT.md`](../NEXT.md)
- [`AGENTS.md`](../AGENTS.md)
- [`spec/SPEC.md`](../spec/SPEC.md)
- [`results/baseline/g8/campaign_manifest.json`](../results/baseline/g8/campaign_manifest.json)
- [`results/baseline/g8/campaign_state.json`](../results/baseline/g8/campaign_state.json)
- [`results/baseline/g8/bler_characterization_source_manifest_v2.json`](../results/baseline/g8/bler_characterization_source_manifest_v2.json)
- [`results/baseline/g8/bler_runner_contract.json`](../results/baseline/g8/bler_runner_contract.json)
