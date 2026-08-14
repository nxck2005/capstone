# Dedicated Pascal worker adoption audit — 2026-08-14

> **Status: discussion memo only.** This document records a read-only technical
> and specification-impact audit for the next cold start. It does not authorize
> installation, delegation, G8_C execution, campaign migration, training,
> validation/test access, or a specification amendment. `spec/SPEC.md`, the
> authenticated campaign state, and `instructions/RESUME.md` remain authoritative.

## Question to debate

Should the dedicated worker containing a GeForce GTX 1080 Ti (11 GB) and TITAN
Xp (12 GB) be qualified as an auxiliary machine for independent future training
runs, while the laptop remains the primary environment and canonical evidence
repository?

The technically safest proposal is:

1. do not attach it to the live G8_C suffix;
2. qualify both Pascal GPUs using a separate, hashed CUDA 12.6 environment;
3. use it later for complete independent training jobs, not cross-machine DDP;
4. freeze the train-seed-to-environment assignment before final training; and
5. perform final inference/scoring centrally on the primary environment.

## Technical conclusion

The worker is technically usable, but not with the project's CUDA 13 lane.
Both GPUs are Pascal compute capability 6.1. CUDA Toolkit 13 removed offline
compilation and library support for Pascal. A researched viable package tuple is:

- Python 3.12;
- `torch==2.9.1+cu126`;
- `torchvision==0.24.1+cu126`;
- `sionna-no-rt==2.0.1`; and
- `numpy==2.5.1`.

The current `torch==2.13.0+cu130` / Python 3.14.6 environment must remain the
primary lane. In particular, `torch==2.13.0+cu126` was not established and must
not be recorded as the Pascal solution.

Evidence behind this conclusion:

- NVIDIA CUDA 13 release notes: <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures>
- NVIDIA legacy GPU compute-capability table: <https://developer.nvidia.com/cuda-legacy-gpus>
- NVIDIA binary compatibility: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility>
- NVIDIA driver/toolkit compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
- Official PyTorch previous-version matrix: <https://pytorch.org/get-started/previous-versions/>
- Official PyTorch CUDA 12.6 wheel index: <https://download.pytorch.org/whl/cu126/torch/>
- cuDNN 9.10.2 support matrix: <https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html>
- Sionna installation requirements: <https://nvlabs.github.io/sionna/installation.html>

The official PyTorch 2.9.1 CUDA 12.6 wheel was inspected and carries suitable
native CUDA targets including `sm_60`; NVIDIA's binary-compatibility rule permits
that cubin on desktop `sm_61`. The reported R580 driver can execute CUDA 12.x
applications through backward compatibility. Pascal is deprecated in modern
cuDNN rather than absent, so actual DJSCC speed—especially AMP performance—must
be measured instead of assumed.

Python 3.14 is not a viable compatibility shortcut for this lane: importing the
tested Torch/Sionna combination under Python 3.14 reached Torch's explicit
`torch.compile is not supported on Python 3.14+` failure. Python 3.12 worked.
An isolated Python 3.12 CPU-side API audit collected 64 relevant tests across
LDPC, DJSCC, loss, channel and power code: 61 passed and the three CUDA-only
tests skipped. This establishes API compatibility, not GPU qualification.

## Qualification required before production use

Qualification must be non-scientific and must not contribute to G8 or training
evidence. At minimum it should establish, separately for each GPU:

- exact OS, Python, driver, Torch, CUDA, cuDNN, Sionna and NumPy versions;
- a CUDA initialization and real-kernel test;
- device name, UUID, index, compute capability and usable VRAM;
- deterministic-backend application and repeatability;
- DJSCC forward/backward, checkpoint save/load and resume;
- achievable batch size, epoch time and peak memory;
- whether AMP is correct and faster; and
- NVMe mount, capacity, free space, health, remote access and authenticated
  artifact-return procedure.

The preferred scheduling unit is an entire independent training run assigned to
one GPU. That avoids cross-machine collective communication and makes failures,
checkpoints and provenance easier to isolate. The two GPUs are not identical in
VRAM, so each must be profiled rather than treating “Pascal” as one measured
device.

## Spec and document impact if adopted for future training

Adoption for production training requires an append-only amendment, probably
the next AM entry, before such training begins. The amendment should:

- preserve `params.compute.primary_device` and the passed G-7 RTX 4060 profile;
- add a named auxiliary compute/environment profile rather than replace the
  current environment;
- add `requirements-pascal.in` and a separately generated, hashed
  `requirements-pascal.lock`;
- amend DEC-4 and SR-11 to cover independently qualified auxiliary devices;
- amend SR-12 to freeze the train-seed-to-environment assignment and define
  centralized final evaluation;
- amend SR-21 from a singular CUDA environment to authenticated named profiles;
- record `environment_profile`, device index/UUID, compute capability and the
  selected lockfile/hash in run metadata;
- put the environment profile into resolved run identity; and
- bump `params.config.fingerprint_schema_version` for that schema change, while
  leaving `analysis_version` unchanged unless the estimand or analysis changes.

After editing `spec/SPEC.md`, regenerate rather than hand-edit:

- `spec/params.generated.yaml`;
- `spec/DATASHEET.md`; and
- the generated files under `spec/concerns/`.

Current hand-written documentation that would then need corresponding updates:

- `AGENTS.md`;
- `NEXT.md` live sections only, retaining historical records;
- `README.md`;
- `docs/standards-and-tools-register.md`;
- `docs/PROJECT-KNOWLEDGE-TRANSFER.md`;
- `configs/README.md`; and
- a new Pascal qualification artifact and worklog.

The First Review contract, Gantt dates and radio deployment dossier do not need
their scope changed merely because an auxiliary training worker exists. Existing
G-1, G-2, G-7 and W4 evidence and historical worklogs must not be rewritten.

## Historical configuration-hash trap

`params.compute` and `params.environment` are both in
`params.config.fingerprint_parameter_roots`. An additive profile therefore
changes new run fingerprints. More subtly,
`tools/verify_w4_baseline_integration.py` currently requires every archived
parameter-root snapshot to equal the whole current root. Adding an off-path
Pascal profile would consequently make valid historical W4 evidence fail.

The amendment and implementation must add a narrow, amendment-bound historical
compatibility rule for this exact off-measurement-path addition. It must preserve
archived W4 configurations and hashes; it must not regenerate, rewrite or rerun
W4 merely to make the verifier green.

## Why the worker must not join live G8_C as things stand

The live runner contract records Torch `2.13.0+cu130` and Torch CUDA 13.0. The
current runner checks exact NumPy and Sionna versions but only checks that Torch
has some CUDA build, so a CUDA 12.6 runtime could pass the local guard while
contradicting its registered contract.

There are also two orchestration problems:

- the v2 coordinator reports one worker per visible GPU but currently starts
  only one process; and
- `SionnaLDPCAdapter` maps generic `cuda` to `cuda:0`, so the second GPU is not
  actually selected.

The state protocol uses local `fcntl.flock`. It is not an authenticated
multi-host evidence-transfer protocol and must not be treated as safe merely by
putting two clones on a shared or network filesystem.

If the Pascal host were deliberately admitted to G8_C, the minimum safe design
would be an additive source epoch after the current accepted boundary, a
superseding fail-closed runner/environment contract, explicit global shard and
device assignments, a non-scientific cross-environment parity probe, and an
authenticated export/import protocol in which only the canonical repository
writes campaign state. Epochs 1 and 2 and all accepted work-unit bytes must stay
immutable. Because the G8 campaign manifest binds the exact bytes of
`spec/SPEC.md` and `spec/params.generated.yaml`, even a future-training-only
spec amendment made before G8 closes needs deliberate campaign-manifest
supersession; it cannot be slipped in as unrelated documentation.

## Decision options for the next cold start

1. **Adopt after G8 for future training — recommended.** Prepare the amendment,
   lock and qualification work only after the live BLER campaign no longer
   depends on the current normative-source binding.
2. **Admit it to G8_C.** Technically possible but disproportionately expensive:
   requires a new authenticated epoch, runner contract, coordinator and remote
   evidence protocol before it can contribute one work unit.
3. **Keep it non-production.** Use it for disposable development and performance
   exploration only; nothing it produces becomes project evidence.

Questions worth settling in the debate:

- Is the goal only to accelerate W5/W7/W8 training, or also the live BLER suffix?
- Must every final seed run on one hardware class, or is a preregistered balanced
  seed-to-environment allocation acceptable?
- Is centralized final inference on the RTX 4060 acceptable?
- Is the schedule gain worth maintaining a second Python/Torch lock?
- Should the worker remain optional overflow so the capstone does not acquire a
  new critical dependency?

Until those questions are decided, the operational answer is **no delegation**.
