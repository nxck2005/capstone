# Engineering Standards and Tools Register

**Register baseline:** 2026-08-11  
**Normative source:** [`spec/SPEC.md`](../spec/SPEC.md)  
**Environment pins:** [`spec/params.generated.yaml`](../spec/params.generated.yaml) and [`requirements.lock`](../requirements.lock)

This register satisfies PR-3. It distinguishes a claimed conformance boundary from contextual standards and implementation tools. A library implementing part of a standard does not make the whole system standards-conformant.

## 1. Status codes

| Code | Meaning |
|---|---|
| **C** | Claimed and tested conformance within the stated boundary |
| **I** | Implementation dependency with a frozen version or authenticated source |
| **R** | Informative reference; used to define terms or compare behavior |
| **P** | Planned tool or interface; not yet selected/frozen |
| **X** | Explicitly outside the project's conformance claim |

## 2. Communications and codec standards

| Standard | Edition/pin | Status | Project use and exact boundary | Verification/evidence |
|---|---|---:|---|---|
| [3GPP TS 38.212, *NR; Multiplexing and channel coding*](https://www.3gpp.org/dynareport/38212.htm) | Release 17, v17.13.0, dated 2026-02 | **C** | Transport-block CRC24A; code-block segmentation; CRC24B; BG1/BG2 selection; lifting-size selection; filler handling; 5G NR LDPC encoding, rate matching, and recovery. Project code performs transport-block framing and segmentation; Sionna provides the registered LDPC primitives. | Bit-exact project-owned vectors; authenticated srsRAN release vectors; independent packetisation arithmetic over all 216 registered configurations; G-2 BLER adjudication. |
| [3GPP TS 38.211, *NR; Physical channels and modulation*](https://www.3gpp.org/dynareport/38211.htm) | Release 17 context | **R/X** | Used as context for constellation terminology. Tier 1 maps generic BPSK, QPSK, and 16-QAM onto an abstract complex AWGN channel. It does **not** implement an NR slot, resource grid, synchronization signal, reference signals, or waveform. | The published claim names abstract AWGN and excludes full-link NR conformance. |
| [3GPP TS 38.214, *NR; Physical layer procedures for data*](https://www.3gpp.org/dynareport/38214.htm) | Release 17 context | **X** | No NR MCS table, transport-block-size procedure, scheduling, HARQ, or resource allocation is claimed. Codec quality, code rate, and modulation are research-grid choices under a fixed complex-symbol budget. | Explicit exclusion in `params.baseline.standards_claim_excludes`. |
| [ITU-T T.800, *JPEG 2000 image coding system: Core coding system*](https://www.itu.int/rec/T-REC-T.800) / ISO/IEC 15444-1 | Codec implementation fixed to OpenJPEG 2.5.4 | **C/I** | JPEG 2000 is the source codec of record. The project emits a raw codestream, uses the irreversible 9/7 wavelet, controls compression by ratio, and accounts for every emitted byte. | OpenJPEG version preflight occurs before artifact creation; byte-accounting and cache-identity tests; bounded W4 evidence. |
| [ITU-T T.81, *Digital compression and coding of continuous-tone still images*](https://www.itu.int/rec/T-REC-T.81) / ISO/IEC 10918-1 | Pillow 12.3.0 binding | **R** | Baseline JPEG is retained only as a labelled secondary sensitivity curve. It is not the headline codec because container overhead can dominate the smallest channel budgets. | Separate codec identity and cache key; no substitution for JPEG 2000. |
| [IEEE 754-2019, *Floating-Point Arithmetic*](https://standards.ieee.org/ieee/754/6210/) | Host/toolchain implementation | **R** | Defines the arithmetic context for NumPy/PyTorch calculations. The project does not claim cross-device bit-identical floating-point training; it claims seeded procedures, recorded environments, tolerances, and exact identity for serialized contract bytes. | Environment record plus CPU exact-hash and GPU same-seed tolerance policies. |
| [RFC 2119, *Key words for use in RFCs to Indicate Requirement Levels*](https://www.rfc-editor.org/rfc/rfc2119) | March 1997 | **R** | `MUST`, `SHOULD`, and `MAY` semantics used by the project specification and phase contracts. | Spec parser and generated concern views preserve requirement text. |
| [ISO 8601-1:2019, date and time representations](https://www.iso.org/standard/70907.html) | UTC timestamp convention | **R** | Human-readable execution and adjudication timestamps use ISO-8601/RFC-3339-compatible forms where the artifact schema calls for them. | Schema validators reject malformed or missing required fields; cryptographic identity does not rely on wall-clock time. |

### 2.1 Permitted standards claim

The defensible short claim is:

> The simulator implements a TS 38.212-derived 5G NR LDPC coding and rate-matching chain for byte-framed payloads, with generic BPSK/QPSK/16-QAM over an abstract AWGN channel; JPEG 2000 coding is provided by OpenJPEG 2.5.4.

It must always be accompanied by the exclusions: no TS 38.214 transport-block/MCS procedure, scheduling, HARQ, resource allocation, OFDM resource mapping, NR synchronization/pilots, waveform generation, π/2-BPSK, or full 5G NR link conformance.

## 3. Software tools and libraries

| Tool | Frozen/current version | Status | Purpose | Control |
|---|---:|---:|---|---|
| Python | 3.14.6 | **I** | Project runtime and tooling language | Fixed in `params.environment.python_version`; recorded in run metadata |
| PyTorch | 2.13.0+cu130; CPU lane 2.13.0+cpu | **I** | Models, training, tensors, CUDA execution | Hashed CUDA and CPU lockfiles; CUDA build assertion in primary environment |
| torchvision | 0.28.0+cu130; CPU lane 0.28.0+cpu | **I** | ResNet-18 and image operations | Same index/pin policy as PyTorch |
| Sionna | `sionna-no-rt==2.0.1` | **I** | 5G NR LDPC base graphs, lifting/encoding, rate matching/recovery, offset-min-sum decoding | Adapter seam; version pin; golden vectors; independent BLER reference |
| NumPy | 2.5.1 | **I** | Array operations, PCG64 identities, evidence serialization support | Hashed lock; seed/domain separation contracts |
| Pillow | 12.3.0 | **I** | Image decode and secondary JPEG codec | Hashed lock; dataset-specific decode rules |
| scikit-image | 0.26.0 | **I** | PSNR/SSIM metric support | Hashed lock; metric constants loaded from generated parameters |
| glymur | 0.14.3 | **I** | Python binding to OpenJPEG | Hashed lock; loaded codec version independently checked |
| OpenJPEG | 2.5.4 | **I** | JPEG 2000 encode/decode | External system dependency; exact loaded version checked before any J2K artifact path exists |
| PyYAML | 6.0.3 | **I** | Generated-parameter and tooling configuration parsing | Included in both full and CPU locks so exact sync cannot remove spec tooling |
| pytest | 9.1.1 | **I** | Behavioral, mutation, conformance, provenance, and gate tests | Hashed lock; marker taxonomy separates real external requirements |
| uv | >=0.11 for resolution | **I** | Generate hashed lockfiles with recorded index URLs | `unsafe-best-match` and `--emit-index-url` are required; runtime remains plain-pip installable |
| Git | Repository history | **I** | Source identity, immutable checkpoint history, release ancestry | Evidence binds commits and/or source-byte SHA-256 according to artifact role |
| GitHub / GitHub Actions | Workflow actions pinned by commit SHA | **I** | Remote durability, release assets, small hosted CI lanes | CI classifies changes; scientific evidence remains verified offline from committed artifacts |
| Streamlit | Version not yet frozen | **P** | W13 local demonstration UI | Selection and lockfile entry occur only when DR-1–DR-6 implementation starts |
| GNU Radio / SoapySDR or vendor API | Not selected | **P** | Optional Tier 2 SDR framing, synchronization, and replay | Selection is gated by G-5 and hardware availability; no current Tier 1 dependency |

## 4. Data, model, and evidence controls

These are project engineering controls rather than external standards, but they are load-bearing:

| Control | Registered implementation | Evidence |
|---|---|---|
| Dataset provenance | Exact archive byte length and SHA-256; canonical tracked split manifests | `tools/fetch_datasets.py --check`, manifest materialization, provenance-only test scan |
| Test isolation | Sole guarded boundary in `src/data/test_access.py`; release at G-12 | AST import-graph test and per-artifact zero-access counters |
| Randomness | Domain-separated keyed identities for initialization, data order, channel noise, and artifacts | Exact-key and control-flow-invariance tests |
| Configuration | Runtime reads generated YAML; `config_hash` is a versioned complete fingerprint | Literal lint and schema validation |
| Evidence identity | Canonical JSON, SHA-256, byte length, source manifest, contract IDs | Offline verifiers and mutation tests |
| Crash recovery | Atomic publication, per-unit locks, predecessor digests, exact resumable prefixes | G8 B2C/B3 contracts and adversarial tests |
| Statistical analysis | Paired image-level outcomes, preregistered confidence method, failure rows retained | Final ER-1/ER-10 analysis after G-12 |

## 5. Hardware and laboratory register

| Item | Current state | Role | Constraint |
|---|---|---|---|
| NVIDIA GeForce RTX 4060 Laptop GPU | Available and profiled | Training and full-strength simulation | CUDA 13.0 PyTorch build; driver 592.82 recorded in evidence |
| WSL2 `/dev/dxg` CUDA path | Available | GPU access boundary | A visible adapter is insufficient; PyTorch CUDA initialization is the acceptance test |
| HackRF One + RTL-SDR pair | Candidate only | Optional transmit/receive SDR replay | No purchase before G-5; at least 1 Msps required; candidate capability must be rechecked before procurement |
| SMA attenuator/cable chain | Candidate only | Conducted loopback and safe receiver level | Must be specified before connecting transmitter to receiver; antenna-free bench replay preferred |
| Raspberry Pi 4/5-class host | Candidate only | Optional Tier 3 edge demo | Attempt only if Tier 2 lands; otherwise prerecorded demonstration is the mandated fallback |

## 6. Register maintenance

Update this file when any standard boundary, version pin, implementation library, or hardware interface changes. A scientific requirement or parameter change additionally needs an append-only amendment in `spec/SPEC.md`; updating this register alone is insufficient. At each review:

1. compare every version with `spec/params.generated.yaml` and the lockfiles;
2. verify that claimed standards have an executable check;
3. downgrade any unverified claim to reference/planned status;
4. add newly selected demo or hardware tools; and
5. retain explicit exclusions so a subsystem claim cannot be read as end-to-end conformance.
