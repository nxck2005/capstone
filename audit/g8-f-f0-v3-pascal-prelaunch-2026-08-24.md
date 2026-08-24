# G8_F Pascal F0-v3 pre-launch relocation audit — 2026-08-24

## Verdict

**PASS — F0-v3 is an additive, pre-data execution-profile relocation from
`local_4060_cu130` to `confessor_pascal_cu126`; the scientific protocol and
AM-88 assignment are unchanged. F1 launch is separately owner-authorized but
F2/training/pass two/fallback/ratio adjudication/test remain closed.**

No amendment was added: DEC-4/SR-23 already require one eligible profile to be
selected and frozen before the first measurement of a scientific campaign.

## Immutable predecessor evidence

The audit re-read and content-authenticated both predecessor files without
changing either byte:

- F0-v1: `results/baseline/g8_f/f0_execution_authorization.json`, ID
  `g8ff0auth-92189865202e4b6cb400a0a86cee101b8ad8a7bdf5ea9d5a78ae96ab49a365b4`,
  file SHA-256
  `17a88e36201d42b3b2ace190b0b5b5f3b34aeb3afb48f8a84e26db159b86de94`;
  superseded before F1 for incomplete resume/referenced-object authentication,
  production coverage zero.
- F0-v2: `results/baseline/g8_f/f0_v2_execution_authorization.json`, ID
  `g8ff0v2auth-dbcac1f4dcf76238a4222629e590372004f5dad3e4fb1316e28b6fd0b93c6f31`,
  file SHA-256
  `b14691ca26b6086d9b8e08b563027047cdba114b438311208fe6d413f5c29ce9`;
  superseded before F1 only because the owner selected the already-qualified
  Pascal profile instead of its frozen local profile, production coverage zero,
  scientific protocol changed false.

## Active F0-v3

- Path: `results/baseline/g8_f/f0_v3_execution_authorization.json`
- ID: `g8ff0v3auth-e261cd53d3bb9fdee1cdde0778f36c2a686e17507b660ff8ec42891bde102497`
- File SHA-256: `391cd81553ed2de869ddf3ad1f0a401523781289342eefaddc7ad27cb005517e`
- Bound scientific source commit: `6f06aa81ae2d624bae0d406904982f3a61278d93`
- Profile/device: `confessor_pascal_cu126` / Torch `cuda:0`
- GPU: `NVIDIA TITAN Xp`, UUID
  `GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a` (physical `nvidia-smi` index 1)
- Driver / Python / Torch / CUDA build: `580.178.04` / `3.14.6` /
  `2.13.0+cu126` / `12.6`
- Torchvision / NumPy / Sionna / Glymur / OpenJPEG: `0.28.0+cu126` /
  `2.5.1` / `2.0.1` / `0.14.3` / `2.5.4`
- Lock: `requirements-pascal.lock`, SHA-256
  `d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82`
- Storage at freeze: 393,746,763,776 bytes available; required reserve
  7,323,313,680 bytes.

The worker environment contained exactly the 56 lock distributions, with no
missing, extra or version-mismatched distribution and no `pytest-xdist`/`xdist`.
The Imagenette archive authenticated at 99,003,388 bytes and SHA-256
`64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`;
only the train payload is admitted by F1.

## Independent protocol reconstruction

Offline and live verification independently reproduced:

- AM-87 ID
  `g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148`,
  file SHA-256
  `733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c`;
- AM-88 ID
  `g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6`,
  file SHA-256
  `eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23`;
- 120 support qualities, 8,469 exact train stable IDs, six distinct qualities
  per image, and exactly 50,814 ordered assignments;
- ordered-pair SHA-256
  `c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229`;
- pair-set SHA-256
  `255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e`;
- typed codec infeasibility means omission without replacement/resampling;
  every other failure is HOLD;
- G8_E E7 and pass one exactly once, G1 unchanged, and test sealed.

At F0-v3 freeze, the F1 runtime and launch authorization were absent; requests,
results, materialized objects, real F1 JPEG2000 invocations, inference,
optimizer steps, pass two, fallback, ratio adjudication, learned training and
test access were all zero. The later separate owner launch artifact is
`results/baseline/g8_f/f1_launch_authorization.json`, ID
`g8ff1launch-a88fc23774b38763858e2fec717bf27f0f79893bb6b768708c0f3d38a570ee74`,
file SHA-256
`4265b69660d40fe72cc7957a1709e33c46384fc82ea59e12e075d2363f36a111`.
It binds only this F0-v3 and F1 source and authorizes F1 materialization only.
