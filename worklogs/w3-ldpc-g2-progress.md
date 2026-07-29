# W3 LDPC / G-2 progress

Started 2026-07-30 after the lineage-bound transparency evidence was pushed and
verified through the public remote. Scope is deliberately limited to the
separated physical-layer foundation and G-2: no BR-4 image sweep, G-8 choice,
training, ER-9, lambda work, or test evaluation.

## Frozen implementation design

- Sionna `2.0.1` remains behind the BR-14 adapter.
- The project owns TB CRC, segmentation, CB CRC, filler placement, exact
  `E_r` distribution, concatenation, CRC failure handling and packet metadata.
- Rung 2 uses srsRAN `release_25_10`, asset `phy_testvectors.tar`, complete
  asset SHA-256 `816d75db7c0d175ea906cae1c515a6ea5295d91e1db14b5285a950c452fa70b5`.
- The two converted cases pin BG1/Z36 and BG2/Z64. The generated NPZ stays
  ignored. The committed offline floor is a hand-derived BG2/Z2 case.
- The statistical reference is the MIT-licensed Lcrypto 5G LDPC platform at
  commit `2fde4c43bae04c0d8397b3e7e46eaa6070e16b3c`; codeload SHA-256
  `457a90726a40dd40f7c115b2477f4b5ce11cecea968f8a48e51f52d0aff43ffc`.
  Its flooding OMS update is reconstructed literally, with its optional
  parity stop disabled so both arms run the configured 50 iterations.
- The G-2 comparison fixes K=128, N=256, BG2, Z=22, rate 1/2, offset 0.5 and
  all three configured modulations. Source Eb/N0 is converted as
  `Es/N0 = Eb/N0 + 10 log10(R Q_m)`.
- The 1/2/4-packet sensitivity design is frozen but not executed: its ratio is
  the future G-8 validation-selected headline ratio.

## Boundary audit

The G-2 runner does not import `src/data/test_access.py`, load an image,
canonicalize pixels, run a classifier, or train a model. Its only random data
are synthetic information bits and keyed AWGN.

## G-2 adjudication — 2026-07-30

W3 implementation is frozen at
`968e907237bbe571adf6ec48e4711ea021831719`. The attempt to wait on
`PYTHONPATH=src .venv/bin/python tools/run_ldpc_g2.py` was interrupted by the
user, but the runner had already finished normally; a process check confirmed
there is no surviving G-2 process.

Eight complete evidence files are committed under `results/baseline/g2/`. The
CSV has 24 data rows. The adjudication binds the clean W3 commit and reports
PASS for golden vectors, offline fixture,
known answers, independent-reference provenance, simulation sufficiency,
waterfall displacement, 216-cell runtime packetisation, the frozen-but-unrun
progressive design, commit cleanliness and the test seal.

Measured displacements:

- BPSK: `0.0 dB` (`Es/N0` waterfall `-0.44272625389121256 dB` in both arms).
- QPSK: `+0.0036302378989723216 dB` (reference `2.5213324689357486 dB`,
  Sionna `2.524962706834721 dB`).
- 16-QAM: `0.0 dB` (`7.894208261597202 dB` in both arms).

All three are inside the configured 0.5 dB limit. Both implementations met the
configured simulation-sufficiency check at every modulation, every known-answer
and golden-vector component passed, runtime packetisation reconciled all 216
cells, and test-access counters remained zero. The progressive 1/2/4-packet
design is frozen but deliberately unrun until G-8 supplies the validation-selected
headline ratio. The single next task is bounded W4 classical-baseline integration
before G-8; no W4 work was started here.
