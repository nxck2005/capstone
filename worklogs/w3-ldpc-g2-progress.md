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
