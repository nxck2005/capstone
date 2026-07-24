<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Hardware (Tier 2/3)

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## HR

- **HR-1** — Tier 2/3 hardware MUST satisfy `params.hardware_tier23.needs` at no less than `params.hardware_tier23.min_sample_rate_msps`, within `params.hardware_tier23.budget_inr_range` and subject to `params.hardware_tier23.budget_note`. Devices in `params.hardware_tier23.candidates` are indicative, not selected. *(verify: procurement checklist against the capability list)*
- **HR-2** — No hardware may be purchased before `params.hardware_tier23.purchase_gate` passes. *(verify: gate record)*
- **HR-3** — **Tier 2 (stretch).** Encoder output SHOULD be replayed as IQ through a real link (wired loopback with attenuator) and captured, then decoded offline and compared against the simulated result at matched measured SNR. *(verify: measured-vs-simulated accuracy table, or a recorded decision not to attempt)*
- **HR-4** — **Tier 3 (stretch).** A live encoder/decoder demo on `params.hardware_tier23.edge_node` SHOULD meet `params.hardware_tier23.live_demo_latency_budget_ms` end-to-end. If it cannot, resolve at G-6 by pre-recording per `params.hardware_tier23.expected_demonstration`. *(verify: measured latency distribution, or the pre-recorded artifact)*
- **HR-5** — No Tier 1 requirement may depend on hardware availability. Tier 1 MUST be completable, reportable and defensible with simulation alone. *(verify: review of SR/BR/ER for hardware dependencies)*
- **HR-6** — Any Tier 2 replay MUST specify and implement the full physical-layer wrapper in `params.hardware_tier23.framing`, and MUST state how channel SNR is *measured* rather than assumed, so the measured-versus-simulated comparison in HR-3 is meaningful. Rationale: an IQ replay without framing, timing recovery and frequency-offset correction does not produce a comparable link, and pilot overhead changes the channel-use accounting that BR-3 depends on. *(verify: framing design note plus a loopback capture showing locked timing and residual CFO within tolerance)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `hardware_tier23.budget_inr_range` | 25000, 40000 |
| `hardware_tier23.budget_note` | two Plutos sit at or above the top of this range once import duty lands; the HackRF + RTL-SDR pairing is what the range actually buys |
| `hardware_tier23.candidates` | adalm_pluto_x2, hackrf_one_plus_rtlsdr |
| `hardware_tier23.edge_node` | raspberry_pi_4_or_5 |
| `hardware_tier23.expected_demonstration` | pre_recorded |
| `hardware_tier23.framing` | preamble_correlation, rrc_pulse_shaping, timing_sync, cfo_estimation, pilot_aided_snr_measurement |
| `hardware_tier23.live_demo_latency_budget_ms` | 500 |
| `hardware_tier23.min_sample_rate_msps` | 1 |
| `hardware_tier23.needs` | iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator |
| `hardware_tier23.purchase_gate` | G-5 |
