<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Hardware (Tier 2/3)

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## HR

- **HR-1** — Tier 2/3 hardware MUST satisfy `params.hardware_tier23.needs` at no less than `params.hardware_tier23.min_sample_rate_msps`, within `params.hardware_tier23.budget_inr_range`. Devices in `params.hardware_tier23.candidates` are indicative, not selected. *(verify: procurement checklist against the capability list)*
- **HR-2** — No hardware may be purchased before `params.hardware_tier23.purchase_gate` passes. *(verify: gate record)*
- **HR-3** — **Tier 2.** Encoder output MUST be replayed as IQ through a real link (wired loopback with attenuator) and captured, then decoded offline and compared against the simulated result at matched measured SNR. *(verify: measured-vs-simulated accuracy table)*
- **HR-4** — **Tier 3.** A live encoder/decoder demo on `params.hardware_tier23.edge_node` MUST meet `params.hardware_tier23.live_demo_latency_budget_ms` end-to-end. If it cannot at the headline dataset, resolve at G-6 by DEC-1 demotion or by pre-recording. *(verify: measured latency distribution)*
- **HR-5** — No Tier 1 requirement may depend on hardware availability. Tier 1 MUST be completable, reportable and defensible with simulation alone. *(verify: review of SR/BR/ER for hardware dependencies)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `hardware_tier23.budget_inr_range` | 25000, 40000 |
| `hardware_tier23.candidates` | adalm_pluto_x2, hackrf_one_plus_rtlsdr |
| `hardware_tier23.edge_node` | raspberry_pi_4_or_5 |
| `hardware_tier23.live_demo_latency_budget_ms` | 500 |
| `hardware_tier23.min_sample_rate_msps` | 1 |
| `hardware_tier23.needs` | iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator |
| `hardware_tier23.purchase_gate` | G-5 |
