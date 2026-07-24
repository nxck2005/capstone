<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Baseline

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## BR

- **BR-1** — The classical chain MUST be `params.baseline.source_codec` → `params.baseline.channel_code` → modulation → the same channel implementation the learned system uses. *(verify: integration test asserting the shared channel object)*
- **BR-2** — The LDPC implementation MUST be validated against published 5G NR BER curves before any learned-vs-classical comparison is reported. *(verify: archived BER-vs-SNR plot with reference curve overlaid)*
- **BR-3** — The baseline MUST receive exactly the same number of complex channel uses `k` as the learned system, per `params.baseline.budget_rule`. Bandwidth matching is counted in channel uses, not in bytes. *(verify: unit test counting emitted symbols for both systems)*
- **BR-4** — At each test SNR the baseline MUST be tuned in its own favour: sweep `params.baseline.jpeg_quality_grid` × `params.baseline.ldpc_rates` and report the **best feasible** configuration, per `params.baseline.tuning`. Reporting a single fixed configuration across all SNRs is prohibited. *(verify: sweep artifact showing the selected config per SNR)*
- **BR-5** — If no JPEG quality produces a file fitting the payload budget, the transmission MUST be recorded as infeasible and scored per `params.baseline.outage_policy`, never silently skipped. *(verify: unit test at the smallest ratio where infeasibility is expected)*
- **BR-6** — A file that cannot be decoded after LDPC decoding MUST be scored per `params.baseline.outage_policy` and counted in `decode_failure_rate`. *(verify: unit test injecting an undecodable block)*
- **BR-7** — Both systems MUST see identical test images and identical noise realisations at a given seed and SNR. *(verify: test asserting bitwise-identical noise draws across the two pipelines)*
- **BR-8** — A frozen `params.reference_classifier` trained on clean images MUST meet each dataset's `clean_acc_floor`, and the same instance MUST score both the classical reconstructions and the semantic reconstruction ablation (ER-4). *(verify: measured clean accuracy per dataset, archived)*
- **BR-9** — `params.baseline.core_modulation` is used for headline results; other entries in `params.baseline.modulations` MAY be reported as supporting evidence. *(verify: config test)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `baseline.budget_rule` | payload_bits = floor(k * bits_per_symbol * rate); the JPEG file MUST fit within payload_bits |
| `baseline.channel_code` | 5g_nr_ldpc |
| `baseline.core_modulation` | qpsk |
| `baseline.jpeg_quality_grid` | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 |
| `baseline.ldpc_rates` | 1/3, 1/2, 2/3, 5/6 |
| `baseline.modulations` | bpsk, qpsk |
| `baseline.outage_policy` | chance_level |
| `baseline.source_codec` | jpeg |
| `baseline.tuning` | best_feasible_config_per_test_snr |
| `reference_classifier` | *(see datasheet)* |
