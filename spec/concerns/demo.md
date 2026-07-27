<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Demo

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## DR

- **DR-1** — An SNR slider spanning `params.channel.test_snr_grid_db` MUST drive both pipelines live on the same input image. *(verify: manual demo script walkthrough)*
- **DR-2** — The interface MUST show, side by side, the classical output, the semantic reconstruction, and each system's predicted label with confidence. *(verify: manual walkthrough)*
- **DR-3** — The accuracy-vs-SNR plot MUST update live with a marker at the current slider position. *(verify: manual walkthrough)*
- **DR-4** — The demo MUST render figures through `params.demo.figure_style_module`, the same module that renders thesis figures, using `params.demo.fonts` and `params.demo.palette`, with default framework chrome suppressed (DEC-7). *(verify: visual review that a demo figure and its thesis counterpart come from the same style module — pixel comparison is not required and is not achievable across DPI, backend and font hinting)*
- **DR-5** — The demo MUST run with `params.demo.offline` true and remain usable on CPU only. *(verify: run with networking disabled on a CPU-only machine)*
- **DR-6** — The demo MUST consume frozen checkpoints and committed result CSVs; it MUST NOT train, fine-tune or recompute reported metrics. *(verify: code review)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `channel.test_snr_grid_db` | -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18 |
| `demo.figure_style_module` | src/viz/style.py |
| `demo.fonts` | serif_computer_modern |
| `demo.offline` | true |
| `demo.palette` | colorblind_safe |
