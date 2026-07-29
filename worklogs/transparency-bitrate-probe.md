# Validation-only JPEG 2000 transparency-bitrate probe

**Status:** COMPLETE

Validation-only engineering probe. Does not select or replace G-8 operating points.

No training ran, the classifier remained frozen, and the test split stayed sealed.
G-8 remains unresolved; both threshold outputs below are probe forecasts only.

The design was committed before measurement. Its implementation identity is
`94f448db302f68c2f046cdd2f560d9e4805387db`, and the clean measurement checkout was
`3ccb0aa970e8d2d36f65f89a840785f92f73dec5`. The fixed 10-ID pilot exercised all 680
budget/axis/sample cells in 37.863 s, demonstrated repeat cache hits and projected the full probe at
1.052 h. The restartable four-axis full run took 4,612.092 s (76.868 min) in total. The ignored
content-addressed cache contains 68,000 entries and 464,997,315 logical bytes; tracked evidence
contains hashes and metadata but no pixels or codestream bytes.

The frozen 17-budget grid, in bytes, was:
`663, 800, 1330, 1600, 2400, 2661, 3200, 4000, 4800, 5328, 5344, 6400, 8000, 9600,
10656, 12800, 15997`. The 663/1330/2661/5328/10656/15997 points are the exact Imagenette source
budgets from the committed packetisation record at `r_1_48` through `r_1_2`, using 16-QAM and LDPC
5/6. The remaining labels are the committed direct-bpp points; 10,656 bytes carries both sources.

## Fixed codec

OpenJPEG 2.5.4 through Glymur 0.14.3; raw codestream, irreversible 9/7,
RPCL, six resolutions, 64×64 code blocks, whole-image tile, and bounded
compression-ratio bisection retaining the largest observed codestream at or
below the byte budget.

## Selected validation curve

| Budget bytes | Requested bpp | Axis | Correct | Accuracy | Δ clean | Mean realised bpp |
|---:|---:|---:|---:|---:|---:|---:|
| 663 | 0.2071875 | 96 | 774/1000 | 0.774000 | -0.124000 | 0.203901 |
| 800 | 0.25 | 96 | 806/1000 | 0.806000 | -0.092000 | 0.245996 |
| 1330 | 0.415625 | 128 | 870/1000 | 0.870000 | -0.028000 | 0.408654 |
| 1600 | 0.5 | 128 | 869/1000 | 0.869000 | -0.029000 | 0.492032 |
| 2400 | 0.75 | 160 | 883/1000 | 0.883000 | -0.015000 | 0.739953 |
| 2661 | 0.8315625 | 160 | 884/1000 | 0.884000 | -0.014000 | 0.821044 |
| 3200 | 1 | 160 | 886/1000 | 0.886000 | -0.012000 | 0.987788 |
| 4000 | 1.25 | 160 | 891/1000 | 0.891000 | -0.007000 | 1.235977 |
| 4800 | 1.5 | 160 | 894/1000 | 0.894000 | -0.004000 | 1.485043 |
| 5328 | 1.665 | 160 | 897/1000 | 0.897000 | -0.001000 | 1.648555 |
| 5344 | 1.67 | 160 | 897/1000 | 0.897000 | -0.001000 | 1.653524 |
| 6400 | 2 | 160 | 895/1000 | 0.895000 | -0.003000 | 1.981038 |
| 8000 | 2.5 | 160 | 896/1000 | 0.896000 | -0.002000 | 2.473208 |
| 9600 | 3 | 160 | 899/1000 | 0.899000 | 0.001000 | 2.960606 |
| 10656 | 3.33 | 160 | 898/1000 | 0.898000 | 0.000000 | 3.272807 |
| 12800 | 4 | 160 | 898/1000 | 0.898000 | 0.000000 | 3.875457 |
| 15997 | 4.999062 | 160 | 898/1000 | 0.898000 | 0.000000 | 4.672511 |

## Forecasts

```json
{
  "probe_crossover_threshold": {
    "label": "probe_crossover_threshold",
    "left_censored": false,
    "result": {
      "budget_bytes": 3200,
      "clean_accuracy": 0.898,
      "mean_realized_bpp": 0.9877875,
      "meets_2pp": true,
      "meets_5pp": true,
      "one_sided_95_lower_bound": -0.018000000000000016,
      "point_difference": -0.01200000000000001,
      "requested_bpp": 1.0,
      "selected_accuracy": 0.886,
      "selected_encode_axis": 160,
      "selected_n_correct": 886
    },
    "right_censored": false,
    "status": "measured"
  },
  "probe_efficiency_threshold": {
    "label": "probe_efficiency_threshold",
    "left_censored": false,
    "result": {
      "budget_bytes": 1330,
      "clean_accuracy": 0.898,
      "mean_realized_bpp": 0.408654375,
      "meets_2pp": false,
      "meets_5pp": true,
      "one_sided_95_lower_bound": -0.041000000000000036,
      "point_difference": -0.028000000000000025,
      "requested_bpp": 0.415625,
      "selected_accuracy": 0.87,
      "selected_encode_axis": 128,
      "selected_n_correct": 870
    },
    "right_censored": false,
    "status": "measured"
  }
}
```

Completed cells: 68000. Infeasible: 0. Decode failures: 0.
