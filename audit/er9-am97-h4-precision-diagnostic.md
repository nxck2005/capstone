# AM-97 H4 pointwise precision diagnostic

AM-97 is a prospective, validation-only clarification made after the AM-96
semantic freeze and before any accepted ER-9 real-chain observation. The H4
learned arm is ordinary learned W8/G-10 `r_1_6` in the three frozen zipped seed
cells. The comparator is the corresponding final ER-9 digital control.
Randomized ER-2 is a separate robustness experiment and is not an H4 arm.

For each stable image and SNR, the diagnostic computes the signed correctness
difference in each of the three cells and averages those three values within
image. A deterministic Philox bootstrap resamples complete stable-image
trajectories, never individual cells, with 10,000 resamples. The 2 percentage
point value is a pointwise descriptive reference only.

This artifact does not certify the full calibrated H4 decision procedure or
its run-calibration requirement. Correlation across SNR points means pointwise
detection probabilities are not multiplied. A negative H4 result cannot,
from this diagnostic alone, be interpreted as strong evidence that no
meaningful learned-versus-ER-9 advantage exists; negative H4 conclusions remain
conservative. The status vocabulary is therefore
`pointwise_precision_diagnostic_complete`, optionally accompanied by
`pointwise_reference_met` or `pointwise_reference_not_met`; `full_strength`
and a claim that H4 has 80% power are prohibited.

No optimizer step, real-chain candidate evaluation, Stage-1 selection,
randomized-ER-2 scientific run, G-11 terminal adjudication, W10 action, or test
access is authorized by AM-97.
