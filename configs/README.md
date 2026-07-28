# Experiment configuration files

Files here are committed, reviewable descriptions of an experiment. They contain:

- `experiment`: a stable descriptive name;
- `choices`: symbolic choices such as `bw_ratio: crossover_ratio`; and
- `sweep_axes`: parameter-list names whose concrete value is supplied to
  `load_experiment()` for one run.

They do not duplicate values derived from `spec/params.generated.yaml`. For
example, a file names `crossover_ratio`, not `r_1_3`, and never repeats `k`.
Loading a file stores both the symbolic choice and what it currently resolves
to. The fully resolved `RunConfig` and its SHA-256 hash are archived beside the
run results.

Add files when an experiment needs them; do not generate one committed YAML file
per sweep cell.
