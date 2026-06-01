# Documentation

## Rename and scope

Aiono has been renamed to Aionoscope. In this codebase:
- the canonical Python package is `aiono`
- canonical synthetic config fields are `aiono` and `offline_probe_aiono`
- the old `aiono` names remain only as compatibility aliases

Only runs that explicitly declare:
- `benchmark_family: aiono_basic_components`
- `benchmark_version: v1`

are treated as benchmark-comparable Aionoscope `basic_components` runs.

## Canonical benchmark presets

Training presets:
- `configs/pretrain/dataset/aiono_basic_components_balanced.yaml`
- `configs/pretrain/dataset/aiono_basic_components_imbalanced.yaml`

Offline-probe presets:
- `configs/pretrain/offline_probe_dataset/aiono_basic_components_balanced.yaml`
- `configs/pretrain/offline_probe_dataset/aiono_basic_components_imbalanced.yaml`
- `configs/pretrain/offline_probe_dataset/both_basic_components.yaml`
- `configs/pretrain/offline_probe_dataset/same_as_train.yaml`

Historical legacy alias presets:
- `configs/pretrain/dataset/aiono_basic_components_balanced.yaml`
- `configs/pretrain/dataset/aiono_basic_components_imbalanced.yaml`
- `configs/pretrain/offline_probe_dataset/aiono_basic_components_balanced.yaml`
- `configs/pretrain/offline_probe_dataset/aiono_basic_components_imbalanced.yaml`

Those legacy presets remain useful for reproducing old diagnostics, but they are
not benchmark-comparable.

## `aiono_basic_components/v1` contract

The canonical benchmark settings are:
- `benchmark_family = "aiono_basic_components"`
- `benchmark_version = "v1"`
- `sampling_frequency = 500`
- `frequency_hz = auto`
- `min_full_periods = 1.0`
- `nyquist_fraction = 0.9`
- `sawtooth_min_points_per_period = 5`
- `square_min_points_in_shorter_plateau = 2`
- `square_duty_cycle.low = 0.1`
- `square_duty_cycle.high = 0.9`

The canonical presets also use:
- `channels: [I]`
- `channel_size: 5000`
- `view_name: mix`
- `validation_seed_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- `validation_seed_offset: 100`

For this preset length, the contract duration is:

$$
T = \frac{L - 1}{f_s} = \frac{5000 - 1}{500} = 9.998 \text{ sec}
$$

## Recoverability rules

The benchmark requirement is recoverability, not a vague "no aliasing" rule.
The shared upstream resolver computes bounds from:

$$
T = \frac{L - 1}{f_s}
$$

$$
f_{\min} = \frac{n_{\text{full periods}}}{T}
$$

$$
f_{\max}^{\text{sine}} = \alpha_{\text{Nyquist}} \cdot \frac{f_s}{2}
$$

$$
f_{\max}^{\text{sawtooth}} = \min\left(f_{\max}^{\text{sine}}, \frac{f_s}{p_{\text{sawtooth}}}\right)
$$

$$
r_{\text{plateau}} = \min(d_{\min}, 1 - d_{\max})
$$

$$
f_{\max}^{\text{square}} = \min\left(f_{\max}^{\text{sine}}, \frac{f_s \cdot r_{\text{plateau}}}{p_{\text{square plateau}}}\right)
$$

With the canonical `v1` defaults at `L=5000`, `f_s=500`:
- resolved minimum periodic frequency is `0.10002000400080016 Hz`
- sine upper bound is `225.0 Hz`
- sawtooth upper bound is `100.0 Hz`
- square upper bound is `24.999999999999993 Hz`

For the square-wave cap:

$$
r_{\text{plateau}} = \min(0.1, 1 - 0.9) = 0.1
$$

$$
f_{\max}^{\text{square}} = 500 \cdot \frac{0.1}{2} = 25
$$

That is why the manifest exposes
`square_frequency_hz_recoverability_upper_bound` and why the YAML comments refer
to shortest-plateau recoverability instead of generic aliasing.

## Validation-seed protocol

Benchmark offline probing uses:
- one train split
- multiple validation splits
- logical validation seed values stored in `validation_seed_values`
- actual generator seeds derived as
  `validation_seed_offset + validation_seed_value`

For the canonical presets:
- logical validation seeds are `0..9`
- generator seeds are `100..109`

This codebase materializes:
- `val_loaders_by_seed: dict[int, DataLoader]`
- `validation_seed_values`
- `validation_seed_offset`
- `validation_seed_to_generator_seed`
- `validation_seed_count`

Offline probe selection works on medians across validation seeds. The matching
sample-standard-deviation values are logged in `*_std` fields.

Examples:
- `macro_auc` and `macro_auprc` are validation-seed medians
- `macro_auc_std` and `macro_auprc_std` are spread across validation seeds
- dense metrics likewise expose `macro_mse_std`, `macro_mae_std`,
  `macro_r2_std`, and `macro_pearson_std`

## W&B metadata keys

Contract metadata is flattened under `offline_probe_contract/aiono/...`.
Important keys include:
- `offline_probe_contract/aiono/benchmark_family`
- `offline_probe_contract/aiono/benchmark_version`
- `offline_probe_contract/aiono/baseline_sampling_frequency_hz`
- `offline_probe_contract/aiono/actual_sampling_frequency_hz`
- `offline_probe_contract/aiono/duration_sec`
- `offline_probe_contract/aiono/periodic_frequency_mode`
- `offline_probe_contract/aiono/periodic_frequency_resolution_source`
- `offline_probe_contract/aiono/periodic_frequency_min_full_periods`
- `offline_probe_contract/aiono/periodic_frequency_nyquist_fraction`
- `offline_probe_contract/aiono/sine_frequency_hz_resolved_low`
- `offline_probe_contract/aiono/sine_frequency_hz_resolved_high`
- `offline_probe_contract/aiono/sawtooth_frequency_hz_resolved_low`
- `offline_probe_contract/aiono/sawtooth_frequency_hz_resolved_high`
- `offline_probe_contract/aiono/square_frequency_hz_resolved_low`
- `offline_probe_contract/aiono/square_frequency_hz_resolved_high`
- `offline_probe_contract/aiono/square_frequency_hz_recoverability_upper_bound`
- `offline_probe_contract/aiono/sawtooth_min_points_per_period`
- `offline_probe_contract/aiono/square_min_points_in_shorter_plateau`
- `offline_probe_contract/aiono/square_duty_cycle_min`
- `offline_probe_contract/aiono/square_duty_cycle_max`
- `offline_probe_contract/aiono/square_shorter_plateau_fraction_min`
- `offline_probe_contract/aiono/view_name`
- `offline_probe_contract/aiono/component_keys`
- `offline_probe_contract/aiono/num_enabled`
- `offline_probe_contract/aiono/validation_seed_values`
- `offline_probe_contract/aiono/validation_seed_offset`
- `offline_probe_contract/aiono/validation_generator_seeds`
- `offline_probe_contract/aiono/validation_seed_to_generator_seed/0`
- `offline_probe_contract/aiono/validation_seed_count`
- `offline_probe_contract/aiono/benchmark_comparable`

Nested periodic sampler fields are also emitted, for example:
- `offline_probe_contract/aiono/periodic_sampler_specs/sine/frequency_hz/high`
- `offline_probe_contract/aiono/periodic_sampler_specs/square/duty_cycle/low`

Absolute Aionoscope radar/reporting utilities should treat a run as
benchmark-comparable only when those metadata fields are present and
`benchmark_comparable=true`.

## Checkpoint evaluation semantics

During normal resume, checkpoint payloads are canonicalized onto the `aiono`
field names before `configs.pretrain.Config` is rebuilt.

During `offline_probe_eval_only=true`:
- current Hydra `offline_probe_*` overrides are merged after checkpoint load
- current top-level `sampling_frequency` is also overlaid
- benchmark validation happens on the merged config

Failure modes are explicit:
- if the checkpoint payload does not contain `sampling_frequency`, benchmark
  evaluation fails
- if the checkpoint was trained with `sampling_frequency != 500`, and the
  requested offline-probe dataset is `aiono_basic_components/v1`, evaluation
  fails

This prevents old checkpoints, including historical low-frequency experiments,
from being silently evaluated as if they were benchmark-compatible Aionoscope
runs.

## Troubleshooting

Run is missing benchmark comparability:
- Check that the dataset preset is `aiono_basic_components_balanced` or
  `aiono_basic_components_imbalanced`.
- Check that `benchmark_family` and `benchmark_version` are explicit.
- Check that `sampling_frequency` is still `500`.
- Check that W&B summary contains
  `offline_probe_contract/aiono/validation_seed_count`.

Run looks similar to old Aiono output but is rejected by absolute Aionoscope plots:
- That is expected for legacy `aiono_basic_components_*` presets and old
  single-seed runs.
- Use the canonical `aiono_*` presets if you want benchmark-comparable numbers.

`aiono` import fails:
- Install dependencies with `uv`.
- The project expects the upstream Aionoscope repository via the SSH git source
  in `pyproject.toml`:

```toml
aiono = { git = "ssh://git@github.com/langotime/aionoscope.git" }
```

Hydra complains when composing PTB-XL runtime data over a null `data:` field:
- Use `+data.ptb-xl=/path/to/dump.npy` so Hydra appends the nested key instead
  of overriding a missing dict entry.
