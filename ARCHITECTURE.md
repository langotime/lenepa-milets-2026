# Architecture

## Purpose

This reproduction repo owns training, checkpointing, offline probing, and W&B logging. It
does not own the `aiono_basic_components/v1` benchmark contract. That contract
lives upstream in the public `aiono` package and is consumed here through the
SSH git dependency configured in `pyproject.toml`:

```toml
aiono = { git = "ssh://git@github.com/langotime/aionoscope.git" }
```

The old `aiono` name remains only as a documented compatibility alias. New
configs, metadata, and user-facing benchmark guidance should use Aionoscope /
`aiono`.

## Contract boundary

Upstream `aiono` defines:
- benchmark family/version constants
- the public `AionoBasicComponentsPeriodicConfig`
- `resolve_aiono_basic_components_periodic_contract(...)`
- the resolved periodic sampler bounds and manifest fields

This reproduction repo defines:
- Hydra config composition
- dataset/backend dispatch
- conversion of the upstream manifest into `PretrainDataBundle` and `OfflineProbeDataBundle`
- offline probe training/evaluation over multiple validation seeds
- W&B logging under `offline_probe_contract/aiono/...`
- checkpoint resume and offline-probe-only evaluation semantics

## Canonical control plane

Canonical synthetic fields:
- `data_backend: aiono`
- `probe_source: aiono`
- `aiono: {...}`
- `offline_probe_source: aiono`
- `offline_probe_aiono: {...}`

Legacy compatibility aliases still accepted:
- `data_backend: aiono`
- `probe_source: aiono`
- `aiono: {...}`
- `offline_probe_source: aiono`
- `offline_probe_aiono: {...}`

The aliases are normalized onto the canonical `aiono` fields before checkpoint
reuse and benchmark checks. Historical legacy presets remain intentionally
non-comparable unless they carry explicit benchmark metadata.

## Benchmark data flow

1. Hydra composes a pretrain config, typically `ViTXS_aiono` plus
   `dataset=aiono_basic_components_balanced` or
   `dataset=aiono_basic_components_imbalanced`.
2. `pretrain_data_backend.pretrain_data_build(...)` dispatches the synthetic
   path to `data.backend_aiono.build_aiono_backend(...)`.
3. `data.backend_aiono` delegates to the shared implementation file
   `data.backend_aiono_impl`, which contains the synthetic backend code.
4. For `kind=basic_components`, `benchmark_family=aiono_basic_components`, and
   `benchmark_version=v1`, the backend:
   - requires top-level `sampling_frequency == 500`
   - parses `basic_components.periodic` through
     `AionoBasicComponentsPeriodicConfig.from_mapping(...)`
   - resolves bounds through
     `resolve_aiono_basic_components_periodic_contract(...)`
   - materializes a benchmark manifest and comparability flag
   - builds per-validation-seed offline-probe loaders
5. `pretrain.py` reuses that metadata during training/offline probing and writes
   the flattened manifest to W&B summary keys under
   `offline_probe_contract/aiono/...`.

## Benchmark invariants

`aiono_basic_components/v1` is defined here as a cross-repo contract, not as a
local convention. The important invariants are:
- `sampling_frequency == 500`
- `frequency_hz: auto`
- explicit `benchmark_family` and `benchmark_version`
- canonical component order
- validation-seed-aware offline probing

Runs that do not satisfy those invariants are treated as legacy diagnostics,
even if the label names look similar.

## Offline probing

For benchmark-comparable Aionoscope runs, offline probing uses:
- one training split
- multiple validation splits keyed by logical validation seed
- generator seeds computed from
  `validation_seed_offset + validation_seed_value`

The runners cache train features once per layer, evaluate every checkpoint
against every validation split, and report:
- median metrics in the legacy scalar keys
- matching `*_std` spread metrics
- `validation_seed_count`

This is part of the benchmark contract. Single-seed Aionoscope/Aiono runs are
historical and should not be mixed with benchmark-labeled medians.

## W&B contract

Absolute Aionoscope radar/reporting code should decide comparability from W&B
metadata alone. The canonical signal for that is the flattened metadata written
under `offline_probe_contract/aiono/...`, plus
`benchmark_comparable=true`.

`utils.aiono_benchmark.aiono_is_benchmark_comparable(...)` centralizes that
decision and requires:
- `benchmark_family == aiono_basic_components`
- `benchmark_version == v1`
- `baseline_sampling_frequency_hz == 500`
- `validation_seed_count > 0`

## Checkpoint semantics

Checkpoint payloads are canonicalized onto the `aiono` names before they are
reused. During `offline_probe_eval_only=true`, `pretrain.py` overlays current
Hydra `offline_probe_*` fields and the current top-level `sampling_frequency`
after loading the checkpoint payload. This prevents stale checkpoint-era
synthetic config from silently masquerading as the current benchmark contract.

If offline-probe-only evaluation requests `aiono_basic_components/v1`, then the
checkpoint must record `sampling_frequency == 500`. Otherwise evaluation fails
fast instead of producing a non-comparable benchmark-looking run.
