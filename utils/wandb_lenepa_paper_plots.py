"""Export W&B offline-probe histories and build paper-ready LeNEPA plots.

This script targets the LeNEPA paper experiment matrix defined in:
- `plans/110_lenepa_experiments_nebius_parallel_run_plan.md`
- `paper/plan.md`

It generates two plot families for (non-Annex) runs:
1) **Training dynamics**: best layer per step (oracle over layers 0..8).
2) **Best layer location**: all layers at the last eval step.

Appendix-only (seed=0) ablations from plan-110 are also plotted:
  - A1) JEPA masking fragility (keep-ratio sensitivity; seed=0 only).
  - A2) NEPA Conv tokenizer kernel/stride sensitivity (patch_size; seed=0 only).
  - A4) SIGReg component ablations (single component; seed=0 only).

Metrics:
- PTB-XL: AUROC/AUPRC.
- Aionoscope (basic components): AUROC/AUPRC + MSE/MAE/Pearson/R2.

The run set is discovered from:
- `experiments/legacy_job_scripts/plan110/queue_order.txt` (seed0 base)
- `experiments/legacy_job_scripts/plan110/queue_seed_std_s1-4.txt` (extra seeds for variability; seeds 1..4)
- `experiments/legacy_job_scripts/plan110/queue_seed_std_s5-9.txt` (extra seeds for variability; seeds 5..9)

Paper-facing plots/tables aggregate across seeds as `median ± std` (sample std,
ddof=1), using seed pool {0,1,2,3,4,5,6,7,8,9} and aggregating over the subset
of seeds that are available per (config, dataset) group (requires at least 2
seeds to compute sample std).
W&B run IDs are discovered from VM job logs under `experiments/legacy_job_scripts/`.

Usage:
  uv run python -m utils.wandb_lenepa_paper_plots
"""

from __future__ import annotations

import argparse
import json
import time
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import yaml

from utils.aiono_benchmark import aiono_is_benchmark_comparable


_WANDB_URL_RE = re.compile(r'/runs/([a-z0-9]{8})(?:\b|/)', re.IGNORECASE)

_A1_JEPA_BASE_KEEP_RATIO = (0.15, 0.25)
_A2_NEPA_BASE_PATCH_SIZE = 25


@dataclass(frozen=True)
class RunMeta:
  """Metadata for one LeNEPA paper run (one dataset, one method, one seed)."""

  run_name: str
  dataset: str
  method_family: str
  variant: str | None
  projector: bool | None
  seed: int
  wandb_entity: str
  wandb_project: str
  wandb_run_id: str
  job_log_path: Path

  @property
  def wandb_path(self) -> str:
    """Return the W&B API path for this run."""
    return f'{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}'


@dataclass(frozen=True)
class MetricSpec:
  """Specification for a per-layer offline-probe metric series."""

  name: str
  direction: str  # 'max' or 'min'
  key_template: str
  pretty: str

  def key(self, layer: int) -> str:
    """Build the W&B history key for a specific layer."""
    return self.key_template.format(layer=int(layer))


@dataclass(frozen=True)
class LastStepBestLayer:
  """Best-layer metric value at a run's last offline-probe eval step."""

  last_step: int
  best_layer: int
  best_value: float


@dataclass(frozen=True)
class BestLayerAtStep:
  """Best-layer metric value at a specific offline-probe eval step."""

  step: int
  best_layer: int
  best_value: float


@dataclass(frozen=True)
class TrainStepTimeStats:
  """Median per-step wall-clock duration for a training run.

  The value is computed from W&B history key `train/step_time` over a fixed
  step window, and therefore does not include offline-probe evaluation time.
  """

  min_step: int
  max_step: int
  median_s: float
  n: int


@dataclass(frozen=True)
class DenseTargetSpec:
  """Aiono dense probe target spec for per-target analysis tables."""

  name: str
  signal: str
  metric: str


@dataclass(frozen=True)
class SeedStats:
  """Seed-aggregated summary statistics for paper values.

  Aggregation rule (paper-facing):
    - center: median across seeds
    - spread: sample standard deviation across seeds (ddof=1)
  """

  median: float
  std: float
  n: int


def _aiono_basic_components_target_metric_from_param(*, target_name: str, param: str) -> str:
  """Convert a Aiono config `param` into a table-friendly metric name."""
  # Step 1: the config uses time indices for events; the table uses unit-interval fractions.
  if param == 'time_idx' and target_name.endswith('_time_frac'):
    return 'time_frac'
  return param


def _aiono_basic_components_massage_target_metric(*, target_signal: str, target_metric: str) -> str:
  """Normalize dense target parameter names for analysis.

  Rules (requested for Plan-110 Aionoscope blind-spot analysis):
    1) spike/gaussian/level_change amplitude -> magnitude
    2) uniform_noise amplitude + gaussian_noise_std + random_walk_step_std -> std
    3) (quadratic_trend b) and (linear_trend slope) -> slope
    4) (quadratic_trend c) and (linear_trend intercept) -> intercept
  """
  # Step 1: unify singular-event scale parameters.
  if target_metric == 'amplitude' and target_signal in ('spike', 'gaussian', 'level_change'):
    return 'magnitude'

  # Step 2: unify noise scale parameters.
  if target_signal in ('uniform_noise', 'gaussian_noise', 'random_walk_noise'):
    return 'std'

  # Step 3: unify linear coefficient naming across linear/quadratic trends.
  if target_signal == 'quadratic_trend' and target_metric == 'b':
    return 'slope'

  # Step 4: unify intercept naming across linear/quadratic trends.
  if target_signal == 'quadratic_trend' and target_metric == 'c':
    return 'intercept'

  return target_metric


def _load_aiono_basic_components_config_block(config_path: Path) -> dict[str, object]:
  """Load the canonical Aionoscope basic-components config block from YAML."""
  # Step 1: parse YAML and validate the expected structure.
  cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
  if not isinstance(cfg, dict):
    raise ValueError(f'Expected YAML dict at top-level, got {type(cfg).__name__}')

  aiono = cfg.get('aiono')
  if isinstance(aiono, dict):
    return aiono

  raise ValueError('Missing aiono dict in config.')


def _load_aiono_basic_components_dense_targets(config_path: Path) -> list[DenseTargetSpec]:
  """Load Aionoscope basic-components dense targets (name + signal + metric) from a YAML config."""
  # Step 1: read the canonical Aionoscope block.
  aiono = _load_aiono_basic_components_config_block(config_path)

  dense_probe = aiono.get('dense_probe')
  if not isinstance(dense_probe, dict):
    raise ValueError('Missing aiono.dense_probe dict in config.')

  targets = dense_probe.get('targets')
  if not isinstance(targets, list) or not targets:
    raise ValueError('aiono.dense_probe.targets must be a non-empty list.')

  # Step 2: collect dense targets in config order.
  out: list[DenseTargetSpec] = []
  seen_names: set[str] = set()
  for idx, item in enumerate(targets):
    if not isinstance(item, dict):
      raise ValueError(f'aiono.dense_probe.targets[{idx}] must be a dict, got {type(item).__name__}')

    name = item.get('name')
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f'aiono.dense_probe.targets[{idx}].name must be a non-empty str, got {name!r}')

    enabled_key = item.get('enabled_key')
    if not isinstance(enabled_key, str) or not enabled_key.strip():
      raise ValueError(
        f'aiono.dense_probe.targets[{idx}].enabled_key must be a non-empty str, got {enabled_key!r}. '
        f'Target={name!r}')

    param = item.get('param')
    if not isinstance(param, str) or not param.strip():
      raise ValueError(
        f'aiono.dense_probe.targets[{idx}].param must be a non-empty str, got {param!r}. '
        f'Target={name!r}')

    metric_raw = _aiono_basic_components_target_metric_from_param(target_name=name, param=param)
    metric = _aiono_basic_components_massage_target_metric(target_signal=enabled_key, target_metric=metric_raw)

    if name in seen_names:
      raise ValueError(f'Duplicate dense target name in config: {name!r}')
    seen_names.add(name)
    out.append(DenseTargetSpec(name=name, signal=enabled_key, metric=metric))

  return out
def _load_aiono_basic_components_signal_classes(config_path: Path) -> list[str]:
  """Load Aionoscope basic-components signal classes (component keys) from a YAML config."""
  # Step 1: read the canonical Aionoscope block.
  aiono = _load_aiono_basic_components_config_block(config_path)

  basic = aiono.get('basic_components')
  if not isinstance(basic, dict):
    raise ValueError('Missing aiono.basic_components dict in config.')

  component_keys = basic.get('component_keys')
  if not isinstance(component_keys, list) or not component_keys:
    raise ValueError('aiono.basic_components.component_keys must be a non-empty list[str].')
  if not all(isinstance(key, str) and key.strip() for key in component_keys):
    raise ValueError('aiono.basic_components.component_keys must contain only non-empty strings.')
  keys = [str(key) for key in component_keys]
  if len(set(keys)) != len(keys):
    raise ValueError(f'aiono.basic_components.component_keys must be unique, got: {keys}')

  return keys


def _load_aiono_basic_components_signal_classes(config_path: Path) -> list[str]:
  """Legacy wrapper kept for compatibility with older tests/imports."""
  return _load_aiono_basic_components_signal_classes(config_path)


def _require_aiono_benchmark_comparable_run(
  run: wandb.apis.public.Run,
  *,
  context: str,
) -> None:
  """Fail fast when a run lacks benchmark-comparable Aionoscope metadata."""
  if aiono_is_benchmark_comparable(source='aiono', summary=run.summary, config=run.config):
    return
  raise ValueError(
    f'{context} requires benchmark-comparable Aionoscope runs tagged via '
    'offline_probe_contract/aiono for aiono_basic_components/v1. '
    f'Got non-comparable run={run.name!r}.'
  )


def _repo_root() -> Path:
  """Return the repo root path regardless of current working directory."""
  return Path(__file__).resolve().parents[1]


def _load_queue_order(queue_order_path: Path) -> list[str]:
  """Load run names from the queue-order file (one run name per line)."""
  if not queue_order_path.is_file():
    raise FileNotFoundError(f'Queue-order file not found: {queue_order_path}')
  lines = queue_order_path.read_text(encoding='utf-8').splitlines()
  run_names = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
  if not run_names:
    raise ValueError(f'Queue-order file is empty: {queue_order_path}')
  return run_names


def _parse_seed(run_name: str) -> int:
  """Parse a `s{seed}` token into an int seed.

  The LeNEPA paper run names use a trailing `..._s0` suffix, but some newer
  local runs embed the seed token earlier (e.g. `..._s0_CONT200K_LOCAL`).
  We therefore accept `sN` as any underscore-delimited token and require it to
  be unique within the run name.
  """
  parts = [p for p in run_name.split('_') if p]
  if not parts:
    raise ValueError(f'Invalid run name (empty): {run_name!r}')

  matches = [p for p in parts if p.startswith('s') and p.removeprefix('s').isdigit()]
  if not matches:
    raise ValueError(f'Run name is missing seed token "sN": {run_name!r}')

  seeds = {int(p.removeprefix('s')) for p in matches}
  if len(seeds) != 1:
    raise ValueError(f'Run name contains multiple distinct seed tokens: {sorted(seeds)} run_name={run_name!r}')
  return int(next(iter(seeds)))


def _parse_run_meta_stub(run_name: str) -> tuple[str, str, str | None, bool | None, int]:
  """Parse dataset/method metadata from a run name.

  Returns:
    (dataset, method_family, variant, projector, seed)
  """
  # Step 1: basic validation and seed parsing.
  seed = _parse_seed(run_name)
  parts = [p for p in run_name.split('_') if p]
  if len(parts) < 3:
    raise ValueError(f'Invalid run name (too few components): {run_name!r}')

  # Step 2: dataset prefix.
  dataset = parts[0]

  # Step 3: locate the method family token (allows dataset-specific prefixes like L5000).
  family_idx: int | None = None
  for i in range(1, len(parts)):
    if parts[i] in ('JEPA', 'NEPA', 'LENEPA'):
      family_idx = int(i)
      break
  if family_idx is None:
    raise ValueError(
      f'Unexpected run name (missing method family token): {run_name!r}. '
      'Expected one of: JEPA, NEPA, LENEPA.'
    )
  family = parts[family_idx]
  if family == 'JEPA':
    method_family = 'JEPA'
    if len(parts) < family_idx + 2:
      raise ValueError(f'Unexpected JEPA run name (missing variant token): {run_name!r}')
    variant = parts[family_idx + 1]
    if variant not in ('CLS', 'MEAN'):
      raise ValueError(f'Unexpected JEPA variant {variant!r} in {run_name!r}')
    projector = None
    return dataset, method_family, variant, projector, seed

  if family == 'NEPA':
    method_family = 'NEPA_STOPGRAD'
    if len(parts) < family_idx + 5 or parts[family_idx + 1] != 'STOPGRAD':
      raise ValueError(f'Unexpected NEPA run name format: {run_name!r}')
    if parts[family_idx + 2] != 'MEAN':
      raise ValueError(f'Unexpected NEPA pooling (expected MEAN): {run_name!r}')
    proj = parts[family_idx + 3]
    if proj not in ('PROJ', 'NOPROJ'):
      raise ValueError(f'Unexpected projector suffix {proj!r} in {run_name!r}')
    projector = proj == 'PROJ'
    return dataset, method_family, None, projector, seed

  if family == 'LENEPA':
    method_family = 'LENEPA'
    if len(parts) < family_idx + 4:
      raise ValueError(f'Unexpected LeNEPA run name format: {run_name!r}')

    # LeNEPA run names can include optional suffix tokens (e.g. BAL/ECG/TL1)
    # between the projector flag and the seed token.
    projector_idx: int | None = None
    for i in range(len(parts) - 1, family_idx, -1):
      if parts[i] in ('PROJ', 'NOPROJ'):
        projector_idx = int(i)
        break
    if projector_idx is None:
      raise ValueError(f'Unexpected LeNEPA run name (missing PROJ/NOPROJ token): {run_name!r}')

    proj = parts[projector_idx]
    projector = proj == 'PROJ'
    variant = '_'.join(parts[family_idx + 1 : projector_idx])
    if not variant:
      raise ValueError(f'LeNEPA variant is empty in run name: {run_name!r}')
    return dataset, method_family, variant, projector, seed

  raise ValueError(f'Unexpected method family {family!r} in run name {run_name!r}')


def _s5k_run_name_from_run_name(run_name: str) -> str:
  """Convert a plan-110 20k run name into its 5k-horizon counterpart.

  The 5k runs are named by inserting an `S5K` token after the dataset prefix.
  Example:
    `PTBXL_JEPA_CLS_s0` -> `PTBXL_S5K_JEPA_CLS_s0`
  """
  parts = run_name.split('_')
  if len(parts) < 3:
    raise ValueError(f'Invalid run name (too few components): {run_name!r}')
  dataset = parts[0]
  if dataset not in ('PTBXL', 'AIONO'):
    raise ValueError(f'Unexpected dataset prefix {dataset!r} in run name {run_name!r}')
  if parts[1] == 'S5K':
    return run_name
  return '_'.join([dataset, 'S5K', *parts[1:]])


def _format_step_suffix(step: int) -> str:
  """Format a training step as a compact, filename-friendly suffix.

  Rule (stable, backward compatible with the LeNEPA paper exports):
    - step divisible by 1000 -> `"{step/1000}k"` (e.g. 20000 -> "20k")
    - otherwise -> `str(step)` (e.g. 123 -> "123")
  """
  step = int(step)
  if step < 0:
    raise ValueError(f'step must be >= 0, got: {step}')
  if step == 0:
    return '0'
  if step % 1000 == 0:
    return f'{step // 1000}k'
  return str(step)


def _run_label(run_name: str, *, include_projector_suffix: bool = True) -> str:
  """Build a short, paper-friendly label from a plan-110 run name.

  Args:
    run_name: Plan-110 run name (e.g. `PTBXL_NEPA_STOPGRAD_MEAN_PROJ_s0`).
    include_projector_suffix: If True, include `+Proj/-Proj` in NEPA/LeNEPA labels.
      For paper main results (projector-only), set False to avoid redundant suffixes.
  """
  dataset, method_family, variant, projector, _seed = _parse_run_meta_stub(run_name)

  if method_family == 'JEPA' and variant is not None:
    pooling = 'CLS' if variant == 'CLS' else 'Mean'
    return f'JEPA ({pooling})'

  if method_family == 'NEPA_STOPGRAD':
    # Paper convention: "NEPA" refers to the no-projector variant; the projector run
    # is labeled explicitly as "NEPA +Proj" to keep the narrative focused on the
    # projector ablation rather than the internal StopGrad implementation detail.
    return 'NEPA +Proj' if bool(projector) else 'NEPA'

  if method_family == 'LENEPA':
    if variant is None:
      raise ValueError(f'LeNEPA run is missing variant: {run_name!r}')
    variant_pretty = (
      variant.replace('SIGREGT', 'T')
      .replace('SIGREGB', 'B')
      .replace('SIGREGR', 'R')
      .replace('SIGREGI', 'I')
      .replace('SIGREG_BRI', 'BRI')
      .replace('SIGREG_', '')
      .replace('_', ' ')
    )
    # Paper convention: pred_depth=0 is the default in the final run set (PD2 excluded),
    # so we omit "PD0" from labels to reduce clutter.
    variant_pretty = ' '.join([tok for tok in variant_pretty.split() if tok != 'PD0']).strip()
    if include_projector_suffix:
      suffix = '+Proj' if projector else '-Proj'
      return f'LeNEPA ({variant_pretty}, {suffix})'
    return f'LeNEPA ({variant_pretty})'

  raise ValueError(f'Failed to label run (unexpected parse): {run_name!r}, dataset={dataset}')


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
  """Convert a hex color like `#1f77b4` into RGB floats in [0,1]."""
  if not hex_color.startswith('#'):
    raise ValueError(f'hex_color must start with "#", got: {hex_color!r}')
  if len(hex_color) != 7:
    raise ValueError(f'hex_color must be "#RRGGBB", got: {hex_color!r}')
  r = int(hex_color[1:3], 16) / 255.0
  g = int(hex_color[3:5], 16) / 255.0
  b = int(hex_color[5:7], 16) / 255.0
  return float(r), float(g), float(b)


def _blend_rgb(
  rgb_a: tuple[float, float, float],
  rgb_b: tuple[float, float, float],
  *,
  t: float,
) -> tuple[float, float, float]:
  """Blend two RGB colors (t=0 -> a, t=1 -> b)."""
  if not (0.0 <= float(t) <= 1.0):
    raise ValueError(f't must be in [0,1], got: {t!r}')
  return (
    float((1.0 - float(t)) * rgb_a[0] + float(t) * rgb_b[0]),
    float((1.0 - float(t)) * rgb_a[1] + float(t) * rgb_b[1]),
    float((1.0 - float(t)) * rgb_a[2] + float(t) * rgb_b[2]),
  )


def _paper_family_base_rgb(method_family: str) -> tuple[float, float, float]:
  """Paper palette base colors (high-contrast, paper-friendly).

  Design intent:
    - JEPA: baseline vibe (neutral grayscale)
    - NEPA: warm amber (distinct from both gray and green)
    - LeNEPA: positive vibe (fresh teal/green)
  """
  base_hex = {
    # Okabe-Ito inspired palette (colorblind-friendly).
    'JEPA': '#4D4D4D',  # neutral gray
    'NEPA_STOPGRAD': '#E69F00',  # orange/amber
    'LENEPA': '#009E73',  # bluish green
  }.get(method_family)
  if base_hex is None:
    raise ValueError(f'Unexpected method_family for palette: {method_family!r}')
  return _hex_to_rgb01(base_hex)


def _paper_colors_for_runs(run_names_sorted: list[str]) -> dict[str, tuple[float, float, float, float]]:
  """Assign deterministic family-colored shades to runs for paper plots.

  Color rule (requested):
    - JEPA: shades of blue
    - NEPA: shade(s) of yellow
    - LeNEPA: shades of green

  Shades within a family follow the `run_names_sorted` order.
  """
  if not run_names_sorted:
    raise ValueError('run_names_sorted must be non-empty')

  family_to_names: dict[str, list[str]] = {}
  for name in run_names_sorted:
    _dataset, method_family, _variant, _projector, _seed = _parse_run_meta_stub(name)
    family_to_names.setdefault(method_family, []).append(name)

  white = (1.0, 1.0, 1.0)
  colors: dict[str, tuple[float, float, float, float]] = {}
  for method_family, names in family_to_names.items():
    base_rgb = _paper_family_base_rgb(method_family)
    max_lighten = 0.45
    if method_family == 'NEPA_STOPGRAD':
      max_lighten = 0.25
    if len(names) == 1:
      colors[names[0]] = (*base_rgb, 1.0)
      continue
    for i, run_name in enumerate(names):
      t = (float(i) / float(len(names) - 1)) * float(max_lighten)
      rgb = _blend_rgb(base_rgb, white, t=t)
      colors[run_name] = (*rgb, 1.0)

  return colors


def _run_sort_key(run_name: str) -> tuple:
  """Sort runs into a stable, paper-friendly order."""
  dataset, method_family, variant, projector, _seed = _parse_run_meta_stub(run_name)

  family_rank = {
    'JEPA': 0,
    'NEPA_STOPGRAD': 1,
    'LENEPA': 2,
  }

  if method_family == 'JEPA':
    variant_rank = {'CLS': 0, 'MEAN': 1}.get(variant, 99)
    return (dataset, family_rank[method_family], variant_rank, 0)

  if method_family == 'NEPA_STOPGRAD':
    # Prefer showing the baseline (-Proj) before the (+Proj) variant in plots/tables.
    proj_rank = 0 if not projector else 1
    return (dataset, family_rank[method_family], 0, proj_rank)

  if method_family == 'LENEPA':
    proj_rank = 0 if projector else 1
    variant_rank = 99
    if variant is not None:
      if variant.startswith('SIGREGT20') and 'PD0' in variant:
        variant_rank = 0
      elif variant.startswith('SIGREGT5') and 'PD0' in variant:
        variant_rank = 1
      elif variant.startswith('SIGREGT20') and 'PD2' in variant:
        variant_rank = 2
      elif 'BRI20' in variant:
        variant_rank = 3
    return (dataset, family_rank[method_family], variant_rank, proj_rank)

  return (dataset, 99, 99, 99)


def _a1_jepa_keep_ratio_range(run_name: str) -> tuple[float, float]:
  """Return (min_keep_ratio, max_keep_ratio) for A1 JEPA masking-fragility runs.

  A1 uses JEPA (CLS) runs on PTB-XL with different keep-ratio ranges.

  Supported naming:
    - baseline: `PTBXL_JEPA_CLS_s0` (uses the config default keep-ratio range)
    - ablations: `PTBXL_JEPA_CLS_KR05-10_s0`, `PTBXL_JEPA_CLS_KR30-40_s0` where
      `KRxx-yy` encodes `min_keep_ratio=xx/100` and `max_keep_ratio=yy/100`.
  """
  if run_name == 'PTBXL_JEPA_CLS_s0':
    return float(_A1_JEPA_BASE_KEEP_RATIO[0]), float(_A1_JEPA_BASE_KEEP_RATIO[1])

  match = re.search(r'(?:^|_)KR(\d{2})-(\d{2})(?:_|$)', run_name)
  if match is None:
    raise ValueError(f'Run name is not a supported A1 JEPA masking-fragility run: {run_name!r}')

  lo = int(match.group(1)) / 100.0
  hi = int(match.group(2)) / 100.0
  if not (0.0 < float(lo) <= 1.0 and 0.0 < float(hi) <= 1.0):
    raise ValueError(f'Parsed keep ratio must be in (0,1], got: {(lo, hi)} from run={run_name!r}')
  if float(lo) >= float(hi):
    raise ValueError(f'Expected min_keep_ratio < max_keep_ratio, got: {(lo, hi)} from run={run_name!r}')
  return float(lo), float(hi)


def _a2_nepa_patch_size(run_name: str) -> int:
  """Return patch_size for A2 NEPA Conv tokenizer sensitivity runs.

  A2 varies `patch_size` in NEPA +Proj on PTB-XL. Patch size equals the
  Conv1d tokenizer kernel and stride (ViT-style patch embedding).

  Supported naming:
    - baseline: `PTBXL_NEPA_STOPGRAD_MEAN_PROJ_s0` (uses config default patch_size)
    - ablations: `PTBXL_NEPA_STOPGRAD_MEAN_PROJ_PS10_s0`, `..._PS50_s0` where
      `PSN` encodes `patch_size=N`.
  """
  if run_name == 'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_s0':
    return int(_A2_NEPA_BASE_PATCH_SIZE)

  match = re.search(r'(?:^|_)PS(\d+)(?:_|$)', run_name)
  if match is None:
    raise ValueError(f'Run name is not a supported A2 NEPA patch-size run: {run_name!r}')
  patch_size = int(match.group(1))
  if patch_size <= 0:
    raise ValueError(f'patch_size must be > 0, got: {patch_size} (run={run_name!r})')
  return int(patch_size)


def _plot_ptbxl_single_seed_best_layer_dynamics_classification(
  *,
  histories: dict[str, pd.DataFrame],
  run_names: list[str],
  legend_labels: list[str],
  layers: list[int],
  out_dir: Path,
  out_basename: str,
) -> None:
  """Plot AUROC/AUPRC best-layer dynamics for a single-seed PTB-XL run set.

  This is used for appendix-only PTB-XL ablations (A1/A2) that were run with
  seed=0 only, and therefore do not have seed-aggregated error bars.
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not run_names:
    raise ValueError('run_names must be non-empty')
  if len(run_names) != len(legend_labels):
    raise ValueError(f'run_names and legend_labels length mismatch: {len(run_names)} vs {len(legend_labels)}')
  if not out_basename.strip():
    raise ValueError('out_basename must be non-empty')

  # Step 1: ensure inputs exist in exported histories.
  missing = [r for r in run_names if r not in histories]
  if missing:
    raise KeyError(
      'Missing required histories for single-seed PTB-XL plot. '
      f'missing={missing} available={len(histories)}'
    )

  # Step 2: build deterministic colors and distinct linestyles.
  colors = _paper_colors_for_runs(run_names)
  linestyles = ['-', '--', ':', '-.']
  if len(run_names) > len(linestyles):
    raise ValueError(f'Too many runs for the built-in linestyle set: n={len(run_names)}')

  # Step 3: plot AUROC/AUPRC dynamics (best layer per step; oracle over layers).
  metrics = [m for m in _metric_specs_for_dataset('PTBXL') if m.name in ('auroc', 'auprc')]
  if len(metrics) != 2:
    raise RuntimeError(f'Expected PTB-XL metrics to include AUROC and AUPRC, got: {[m.name for m in metrics]}')

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
  axes_list: list[plt.Axes] = list(axes)

  max_step = 0
  for ax, metric in zip(axes_list, metrics, strict=True):
    for i, (run_name, label) in enumerate(zip(run_names, legend_labels, strict=True)):
      df = histories[run_name]
      best = _best_layer_per_step(df, metric=metric, layers=layers)
      x = best['_step'].to_numpy(dtype=int)
      y = best['best_value'].to_numpy(dtype=float)
      max_step = max(max_step, int(x.max()))
      ax.plot(
        x,
        y,
        color=colors[run_name],
        linestyle=linestyles[i],
        linewidth=2.0,
        label=label,
      )
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, int(max_step))
    if int(max_step) == 20000:
      ax.set_xticks([0, 5000, 10000, 15000, 20000])

  handles, labels = axes_list[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=min(4, len(run_names)),
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{out_basename}.png')
  fig.savefig(out_dir / f'{out_basename}.pdf')
  plt.close(fig)


def _plot_sigreg_component_ablations_dynamics(
  *,
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  out_dir: Path,
  out_basename: str,
) -> None:
  """Plot SIGReg component ablations (seed0 only) across PTB-XL + Aionoscope.

  This targets plan-110 Annex A4: single-component SIGReg placements for LeNEPA.

  The plot uses best-layer values (oracle over `layers`) and includes:
    - T20: time-only SIGReg (baseline; from the main matrix)
    - BRI20: batch+rep+innov SIGReg (stable multi-component reference; from the main matrix)
    - B20/R20/I20/BT20: batch-only / rep-only / innov-only / batch-time-only SIGReg

  Output:
    `{out_basename}.pdf/.png` with a 2x4 panel grid:
      Row 1 (classification): PTB-XL AUROC | PTB-XL AUPRC | Aionoscope AUROC | Aionoscope AUPRC
      Row 2 (dense; Aionoscope): MSE | MAE | Pearson | R^2
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()
  plt.rcParams.update({
      'font.size': 14,          # Base font size
      'axes.titlesize': 16,     # Title of subplots
      'axes.labelsize': 14,     # X and Y labels
      'xtick.labelsize': 12,    # X tick numbers
      'ytick.labelsize': 12,    # Y tick numbers
      'legend.fontsize': 12,    # Legend text
  })

  if not out_basename.strip():
    raise ValueError('out_basename must be non-empty')

  variants: list[tuple[str, str]] = [
    ('T20 (time)', 'LENEPA_SIGREGT20_L0-8_PD0_PROJ_s0'),
    ('BRI20 (batch+rep+innov)', 'LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_s0'),
    ('B20 (batch)', 'LENEPA_SIGREGB20_L0-8_PD0_PROJ_s0'),
    ('R20 (rep)', 'LENEPA_SIGREGR20_L8_PD0_PROJ_s0'),
    ('I20 (innov)', 'LENEPA_SIGREGI20_L0-8_PD0_PROJ_s0'),
    ('BT20 (batch-time)', 'LENEPA_SIGREG_BT20_L0-8_PD0_PROJ_s0'),
  ]

  datasets = [
    ('PTBXL', 'PTB-XL'),
    ('AIONO', 'Aionoscope'),
  ]

  run_name_by_variant_and_dataset: dict[tuple[str, str], str] = {}
  for dataset, _dataset_pretty in datasets:
    for label, suffix in variants:
      run_name_by_variant_and_dataset[(label, dataset)] = f'{dataset}_{suffix}'

  required_run_names = sorted(run_name_by_variant_and_dataset.values())
  missing = [r for r in required_run_names if r not in histories]
  if missing:
    raise KeyError(
      'Missing required histories for SIGReg component ablation plot. '
      f'missing={missing} available={len(histories)}'
    )

  metrics_ptbxl = {m.name: m for m in _metric_specs_for_dataset('PTBXL')}
  metrics_aiono = {m.name: m for m in _metric_specs_for_dataset('AIONO')}
  required_ptbxl = {'auroc', 'auprc'}
  required_aiono = {'auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2'}
  if not required_ptbxl.issubset(metrics_ptbxl):
    raise RuntimeError(
      'Expected PTB-XL metric specs to include required keys. '
      f'missing={sorted(required_ptbxl - set(metrics_ptbxl.keys()))} available={sorted(metrics_ptbxl.keys())}'
    )
  if not required_aiono.issubset(metrics_aiono):
    raise RuntimeError(
      'Expected Aionoscope metric specs to include required keys. '
      f'missing={sorted(required_aiono - set(metrics_aiono.keys()))} available={sorted(metrics_aiono.keys())}'
    )

  # Okabe-Ito colors (colorblind-friendly).
  palette = [
    '#009E73',  # green
    '#0072B2',  # blue
    '#E69F00',  # orange
    '#CC79A7',  # purple
    '#D55E00',  # vermillion
    '#4D4D4D',  # gray
  ]
  if len(palette) < len(variants):
    raise RuntimeError(f'Palette must cover all variants, got {len(palette)} colors for {len(variants)} variants')
  colors_by_label = {variants[i][0]: (*_hex_to_rgb01(palette[i]), 1.0) for i in range(len(variants))}

  linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (5, 1))]
  if len(linestyles) < len(variants):
    raise RuntimeError(f'linestyles must cover all variants, got {len(linestyles)} for {len(variants)} variants')
  linestyle_by_label = {variants[i][0]: linestyles[i] for i in range(len(variants))}

  panels = [
    # Row 1: classification.
    (0, 0, 'PTBXL', 'PTB-XL', 'auroc', metrics_ptbxl['auroc']),
    (0, 1, 'PTBXL', 'PTB-XL', 'auprc', metrics_ptbxl['auprc']),
    (0, 2, 'AIONO', 'Aionoscope', 'auroc', metrics_aiono['auroc']),
    (0, 3, 'AIONO', 'Aionoscope', 'auprc', metrics_aiono['auprc']),
    # Row 2: dense (Aionoscope).
    (1, 0, 'AIONO', 'Aionoscope (dense)', 'mse', metrics_aiono['mse']),
    (1, 1, 'AIONO', 'Aionoscope (dense)', 'mae', metrics_aiono['mae']),
    (1, 2, 'AIONO', 'Aionoscope (dense)', 'pearson', metrics_aiono['pearson']),
    (1, 3, 'AIONO', 'Aionoscope (dense)', 'r2', metrics_aiono['r2']),
  ]

  # Step 1: compute best-layer series for all (variant, dataset, metric) triples.
  series_by_key: dict[tuple[str, str, str], pd.DataFrame] = {}
  max_step = 0
  metric_names_by_dataset = {
    'PTBXL': ['auroc', 'auprc'],
    'AIONO': ['auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2'],
  }
  metrics_by_dataset: dict[str, dict[str, MetricSpec]] = {
    'PTBXL': metrics_ptbxl,
    'AIONO': metrics_aiono,
  }
  for variant_label, _suffix in variants:
    for dataset, _dataset_pretty in datasets:
      run_name = run_name_by_variant_and_dataset[(variant_label, dataset)]
      df = histories[run_name]
      for metric_name in metric_names_by_dataset[dataset]:
        metric = metrics_by_dataset[dataset][metric_name]
        best = _best_layer_per_step(df, metric=metric, layers=layers)
        max_step = max(max_step, int(best['_step'].max()))
        series_by_key[(variant_label, dataset, metric_name)] = best

  # Step 2: plot as a single figure (2x4 panels).
  fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.9), sharex=True)
  axes_grid: list[list[plt.Axes]] = [[axes[r][c] for c in range(4)] for r in range(2)]

  for r, c, dataset, dataset_pretty, metric_name, metric in panels:
    ax = axes_grid[r][c]
    for i, (variant_label, _suffix) in enumerate(variants):
      best = series_by_key[(variant_label, dataset, metric_name)]
      x = best['_step'].to_numpy(dtype=int)
      y = best['best_value'].to_numpy(dtype=float)
      label = variant_label if (r == 0 and c == 0) else '_nolegend_'
      ax.plot(
        x,
        y,
        color=colors_by_label[variant_label],
        linestyle=linestyle_by_label[variant_label],
        linewidth=2.0,
        label=label,
      )
    ax.set_title(dataset_pretty)
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, int(max_step))
    if int(max_step) == 20000:
      ax.set_xticks([0, 5000, 10000, 15000, 20000])

  handles, labels = axes_grid[0][0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=max(1, len(variants)),
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{out_basename}.png')
  fig.savefig(out_dir / f'{out_basename}.pdf')
  plt.close(fig)


def _plot_aiono_single_seed_experiment_dynamics(
  *,
  histories: dict[str, pd.DataFrame],
  run_names: list[str],
  layers: list[int],
  experiment_label: str,
  include_projector_suffix: bool,
  out_dir: Path,
  out_basename: str,
) -> None:
  """Plot Aionoscope categorical+dense best-layer dynamics for a single-seed run set.

  This is used for appendix-only Aionoscope experiments that were run with
  seed=0 only (no seed-aggregated uncertainty). The figure combines:
    - AUROC/AUPRC (left column; stacked)
    - MSE/MAE/Pearson/R^2 (right; 2x2 grid)

  Layout (2x3):
    Row 0: AUROC | MSE | MAE
    Row 1: AUPRC | Pearson | R^2
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not run_names:
    raise ValueError('run_names must be non-empty')
  if not experiment_label.strip():
    raise ValueError('experiment_label must be non-empty')
  if not out_basename.strip():
    raise ValueError('out_basename must be non-empty')

  # Step 1: compute best-layer series for all runs (skip runs with missing offline probes).
  metrics = _metric_specs_for_dataset('AIONO')
  metric_by_name = {m.name: m for m in metrics}
  required_metric_names = ['auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2']
  if set(required_metric_names) != set(metric_by_name.keys()):
    raise RuntimeError(
      'Unexpected Aionoscope metric spec set. '
      f'expected={sorted(required_metric_names)} got={sorted(metric_by_name.keys())}'
    )

  run_names_sorted = sorted(run_names, key=_run_sort_key)
  series_by_run_and_metric: dict[tuple[str, str], pd.DataFrame] = {}
  plotted_run_names: list[str] = []
  skip_reasons: dict[str, str] = {}
  max_step = 0

  for run_name in run_names_sorted:
    df = histories.get(run_name)
    if df is None:
      skip_reasons[run_name] = 'missing exported history (offline probes)'
      continue

    ok = True
    for metric_name in required_metric_names:
      metric = metric_by_name[metric_name]
      try:
        best = _best_layer_per_step(df, metric=metric, layers=layers)
      except Exception as e:  # noqa: BLE001
        skip_reasons[run_name] = f'failed to compute best-layer series: {type(e).__name__}: {e}'
        ok = False
        break
      max_step = max(max_step, int(best['_step'].max()))
      series_by_run_and_metric[(run_name, metric_name)] = best

    if ok:
      plotted_run_names.append(run_name)

  if skip_reasons:
    print(
      f'[warn] Skipping {len(skip_reasons)}/{len(run_names_sorted)} run(s) in {out_basename} '
      f'due to missing/invalid offline probes: {skip_reasons}'
    )

  if not plotted_run_names:
    raise RuntimeError(
      'All runs were skipped (no valid offline-probe histories available). '
      f'out_basename={out_basename!r} run_names={run_names_sorted}'
    )

  # Step 2: deterministic colors + linestyles.
  colors = _paper_colors_for_runs(plotted_run_names)
  linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
  if len(plotted_run_names) > len(linestyles):
    raise ValueError(
      'Too many runs for the built-in linestyle set. '
      f'n={len(plotted_run_names)} max={len(linestyles)} out_basename={out_basename!r}'
    )

  base_labels = [_run_label(r, include_projector_suffix=include_projector_suffix) for r in plotted_run_names]
  legend_labels = [_tick_label_from_base_label(lbl).replace('\n', ' ') for lbl in base_labels]

  # Step 3: plot (2x3 combined categorical+dense layout).
  fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.9), sharex=True)
  panels: list[tuple[plt.Axes, str]] = [
    (axes[0, 0], 'auroc'),
    (axes[1, 0], 'auprc'),
    (axes[0, 1], 'mse'),
    (axes[0, 2], 'mae'),
    (axes[1, 1], 'pearson'),
    (axes[1, 2], 'r2'),
  ]

  for ax, metric_name in panels:
    metric = metric_by_name[metric_name]
    for i, run_name in enumerate(plotted_run_names):
      best = series_by_run_and_metric[(run_name, metric_name)]
      x = best['_step'].to_numpy(dtype=int)
      y = best['best_value'].to_numpy(dtype=float)
      label = legend_labels[i] if ax is axes[0, 0] else '_nolegend_'
      ax.plot(
        x,
        y,
        color=colors[run_name],
        linestyle=linestyles[i],
        linewidth=2.0,
        label=label,
      )
    ax.set_title(f'{metric.pretty} (best layer per step)')
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, int(max_step))
    if int(max_step) == 20000:
      ax.set_xticks([0, 5000, 10000, 15000, 20000])

  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=min(5, len(plotted_run_names)),
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
  )
  fig.text(
    0.5,
    0.99,
    experiment_label,
    ha='center',
    va='top',
    fontsize=10,
    fontweight='bold',
  )
  fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{out_basename}.png')
  fig.savefig(out_dir / f'{out_basename}.pdf')
  plt.close(fig)


def _iter_lenepa_job_logs(logs_dir: Path) -> list[Path]:
  """Return candidate job log files for LeNEPA plan-110 runs."""
  if not logs_dir.is_dir():
    raise FileNotFoundError(f'Logs directory not found: {logs_dir}')
  return sorted([p for p in logs_dir.glob('*.log') if 'lenepa' in p.name.lower()])


def _extract_wandb_run_id_from_log(log_path: Path, *, expected_run_name: str) -> tuple[str | None, bool]:
  """Extract W&B run id from a VM job log and whether it reached step=20000."""
  run_id: str | None = None
  found_name = False
  has_step_20000 = False
  needle = f'run_name: {expected_run_name}'

  with log_path.open('r', encoding='utf-8', errors='replace') as f:
    for line in f:
      if needle in line:
        found_name = True
      if '[step 20000]' in line:
        has_step_20000 = True
      if 'View run' in line and '/runs/' in line:
        match = _WANDB_URL_RE.search(line)
        if match:
          run_id = match.group(1)

  if not found_name:
    return None, False
  return run_id, has_step_20000


def _discover_wandb_run_meta(
  run_name: str,
  *,
  logs_dir: Path,
  wandb_entity: str,
  wandb_project: str,
) -> RunMeta:
  """Locate the VM job log for a run_name and parse its W&B run id."""
  # Step 1: scan candidate job logs and collect matches.
  candidates: list[tuple[Path, str, bool]] = []
  for log_path in _iter_lenepa_job_logs(logs_dir):
    run_id, has_step_20000 = _extract_wandb_run_id_from_log(log_path, expected_run_name=run_name)
    if run_id is None:
      continue
    candidates.append((log_path, run_id, has_step_20000))

  if not candidates:
    raise FileNotFoundError(
      'Failed to locate a LeNEPA VM job log containing a W&B run URL for '
      f'run_name={run_name!r}. Looked in logs_dir={logs_dir}.'
    )

  # Step 2: prefer logs that reached step 20000, then pick the newest by mtime.
  candidates.sort(
    key=lambda t: (
      0 if t[2] else 1,  # has_step_20000 first
      -t[0].stat().st_mtime,  # newest first
    )
  )
  best_log_path, best_run_id, _best_has_step_20000 = candidates[0]

  dataset, method_family, variant, projector, seed = _parse_run_meta_stub(run_name)
  return RunMeta(
    run_name=run_name,
    dataset=dataset,
    method_family=method_family,
    variant=variant,
    projector=projector,
    seed=seed,
    wandb_entity=wandb_entity,
    wandb_project=wandb_project,
    wandb_run_id=best_run_id,
    job_log_path=best_log_path,
  )


def _dedupe_wandb_history(df: pd.DataFrame) -> pd.DataFrame:
  """Collapse W&B scan_history rows into one row per `_step`.

  W&B can emit multiple rows for the same step. We forward-fill within each step
  group and keep the last row to approximate "last non-null per metric per step".
  """
  if '_step' not in df.columns:
    raise ValueError('History is missing required column "_step".')

  # Step 1: stable sort by step (and timestamp/runtime when available).
  sort_cols = ['_step']
  for col in ('_timestamp', '_runtime'):
    if col in df.columns:
      sort_cols.append(col)
  df = df.sort_values(sort_cols, kind='stable').reset_index(drop=True)

  # Step 2: forward-fill inside each step and keep last row.
  df_ffill = df.groupby('_step', sort=True).ffill()
  df_ffill = pd.concat([df[['_step']], df_ffill], axis=1)
  return df_ffill.groupby('_step', sort=True).tail(1).reset_index(drop=True)


def _scan_history_row_at_step_chunked(
  run: wandb.apis.public.Run,
  *,
  step: int,
  keys: list[str],
  page_size: int,
) -> dict[str, object]:
  """Scan W&B history at an exact step, requesting metric keys in adaptive chunks.

  W&B GraphQL requests can time out for wide rows (many metrics). This helper
  mitigates that by retrying transient failures and recursively splitting `keys`
  on `wandb.errors.CommError` until the request succeeds. It preserves the paper
  export behavior:
    - scan history in `[step, step+1]`
    - dedupe rows by `_step` (last non-null per key)

  Returns:
    A merged row dict with at least `_step` plus the requested keys.
  """
  step = int(step)
  if step < 0:
    raise ValueError(f'step must be >= 0, got: {step}')
  if page_size <= 0:
    raise ValueError(f'page_size must be > 0, got: {page_size}')
  if not keys:
    raise ValueError('keys must be non-empty')

  metric_keys = [str(k) for k in keys if str(k) and str(k) != '_step']
  if not metric_keys:
    raise ValueError('keys must contain at least one non-_step key')

  def _scan_keys_once(keys_subset: list[str]) -> dict[str, object]:
    """Return a dict of values for a key subset at an exact step."""
    rows = list(
      run.scan_history(
        keys=['_step', *keys_subset],
        page_size=page_size,
        min_step=step,
        max_step=step + 1,
      )
    )
    if not rows:
      raise RuntimeError(
        'W&B returned empty history for the requested step range. '
        f'run={run.path} step={step} requested_keys={len(keys_subset)}'
      )
    df = pd.DataFrame.from_records(rows)
    df = _dedupe_wandb_history(df)
    df['_step'] = pd.to_numeric(df['_step'], errors='coerce')
    df = df.dropna(subset=['_step']).copy()
    df['_step'] = df['_step'].astype(int)

    row_df = df[df['_step'] == step]
    if row_df.empty:
      raise RuntimeError(f'Missing required step in W&B history: run={run.path} step={step}')
    row = row_df.iloc[0].to_dict()
    return {key: row.get(key) for key in keys_subset}

  def _scan_keys_with_retry(keys_subset: list[str]) -> dict[str, object]:
    """Scan a key subset with bounded retries on CommError."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
      try:
        return _scan_keys_once(keys_subset)
      except wandb.errors.CommError as exc:
        if attempt == max_attempts:
          raise
        sleep_s = 2 ** (attempt - 1)
        print(
          f'[warn] W&B scan_history failed (attempt {attempt}/{max_attempts}): '
          f'run={run.path} step={step} keys={len(keys_subset)} error={exc}. Retrying in {sleep_s}s...',
          flush=True,
        )
        time.sleep(sleep_s)
    raise RuntimeError('Unreachable: retry loop exhausted without returning or raising.')

  def _scan_keys_adaptive(keys_subset: list[str]) -> dict[str, object]:
    """Scan keys, splitting on CommError to reduce request width."""
    try:
      return _scan_keys_with_retry(keys_subset)
    except wandb.errors.CommError:
      if len(keys_subset) == 1:
        raise
      mid = len(keys_subset) // 2
      left = _scan_keys_adaptive(keys_subset[:mid])
      right = _scan_keys_adaptive(keys_subset[mid:])
      left.update(right)
      return left

  merged = {'_step': step}
  merged.update(_scan_keys_adaptive(metric_keys))
  return merged


def _download_offline_probe_history(
  api: wandb.Api,
  *,
  run_meta: RunMeta,
  layers: list[int],
  max_step: int,
  page_size: int,
) -> pd.DataFrame:
  """Download a run history restricted to offline-probe (categorical+dense) keys."""
  # Step 1: construct the minimal key set we need for plotting.
  cat_metrics = [
    MetricSpec(
      name='auroc',
      direction='max',
      key_template='offline_probe/layer_{layer}/best_macro_auc',
      pretty='AUROC',
    ),
    MetricSpec(
      name='auprc',
      direction='max',
      key_template='offline_probe/layer_{layer}/best_macro_auprc',
      pretty='AUPRC',
    ),
  ]
  dense_metrics = [
    MetricSpec(name='mse', direction='min', key_template='offline_dense_probe/layer_{layer}/mse', pretty='MSE'),
    MetricSpec(name='mae', direction='min', key_template='offline_dense_probe/layer_{layer}/mae', pretty='MAE'),
    MetricSpec(
      name='pearson',
      direction='max',
      key_template='offline_dense_probe/layer_{layer}/pearson',
      pretty='Pearson',
    ),
    MetricSpec(name='r2', direction='max', key_template='offline_dense_probe/layer_{layer}/r2', pretty='$R^2$'),
  ]

  keys: list[str] = []
  for metric in cat_metrics:
    keys.extend([metric.key(layer) for layer in layers])

  if run_meta.dataset == 'AIONO':
    for metric in dense_metrics:
      keys.extend([metric.key(layer) for layer in layers])

  # Step 2: query W&B.
  run = api.run(run_meta.wandb_path)
  if run.name != run_meta.run_name:
    raise RuntimeError(
      'W&B run name mismatch (wrong run id parsed from logs). '
      f'expected={run_meta.run_name!r} got={run.name!r} path={run_meta.wandb_path}'
    )

  rows = list(run.scan_history(keys=['_step', *keys], page_size=page_size, min_step=0, max_step=max_step))
  if not rows:
    raise RuntimeError(f'W&B returned empty history: {run_meta.wandb_path}')

  df = pd.DataFrame.from_records(rows)
  df = _dedupe_wandb_history(df)

  # Step 3: keep only eval rows (where at least one offline metric is present).
  df['_step'] = pd.to_numeric(df['_step'], errors='coerce')
  df = df.dropna(subset=['_step']).copy()
  df['_step'] = df['_step'].astype(int)

  metric_cols = [col for col in df.columns if col != '_step']
  df = df.dropna(subset=metric_cols, how='all').reset_index(drop=True)
  if df.empty:
    raise RuntimeError(f'No offline-probe rows found in history: {run_meta.wandb_path}')

  return df


def _download_train_step_time_history(
  api: wandb.Api,
  *,
  run_meta: RunMeta,
  min_step: int,
  max_step: int,
  page_size: int,
) -> pd.DataFrame:
  """Download `train/step_time` history for one run in a fixed step window."""
  if min_step < 0:
    raise ValueError(f'min_step must be >= 0, got: {min_step}')
  if max_step < min_step:
    raise ValueError(f'max_step must be >= min_step, got: min={min_step} max={max_step}')

  run = api.run(run_meta.wandb_path)
  if run.name != run_meta.run_name:
    raise RuntimeError(
      'W&B run name mismatch (wrong run id parsed from logs). '
      f'expected={run_meta.run_name!r} got={run.name!r} path={run_meta.wandb_path}'
    )

  rows = list(
    run.scan_history(
      keys=['_step', 'train/step_time'],
      page_size=page_size,
      min_step=min_step,
      max_step=max_step,
    )
  )
  if not rows:
    raise RuntimeError(f'W&B returned empty train/step_time history: {run_meta.wandb_path}')

  df = pd.DataFrame.from_records(rows)
  df = _dedupe_wandb_history(df)

  df['_step'] = pd.to_numeric(df['_step'], errors='coerce')
  df = df.dropna(subset=['_step']).copy()
  df['_step'] = df['_step'].astype(int)

  if 'train/step_time' not in df.columns:
    raise KeyError(f'Missing required column \"train/step_time\" in history: {run_meta.wandb_path}')
  df['train/step_time'] = pd.to_numeric(df['train/step_time'], errors='coerce')
  df = df.dropna(subset=['train/step_time']).copy()
  if df.empty:
    raise RuntimeError(
      'No non-null train/step_time rows found in the requested window. '
      f'path={run_meta.wandb_path} min_step={min_step} max_step={max_step}'
    )

  df['train/step_time'] = df['train/step_time'].astype(float)
  if not np.all(np.isfinite(df['train/step_time'].to_numpy(dtype=float))):
    raise ValueError(f'Non-finite train/step_time values found: {run_meta.wandb_path}')

  return df[['_step', 'train/step_time']].reset_index(drop=True)


def _metric_specs_for_dataset(dataset: str) -> list[MetricSpec]:
  """Return metric specs in plot order for a dataset."""
  if dataset == 'PTBXL':
    return [
      MetricSpec('auroc', 'max', 'offline_probe/layer_{layer}/best_macro_auc', 'AUROC'),
      MetricSpec('auprc', 'max', 'offline_probe/layer_{layer}/best_macro_auprc', 'AUPRC'),
    ]
  if dataset == 'AIONO':
    return [
      MetricSpec('auroc', 'max', 'offline_probe/layer_{layer}/best_macro_auc', 'AUROC'),
      MetricSpec('auprc', 'max', 'offline_probe/layer_{layer}/best_macro_auprc', 'AUPRC'),
      MetricSpec('mse', 'min', 'offline_dense_probe/layer_{layer}/mse', 'MSE'),
      MetricSpec('mae', 'min', 'offline_dense_probe/layer_{layer}/mae', 'MAE'),
      MetricSpec('pearson', 'max', 'offline_dense_probe/layer_{layer}/pearson', 'Pearson'),
      MetricSpec('r2', 'max', 'offline_dense_probe/layer_{layer}/r2', '$R^2$'),
    ]
  raise ValueError(f'Unexpected dataset: {dataset!r}')


def _best_layer_per_step(
  df_history: pd.DataFrame,
  *,
  metric: MetricSpec,
  layers: list[int],
) -> pd.DataFrame:
  """Compute the best (oracle) layer per step for a given metric.

  For each `_step`, selects the layer that maximizes (or minimizes) the metric.
  Returns a dataframe with columns: `_step`, `best_layer`, `best_value`.
  """
  # Step 1: extract per-layer columns.
  cols = [metric.key(layer) for layer in layers]
  missing = [col for col in cols if col not in df_history.columns]
  if missing:
    raise KeyError(f'Missing required metric columns for {metric.name}: {missing[:3]}...')

  df = df_history[['_step', *cols]].copy()
  df = df.dropna(subset=cols, how='all').reset_index(drop=True)
  if df.empty:
    raise RuntimeError(f'No non-NaN rows for metric={metric.name}')

  values = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
  if metric.direction not in ('max', 'min'):
    raise ValueError(f'Invalid metric direction: {metric.direction!r}')

  # Step 2: row-wise argbest over layers (ignoring NaNs).
  mask_all_nan = np.all(~np.isfinite(values), axis=1)
  if np.any(mask_all_nan):
    df = df.loc[~mask_all_nan].reset_index(drop=True)
    values = values[~mask_all_nan]
  if values.size == 0:
    raise RuntimeError(f'All rows are NaN for metric={metric.name}')

  if metric.direction == 'max':
    best_idx = np.nanargmax(values, axis=1)
  else:
    best_idx = np.nanargmin(values, axis=1)

  best_layers = [int(layers[int(i)]) for i in best_idx]
  best_values = values[np.arange(values.shape[0]), best_idx].astype(float)

  return pd.DataFrame({'_step': df['_step'].astype(int), 'best_layer': best_layers, 'best_value': best_values})


def _best_layer_per_step_seed_stats(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  config_label: str,
  dataset: str,
  metric: MetricSpec,
  layers: list[int],
  seed_pool: tuple[int, ...],
) -> pd.DataFrame:
  """Compute best-layer-per-step seed stats for one (config, dataset, metric).

  For each seed, we compute the best-layer-per-step series (oracle over layers),
  then require that all seeds share the exact same `_step` grid, and aggregate
  the per-step values as `median ± std` across seeds.

  Returns:
    DataFrame with columns: `_step`, `median`, `std`, `n`.
  """
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  steps_ref: list[int] | None = None
  values_by_seed: list[np.ndarray] = []

  by_seed = config_groups[config_label][dataset]
  present_seeds = _seed_pool_present_seeds(
    by_seed,
    seed_pool=seed_pool,
    context=f'best_layer_per_step cfg={config_label!r} dataset={dataset!r} metric={metric.name!r}',
    min_seeds=2,
  )
  for seed in present_seeds:
    meta = config_groups[config_label][dataset].get(int(seed))
    if meta is None:
      raise KeyError(f'Missing required seed={int(seed)} for cfg={config_label!r} dataset={dataset!r}')
    df = histories.get(meta.run_name)
    if df is None:
      raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={config_label!r} dataset={dataset!r}')

    best = _best_layer_per_step(df, metric=metric, layers=layers)
    steps = [int(s) for s in best['_step'].tolist()]
    if steps != sorted(steps):
      raise RuntimeError(f'Best-layer series `_step` is not sorted: cfg={config_label!r} dataset={dataset!r} seed={seed}')

    if steps_ref is None:
      steps_ref = steps
    elif steps != steps_ref:
      missing = sorted(set(steps_ref) - set(steps))
      extra = sorted(set(steps) - set(steps_ref))
      raise KeyError(
        'Seeds have different eval-step grids for best-layer-per-step aggregation. '
        f'cfg={config_label!r} dataset={dataset!r} metric={metric.name!r} seed={int(seed)} '
        f'missing_steps={missing} extra_steps={extra}'
      )

    vals = pd.to_numeric(best['best_value'], errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
      raise ValueError(
        'Non-finite values in best-layer-per-step series. '
        f'cfg={config_label!r} dataset={dataset!r} metric={metric.name!r} seed={int(seed)}'
      )
    values_by_seed.append(vals)

  if steps_ref is None:
    raise RuntimeError('Internal error: steps_ref is None after processing present_seeds.')

  mat = np.stack(values_by_seed, axis=1)  # [n_steps, n_seeds]
  if mat.shape[1] != len(present_seeds):
    raise RuntimeError('Internal error: stacked seed matrix has unexpected shape.')

  med = np.median(mat, axis=1).astype(float)
  std = np.std(mat, axis=1, ddof=1).astype(float)
  if not np.all(np.isfinite(med)):
    raise ValueError(f'Non-finite median values in per-step seed stats: cfg={config_label!r} dataset={dataset!r}')
  if not np.all(np.isfinite(std)):
    raise ValueError(f'Non-finite std values in per-step seed stats: cfg={config_label!r} dataset={dataset!r}')

  return pd.DataFrame(
    {
      '_step': np.asarray(steps_ref, dtype=int),
      'median': med,
      'std': std,
      'n': int(len(present_seeds)),
    }
  )


def _best_layer_at_step(
  df_history: pd.DataFrame,
  *,
  metric: MetricSpec,
  layers: list[int],
  step: int,
) -> BestLayerAtStep:
  """Return the best-layer value for a metric at a specific eval step.

  This implements the paper's "oracle over layers" at a fixed step: for a given
  step $t$, choose the layer that maximizes (or minimizes) the metric at $t$.
  """
  if step < 0:
    raise ValueError(f'step must be >= 0, got: {step}')
  df_best = _best_layer_per_step(df_history, metric=metric, layers=layers)
  row = df_best[df_best['_step'] == int(step)]
  if row.empty:
    available = sorted(set(df_best['_step'].tolist()))
    raise KeyError(
      'Requested eval step is missing from best-layer series. '
      f'metric={metric.name!r} step={step} available_steps={available[:10]}... (n={len(available)})'
    )
  row = row.tail(1)
  best_layer = int(row['best_layer'].iloc[0])
  best_value = float(row['best_value'].iloc[0])
  if not np.isfinite(best_value):
    raise ValueError(f'Non-finite best_value at step: metric={metric.name!r} step={step} value={best_value}')
  return BestLayerAtStep(step=int(step), best_layer=best_layer, best_value=best_value)


def _last_step_by_layer(
  df_history: pd.DataFrame,
  *,
  metric: MetricSpec,
  layers: list[int],
  expected_last_step: int | None = None,
) -> tuple[int, dict[int, float]]:
  """Return metric values by layer at the last eval step.

  Returns:
    (last_step, values_by_layer)
  """
  cols_by_layer = {layer: metric.key(layer) for layer in layers}
  missing = [col for col in cols_by_layer.values() if col not in df_history.columns]
  if missing:
    raise KeyError(f'Missing required metric columns for {metric.name}: {missing[:3]}...')

  last_step = int(df_history['_step'].max())
  if expected_last_step is not None and last_step != expected_last_step:
    raise RuntimeError(
      f'Unexpected last_step for metric={metric.name}: got {last_step}, expected {expected_last_step}.'
    )

  row = df_history[df_history['_step'] == last_step]
  if row.empty:
    raise RuntimeError(f'Failed to locate last_step={last_step} in history.')
  row = row.iloc[0]

  values: dict[int, float] = {}
  for layer, col in cols_by_layer.items():
    value = pd.to_numeric(row[col], errors='coerce')
    if pd.isna(value):
      raise ValueError(
        f'Missing value at last_step={last_step} for metric={metric.name} layer={layer} col={col!r}.'
      )
    values[int(layer)] = float(value)

  return last_step, values


def _best_layer_at_last_step(
  df_history: pd.DataFrame,
  *,
  metric: MetricSpec,
  layers: list[int],
) -> LastStepBestLayer:
  """Return the best layer and value for a metric at the last eval step."""
  # Step 1: extract per-layer values at last step.
  last_step, values = _last_step_by_layer(df_history, metric=metric, layers=layers)
  layer_values = [values[layer] for layer in layers]

  # Step 2: pick the best layer for the metric direction.
  if metric.direction == 'max':
    best_idx = int(np.argmax(layer_values))
  elif metric.direction == 'min':
    best_idx = int(np.argmin(layer_values))
  else:
    raise ValueError(f'Invalid metric direction: {metric.direction!r}')

  return LastStepBestLayer(
    last_step=last_step,
    best_layer=int(layers[best_idx]),
    best_value=float(layer_values[best_idx]),
  )


def _set_matplotlib_paper_style() -> None:
  """Set matplotlib rcParams for compact, paper-ready figures (no LaTeX dependency)."""
  import matplotlib as mpl

  mpl.rcParams.update(
    {
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
      'axes.titlesize': 11,
      'axes.labelsize': 10,
      'xtick.labelsize': 9,
      'ytick.labelsize': 9,
      'legend.fontsize': 8,
      'lines.linewidth': 1.5,
      'grid.alpha': 0.3,
      'figure.dpi': 150,
      'savefig.dpi': 300,
    }
  )


def _latex_escape(text: str) -> str:
  """Escape a string for LaTeX tables (minimal set)."""
  if not isinstance(text, str):
    raise TypeError(f'Expected str, got {type(text).__name__}')
  return (
    text.replace('\\', '\\\\')
    .replace('&', '\\&')
    .replace('%', '\\%')
    .replace('#', '\\#')
    .replace('_', '\\_')
  )


def _format_float(value: float, *, digits: int) -> str:
  """Format a float with fixed precision (fail-fast on NaN/inf)."""
  if not np.isfinite(value):
    raise ValueError(f'Expected finite float, got: {value}')
  return f'{float(value):.{digits}f}'


def _seed_pool_present_and_missing(
  by_seed: dict[int, RunMeta],
  *,
  seed_pool: tuple[int, ...],
) -> tuple[list[int], list[int]]:
  """Return (present, missing) seeds from `seed_pool` for a config group."""
  present = [int(s) for s in seed_pool if int(s) in by_seed]
  missing = [int(s) for s in seed_pool if int(s) not in by_seed]
  return present, missing


def _seed_pool_present_seeds(
  by_seed: dict[int, RunMeta],
  *,
  seed_pool: tuple[int, ...],
  context: str,
  min_seeds: int = 2,
) -> list[int]:
  """Return present seeds from `seed_pool`, requiring at least `min_seeds`."""
  if min_seeds < 1:
    raise ValueError(f'min_seeds must be >= 1, got: {min_seeds}')
  present, missing = _seed_pool_present_and_missing(by_seed, seed_pool=seed_pool)
  if len(present) < int(min_seeds):
    raise ValueError(
      'Not enough seeds available to compute seed-aggregated stats. '
      f'context={context} seed_pool={list(seed_pool)} present={present} missing={missing} min_seeds={int(min_seeds)}'
    )
  return present


def _representative_seed(by_seed: dict[int, RunMeta], *, preferred_seed: int = 0) -> int:
  """Pick a stable representative seed for sorting/labels (prefers seed 0)."""
  if not by_seed:
    raise ValueError('by_seed must be non-empty')
  if int(preferred_seed) in by_seed:
    return int(preferred_seed)
  return int(min(by_seed.keys()))


def _seed_stats(values: list[float], *, context: str) -> SeedStats:
  """Compute `median ± std` across seeds (sample std, ddof=1)."""
  if not values:
    raise ValueError(f'values must be non-empty to compute SeedStats: context={context}')
  arr = np.asarray(values, dtype=float)
  if not np.all(np.isfinite(arr)):
    bad = [float(v) for v in arr.tolist() if not np.isfinite(float(v))]
    raise ValueError(f'Non-finite values found while computing SeedStats: context={context} bad={bad}')
  if int(arr.size) < 2:
    raise ValueError(
      'Need at least 2 values to compute sample std (ddof=1). '
      f'context={context} n={int(arr.size)}'
    )
  median = float(np.median(arr))
  std = float(np.std(arr, ddof=1))
  if not np.isfinite(std):
    raise ValueError(f'Non-finite std computed for SeedStats: context={context} std={std}')
  return SeedStats(median=median, std=std, n=int(arr.size))


def _format_seed_stats_latex(stats: SeedStats, *, digits: int) -> str:
  """Format seed stats as `median(std)` for LaTeX tables (compact).

  We display the standard deviation as an integer uncertainty in the same
  precision as the formatted median:

    std_int = int(std * 10^digits)

  Example (digits=3):
    0.893(3) = 0.893 ± 0.003
  """
  med = _format_float(float(stats.median), digits=int(digits))
  std = float(stats.std)
  if not np.isfinite(std):
    raise ValueError(f'Expected finite std, got: {std}')
  if std < 0:
    raise ValueError(f'Expected non-negative std, got: {std}')
  scale = 10 ** int(digits)
  std_int = int(std * float(scale))
  return f'{med}({std_int})'


def _near_best_mask(values: list[float], *, direction: str, frac: float) -> tuple[int, list[bool]]:
  """Return (best_idx, near_mask) for values.

  "Near" is defined as within (1-frac) relative margin of the best value, using
  a ratio rule when the best value is positive:
  - direction='max': value >= frac * best
  - direction='min': value <= best / frac

  and an absolute-value fallback when the best value is <= 0 (to handle negative
  maxima such as $R^2 < 0$).

  Args:
    values: List of per-layer metric values (finite floats).
    direction: 'max' or 'min'.
    frac: Threshold fraction in (0, 1], e.g. 0.9 for "within 90% of best".
  """
  if not values:
    raise ValueError('values must be non-empty')
  if direction not in ('max', 'min'):
    raise ValueError(f'direction must be \"max\" or \"min\", got {direction!r}')
  if not (0.0 < frac <= 1.0):
    raise ValueError(f'frac must be in (0,1], got {frac}')
  if any(not np.isfinite(v) for v in values):
    raise ValueError('values must be finite to compute near-best mask')

  arr = np.asarray(values, dtype=float)
  if direction == 'max':
    best_idx = int(np.argmax(arr))
    best_value = float(arr[best_idx])
    if best_value > 0.0:
      threshold = float(frac) * best_value
    else:
      threshold = best_value - (1.0 - frac) * abs(best_value)
    mask = [float(v) >= threshold for v in arr.tolist()]
    return best_idx, mask

  best_idx = int(np.argmin(arr))
  best_value = float(arr[best_idx])
  if best_value > 0.0:
    threshold = best_value / float(frac)
  else:
    threshold = best_value + (1.0 - frac) * abs(best_value)
  mask = [float(v) <= threshold for v in arr.tolist()]
  return best_idx, mask


def _method_family_pretty(method_family: str) -> str:
  """Map RunMeta.method_family to a paper-friendly family name."""
  if method_family == 'JEPA':
    return 'JEPA'
  if method_family == 'NEPA_STOPGRAD':
    return 'NEPA'
  if method_family == 'LENEPA':
    return 'LeNEPA'
  raise ValueError(f'Unexpected method_family: {method_family!r}')


def _split_model_and_config_from_label(config_label: str) -> tuple[str, str]:
  """Split a paper config label into (model, config).

  Supported label formats:
    - `JEPA (CLS)` -> (`JEPA`, `CLS`)
    - `LeNEPA (T20 L0-8)` -> (`LeNEPA`, `T20 L0-8`)
    - `NEPA` -> (`NEPA`, ``)
    - `NEPA +Proj` -> (`NEPA`, `+Proj`)
  """
  label = str(config_label).strip()
  if not label:
    raise ValueError('config_label must be non-empty')

  # Step 1: parenthesized labels (most methods).
  if '(' in label and label.endswith(')'):
    model = label.split('(', 1)[0].strip()
    config = label.split('(', 1)[1].removesuffix(')').strip()
    if not model:
      raise ValueError(f'Failed to parse model from config_label={config_label!r}')
    return model, config

  # Step 2: NEPA labels without parentheses.
  for suffix in (' +Proj', ' -Proj'):
    if label.endswith(suffix):
      model = label.removesuffix(suffix).strip()
      if not model:
        raise ValueError(f'Failed to parse model from config_label={config_label!r}')
      return model, suffix.strip()

  # Step 3: no config detail.
  return label, ''


def _config_detail_from_label(config_label: str) -> str:
  """Extract config detail from a run label like `JEPA (CLS)`."""
  _model, config = _split_model_and_config_from_label(config_label)
  return config


def _group_run_metas_by_config_and_seed(
  run_metas: list[RunMeta],
  *,
  required_datasets: tuple[str, ...],
  seed_pool: tuple[int, ...],
  include_projector_suffix: bool = True,
) -> dict[str, dict[str, dict[int, RunMeta]]]:
  """Group RunMeta objects by config label, dataset, and seed.

  Returns:
    config_label -> dataset -> seed -> RunMeta

  This is the core grouping used for seed-aggregated paper plots/tables.
  """
  if not run_metas:
    raise ValueError('run_metas must be non-empty')
  if not required_datasets:
    raise ValueError('required_datasets must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  # Step 1: group by config label, dataset, then seed.
  groups: dict[str, dict[str, dict[int, RunMeta]]] = {}
  for meta in run_metas:
    config_label = _run_label(meta.run_name, include_projector_suffix=include_projector_suffix)
    by_dataset = groups.setdefault(config_label, {})
    by_seed = by_dataset.setdefault(meta.dataset, {})
    if meta.seed in by_seed:
      raise ValueError(
        'Duplicate seed in config group. '
        f'config_label={config_label!r} dataset={meta.dataset!r} seed={meta.seed} '
        f'runs=({by_seed[meta.seed].run_name!r}, {meta.run_name!r})'
      )
    by_seed[int(meta.seed)] = meta

  # Step 2: require full dataset pairing for every config.
  missing_datasets: dict[str, list[str]] = {}
  for config_label, by_dataset in groups.items():
    absent = [ds for ds in required_datasets if ds not in by_dataset]
    if absent:
      missing_datasets[config_label] = absent
  if missing_datasets:
    raise ValueError(
      'Some config labels are missing required datasets. '
      f'required={required_datasets} missing={missing_datasets}'
    )

  # Step 3: require that each (config, dataset) group has at least one seed from the seed pool.
  missing_all_pool_seeds: dict[str, dict[str, list[int]]] = {}
  for config_label, by_dataset in groups.items():
    for dataset in required_datasets:
      by_seed = by_dataset.get(dataset)
      if by_seed is None:
        continue
      present, missing = _seed_pool_present_and_missing(by_seed, seed_pool=seed_pool)
      if not present:
        missing_all_pool_seeds.setdefault(config_label, {})[dataset] = missing
  if missing_all_pool_seeds:
    raise ValueError(
      'Some (config, dataset) groups have no runs from the configured seed pool. '
      f'seed_pool={tuple(int(s) for s in seed_pool)} missing={missing_all_pool_seeds}'
    )

  return groups


def _compute_last_step_best_layer_seed_stats(
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  *,
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  seed_pool: tuple[int, ...],
) -> dict[str, dict[str, dict[str, SeedStats]]]:
  """Compute seed-aggregated best-layer metrics at the last step.

  For each (config, dataset, metric), we compute the best layer at the run's last
  offline-probe eval step separately per seed, then aggregate values as
  `median ± std` across seeds.
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  out: dict[str, dict[str, dict[str, SeedStats]]] = {}
  for config_label, by_dataset in config_groups.items():
    out[config_label] = {}
    for dataset, by_seed in by_dataset.items():
      out[config_label][dataset] = {}
      present_seeds = _seed_pool_present_seeds(
        by_seed,
        seed_pool=seed_pool,
        context=f'last_step_best cfg={config_label!r} dataset={dataset!r}',
        min_seeds=2,
      )
      for metric in _metric_specs_for_dataset(dataset):
        values: list[float] = []
        last_steps: list[int] = []
        for seed in present_seeds:
          meta = by_seed.get(int(seed))
          if meta is None:
            raise KeyError(
              'Missing required seed in config group. '
              f'config_label={config_label!r} dataset={dataset!r} seed={int(seed)}'
            )
          df = histories.get(meta.run_name)
          if df is None:
            raise KeyError(
              f'Missing history for run_name={meta.run_name!r} (config_label={config_label!r} seed={int(seed)})'
            )
          best = _best_layer_at_last_step(df, metric=metric, layers=layers)
          values.append(float(best.best_value))
          last_steps.append(int(best.last_step))
        if len(set(last_steps)) != 1:
          raise RuntimeError(
            'Inconsistent last_step across seeds for last-step best-layer aggregation. '
            f'config_label={config_label!r} dataset={dataset!r} metric={metric.name!r} last_steps={sorted(set(last_steps))}'
          )
        out[config_label][dataset][metric.name] = _seed_stats(
          values, context=f'last_step_best cfg={config_label} dataset={dataset} metric={metric.name}'
        )

  return out


def _compute_best_layer_seed_stats_at_step(
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  *,
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  step: int,
  seed_pool: tuple[int, ...],
) -> dict[str, dict[str, dict[str, SeedStats]]]:
  """Compute seed-aggregated best-layer metrics at a fixed eval step.

  For each (config, dataset, metric), selects the best layer (oracle over layers)
  at the requested offline-probe eval step separately per seed, then aggregates
  values as `median ± std` across seeds.
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')
  step_i = int(step)
  if step_i < 0:
    raise ValueError(f'step must be >= 0, got: {step}')

  out: dict[str, dict[str, dict[str, SeedStats]]] = {}
  for config_label, by_dataset in config_groups.items():
    out[config_label] = {}
    for dataset, by_seed in by_dataset.items():
      out[config_label][dataset] = {}
      present_seeds = _seed_pool_present_seeds(
        by_seed,
        seed_pool=seed_pool,
        context=f'best_at_step cfg={config_label!r} dataset={dataset!r} step={step_i}',
        min_seeds=2,
      )
      for metric in _metric_specs_for_dataset(dataset):
        values: list[float] = []
        for seed in present_seeds:
          meta = by_seed.get(int(seed))
          if meta is None:
            raise KeyError(
              'Missing required seed in config group. '
              f'config_label={config_label!r} dataset={dataset!r} seed={int(seed)}'
            )
          df = histories.get(meta.run_name)
          if df is None:
            raise KeyError(
              f'Missing history for run_name={meta.run_name!r} (config_label={config_label!r} seed={int(seed)})'
            )
          try:
            best = _best_layer_at_step(df, metric=metric, layers=layers, step=step_i)
          except KeyError as e:
            raise KeyError(
              'Missing required eval step in history for seed-aggregated output. '
              f'config_label={config_label!r} dataset={dataset!r} metric={metric.name!r} step={step_i} '
              f'seed={int(seed)} run={meta.run_name!r}'
            ) from e
          values.append(float(best.best_value))
        out[config_label][dataset][metric.name] = _seed_stats(
          values, context=f'best_at_step cfg={config_label} dataset={dataset} metric={metric.name} step={step_i}'
        )

  return out


def _compute_best_layer_delta_last_minus_step_seed_stats(
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  *,
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  step: int,
  seed_pool: tuple[int, ...],
) -> dict[str, dict[str, dict[str, SeedStats]]]:
  """Compute seed-aggregated deltas: (last step) minus (fixed step).

  Delta is computed per seed as:
    delta_s = best_layer(last_step)_s - best_layer(step)_s,
  where best_layer is the paper oracle over layers 0..8 for the given metric.

  We then aggregate deltas across seeds as `median ± std`.
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')
  step_i = int(step)
  if step_i < 0:
    raise ValueError(f'step must be >= 0, got: {step}')

  out: dict[str, dict[str, dict[str, SeedStats]]] = {}
  for config_label, by_dataset in config_groups.items():
    out[config_label] = {}
    for dataset, by_seed in by_dataset.items():
      out[config_label][dataset] = {}
      present_seeds = _seed_pool_present_seeds(
        by_seed,
        seed_pool=seed_pool,
        context=f'delta_last_minus_step cfg={config_label!r} dataset={dataset!r} step={step_i}',
        min_seeds=2,
      )
      for metric in _metric_specs_for_dataset(dataset):
        deltas: list[float] = []
        last_steps: list[int] = []
        for seed in present_seeds:
          meta = by_seed.get(int(seed))
          if meta is None:
            raise KeyError(
              'Missing required seed in config group. '
              f'config_label={config_label!r} dataset={dataset!r} seed={int(seed)}'
            )
          df = histories.get(meta.run_name)
          if df is None:
            raise KeyError(
              f'Missing history for run_name={meta.run_name!r} (config_label={config_label!r} seed={int(seed)})'
            )

          best_last = _best_layer_at_last_step(df, metric=metric, layers=layers)
          try:
            best_step = _best_layer_at_step(df, metric=metric, layers=layers, step=step_i)
          except KeyError as e:
            raise KeyError(
              'Missing required eval step in history for delta aggregation. '
              f'config_label={config_label!r} dataset={dataset!r} metric={metric.name!r} step={step_i} '
              f'seed={int(seed)} run={meta.run_name!r}'
            ) from e

          deltas.append(float(best_last.best_value) - float(best_step.best_value))
          last_steps.append(int(best_last.last_step))

        if len(set(last_steps)) != 1:
          raise RuntimeError(
            'Inconsistent last_step across seeds for delta aggregation. '
            f'config_label={config_label!r} dataset={dataset!r} metric={metric.name!r} last_steps={sorted(set(last_steps))}'
          )

        out[config_label][dataset][metric.name] = _seed_stats(
          deltas,
          context=f'delta_last_minus_step cfg={config_label} dataset={dataset} metric={metric.name} step={step_i}',
        )

  return out


def _select_best_config_per_family(
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  *,
  last_step_best: dict[str, dict[str, dict[str, SeedStats]]],
) -> tuple[dict[str, str], pd.DataFrame]:
  """Select one best config per family via mean-rank aggregation.

  Selection rule (explicit, paper-stable):
  - For each family (JEPA/NEPA/LeNEPA), score each config label by:
    1) Computing per-dataset mean rank over all last-step metrics (AUROC/AUPRC for PTB-XL;
       AUROC/AUPRC/MSE/MAE/Pearson/R2 for Aionoscope).
    2) Averaging the dataset mean ranks (PTB-XL and Aionoscope weighted equally).
  - Lower score is better.
  - Tie-breakers: higher mean AUPRC across datasets, lower Aionoscope MSE, then lexicographic label.
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not last_step_best:
    raise ValueError('last_step_best must be non-empty')

  # Step 1: assign family for each config label.
  family_by_config: dict[str, str] = {}
  for config_label, by_dataset in config_groups.items():
    # Both datasets share method family for a config.
    meta_any = next(iter(next(iter(by_dataset.values())).values()))
    family_by_config[config_label] = _method_family_pretty(meta_any.method_family)

  # Step 2: compute mean ranks within each family.
  records: list[dict[str, object]] = []
  selected: dict[str, str] = {}
  families = sorted(set(family_by_config.values()))

  for family in families:
    configs = sorted([cfg for cfg, fam in family_by_config.items() if fam == family])
    if not configs:
      raise RuntimeError(f'No configs found for family={family!r}')

    # Dataset list is fixed by the plan.
    datasets = ('PTBXL', 'AIONO')
    dataset_scores: dict[str, dict[str, float]] = {ds: {} for ds in datasets}

    # Step 2a: build per-metric ranks for each dataset.
    for dataset in datasets:
      metrics = _metric_specs_for_dataset(dataset)
      for metric in metrics:
        values = pd.Series(
          {cfg: float(last_step_best[cfg][dataset][metric.name].median) for cfg in configs},
          dtype=float,
        )
        ascending = metric.direction == 'min'
        ranks = values.rank(method='min', ascending=ascending)
        for cfg in configs:
          dataset_scores[dataset].setdefault(cfg, 0.0)
          dataset_scores[dataset][cfg] += float(ranks[cfg])

      # Convert sum of ranks to mean rank.
      for cfg in configs:
        dataset_scores[dataset][cfg] /= float(len(metrics))

    # Step 2b: overall score is mean of dataset mean ranks.
    overall: dict[str, float] = {}
    for cfg in configs:
      overall[cfg] = 0.5 * (dataset_scores['PTBXL'][cfg] + dataset_scores['AIONO'][cfg])

    # Step 2c: tie-breakers: mean AUPRC, Aionoscope MSE, label.
    def _tie_key(cfg: str) -> tuple[float, float, float, str]:
      mean_auprc = 0.5 * (
        float(last_step_best[cfg]['PTBXL']['auprc'].median)
        + float(last_step_best[cfg]['AIONO']['auprc'].median)
      )
      aiono_mse = float(last_step_best[cfg]['AIONO']['mse'].median)
      return (overall[cfg], -mean_auprc, aiono_mse, cfg)

    best_cfg = min(configs, key=_tie_key)
    selected[family] = best_cfg

    # Step 2d: emit a score record for every config in this family.
    for cfg in configs:
      records.append(
        {
          'family': family,
          'config_label': cfg,
          'selected': cfg == best_cfg,
          'score': overall[cfg],
          'ptbxl_mean_rank': dataset_scores['PTBXL'][cfg],
          'aiono_mean_rank': dataset_scores['AIONO'][cfg],
          'ptbxl_auprc': float(last_step_best[cfg]['PTBXL']['auprc'].median),
          'aiono_auprc': float(last_step_best[cfg]['AIONO']['auprc'].median),
          'aiono_mse': float(last_step_best[cfg]['AIONO']['mse'].median),
        }
      )

  df_scores = pd.DataFrame.from_records(records).sort_values(
    ['family', 'score', 'config_label'], kind='stable'
  )
  return selected, df_scores.reset_index(drop=True)


def _write_latex_table(
  *,
  out_path: Path,
  caption: str,
  label: str,
  col_spec: str,
  header: list[str],
  rows: list[list[str]],
  resize_to_linewidth: bool = False,
) -> None:
  """Write a standalone LaTeX table environment."""
  if not caption.strip():
    raise ValueError('caption must be non-empty')
  if not label.strip():
    raise ValueError('label must be non-empty')
  if not col_spec.strip():
    raise ValueError('col_spec must be non-empty')
  if not header:
    raise ValueError('header must be non-empty')
  if not rows:
    raise ValueError('rows must be non-empty')

  lines: list[str] = []
  lines.append('\\begin{table}[t]')
  lines.append('\\centering')
  lines.append('\\scriptsize')
  lines.append(f'\\caption{{{caption}}}')
  lines.append(f'\\label{{{label}}}')
  if resize_to_linewidth:
    lines.append('\\resizebox{\\linewidth}{!}{%')
  lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
  lines.append('\\toprule')
  lines.append(' & '.join(header) + ' \\\\')
  lines.append('\\midrule')
  for row in rows:
    if len(row) != len(header):
      raise ValueError(f'Row length mismatch: expected {len(header)}, got {len(row)}')
    lines.append(' & '.join(row) + ' \\\\')
  lines.append('\\bottomrule')
  lines.append('\\end{tabular}')
  if resize_to_linewidth:
    lines.append('}')
  lines.append('\\end{table}')
  lines.append('')

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text('\n'.join(lines), encoding='utf-8')


def _export_steps_5k_vs_20k_best_layer_table(
  *,
  runs_20k: list[RunMeta],
  runs_5k: list[RunMeta],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  steps: tuple[int, int],
  include_projector_suffix: bool,
  digits: int,
  csv_dir: Path,
  tex_dir: Path,
  out_basename: str,
  caption: str,
  label: str,
) -> None:
  """Export best-layer metric values for matched (5k vs 20k) training runs.

  The 5k runs are separate trainings (not early-stopping) whose run names
  include an `S5K` token after the dataset prefix. For each metric and horizon,
  we select the best layer separately (oracle over layers), matching the paper
  appendix convention.
  """
  if len(steps) != 2:
    raise ValueError(f'steps must be a pair (2 items), got: {steps}')
  step_a, step_b = (int(steps[0]), int(steps[1]))
  if step_a < 0 or step_b < 0:
    raise ValueError(f'steps must be >= 0, got: {steps}')
  if step_a == step_b:
    raise ValueError(f'steps must be distinct, got: {steps}')
  if not out_basename.strip():
    raise ValueError('out_basename must be non-empty')
  if not caption.strip():
    raise ValueError('caption must be non-empty')
  if not label.strip():
    raise ValueError('label must be non-empty')

  run_names_20k = [r.run_name for r in runs_20k if r.dataset == dataset]
  run_names_20k_sorted = sorted(run_names_20k, key=_run_sort_key)
  run_names_5k_set = {r.run_name for r in runs_5k if r.dataset == dataset}

  pairs: list[tuple[str, str]] = []
  missing_s5k: list[str] = []
  for run_name_20k in run_names_20k_sorted:
    run_name_5k = _s5k_run_name_from_run_name(run_name_20k)
    if run_name_5k not in run_names_5k_set:
      missing_s5k.append(run_name_5k)
      continue
    pairs.append((run_name_20k, run_name_5k))

  if missing_s5k:
    print(
      f'[filter] Excluding {len(missing_s5k)} run(s) from {out_basename} (missing S5K counterpart): {missing_s5k}'
    )
  if not pairs:
    raise ValueError(
      'No (5k, 20k) run pairs available for table export. '
      f'dataset={dataset} out_basename={out_basename} expected_s5k={missing_s5k}'
    )

  records: list[dict[str, object]] = []
  tex_rows: list[list[str]] = []

  for run_name_20k, run_name_5k in pairs:
    df_20k = histories[run_name_20k]
    df_5k = histories[run_name_5k]

    last_step_20k = int(df_20k['_step'].max())
    last_step_5k = int(df_5k['_step'].max())
    if last_step_20k != int(step_b):
      raise ValueError(
        'Unexpected last eval step for the 20k run. '
        f'run={run_name_20k!r} last_step={last_step_20k} expected={step_b}'
      )
    if last_step_5k != int(step_a):
      raise ValueError(
        'Unexpected last eval step for the 5k run. '
        f'run={run_name_5k!r} last_step={last_step_5k} expected={step_a}'
      )

    label_run = _run_label(run_name_20k, include_projector_suffix=include_projector_suffix)
    record: dict[str, object] = {
      'run_name_20k': run_name_20k,
      'run_name_5k': run_name_5k,
      'label': label_run,
      'dataset': dataset,
      'last_step_5k': int(last_step_5k),
      'last_step_20k': int(last_step_20k),
    }
    row_tex: list[str] = [_latex_escape(label_run)]

    for metric in metrics:
      best_5k = _best_layer_at_last_step(df_5k, metric=metric, layers=layers)
      best_20k = _best_layer_at_last_step(df_20k, metric=metric, layers=layers)

      record[f'{metric.name}@{step_a}'] = float(best_5k.best_value)
      record[f'best_layer_{metric.name}@{step_a}'] = int(best_5k.best_layer)
      record[f'{metric.name}@{step_b}'] = float(best_20k.best_value)
      record[f'best_layer_{metric.name}@{step_b}'] = int(best_20k.best_layer)

      row_tex.append(_format_float(float(best_5k.best_value), digits=int(digits)))
      row_tex.append(_format_float(float(best_20k.best_value), digits=int(digits)))

    records.append(record)
    tex_rows.append(row_tex)

  df_out = pd.DataFrame.from_records(records)
  csv_dir.mkdir(parents=True, exist_ok=True)
  df_out.to_csv(csv_dir / f'{out_basename}.csv', index=False)

  header: list[str] = ['method']
  step_suffix = {step_a: '5k' if step_a == 5000 else str(step_a), step_b: '20k' if step_b == 20000 else str(step_b)}
  for metric in metrics:
    header.append(f'{metric.pretty}@{step_suffix[step_a]}')
    header.append(f'{metric.pretty}@{step_suffix[step_b]}')

  col_spec = 'l' + ('r' * (len(header) - 1))
  tex_dir.mkdir(parents=True, exist_ok=True)
  _write_latex_table(
    out_path=tex_dir / f'{out_basename}.tex',
    caption=caption,
    label=label,
    col_spec=col_spec,
    header=header,
    rows=tex_rows,
  )


def _normalized_delta_from_step0(*, metric: str, step0: float, step1: float) -> float:
  """Compute a step0-normalized delta for cross-target comparability.

  Normalization rules:
    - For error metrics (MSE/MAE): relative error reduction, (step0 - step1) / step0.
      Positive values mean improvement (lower error).
    - For bounded 'higher-is-better' metrics with natural upper bound 1 (AUROC/AUPRC/Pearson/$R^2$): fraction of
      headroom-to-1 captured, (step1 - step0) / (1 - step0). Positive values mean improvement.
  """
  metric = str(metric)
  if metric in ('mse', 'mae'):
    if not np.isfinite(step0) or not np.isfinite(step1):
      raise ValueError(f'Expected finite values for metric={metric!r}, got step0={step0} step1={step1}')
    if step0 <= 0:
      raise ValueError(
        'Step0 normalization for error metrics requires step0 > 0. '
        f'Got metric={metric!r} step0={step0} step1={step1}'
      )
    return float((step0 - step1) / step0)

  if metric in ('auroc', 'auprc', 'pearson', 'r2'):
    if not np.isfinite(step0) or not np.isfinite(step1):
      raise ValueError(f'Expected finite values for metric={metric!r}, got step0={step0} step1={step1}')
    denom = 1.0 - float(step0)
    if denom <= 0:
      raise ValueError(
        'Step0 normalization for bounded metrics requires (1 - step0) > 0. '
        f'Got metric={metric!r} step0={step0} step1={step1}'
      )
    return float((step1 - step0) / denom)

  raise ValueError(f'Unsupported metric for normalization: {metric!r}')


def _export_aiono_dense_targets_step0_vs_20k_table(
  api: wandb.Api,
  *,
  run_meta_step0: RunMeta,
  run_meta_step1: RunMeta,
  layers: list[int],
  steps: tuple[int, int],
  targets_config_path: Path,
  page_size: int,
  digits: int,
  out_csv_path: Path,
) -> None:
  """Export per-target dense probe deltas (oracle over layers) for Aionoscope.

  The table is designed for "blind spot" analysis: it reports best-layer values
  per target at step0 and a later step, plus deltas and step0-normalized deltas.

  For resumed runs where step0 was logged under a different W&B run id, provide
  `run_meta_step0` and `run_meta_step1` accordingly (step numbers remain absolute).
  """
  if len(steps) != 2:
    raise ValueError(f'steps must be a pair (2 items), got: {steps}')
  step0, step1 = (int(steps[0]), int(steps[1]))
  if step0 < 0 or step1 < 0:
    raise ValueError(f'steps must be >= 0, got: {steps}')
  if step0 == step1:
    raise ValueError(f'steps must be distinct, got: {steps}')
  if step0 != 0:
    raise ValueError(f'This export expects step0==0, got: {step0}')
  if digits < 0:
    raise ValueError(f'digits must be >= 0, got: {digits}')
  if page_size <= 0:
    raise ValueError(f'page_size must be > 0, got: {page_size}')

  layers = list(dict.fromkeys(int(layer) for layer in layers))
  if not layers:
    raise ValueError('layers must be non-empty')

  # Step 1: load target specs from the canonical Aionoscope config.
  targets = _load_aiono_basic_components_dense_targets(targets_config_path)
  target_names = [t.name for t in targets]
  if not target_names:
    raise ValueError('No dense targets loaded from config (unexpected).')

  # Step 2: download per-layer per-target metrics at the required steps.
  dense_metrics = ('mse', 'mae', 'pearson', 'r2')
  direction_by_metric = {'mse': 'min', 'mae': 'min', 'pearson': 'max', 'r2': 'max'}

  values_by_step: dict[int, dict[str, np.ndarray]] = {}
  for step in (step0, step1):
    values_by_step[step] = {
      metric: np.full((len(layers), len(target_names)), np.nan, dtype=float) for metric in dense_metrics
    }

  meta_by_step = {step0: run_meta_step0, step1: run_meta_step1}
  run_by_step: dict[int, wandb.apis.public.Run] = {}
  for step in (step0, step1):
    meta = meta_by_step[step]
    run = api.run(meta.wandb_path)
    if run.name != meta.run_name:
      raise RuntimeError(
        'W&B run name mismatch (wrong run id parsed from logs / CLI). '
        f'expected={meta.run_name!r} got={run.name!r} path={meta.wandb_path}'
      )
    _require_aiono_benchmark_comparable_run(
      run,
      context='AIONO dense-target step0-vs-step1 export',
    )
    run_by_step[step] = run

  for layer_index, layer in enumerate(layers):
    keys = ['_step']
    for target in target_names:
      for metric in dense_metrics:
        keys.append(f'offline_dense_probe/layer_{layer}/target_{metric}/{target}')

    for step in (step0, step1):
      run = run_by_step[step]
      row = _scan_history_row_at_step_chunked(
        run,
        step=step,
        keys=keys,
        page_size=page_size,
      )
      for target_index, target in enumerate(target_names):
        for metric in dense_metrics:
          key = f'offline_dense_probe/layer_{layer}/target_{metric}/{target}'
          value = pd.to_numeric(row.get(key), errors='coerce')
          values_by_step[step][metric][layer_index, target_index] = float(value)

  # Step 3: compute best-layer (oracle over layers) values per target/metric/step.
  records: list[dict[str, object]] = []
  for target_index, target_spec in enumerate(targets):
    record: dict[str, object] = {
      'target': target_spec.name,
      'target_signal': target_spec.signal,
      'target_metric': target_spec.metric,
    }
    for metric in dense_metrics:
      direction = direction_by_metric[metric]
      v0_all_layers = values_by_step[step0][metric][:, target_index]
      v1_all_layers = values_by_step[step1][metric][:, target_index]

      if direction == 'min':
        best_layer_idx0 = int(np.nanargmin(v0_all_layers))
        best_layer_idx1 = int(np.nanargmin(v1_all_layers))
      else:
        best_layer_idx0 = int(np.nanargmax(v0_all_layers))
        best_layer_idx1 = int(np.nanargmax(v1_all_layers))

      best_value0 = float(v0_all_layers[best_layer_idx0])
      best_value1 = float(v1_all_layers[best_layer_idx1])
      best_layer0 = int(layers[best_layer_idx0])
      best_layer1 = int(layers[best_layer_idx1])

      delta = float(best_value1 - best_value0)
      delta_norm0 = _normalized_delta_from_step0(metric=metric, step0=best_value0, step1=best_value1)

      record[f'{metric}_step{step0}_best'] = best_value0
      record[f'{metric}_step{step0}_layer'] = best_layer0
      record[f'{metric}_step{step1}_best'] = best_value1
      record[f'{metric}_step{step1}_layer'] = best_layer1
      record[f'{metric}_delta'] = delta
      record[f'{metric}_delta_norm0'] = delta_norm0

    records.append(record)

  df_out = pd.DataFrame.from_records(records)

  # Step 4: enforce column order: target + meta, then metric blocks in a fixed order.
  ordered_cols: list[str] = ['target', 'target_signal', 'target_metric']
  for metric in dense_metrics:
    ordered_cols.extend(
      [
        f'{metric}_step{step0}_best',
        f'{metric}_step{step0}_layer',
        f'{metric}_step{step1}_best',
        f'{metric}_step{step1}_layer',
        f'{metric}_delta',
        f'{metric}_delta_norm0',
      ]
    )
  df_out = df_out[ordered_cols].copy()

  # Step 5: write CSV with fixed float formatting.
  out_csv_path.parent.mkdir(parents=True, exist_ok=True)
  float_format = f'%.{int(digits)}f'
  df_out.to_csv(out_csv_path, index=False, float_format=float_format)
  print(f'[ok] Wrote dense-target table: {out_csv_path} (rows={len(df_out)} cols={len(df_out.columns)})')


def _export_aiono_categorical_signals_step0_vs_20k_table(
  api: wandb.Api,
  *,
  run_meta_step0: RunMeta,
  run_meta_step1: RunMeta,
  layers: list[int],
  steps: tuple[int, int],
  signals_config_path: Path,
  page_size: int,
  digits: int,
  out_csv_path: Path,
) -> None:
  """Export per-signal categorical probe deltas (oracle over layers) for Aionoscope.

  The table is designed for signal-level "blind spot" analysis: it reports best-layer
  per-class AUROC/AUPRC values at step0 and a later step, plus deltas and step0-normalized deltas.
  """
  if len(steps) != 2:
    raise ValueError(f'steps must be a pair (2 items), got: {steps}')
  step0, step1 = (int(steps[0]), int(steps[1]))
  if step0 < 0 or step1 < 0:
    raise ValueError(f'steps must be >= 0, got: {steps}')
  if step0 == step1:
    raise ValueError(f'steps must be distinct, got: {steps}')
  if step0 != 0:
    raise ValueError(f'This export expects step0==0, got: {step0}')
  if digits < 0:
    raise ValueError(f'digits must be >= 0, got: {digits}')
  if page_size <= 0:
    raise ValueError(f'page_size must be > 0, got: {page_size}')

  layers = list(dict.fromkeys(int(layer) for layer in layers))
  if not layers:
    raise ValueError('layers must be non-empty')

  # Step 1: load signal classes from the canonical Aionoscope config.
  signals = _load_aiono_basic_components_signal_classes(signals_config_path)
  if not signals:
    raise ValueError('No signal classes loaded from config (unexpected).')

  # Step 2: download per-layer per-signal metrics at the required steps.
  cat_metrics = ('auroc', 'auprc')
  direction_by_metric = {'auroc': 'max', 'auprc': 'max'}

  values_by_step: dict[int, dict[str, np.ndarray]] = {}
  for step in (step0, step1):
    values_by_step[step] = {
      metric: np.full((len(layers), len(signals)), np.nan, dtype=float) for metric in cat_metrics
    }

  meta_by_step = {step0: run_meta_step0, step1: run_meta_step1}
  run_by_step: dict[int, wandb.apis.public.Run] = {}
  for step in (step0, step1):
    meta = meta_by_step[step]
    run = api.run(meta.wandb_path)
    if run.name != meta.run_name:
      raise RuntimeError(
        'W&B run name mismatch (wrong run id parsed from logs / CLI). '
        f'expected={meta.run_name!r} got={run.name!r} path={meta.wandb_path}'
      )
    _require_aiono_benchmark_comparable_run(
      run,
      context='AIONO categorical-signal step0-vs-step1 export',
    )
    run_by_step[step] = run

  for layer_index, layer in enumerate(layers):
    keys = ['_step']
    for signal in signals:
      keys.append(f'offline_probe/layer_{layer}/val/probe_auc/{signal}')
      keys.append(f'offline_probe_best_auprc/layer_{layer}/val/probe_auprc/{signal}')

    for step in (step0, step1):
      run = run_by_step[step]
      row = _scan_history_row_at_step_chunked(
        run,
        step=step,
        keys=keys,
        page_size=page_size,
      )
      for signal_index, signal in enumerate(signals):
        key_auroc = f'offline_probe/layer_{layer}/val/probe_auc/{signal}'
        key_auprc = f'offline_probe_best_auprc/layer_{layer}/val/probe_auprc/{signal}'
        values_by_step[step]['auroc'][layer_index, signal_index] = float(
          pd.to_numeric(row.get(key_auroc), errors='coerce')
        )
        values_by_step[step]['auprc'][layer_index, signal_index] = float(
          pd.to_numeric(row.get(key_auprc), errors='coerce')
        )

  # Step 3: compute best-layer (oracle over layers) values per signal/metric/step.
  records: list[dict[str, object]] = []
  for signal_index, signal in enumerate(signals):
    record: dict[str, object] = {'signal': signal}
    for metric in cat_metrics:
      direction = direction_by_metric[metric]
      v0_all_layers = values_by_step[step0][metric][:, signal_index]
      v1_all_layers = values_by_step[step1][metric][:, signal_index]
      if np.all(np.isnan(v0_all_layers)):
        raise RuntimeError(
          f'All-NaN categorical metric at step0: run={run_meta_step0.wandb_path} metric={metric} signal={signal!r}'
        )
      if np.all(np.isnan(v1_all_layers)):
        raise RuntimeError(
          f'All-NaN categorical metric at step1: run={run_meta_step1.wandb_path} metric={metric} signal={signal!r}'
        )

      if direction == 'max':
        best_layer_idx0 = int(np.nanargmax(v0_all_layers))
        best_layer_idx1 = int(np.nanargmax(v1_all_layers))
      else:
        raise ValueError(f'Unexpected direction for categorical metric={metric!r}: {direction!r}')

      best_value0 = float(v0_all_layers[best_layer_idx0])
      best_value1 = float(v1_all_layers[best_layer_idx1])
      best_layer0 = int(layers[best_layer_idx0])
      best_layer1 = int(layers[best_layer_idx1])

      delta = float(best_value1 - best_value0)
      delta_norm0 = _normalized_delta_from_step0(metric=metric, step0=best_value0, step1=best_value1)

      record[f'{metric}_step{step0}_best'] = best_value0
      record[f'{metric}_step{step0}_layer'] = best_layer0
      record[f'{metric}_step{step1}_best'] = best_value1
      record[f'{metric}_step{step1}_layer'] = best_layer1
      record[f'{metric}_delta'] = delta
      record[f'{metric}_delta_norm0'] = delta_norm0

    records.append(record)

  df_out = pd.DataFrame.from_records(records)

  # Step 4: enforce column order: signal, then metric blocks in a fixed order.
  ordered_cols: list[str] = ['signal']
  for metric in cat_metrics:
    ordered_cols.extend(
      [
        f'{metric}_step{step0}_best',
        f'{metric}_step{step0}_layer',
        f'{metric}_step{step1}_best',
        f'{metric}_step{step1}_layer',
        f'{metric}_delta',
        f'{metric}_delta_norm0',
      ]
    )
  df_out = df_out[ordered_cols].copy()

  # Step 5: write CSV with fixed float formatting.
  out_csv_path.parent.mkdir(parents=True, exist_ok=True)
  float_format = f'%.{int(digits)}f'
  df_out.to_csv(out_csv_path, index=False, float_format=float_format)
  print(f'[ok] Wrote categorical-per-signal table: {out_csv_path} (rows={len(df_out)} cols={len(df_out.columns)})')


def _export_train_step_time_table(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  step_time_by_run_name: dict[str, TrainStepTimeStats],
  seed_pool: tuple[int, ...],
  out_csv_path: Path,
  out_tex_path: Path,
) -> None:
  """Export `train/step_time` table aggregated across seeds as `median(std)`."""
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not step_time_by_run_name:
    raise ValueError('step_time_by_run_name must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    run_name = config_groups[cfg]['PTBXL'][int(seed)].run_name
    return _run_sort_key(run_name)

  cfgs = sorted(config_groups.keys(), key=_cfg_sort_key)

  records: list[dict[str, object]] = []
  tex_rows: list[list[str]] = []
  for cfg in cfgs:
    ptbxl_values: list[float] = []
    aiono_values: list[float] = []
    min_steps: list[int] = []
    max_steps: list[int] = []

    ptbxl_by_seed = config_groups[cfg]['PTBXL']
    aiono_by_seed = config_groups[cfg]['AIONO']
    paired_seeds = [int(s) for s in seed_pool if int(s) in ptbxl_by_seed and int(s) in aiono_by_seed]
    if len(paired_seeds) < 2:
      raise ValueError(
        'Need at least 2 paired seeds across PTBXL and AIONO for train/step_time aggregation. '
        f'cfg={cfg!r} seed_pool={list(seed_pool)} paired_seeds={paired_seeds}'
      )

    paired_with_step_time: list[int] = []
    missing_step_time_seeds: list[int] = []
    for seed in paired_seeds:
      meta_ptbxl = config_groups[cfg]['PTBXL'].get(int(seed))
      meta_aiono = config_groups[cfg]['AIONO'].get(int(seed))
      if meta_ptbxl is None or meta_aiono is None:
        raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} in config_groups.')

      ptbxl_stats = step_time_by_run_name.get(meta_ptbxl.run_name)
      aiono_stats = step_time_by_run_name.get(meta_aiono.run_name)
      if ptbxl_stats is None or aiono_stats is None:
        missing_step_time_seeds.append(int(seed))
        continue
      paired_with_step_time.append(int(seed))

      ptbxl_values.append(float(ptbxl_stats.median_s))
      aiono_values.append(float(aiono_stats.median_s))
      min_steps.append(int(ptbxl_stats.min_step))
      max_steps.append(int(ptbxl_stats.max_step))

    if missing_step_time_seeds:
      print(
        '[warn] Missing train/step_time exports for some seeds (skipping them in train_step_time_median table). '
        f'cfg={cfg!r} missing_seeds={missing_step_time_seeds}'
      )
    if len(paired_with_step_time) < 2:
      raise ValueError(
        'Need at least 2 paired seeds with train/step_time exports for aggregation. '
        f'cfg={cfg!r} seed_pool={list(seed_pool)} paired_seeds={paired_seeds} paired_with_step_time={paired_with_step_time}'
      )

    if len(set(min_steps)) != 1 or len(set(max_steps)) != 1:
      raise RuntimeError(
        'Inconsistent (min_step, max_step) across seeds for train/step_time aggregation. '
        f'cfg={cfg!r} min_steps={sorted(set(min_steps))} max_steps={sorted(set(max_steps))}'
      )

    ptbxl_seed_stats = _seed_stats(ptbxl_values, context=f'train_step_time cfg={cfg} dataset=PTBXL')
    aiono_seed_stats = _seed_stats(aiono_values, context=f'train_step_time cfg={cfg} dataset=AIONO')

    seed_rep = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    family = _method_family_pretty(config_groups[cfg]['PTBXL'][int(seed_rep)].method_family)
    detail = _config_detail_from_label(cfg)
    records.append(
      {
        'family': family,
        'config_label': cfg,
        'config_detail': detail,
        'n_seeds': int(len(paired_with_step_time)),
        'ptbxl_step_time_median_s': float(ptbxl_seed_stats.median),
        'ptbxl_step_time_std_s': float(ptbxl_seed_stats.std),
        'aiono_step_time_median_s': float(aiono_seed_stats.median),
        'aiono_step_time_std_s': float(aiono_seed_stats.std),
        'min_step': int(min_steps[0]),
        'max_step': int(max_steps[0]),
      }
    )

    tex_rows.append(
      [
        _latex_escape(family),
        _latex_escape(detail),
        _format_seed_stats_latex(ptbxl_seed_stats, digits=3),
        _format_seed_stats_latex(aiono_seed_stats, digits=3),
      ]
    )

  df_out = pd.DataFrame.from_records(records)
  out_csv_path.parent.mkdir(parents=True, exist_ok=True)
  df_out.to_csv(out_csv_path, index=False)

  _write_latex_table(
    out_path=out_tex_path,
    caption=(
      'Median per-step training time (W\\&B: train/step\\_time) over steps 1000..20000, '
      'aggregated across seeds as $\\mathrm{median}(\\mathrm{std})$ (ddof=1; std shown in $10^{-3}$ units). '
      'Logged per training step and excludes offline-probe evaluation time.'
    ),
    label='tab:train_step_time_median',
    col_spec='ll S[table-format=1.3(3)] S[table-format=1.3(3)]',
    header=[
      'model',
      'config',
      '\\multicolumn{1}{c}{PTB-XL (s)}',
      '\\multicolumn{1}{c}{Aionoscope (s)}',
    ],
    rows=tex_rows,
  )


def _export_last_step_by_layer_tables(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  seed_pool: tuple[int, ...],
  near_frac: float,
  digits_by_metric: dict[str, int],
  csv_dir: Path,
  tex_dir: Path,
) -> None:
  """Export last-step metric-by-layer tables aggregated across seeds."""
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg][dataset], preferred_seed=0)
    return _run_sort_key(config_groups[cfg][dataset][int(seed)].run_name)

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)
  dataset_pretty = 'PTB-XL' if dataset == 'PTBXL' else 'Aionoscope'

  for metric in metrics:
    data_rows: list[dict[str, object]] = []
    formatted_rows: list[list[str]] = []

    # Step 1: compute last-step per-layer seed stats for each config.
    last_steps_seen: list[int] = []
    per_cfg_stats: dict[str, dict[int, SeedStats]] = {}
    for cfg in cfg_labels:
      values_by_layer_by_seed: dict[int, list[float]] = {int(layer): [] for layer in layers}
      last_steps: list[int] = []

      present_seeds = _seed_pool_present_seeds(
        config_groups[cfg][dataset],
        seed_pool=seed_pool,
        context=f'last_step_by_layer cfg={cfg!r} dataset={dataset!r} metric={metric.name!r}',
        min_seeds=2,
      )
      for seed in present_seeds:
        meta = config_groups[cfg][dataset].get(int(seed))
        if meta is None:
          raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
        df = histories.get(meta.run_name)
        if df is None:
          raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')

        last_step, values_by_layer = _last_step_by_layer(df, metric=metric, layers=layers)
        last_steps.append(int(last_step))
        for layer in layers:
          values_by_layer_by_seed[int(layer)].append(float(values_by_layer[int(layer)]))

      if len(set(last_steps)) != 1:
        raise RuntimeError(
          'Inconsistent last_step across seeds for last-step-by-layer aggregation. '
          f'cfg={cfg!r} dataset={dataset!r} metric={metric.name!r} last_steps={sorted(set(last_steps))}'
        )
      last_step = int(last_steps[0])
      last_steps_seen.append(last_step)

      stats_by_layer: dict[int, SeedStats] = {}
      for layer in layers:
        stats_by_layer[int(layer)] = _seed_stats(
          values_by_layer_by_seed[int(layer)],
          context=f'last_step_by_layer cfg={cfg} dataset={dataset} metric={metric.name} layer={int(layer)}',
        )
      per_cfg_stats[cfg] = stats_by_layer

    unique_last_steps = sorted(set(last_steps_seen))
    if len(unique_last_steps) != 1:
      raise RuntimeError(
        'Expected a single last_step across all configs for a dataset (paper fixed horizon). '
        f'dataset={dataset!r} metric={metric.name!r} last_steps={unique_last_steps}'
      )

    # Step 2: format CSV + TeX rows (bold/underline based on medians).
    for cfg in cfg_labels:
      stats_by_layer = per_cfg_stats[cfg]
      medians = [float(stats_by_layer[int(layer)].median) for layer in layers]
      best_idx, near_mask = _near_best_mask(medians, direction=metric.direction, frac=near_frac)
      best_layer = int(layers[best_idx])
      digits = int(digits_by_metric.get(metric.name, 3))

      record: dict[str, object] = {
        'config_label': cfg,
        'dataset': dataset,
        'last_step': int(unique_last_steps[0]),
        'best_layer_median': int(best_layer),
      }
      for layer in layers:
        s = stats_by_layer[int(layer)]
        record[f'L{int(layer)}_median'] = float(s.median)
        record[f'L{int(layer)}_std'] = float(s.std)
      data_rows.append(record)

      row_tex: list[str] = [_latex_escape(cfg)]
      for j, layer in enumerate(layers):
        cell = _format_seed_stats_latex(stats_by_layer[int(layer)], digits=digits)
        if int(layer) == int(best_layer):
          cell = f'\\textbf{{{cell}}}'
        elif bool(near_mask[j]):
          cell = f'\\underline{{{cell}}}'
        row_tex.append(cell)
      formatted_rows.append(row_tex)

    df_out = pd.DataFrame.from_records(data_rows)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'{dataset.lower()}_last_step_by_layer_{metric.name}.csv'
    df_out.to_csv(csv_path, index=False)

    header = ['method'] + [f'L{layer}' for layer in layers]
    col_spec = 'l' + ('r' * len(layers))
    step_note = f'step={unique_last_steps[0]}'
    tex_path = tex_dir / f'{dataset.lower()}_last_step_by_layer_{metric.name}.tex'
    _write_latex_table(
      out_path=tex_path,
      caption=(
        f'{dataset_pretty}: {metric.pretty} by layer at the last eval step ({step_note}). '
        'Each cell reports $\\mathrm{median}(\\mathrm{std})$ across seeds (std shown in $10^{-3}$ units); '
        f'best layer by median is bold; layers within {int(near_frac * 100)}\\% of best (by median) are underlined.'
      ),
      label=f'tab:{dataset.lower()}_last_step_by_layer_{metric.name}',
      col_spec=col_spec,
      header=header,
      rows=formatted_rows,
    )


def _export_transfer_best_layer_dynamics_tables(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  steps: list[int],
  seed_pool: tuple[int, ...],
  csv_dir: Path,
  tex_dir: Path,
) -> None:
  """Export transfer training-dynamics tables for AUROC/AUPRC (best layer per step).

  For each (config, dataset, metric), we compute the oracle best layer at each
  offline-probe eval step, separately per seed, then aggregate across seeds.

  LaTeX cell format (compact):
    `abs_med±abs_std / pct_med%`

  Percent-of-improvement is computed per seed as a fraction of
  $(v_{20k} - v_0)$ (direction-aware), then aggregated as the median across
  seeds (std is exported to CSV only).
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')
  if not steps:
    raise ValueError('steps must be non-empty')

  steps_i = [int(s) for s in steps]
  if any(s < 0 for s in steps_i):
    raise ValueError(f'steps must be >= 0, got: {steps_i}')
  if len(set(steps_i)) != len(steps_i):
    raise ValueError(f'steps must not contain duplicates, got: {steps_i}')
  if steps_i != sorted(steps_i):
    raise ValueError(f'steps must be sorted ascending, got: {steps_i}')
  if steps_i[0] != 0:
    raise ValueError(f'steps must start at 0 to compute % improvement from step 0, got: {steps_i}')

  final_step = int(steps_i[-1])
  if final_step != 20000:
    raise ValueError(f'Expected final_step=20000 for transfer dynamics tables, got: {final_step}')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    return _run_sort_key(config_groups[cfg]['PTBXL'][int(seed)].run_name)

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)

  metrics_by_name = {m.name: m for m in _metric_specs_for_dataset('PTBXL') if m.name in ('auroc', 'auprc')}
  if set(metrics_by_name.keys()) != {'auroc', 'auprc'}:
    raise RuntimeError(f'Unexpected metric set for transfer tables: {sorted(metrics_by_name.keys())}')
  metrics = [metrics_by_name['auroc'], metrics_by_name['auprc']]

  datasets: list[tuple[str, str]] = [
    ('PTBXL', 'PTB-XL'),
    ('AIONO', 'Aionoscope: basic-components ($k=2$, imbalanced)'),
  ]

  def _step_label(step: int) -> str:
    if step == 0:
      return '0'
    if step % 1000 != 0:
      raise ValueError(f'Unexpected step (expected multiples of 1000): {step}')
    return f'{int(step // 1000)}k'

  step_labels = [_step_label(s) for s in steps_i]
  dataset_row = [
    '\\multicolumn{2}{c}{}',
    f'\\multicolumn{{{len(steps_i)}}}{{c}}{{{datasets[0][1]}}}',
    f'\\multicolumn{{{len(steps_i)}}}{{c}}{{{datasets[1][1]}}}',
  ]

  header = ['model', 'config', *step_labels, *step_labels]
  col_spec = 'll' + ('r' * (len(header) - 2))
  cmidrule = f'\\cmidrule(lr){{3-{2 + len(steps_i)}}} \\cmidrule(lr){{{3 + len(steps_i)}-{2 + 2 * len(steps_i)}}}'

  def _format_cell(abs_stats: SeedStats, pct_median: float) -> str:
    abs_str = _format_seed_stats_latex(abs_stats, digits=3)
    pct_str = f'{int(round(float(pct_median)))}\\%'
    return f'{abs_str}/{pct_str}'

  for metric in metrics:
    records: list[dict[str, object]] = []
    tex_rows: list[list[str]] = []

    for cfg in cfg_labels:
      model, detail = _split_model_and_config_from_label(cfg)
      row: list[str] = [_latex_escape(model), _latex_escape(detail)]
      record: dict[str, object] = {'family': model, 'config': detail, 'config_label': cfg}

      for dataset, _dataset_pretty in datasets:
        abs_values_by_step: dict[int, list[float]] = {int(s): [] for s in steps_i}
        pct_values_by_step: dict[int, list[float]] = {int(s): [] for s in steps_i}

        present_seeds = _seed_pool_present_seeds(
          config_groups[cfg][dataset],
          seed_pool=seed_pool,
          context=f'transfer_dynamics cfg={cfg!r} dataset={dataset!r} metric={metric.name!r}',
          min_seeds=2,
        )
        for seed in present_seeds:
          meta = config_groups[cfg][dataset].get(int(seed))
          if meta is None:
            raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
          df = histories.get(meta.run_name)
          if df is None:
            raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')

          best = _best_layer_per_step(df, metric=metric, layers=layers)
          best_by_step = {
            int(s): float(v)
            for s, v in zip(best['_step'].tolist(), best['best_value'].tolist(), strict=True)
          }
          missing = [int(s) for s in steps_i if int(s) not in best_by_step]
          if missing:
            available = sorted(best_by_step.keys())
            raise KeyError(
              'Some requested steps are missing from best-layer series for seed-aggregated transfer table. '
              f'cfg={cfg!r} dataset={dataset!r} metric={metric.name!r} seed={int(seed)} '
              f'missing={missing} available_steps={available[:10]}... (n={len(available)})'
            )

          v0 = float(best_by_step[0])
          v_final = float(best_by_step[final_step])
          if not np.isfinite(v0):
            raise ValueError(
              f'Non-finite step0 value: cfg={cfg!r} dataset={dataset!r} metric={metric.name} seed={int(seed)} v0={v0}'
            )
          if not np.isfinite(v_final):
            raise ValueError(
              f'Non-finite final value: cfg={cfg!r} dataset={dataset!r} metric={metric.name} seed={int(seed)} v_final={v_final}'
            )

          if metric.direction == 'max':
            denom = float(v_final) - float(v0)
          elif metric.direction == 'min':
            denom = float(v0) - float(v_final)
          else:
            raise ValueError(f'Unexpected metric direction: {metric.direction!r}')
          if float(denom) == 0.0:
            raise ValueError(
              'No net improvement between step 0 and step 20k (cannot compute % improvement). '
              f'cfg={cfg!r} dataset={dataset!r} metric={metric.name} seed={int(seed)} v0={v0} v_final={v_final}'
            )

          for step in steps_i:
            v = float(best_by_step[int(step)])
            if not np.isfinite(v):
              raise ValueError(
                f'Non-finite value: cfg={cfg!r} dataset={dataset!r} metric={metric.name} seed={int(seed)} '
                f'step={int(step)} v={v}'
              )
            if metric.direction == 'max':
              delta_t = float(v) - float(v0)
            elif metric.direction == 'min':
              delta_t = float(v0) - float(v)
            else:
              raise ValueError(f'Unexpected metric direction: {metric.direction!r}')

            pct = 100.0 * float(delta_t) / float(denom)
            if int(step) == 0:
              pct = 0.0
            if int(step) == final_step:
              pct = 100.0

            abs_values_by_step[int(step)].append(float(v))
            pct_values_by_step[int(step)].append(float(pct))

        for step in steps_i:
          abs_stats = _seed_stats(
            abs_values_by_step[int(step)],
            context=f'transfer_abs cfg={cfg} dataset={dataset} metric={metric.name} step={int(step)}',
          )
          pct_stats = _seed_stats(
            pct_values_by_step[int(step)],
            context=f'transfer_pct cfg={cfg} dataset={dataset} metric={metric.name} step={int(step)}',
          )

          record[f'{dataset.lower()}_{metric.name}_value_median_step_{int(step)}'] = float(abs_stats.median)
          record[f'{dataset.lower()}_{metric.name}_value_std_step_{int(step)}'] = float(abs_stats.std)
          record[f'{dataset.lower()}_{metric.name}_pct_median_step_{int(step)}'] = float(pct_stats.median)
          record[f'{dataset.lower()}_{metric.name}_pct_std_step_{int(step)}'] = float(pct_stats.std)

          row.append(_format_cell(abs_stats, float(pct_stats.median)))

      records.append(record)
      tex_rows.append(row)

    df_out = pd.DataFrame.from_records(records)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'dynamics_best_layer_transfer_{metric.name}.csv'
    df_out.to_csv(csv_path, index=False)

    caption = (
      'Transfer dynamics: best-layer value at each offline-probe eval step (oracle over layers $0$--$8$), '
      'aggregated across seeds. Each cell reports $\\mathrm{median}(\\mathrm{std})$ (ddof=1; std shown in $10^{-3}$ units) and the median '
      'percent-of-improvement from step $0$ to step $20{,}000$ (same row; step $0$ is 0\\%, step $20{,}000$ is 100\\%).'
    )
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / f'dynamics_best_layer_transfer_{metric.name}.tex'

    lines: list[str] = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\scriptsize')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{tab:dynamics_best_layer_transfer_{metric.name}}}')
    lines.append('\\resizebox{\\linewidth}{!}{%')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(dataset_row) + ' \\\\')
    lines.append(cmidrule)
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\midrule')
    for r in tex_rows:
      if len(r) != len(header):
        raise ValueError(f'Row length mismatch: expected {len(header)}, got {len(r)}')
      lines.append(' & '.join(r) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('}')
    lines.append('\\end{table}')
    lines.append('')

    tex_path.write_text('\n'.join(lines), encoding='utf-8')


def _plot_best_layer_dynamics(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  seed_pool: tuple[int, ...],
  plot_classification: bool = True,
  plot_dense: bool = True,
  out_suffix: str | None = None,
  out_dir: Path,
) -> None:
  """Write best-layer-per-step dynamics plots aggregated across seeds."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')
  if not plot_classification and not plot_dense:
    raise ValueError('At least one of plot_classification or plot_dense must be True.')
  suffix = f'_{out_suffix}' if out_suffix else ''

  if dataset not in ('PTBXL', 'AIONO'):
    raise ValueError(f'Unexpected dataset: {dataset!r}')
  if dataset != 'AIONO' and not plot_classification:
    raise ValueError(f'plot_classification must be True for dataset={dataset!r} (no dense probes).')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg][dataset], preferred_seed=0)
    return _run_sort_key(config_groups[cfg][dataset][int(seed)].run_name)

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)
  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg][dataset], preferred_seed=0)
      meta = config_groups[cfg][dataset][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(bucket, key=_cfg_sort_key)
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  # Step 1: classification (AUROC/AUPRC) best-layer-per-step dynamics.
  if plot_classification:
    panel_metrics = [m for m in metrics if m.name in ('auroc', 'auprc')]
    if len(panel_metrics) != 2:
      raise ValueError(f'Expected AUROC and AUPRC in metrics, got: {[m.name for m in panel_metrics]}')

    if dataset == 'PTBXL':
      fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    elif dataset == 'AIONO':
      fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharex=True)
    else:
      raise ValueError(f'Unexpected dataset: {dataset!r}')
    axes_list = list(axes)

    max_step = 0
    for ax, metric in zip(axes_list, panel_metrics, strict=True):
      for i, cfg in enumerate(cfg_labels):
        label = legend_labels[i]
        stats = _best_layer_per_step_seed_stats(
          config_groups=config_groups,
          histories=histories,
          config_label=cfg,
          dataset=dataset,
          metric=metric,
          layers=layers,
          seed_pool=seed_pool,
        )
        x = stats['_step'].to_numpy(dtype=int)
        y_med = stats['median'].to_numpy(dtype=float)
        y_std = stats['std'].to_numpy(dtype=float)
        max_step = max(max_step, int(x.max()))
        ax.plot(x, y_med, color=cfg_colors[cfg], linewidth=2.0, label=label)
        ax.fill_between(x, y_med - y_std, y_med + y_std, color=cfg_colors[cfg], alpha=0.18, linewidth=0.0)
      ax.set_xlabel('step')
      ax.set_ylabel(metric.pretty)
      ax.grid(True)
      ax.set_xlim(0, int(max_step))
      if int(max_step) == 20000:
        ax.set_xticks([0, 5000, 10000, 15000, 20000])

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
      handles,
      labels,
      loc='lower center',
      ncol=4,
      frameon=False,
      bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{dataset.lower()}_dynamics_best_layer_classification{suffix}.png')
    fig.savefig(out_dir / f'{dataset.lower()}_dynamics_best_layer_classification{suffix}.pdf')
    plt.close(fig)

  # Step 2: Aionoscope dense metrics in a separate figure.
  if dataset != 'AIONO' or not plot_dense:
    return

  dense_panel = [m for m in metrics if m.name in ('mse', 'mae', 'pearson', 'r2')]
  if len(dense_panel) != 4:
    raise ValueError(f'Expected 4 dense metrics in metrics, got: {[m.name for m in dense_panel]}')

  fig2, axes2 = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
  axes2_list = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]

  max_step = 0
  for ax, metric in zip(axes2_list, dense_panel, strict=True):
    for i, cfg in enumerate(cfg_labels):
      label = legend_labels[i] if ax is axes2_list[0] else '_nolegend_'
      stats = _best_layer_per_step_seed_stats(
        config_groups=config_groups,
        histories=histories,
        config_label=cfg,
        dataset='AIONO',
        metric=metric,
        layers=layers,
        seed_pool=seed_pool,
      )
      x = stats['_step'].to_numpy(dtype=int)
      y_med = stats['median'].to_numpy(dtype=float)
      y_std = stats['std'].to_numpy(dtype=float)
      max_step = max(max_step, int(x.max()))
      ax.plot(x, y_med, color=cfg_colors[cfg], linewidth=2.0, label=label)
      ax.fill_between(x, y_med - y_std, y_med + y_std, color=cfg_colors[cfg], alpha=0.18, linewidth=0.0)
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, int(max_step))
    if int(max_step) == 20000:
      ax.set_xticks([0, 5000, 10000, 15000, 20000])

  handles, labels = axes2_list[0].get_legend_handles_labels()
  fig2.legend(
    handles,
    labels,
    loc='lower center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
  )
  fig2.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
  out_dir.mkdir(parents=True, exist_ok=True)
  fig2.savefig(out_dir / f'aiono_dynamics_best_layer_dense{suffix}.png')
  fig2.savefig(out_dir / f'aiono_dynamics_best_layer_dense{suffix}.pdf')
  plt.close(fig2)


def _plot_transfer_best_layer_dynamics_classification(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  metrics: list[MetricSpec],
  layers: list[int],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Plot AUROC/AUPRC best-layer-per-step dynamics for PTB-XL vs Aionoscope.

  This is a paper-facing transfer figure: we place all 4 panels in one row
  (PTB-XL/Aionoscope × AUROC/AUPRC) with a shared legend, but independent
  y-axes (no shared y-limits).
  """
  import matplotlib
  from matplotlib.ticker import FuncFormatter

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  metric_by_name = {m.name: m for m in metrics if m.name in ('auroc', 'auprc')}
  if set(metric_by_name.keys()) != {'auroc', 'auprc'}:
    raise ValueError(
      'Expected `metrics` to contain AUROC and AUPRC. '
      f'Got: {sorted(metric_by_name.keys())}'
    )
  panel_metrics = [metric_by_name['auroc'], metric_by_name['auprc']]

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    return _run_sort_key(config_groups[cfg]['PTBXL'][int(seed)].run_name)

  def format_k(x, pos):
      return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)
  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
      meta = config_groups[cfg]['PTBXL'][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(bucket, key=_cfg_sort_key)
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  # Step 1: precompute best-layer series for plotting.
  best_series: dict[tuple[str, str, str], pd.DataFrame] = {}
  max_step = 0
  for cfg in cfg_labels:
    for dataset in ('PTBXL', 'AIONO'):
      for metric in panel_metrics:
        stats = _best_layer_per_step_seed_stats(
          config_groups=config_groups,
          histories=histories,
          config_label=cfg,
          dataset=dataset,
          metric=metric,
          layers=layers,
          seed_pool=seed_pool,
        )
        max_step = max(max_step, int(stats['_step'].max()))
        best_series[(cfg, dataset, metric.name)] = stats

  # Increase base font sizes globally
  plt.rcParams.update({
      'font.size': 14,          # Base font size
      'axes.titlesize': 16,     # Title of subplots
      'axes.labelsize': 14,     # X and Y labels
      'xtick.labelsize': 12,    # X tick numbers
      'ytick.labelsize': 12,    # Y tick numbers
      'legend.fontsize': 12,    # Legend text
  })
  
  # Step 2: plot as a single figure (1x4 panels, independent y-axes).
  fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharex=True)
  axes_list: list[plt.Axes] = list(axes)

  panels = [
    ('PTBXL', 'PTB-XL', 'auroc'),
    ('AIONO', 'Aionoscope', 'auroc'),
    ('PTBXL', 'PTB-XL', 'auprc'),
    ('AIONO', 'Aionoscope', 'auprc'),
  ]
  for idx, (dataset, dataset_pretty, metric_name) in enumerate(panels):
    metric = metric_by_name[metric_name]
    ax = axes_list[idx]
    for i, cfg in enumerate(cfg_labels):
      label = legend_labels[i] if idx == 0 else '_nolegend_'
      best = best_series[(cfg, dataset, metric.name)]
      y_median = best['median'].to_numpy(dtype=float)
      y_std = best['std'].to_numpy(dtype=float)
      x = best['_step'].to_numpy(dtype=int)
      ax.plot(
        x,
        y_median,
        color=cfg_colors[cfg],
        linestyle='-',
        linewidth=1.0,
        label=label,
      )
      ax.fill_between(
        x,
        y_median - y_std,
        y_median + y_std,
        color=cfg_colors[cfg],
        alpha=0.18,
        linewidth=0.0,
      )
    ax.set_title(dataset_pretty)
    ax.set_xlabel('step')
    if dataset == 'PTBXL':
      ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, int(max_step))
    if int(max_step) == 20000:
      ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))

  handles, labels = axes_list[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=max(1, len(cfg_labels)),
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'dynamics_best_layer_classification_transfer.png')
  fig.savefig(out_dir / 'dynamics_best_layer_classification_transfer.pdf')
  plt.close(fig)


def _plot_best_layer_dynamics_5k_vs_20k(
  *,
  runs_20k: list[RunMeta],
  runs_5k: list[RunMeta],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  include_projector_suffix: bool,
  out_suffix: str | None = None,
  out_dir: Path,
) -> None:
  """Write best-layer-per-step dynamics plots overlaying 5k and 20k runs."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402
  from matplotlib.lines import Line2D  # noqa: E402

  _set_matplotlib_paper_style()

  suffix = f'_{out_suffix}' if out_suffix else ''

  # Step 1: pick metric panels for this dataset.
  if dataset == 'PTBXL':
    panel_metrics = [m for m in metrics if m.name in ('auroc', 'auprc')]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    axes_list = list(axes)
  elif dataset == 'AIONO':
    panel_metrics = [m for m in metrics if m.name in ('auroc', 'auprc')]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharex=True)
    axes_list = list(axes)
  else:
    raise ValueError(f'Unexpected dataset: {dataset!r}')

  # Step 2: compute 20k->5k run pairs and stable colors (based on 20k runs only).
  run_names_20k = [r.run_name for r in runs_20k if r.dataset == dataset]
  run_names_20k_sorted = sorted(run_names_20k, key=_run_sort_key)
  colors = _paper_colors_for_runs(run_names_20k_sorted)
  run_names_5k_set = {r.run_name for r in runs_5k if r.dataset == dataset}

  pairs: list[tuple[str, str]] = []
  missing: list[str] = []
  for run_name_20k in run_names_20k_sorted:
    run_name_5k = _s5k_run_name_from_run_name(run_name_20k)
    if run_name_5k not in run_names_5k_set:
      missing.append(run_name_5k)
      continue
    pairs.append((run_name_20k, run_name_5k))

  if missing:
    print(
      f'[filter] Excluding {len(missing)} run(s) from {dataset.lower()} dynamics{suffix} '
      f'(missing S5K counterpart): {missing}'
    )
  if not pairs:
    raise ValueError(f'No (5k, 20k) run pairs to plot for dataset={dataset!r} out_suffix={out_suffix!r}')

  max_step_20k = max(int(histories[name]['_step'].max()) for name, _ in pairs)
  if max_step_20k != 20000:
    raise ValueError(
      'Expected 20k histories to end at step=20000 for 5k-vs-20k plots. '
      f'dataset={dataset!r} max_step_20k={max_step_20k}'
    )

  # Step 3: plot each metric panel.
  for ax, metric in zip(axes_list, panel_metrics, strict=True):
    for run_name_20k, run_name_5k in pairs:
      df_20k = histories[run_name_20k]
      df_5k = histories[run_name_5k]

      best_20k = _best_layer_per_step(df_20k, metric=metric, layers=layers)
      best_5k = _best_layer_per_step(df_5k, metric=metric, layers=layers)

      ax.plot(
        best_20k['_step'],
        best_20k['best_value'],
        color=colors[run_name_20k],
        linestyle='-',
        linewidth=2.0,
        label=_run_label(run_name_20k, include_projector_suffix=include_projector_suffix),
      )
      ax.plot(
        best_5k['_step'],
        best_5k['best_value'],
        color=colors[run_name_20k],
        linestyle='--',
        linewidth=1.8,
        marker='o',
        markersize=3.0,
        label='_nolegend_',
        zorder=3,
      )

    ax.set_title(f'{metric.pretty} (best layer per step)')
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])

  style_handles = [
    Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, label='20k'),
    Line2D(
      [0],
      [0],
      color='black',
      linestyle='--',
      linewidth=1.8,
      marker='o',
      markersize=3.0,
      label='5k',
    ),
  ]
  axes_list[0].legend(handles=style_handles, loc='upper left', frameon=False)

  # Step 4: figure-level legend and save.
  handles, labels = axes_list[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / f'{dataset.lower()}_dynamics_best_layer_classification{suffix}.png')
  fig.savefig(out_dir / f'{dataset.lower()}_dynamics_best_layer_classification{suffix}.pdf')
  plt.close(fig)

  # Step 5: Aionoscope dense metrics in a separate figure.
  if dataset != 'AIONO':
    return

  dense_panel = [m for m in metrics if m.name in ('mse', 'mae', 'pearson', 'r2')]
  fig2, axes2 = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
  axes2_list = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
  for ax, metric in zip(axes2_list, dense_panel, strict=True):
    for run_name_20k, run_name_5k in pairs:
      df_20k = histories[run_name_20k]
      df_5k = histories[run_name_5k]

      best_20k = _best_layer_per_step(df_20k, metric=metric, layers=layers)
      best_5k = _best_layer_per_step(df_5k, metric=metric, layers=layers)

      ax.plot(
        best_20k['_step'],
        best_20k['best_value'],
        color=colors[run_name_20k],
        linestyle='-',
        linewidth=2.0,
        label=_run_label(run_name_20k, include_projector_suffix=include_projector_suffix),
      )
      ax.plot(
        best_5k['_step'],
        best_5k['best_value'],
        color=colors[run_name_20k],
        linestyle='--',
        linewidth=1.8,
        marker='o',
        markersize=3.0,
        label='_nolegend_',
        zorder=3,
      )

    ax.set_title(f'{metric.pretty} (best layer per step)')
    ax.set_xlabel('step')
    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])

  handles, labels = axes2_list[0].get_legend_handles_labels()
  fig2.legend(
    handles,
    labels,
    loc='lower center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
  )
  axes2_list[0].legend(handles=style_handles, loc='upper left', frameon=False)
  fig2.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
  fig2.savefig(out_dir / f'aiono_dynamics_best_layer_dense{suffix}.png')
  fig2.savefig(out_dir / f'aiono_dynamics_best_layer_dense{suffix}.pdf')
  plt.close(fig2)


def _plot_best_layer_dynamics_5k_vs_20k_combined(
  *,
  runs_20k: list[RunMeta],
  runs_5k: list[RunMeta],
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  include_projector_suffix: bool,
  out_dir: Path,
) -> None:
  """Write a single PTB-XL + Aionoscope best-layer-per-step 5k-vs-20k figure.

  Layout (stacked; 4x2):
    - Row 0: PTB-XL (AUROC, AUPRC)
    - Rows 1-3: Aionoscope (AUROC, AUPRC, MSE, MAE, Pearson, R^2)

  The plot overlays the 20k runs (solid) with their 5k-horizon counterparts
  (dashed + markers) using the same color per configuration.
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402
  from matplotlib.lines import Line2D  # noqa: E402

  _set_matplotlib_paper_style()

  if not runs_20k:
    raise ValueError('runs_20k must be non-empty')
  if not runs_5k:
    raise ValueError('runs_5k must be non-empty')

  ptbxl_metrics = _metric_specs_for_dataset('PTBXL')
  aiono_metrics = _metric_specs_for_dataset('AIONO')

  def _pairs_by_label_for_dataset(dataset: str) -> dict[str, tuple[str, str]]:
    # Step 1: find 20k runs and their S5K counterparts for a dataset.
    run_names_20k = [r.run_name for r in runs_20k if r.dataset == dataset]
    run_names_20k_sorted = sorted(run_names_20k, key=_run_sort_key)
    run_names_5k_set = {r.run_name for r in runs_5k if r.dataset == dataset}

    out: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    for run_name_20k in run_names_20k_sorted:
      run_name_5k = _s5k_run_name_from_run_name(run_name_20k)
      if run_name_5k not in run_names_5k_set:
        missing.append(run_name_5k)
        continue
      label = _run_label(run_name_20k, include_projector_suffix=include_projector_suffix)
      if label in out:
        raise ValueError(
          'Duplicate config label while building 5k-vs-20k pairs. '
          f'dataset={dataset!r} label={label!r} runs={[out[label][0], run_name_20k]!r}'
        )
      out[label] = (run_name_20k, run_name_5k)

    if missing:
      print(
        f'[filter] Excluding {len(missing)} run(s) from {dataset.lower()} dynamics_5k_combined '
        f'(missing S5K counterpart): {missing}'
      )
    if not out:
      raise ValueError(f'No (5k, 20k) run pairs to plot for dataset={dataset!r}')

    # Step 2: sanity-check history availability and training horizon.
    max_step_20k = max(int(histories[name]['_step'].max()) for name, _ in out.values())
    if max_step_20k != 20000:
      raise ValueError(
        'Expected 20k histories to end at step=20000 for 5k-vs-20k combined plots. '
        f'dataset={dataset!r} max_step_20k={max_step_20k}'
      )
    max_step_5k = max(int(histories[name]['_step'].max()) for _, name in out.values())
    if max_step_5k != 5000:
      raise ValueError(
        'Expected S5K histories to end at step=5000 for 5k-vs-20k combined plots. '
        f'dataset={dataset!r} max_step_5k={max_step_5k}'
      )
    return out

  ptbxl_pairs = _pairs_by_label_for_dataset('PTBXL')
  aiono_pairs = _pairs_by_label_for_dataset('AIONO')

  # Step 1: require a consistent configuration set across datasets.
  common = sorted(set(ptbxl_pairs.keys()) & set(aiono_pairs.keys()))
  if not common:
    raise ValueError(
      'No common configurations between PTB-XL and Aionoscope for 5k-vs-20k combined plotting. '
      f'ptbxl={sorted(ptbxl_pairs.keys())} aiono={sorted(aiono_pairs.keys())}'
    )
  dropped = sorted((set(ptbxl_pairs.keys()) ^ set(aiono_pairs.keys())))
  if dropped:
    print(
      '[filter] Excluding config(s) from dynamics_5k_combined due to missing dataset counterpart: '
      f'{dropped}'
    )

  cfg_labels = sorted(common, key=lambda lbl: _run_sort_key(ptbxl_pairs[lbl][0]))
  run_names_sorted = [ptbxl_pairs[cfg][0] for cfg in cfg_labels]
  colors_by_run = _paper_colors_for_runs(run_names_sorted)
  cfg_colors = {cfg: colors_by_run[ptbxl_pairs[cfg][0]] for cfg in cfg_labels}
  legend_labels = [_tick_label_from_base_label(cfg).replace('\n', ' ') for cfg in cfg_labels]

  # Step 2: build axes in the same layout as `last_step_by_layer_lines`.
  fig, axes = plt.subplots(4, 2, figsize=(10.5, 10.7), sharex=True)
  axes_list = [
    axes[0, 0],
    axes[0, 1],
    axes[1, 0],
    axes[1, 1],
    axes[2, 0],
    axes[2, 1],
    axes[3, 0],
    axes[3, 1],
  ]
  plot_specs: list[tuple[str, MetricSpec]] = [
    ('PTBXL', ptbxl_metrics[0]),
    ('PTBXL', ptbxl_metrics[1]),
    ('AIONO', aiono_metrics[0]),
    ('AIONO', aiono_metrics[1]),
    ('AIONO', aiono_metrics[2]),
    ('AIONO', aiono_metrics[3]),
    ('AIONO', aiono_metrics[4]),
    ('AIONO', aiono_metrics[5]),
  ]

  # Step 3: plot each metric panel.
  for ax, (dataset, metric) in zip(axes_list, plot_specs, strict=True):
    pairs = ptbxl_pairs if dataset == 'PTBXL' else aiono_pairs
    for i, cfg in enumerate(cfg_labels):
      run_name_20k, run_name_5k = pairs[cfg]
      df_20k = histories.get(run_name_20k)
      df_5k = histories.get(run_name_5k)
      if df_20k is None:
        raise KeyError(f'Missing history for run_name={run_name_20k!r} (cfg={cfg!r})')
      if df_5k is None:
        raise KeyError(f'Missing history for run_name={run_name_5k!r} (cfg={cfg!r})')

      best_20k = _best_layer_per_step(df_20k, metric=metric, layers=layers)
      best_5k = _best_layer_per_step(df_5k, metric=metric, layers=layers)

      label = legend_labels[i] if ax is axes_list[0] else '_nolegend_'
      ax.plot(
        best_20k['_step'],
        best_20k['best_value'],
        color=cfg_colors[cfg],
        linestyle='-',
        linewidth=2.0,
        label=label,
      )
      ax.plot(
        best_5k['_step'],
        best_5k['best_value'],
        color=cfg_colors[cfg],
        linestyle='--',
        linewidth=1.8,
        marker='o',
        markersize=3.0,
        label='_nolegend_',
        zorder=3,
      )

    ax.set_ylabel(metric.pretty)
    ax.grid(True)
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])

  # Step 4: axis-level legend for the horizon style.
  style_handles = [
    Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, label='20k'),
    Line2D(
      [0],
      [0],
      color='black',
      linestyle='--',
      linewidth=1.8,
      marker='o',
      markersize=3.0,
      label='5k',
    ),
  ]
  axes[0, 0].legend(handles=style_handles, loc='upper left', frameon=False)

  for ax in (axes[3, 0], axes[3, 1]):
    ax.set_xlabel('step', labelpad=0.0)
    ax.tick_params(axis='x', pad=0.0)

  # Step 5: figure legend and dataset labels (tight layout with legend-aware bottom margin).
  handles, labels = axes_list[0].get_legend_handles_labels()
  legend = fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=max(1, len(cfg_labels)),
    frameon=False,
    bbox_to_anchor=(0.5, 0.0),
  )

  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
  legend_pad = 0.003
  bottom_rect = float(legend_bbox.y1) + float(legend_pad)
  fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 0.985), pad=0.6, h_pad=2.0)

  ptbxl_top = max(axes[0, 0].get_position().y1, axes[0, 1].get_position().y1)
  ptbxl_bottom = min(axes[0, 0].get_position().y0, axes[0, 1].get_position().y0)
  aiono_top = max(axes[1, 0].get_position().y1, axes[1, 1].get_position().y1)
  aiono_label_y = (ptbxl_bottom + aiono_top) / 2.0
  fig.text(
    0.5,
    min(ptbxl_top + 0.0002, 0.99),
    'PTB-XL',
    ha='center',
    va='bottom',
    fontsize=10,
    fontweight='bold',
  )
  fig.text(
    0.5,
    aiono_label_y,
    'Aionoscope: basic-components (k=2, imbalanced)',
    ha='center',
    va='center',
    fontsize=10,
    fontweight='bold',
  )

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'dynamics_best_layer_5k_vs_20k.png')
  fig.savefig(out_dir / 'dynamics_best_layer_5k_vs_20k.pdf')
  plt.close(fig)


def _plot_last_step_heatmaps(
  *,
  runs: list[RunMeta],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  include_projector_suffix: bool,
  out_dir: Path,
) -> None:
  """Write last-step per-layer heatmaps for a dataset."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  dataset_pretty = 'PTB-XL' if dataset == 'PTBXL' else 'Aionoscope'
  run_names = [r.run_name for r in runs if r.dataset == dataset]
  run_names_sorted = sorted(run_names, key=_run_sort_key)
  last_step_by_run = {name: int(histories[name]['_step'].max()) for name in run_names_sorted}
  unique_last_steps = sorted(set(last_step_by_run.values()))

  y_labels: list[str] = []
  for name in run_names_sorted:
    label = _run_label(name, include_projector_suffix=include_projector_suffix)
    if len(unique_last_steps) > 1:
      label = f'{label} @ {last_step_by_run[name]}'
    y_labels.append(label)

  def _heatmap_for_metric(ax: plt.Axes, metric: MetricSpec) -> None:
    # Step 1: build matrix [runs, layers] at the last step.
    mat = np.zeros((len(run_names_sorted), len(layers)), dtype=float)
    best_layer_idx: list[int] = []
    for i, run_name in enumerate(run_names_sorted):
      df = histories[run_name]
      last_step, values = _last_step_by_layer(df, metric=metric, layers=layers)
      for j, layer in enumerate(layers):
        mat[i, j] = float(values[layer])
      if metric.direction == 'max':
        best_layer_idx.append(int(np.nanargmax(mat[i])))
      else:
        best_layer_idx.append(int(np.nanargmin(mat[i])))
      if last_step != last_step_by_run[run_name]:
        raise RuntimeError(
          'Inconsistent last_step_by_run mapping: '
          f'run={run_name} expected={last_step_by_run[run_name]} got={last_step}'
        )

    # Step 2: plot.
    cmap = 'viridis' if metric.direction == 'max' else 'viridis_r'
    im = ax.imshow(mat, aspect='auto', cmap=cmap)
    ax.set_title(metric.pretty)
    ax.set_xlabel('layer')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Step 3: mark best layer per run.
    ax.scatter(best_layer_idx, list(range(len(best_layer_idx))), s=28, facecolors='none', edgecolors='black')

    # Step 4: colorbar.
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Note: step mismatch checks are validated inside the per-run loop above.

  # Step 1: categorical heatmaps.
  if dataset in ('PTBXL', 'AIONO'):
    cat_metrics = [m for m in metrics if m.name in ('auroc', 'auprc')]
    height = 5.6 if dataset == 'AIONO' else 7.2
    fig, axes = plt.subplots(1, 2, figsize=(10.5, height), sharex=True)
    for ax, metric in zip(list(axes), cat_metrics, strict=True):
      _heatmap_for_metric(ax, metric)
    if len(unique_last_steps) == 1:
      title_step = f'step={unique_last_steps[0]}'
    else:
      title_step = 'step varies (see labels)'
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{dataset.lower()}_last_step_by_layer_classification.png')
    fig.savefig(out_dir / f'{dataset.lower()}_last_step_by_layer_classification.pdf')
    plt.close(fig)

  # Step 2: Aionoscope dense heatmaps.
  if dataset != 'AIONO':
    return

  dense_metrics = [m for m in metrics if m.name in ('mse', 'mae', 'pearson', 'r2')]
  fig2, axes2 = plt.subplots(2, 2, figsize=(10.5, 8.4), sharex=True)
  axes2_list = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
  for ax, metric in zip(axes2_list, dense_metrics, strict=True):
    _heatmap_for_metric(ax, metric)
  if len(unique_last_steps) == 1:
    title_step = f'step={unique_last_steps[0]}'
  else:
    title_step = 'step varies (see labels)'
  fig2.tight_layout()
  out_dir.mkdir(parents=True, exist_ok=True)
  fig2.savefig(out_dir / 'aiono_last_step_by_layer_dense.png')
  fig2.savefig(out_dir / 'aiono_last_step_by_layer_dense.pdf')
  plt.close(fig2)


def _plot_last_step_layer_lineplots(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  dataset: str,
  metrics: list[MetricSpec],
  layers: list[int],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Write last-step metric-by-layer line plots aggregated across seeds."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg][dataset], preferred_seed=0)
    return _run_sort_key(config_groups[cfg][dataset][int(seed)].run_name)

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)
  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg][dataset], preferred_seed=0)
      meta = config_groups[cfg][dataset][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(bucket, key=_cfg_sort_key)
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  def _seed_stats_by_layer(cfg: str, metric: MetricSpec) -> tuple[int, dict[int, SeedStats]]:
    values_by_layer: dict[int, list[float]] = {int(layer): [] for layer in layers}
    last_steps: list[int] = []
    present_seeds = _seed_pool_present_seeds(
      config_groups[cfg][dataset],
      seed_pool=seed_pool,
      context=f'plot_last_step_by_layer cfg={cfg!r} dataset={dataset!r} metric={metric.name!r}',
      min_seeds=2,
    )
    for seed in present_seeds:
      meta = config_groups[cfg][dataset].get(int(seed))
      if meta is None:
        raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
      df = histories.get(meta.run_name)
      if df is None:
        raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')
      last_step, values = _last_step_by_layer(df, metric=metric, layers=layers)
      last_steps.append(int(last_step))
      for layer in layers:
        values_by_layer[int(layer)].append(float(values[int(layer)]))
    if len(set(last_steps)) != 1:
      raise RuntimeError(
        'Inconsistent last_step across seeds for last-step-by-layer plotting. '
        f'cfg={cfg!r} dataset={dataset!r} metric={metric.name!r} last_steps={sorted(set(last_steps))}'
      )
    stats_by_layer: dict[int, SeedStats] = {}
    for layer in layers:
      stats_by_layer[int(layer)] = _seed_stats(
        values_by_layer[int(layer)],
        context=f'plot_last_step_by_layer cfg={cfg} dataset={dataset} metric={metric.name} layer={int(layer)}',
      )
    return int(last_steps[0]), stats_by_layer

  cat_metrics = [m for m in metrics if m.name in ('auroc', 'auprc')]
  dense_metrics = [m for m in metrics if m.name in ('mse', 'mae', 'pearson', 'r2')]

  if dataset != 'AIONO':
    # Step 2: classification line plots.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for ax, metric in zip(list(axes), cat_metrics, strict=True):
      for i, cfg in enumerate(cfg_labels):
        _last_step, stats_by_layer = _seed_stats_by_layer(cfg, metric)
        ys = [float(stats_by_layer[int(layer)].median) for layer in layers]
        yerr = [float(stats_by_layer[int(layer)].std) for layer in layers]
        label = legend_labels[i]
        ax.errorbar(
          layers,
          ys,
          yerr=yerr,
          marker='o',
          markersize=3,
          capsize=2.5,
          color=cfg_colors[cfg],
          linewidth=1.8,
          label=label,
        )
      ax.set_xlabel('layer')
      ax.set_ylabel(metric.pretty)
      ax.set_xticks(layers)
      ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
      handles,
      labels,
      loc='lower center',
      ncol=4,
      frameon=False,
      bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{dataset.lower()}_last_step_by_layer_classification_lines.png')
    fig.savefig(out_dir / f'{dataset.lower()}_last_step_by_layer_classification_lines.pdf')
    plt.close(fig)
    return

  # Step 2: Aionoscope combined classification+dense line plot (single figure).
  all_metrics = [*cat_metrics, *dense_metrics]
  if [m.name for m in all_metrics] != ['auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2']:
    raise ValueError(f'Unexpected metric order for Aionoscope combined by-layer plot: {[m.name for m in all_metrics]}')

  fig2, axes2 = plt.subplots(3, 2, figsize=(10.5, 9.2), sharex=True)
  axes2_list = [
    axes2[0, 0],
    axes2[0, 1],
    axes2[1, 0],
    axes2[1, 1],
    axes2[2, 0],
    axes2[2, 1],
  ]
  for ax, metric in zip(axes2_list, all_metrics, strict=True):
    for i, cfg in enumerate(cfg_labels):
      _last_step, stats_by_layer = _seed_stats_by_layer(cfg, metric)
      ys = [float(stats_by_layer[int(layer)].median) for layer in layers]
      yerr = [float(stats_by_layer[int(layer)].std) for layer in layers]
      label = legend_labels[i] if ax is axes2_list[0] else '_nolegend_'
      ax.errorbar(
        layers,
        ys,
        yerr=yerr,
        marker='o',
        markersize=3,
        capsize=2.5,
        color=cfg_colors[cfg],
        linewidth=1.8,
        label=label,
      )
    ax.set_xlabel('layer')
    ax.set_ylabel(metric.pretty)
    ax.set_xticks(layers)
    ax.grid(True)

  handles, labels = axes2_list[0].get_legend_handles_labels()
  fig2.legend(
    handles,
    labels,
    loc='lower center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
  )
  fig2.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
  out_dir.mkdir(parents=True, exist_ok=True)
  fig2.savefig(out_dir / 'aiono_last_step_by_layer_lines.png')
  fig2.savefig(out_dir / 'aiono_last_step_by_layer_lines.pdf')
  plt.close(fig2)


def _plot_last_step_layer_lineplots_ptbxl_aiono_combined(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Write a single PTB-XL + Aionoscope layer-wise last-step figure.

  Layout (stacked; 4x2):
    - Row 0: PTB-XL (AUROC, AUPRC)
    - Rows 1-3: Aionoscope (AUROC, AUPRC, MSE, MAE, Pearson, R^2)
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  metrics_ptbxl = _metric_specs_for_dataset('PTBXL')
  metrics_aiono = _metric_specs_for_dataset('AIONO')
  ptbxl_metrics = [m for m in metrics_ptbxl if m.name in ('auroc', 'auprc')]
  aiono_metrics = [m for m in metrics_aiono if m.name in ('auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2')]
  if [m.name for m in ptbxl_metrics] != ['auroc', 'auprc']:
    raise ValueError(f'Unexpected PTB-XL metric set for combined by-layer plot: {[m.name for m in ptbxl_metrics]}')
  if [m.name for m in aiono_metrics] != ['auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2']:
    raise ValueError(f'Unexpected Aionoscope metric set for combined by-layer plot: {[m.name for m in aiono_metrics]}')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    return _run_sort_key(config_groups[cfg]['PTBXL'][int(seed)].run_name)

  cfg_labels = sorted(config_groups.keys(), key=_cfg_sort_key)
  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
      meta = config_groups[cfg]['PTBXL'][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(bucket, key=_cfg_sort_key)
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  def _seed_stats_by_layer(cfg: str, dataset: str, metric: MetricSpec) -> dict[int, SeedStats]:
    values_by_layer: dict[int, list[float]] = {int(layer): [] for layer in layers}
    last_steps: list[int] = []
    present_seeds = _seed_pool_present_seeds(
      config_groups[cfg][dataset],
      seed_pool=seed_pool,
      context=f'plot_last_step_by_layer_combined cfg={cfg!r} dataset={dataset!r} metric={metric.name!r}',
      min_seeds=2,
    )
    for seed in present_seeds:
      meta = config_groups[cfg][dataset].get(int(seed))
      if meta is None:
        raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
      df = histories.get(meta.run_name)
      if df is None:
        raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')
      last_step, values = _last_step_by_layer(df, metric=metric, layers=layers)
      last_steps.append(int(last_step))
      for layer in layers:
        values_by_layer[int(layer)].append(float(values[int(layer)]))
    if len(set(last_steps)) != 1:
      raise RuntimeError(
        'Inconsistent last_step across seeds for last-step-by-layer plotting. '
        f'cfg={cfg!r} dataset={dataset!r} metric={metric.name!r} last_steps={sorted(set(last_steps))}'
      )
    stats_by_layer: dict[int, SeedStats] = {}
    for layer in layers:
      stats_by_layer[int(layer)] = _seed_stats(
        values_by_layer[int(layer)],
        context=f'plot_last_step_by_layer cfg={cfg} dataset={dataset} metric={metric.name} layer={int(layer)}',
      )
    return stats_by_layer

  fig, axes = plt.subplots(4, 2, figsize=(10.5, 10.7), sharex=True)
  axes_list = [
    axes[0, 0],
    axes[0, 1],
    axes[1, 0],
    axes[1, 1],
    axes[2, 0],
    axes[2, 1],
    axes[3, 0],
    axes[3, 1],
  ]
  plot_specs: list[tuple[str, MetricSpec]] = [
    ('PTBXL', ptbxl_metrics[0]),
    ('PTBXL', ptbxl_metrics[1]),
    ('AIONO', aiono_metrics[0]),
    ('AIONO', aiono_metrics[1]),
    ('AIONO', aiono_metrics[2]),
    ('AIONO', aiono_metrics[3]),
    ('AIONO', aiono_metrics[4]),
    ('AIONO', aiono_metrics[5]),
  ]
  for ax, (dataset, metric) in zip(axes_list, plot_specs, strict=True):
    for i, cfg in enumerate(cfg_labels):
      stats_by_layer = _seed_stats_by_layer(cfg, dataset, metric)
      ys = [float(stats_by_layer[int(layer)].median) for layer in layers]
      yerr = [float(stats_by_layer[int(layer)].std) for layer in layers]
      label = legend_labels[i] if ax is axes_list[0] else '_nolegend_'
      ax.errorbar(
        layers,
        ys,
        yerr=yerr,
        marker='o',
        markersize=3,
        capsize=2.5,
        color=cfg_colors[cfg],
        linewidth=1.8,
        label=label,
      )
    ax.set_ylabel(metric.pretty)
    ax.set_xticks(layers)
    ax.grid(True)

  axes[3, 0].set_xlabel('layer')
  axes[3, 1].set_xlabel('layer')

  handles, labels = axes_list[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
  )
  fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.95), h_pad=2.4)

  ptbxl_top = max(axes[0, 0].get_position().y1, axes[0, 1].get_position().y1)
  ptbxl_bottom = min(axes[0, 0].get_position().y0, axes[0, 1].get_position().y0)
  aiono_top = max(axes[1, 0].get_position().y1, axes[1, 1].get_position().y1)
  aiono_label_y = (ptbxl_bottom + aiono_top) / 2.0
  fig.text(
    0.5,
    min(ptbxl_top + 0.006, 0.99),
    'PTB-XL',
    ha='center',
    va='bottom',
    fontsize=10,
    fontweight='bold',
  )
  fig.text(
    0.5,
    aiono_label_y,
    'Aionoscope: basic-components (k=2, imbalanced)',
    ha='center',
    va='center',
    fontsize=10,
    fontweight='bold',
  )

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'last_step_by_layer_lines.png')
  fig.savefig(out_dir / 'last_step_by_layer_lines.pdf')
  plt.close(fig)


def _summary_config_labels(config_groups: dict[str, dict[str, dict[int, RunMeta]]]) -> list[str]:
  """Return the config labels used in `summary_last_step_bars` and its table.

  Paper-facing summary includes:
    - JEPA (CLS)
    - JEPA (Mean)
    - NEPA
    - NEPA +Proj
    - LeNEPA (T20 L0-8)

  and excludes LeNEPA BRI20 variants to keep the plot/table compact.
  """
  if not config_groups:
    raise ValueError('config_groups must be non-empty')

  def _cfg_sort_key(cfg: str) -> tuple:
    seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
    return _run_sort_key(config_groups[cfg]['PTBXL'][int(seed)].run_name)

  cfg_labels_all = sorted(config_groups.keys(), key=_cfg_sort_key)
  cfg_labels = [cfg for cfg in cfg_labels_all if 'BRI20' not in cfg]
  if len(cfg_labels) != len(cfg_labels_all):
    removed = sorted(set(cfg_labels_all) - set(cfg_labels))
    print(f'[filter] Excluding {len(removed)} config(s) from summary_last_step: {removed}')

  expected = {
    'JEPA (CLS)',
    'JEPA (Mean)',
    'NEPA',
    'NEPA +Proj',
    'LeNEPA (T20 L0-8)',
  }
  missing = sorted(expected - set(cfg_labels))
  extra = sorted(set(cfg_labels) - expected)
  if missing or extra:
    raise ValueError(
      'Unexpected config set for summary_last_step (after filtering). '
      f'missing={missing} extra={extra} got={cfg_labels}'
    )

  return cfg_labels


def _plot_summary_last_step_bar_charts(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  last_step_best: dict[str, dict[str, dict[str, SeedStats]]],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Plot last-step best-layer metrics (all paper configs) as bar charts."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  cfg_labels = _summary_config_labels(config_groups)

  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
      meta = config_groups[cfg]['PTBXL'][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(
        bucket,
        key=lambda cfg: _run_sort_key(
          config_groups[cfg]['PTBXL'][int(_representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0))].run_name
        ),
      )
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  def _plot_on_axis(
    *,
    ax: plt.Axes,
    dataset: str,
    metrics: list[MetricSpec],
    y_text_offset: float,
  ) -> None:
    if not metrics:
      raise ValueError('metrics must be non-empty')
    xs = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = (np.arange(len(cfg_labels), dtype=float) - (len(cfg_labels) - 1) / 2.0) * width

    for i, cfg in enumerate(cfg_labels):
      vals = [float(last_step_best[cfg][dataset][m.name].median) for m in metrics]
      errs = [float(last_step_best[cfg][dataset][m.name].std) for m in metrics]
      bars = ax.bar(
        xs + float(offsets[i]),
        vals,
        width=width,
        color=cfg_colors[cfg],
        yerr=errs,
        capsize=2.0,
        error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        edgecolor='black',
        linewidth=0.35,
        label=legend_labels[i],
      )
      for rect, v in zip(list(bars), vals, strict=True):
        if not np.isfinite(float(v)):
          raise ValueError(f'Non-finite value in summary chart: dataset={dataset} cfg={cfg!r} v={v}')
        x = float(rect.get_x() + rect.get_width() / 2.0)
        y_tip = float(rect.get_y() + rect.get_height())
        if float(v) >= 0.0:
          y_text = y_tip + float(y_text_offset)
          va = 'bottom'
        else:
          y_text = y_tip - float(y_text_offset)
          va = 'top'
        ax.text(
          x,
          y_text,
          _format_float(float(v), digits=3),
          ha='center',
          va=va,
          fontsize=6,
          rotation=0,
          clip_on=True,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([m.pretty for m in metrics])
    ax.grid(True, axis='y')
    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.25)

  # Step 2: build metrics lists.
  ptbxl_metrics = [
    MetricSpec('auroc', 'max', '', 'AUROC $\\uparrow$'),
    MetricSpec('auprc', 'max', '', 'AUPRC $\\uparrow$'),
  ]
  aiono_metrics = [
    MetricSpec('auroc', 'max', '', 'AUROC $\\uparrow$'),
    MetricSpec('auprc', 'max', '', 'AUPRC $\\uparrow$'),
    MetricSpec('mse', 'min', '', 'MSE $\\downarrow$'),
    MetricSpec('mae', 'min', '', 'MAE $\\downarrow$'),
    MetricSpec('pearson', 'max', '', 'Pearson $\\uparrow$'),
    MetricSpec('r2', 'max', '', '$R^2$ $\\uparrow$'),
  ]

  # Step 3: compute per-panel y-limits (independent scale per subplot).
  def _y_scale_for_panel(
    *,
    dataset: str,
    metrics: list[MetricSpec],
    ymin_at_zero: bool,
  ) -> tuple[tuple[float, float], float]:
    values: list[float] = []
    for cfg in cfg_labels:
      for metric in metrics:
        s = last_step_best[cfg][dataset][metric.name]
        values.extend([float(s.median) - float(s.std), float(s.median) + float(s.std)])
    if not values or not np.all(np.isfinite(np.array(values, dtype=float))):
      raise ValueError(f'Non-finite values found while computing y-limits for summary bar charts: dataset={dataset!r}')
    y_min_data = float(np.min(values))
    y_max = float(np.max(values))
    y_min = 0.0 if bool(ymin_at_zero) else float(y_min_data)
    y_range = max(1e-12, (y_max - y_min))
    y_text_offset = float(0.012 * y_range)
    pad = float(0.06 * y_range + y_text_offset)
    y_low = 0.0 if bool(ymin_at_zero) else (y_min - pad)
    return (y_low, y_max + pad), y_text_offset

  aiono_clf_metrics = [m for m in aiono_metrics if m.name in ('auroc', 'auprc')]
  aiono_reg_metrics = [m for m in aiono_metrics if m.name not in ('auroc', 'auprc')]
  if len(aiono_clf_metrics) != 2:
    raise RuntimeError(f'Expected exactly 2 Aionoscope classification metrics, got: {[m.name for m in aiono_clf_metrics]}')
  if not aiono_reg_metrics:
    raise RuntimeError('Expected at least one Aionoscope regression metric.')

  y_lim_ptbxl, y_text_offset_ptbxl = _y_scale_for_panel(dataset='PTBXL', metrics=ptbxl_metrics, ymin_at_zero=True)
  y_lim_aiono_clf, y_text_offset_aiono_clf = _y_scale_for_panel(
    dataset='AIONO',
    metrics=aiono_clf_metrics,
    ymin_at_zero=True,
  )
  y_lim_aiono_reg, y_text_offset_aiono_reg = _y_scale_for_panel(
    dataset='AIONO',
    metrics=aiono_reg_metrics,
    ymin_at_zero=False,
  )

  # Step 4: plot as one figure (three subplots, independent y-scales, one legend).
  fig, (ax_ptbxl, ax_aiono_clf, ax_aiono_reg) = plt.subplots(
    1,
    3,
    figsize=(14.5, 3.8),
    sharey=False,
    gridspec_kw={'width_ratios': [len(ptbxl_metrics), len(aiono_clf_metrics), len(aiono_reg_metrics)]},
  )
  _plot_on_axis(ax=ax_ptbxl, dataset='PTBXL', metrics=ptbxl_metrics, y_text_offset=y_text_offset_ptbxl)
  _plot_on_axis(ax=ax_aiono_clf, dataset='AIONO', metrics=aiono_clf_metrics, y_text_offset=y_text_offset_aiono_clf)
  _plot_on_axis(ax=ax_aiono_reg, dataset='AIONO', metrics=aiono_reg_metrics, y_text_offset=y_text_offset_aiono_reg)
  ax_ptbxl.set_ylim(*y_lim_ptbxl)
  ax_aiono_clf.set_ylim(*y_lim_aiono_clf)
  ax_aiono_reg.set_ylim(*y_lim_aiono_reg)
  ax_ptbxl.set_title('PTB-XL')
  ax_aiono_clf.set_title('Aionoscope (AUROC/AUPRC)')
  ax_aiono_reg.set_title('Aionoscope (regression)')

  handles, labels = ax_ptbxl.get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'summary_last_step_bars.png')
  fig.savefig(out_dir / 'summary_last_step_bars.pdf')
  plt.close(fig)


def _plot_summary_last_step_delta_from_step0_bar_charts(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  delta_last_minus_step0: dict[str, dict[str, dict[str, SeedStats]]],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Plot last-step deltas vs step 0 for best-layer probe metrics.

  Delta definition:
    delta = best_layer(last_step) - best_layer(step=0),
  where best_layer is selected separately per metric and step (oracle over
  layers 0..8).
  """
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  _set_matplotlib_paper_style()

  if not config_groups:
    raise ValueError('config_groups must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  cfg_labels = _summary_config_labels(config_groups)
  legend_labels = [_tick_label_from_base_label(cfg) for cfg in cfg_labels]

  def _colors_for_cfgs(cfgs: list[str]) -> dict[str, tuple[float, float, float, float]]:
    family_to_cfgs: dict[str, list[str]] = {}
    for cfg in cfgs:
      seed = _representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0)
      meta = config_groups[cfg]['PTBXL'][int(seed)]
      family_to_cfgs.setdefault(meta.method_family, []).append(cfg)

    white = (1.0, 1.0, 1.0)
    colors: dict[str, tuple[float, float, float, float]] = {}
    for method_family, bucket in family_to_cfgs.items():
      base_rgb = _paper_family_base_rgb(method_family)
      max_lighten = 0.45
      if method_family == 'NEPA_STOPGRAD':
        max_lighten = 0.25
      bucket_sorted = sorted(
        bucket,
        key=lambda cfg: _run_sort_key(
          config_groups[cfg]['PTBXL'][int(_representative_seed(config_groups[cfg]['PTBXL'], preferred_seed=0))].run_name
        ),
      )
      if len(bucket_sorted) == 1:
        colors[bucket_sorted[0]] = (*base_rgb, 1.0)
        continue
      for i, cfg in enumerate(bucket_sorted):
        t = (float(i) / float(len(bucket_sorted) - 1)) * float(max_lighten)
        rgb = _blend_rgb(base_rgb, white, t=t)
        colors[cfg] = (*rgb, 1.0)
    return colors

  cfg_colors = _colors_for_cfgs(cfg_labels)

  def _delta(cfg: str, dataset: str, metric: str) -> float:
    return float(delta_last_minus_step0[cfg][dataset][metric].median)

  def _delta_std(cfg: str, dataset: str, metric: str) -> float:
    return float(delta_last_minus_step0[cfg][dataset][metric].std)

  def _plot_on_axis(
    *,
    ax: plt.Axes,
    dataset: str,
    metrics: list[MetricSpec],
    y_text_offset: float,
  ) -> None:
    if not metrics:
      raise ValueError('metrics must be non-empty')
    xs = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = (np.arange(len(cfg_labels), dtype=float) - (len(cfg_labels) - 1) / 2.0) * width

    for i, cfg in enumerate(cfg_labels):
      vals = [_delta(cfg, dataset, m.name) for m in metrics]
      errs = [_delta_std(cfg, dataset, m.name) for m in metrics]
      bars = ax.bar(
        xs + float(offsets[i]),
        vals,
        width=width,
        color=cfg_colors[cfg],
        yerr=errs,
        capsize=2.0,
        error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        edgecolor='black',
        linewidth=0.35,
        label=legend_labels[i],
      )
      for rect, v in zip(list(bars), vals, strict=True):
        if not np.isfinite(float(v)):
          raise ValueError(f'Non-finite delta in summary chart: dataset={dataset} cfg={cfg!r} v={v}')
        x = float(rect.get_x() + rect.get_width() / 2.0)
        y_tip = float(rect.get_y() + rect.get_height())
        if float(v) >= 0.0:
          y_text = y_tip + float(y_text_offset)
          va = 'bottom'
        else:
          y_text = y_tip - float(y_text_offset)
          va = 'top'
        ax.text(
          x,
          y_text,
          _format_float(float(v), digits=3),
          ha='center',
          va=va,
          fontsize=6,
          rotation=0,
          clip_on=True,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([m.pretty for m in metrics])
    ax.grid(True, axis='y')
    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.25)

  ptbxl_metrics = [
    MetricSpec('auroc', 'max', '', 'AUROC $\\uparrow$'),
    MetricSpec('auprc', 'max', '', 'AUPRC $\\uparrow$'),
  ]
  aiono_metrics = [
    MetricSpec('auroc', 'max', '', 'AUROC $\\uparrow$'),
    MetricSpec('auprc', 'max', '', 'AUPRC $\\uparrow$'),
    MetricSpec('mse', 'min', '', 'MSE $\\downarrow$'),
    MetricSpec('mae', 'min', '', 'MAE $\\downarrow$'),
    MetricSpec('pearson', 'max', '', 'Pearson $\\uparrow$'),
    MetricSpec('r2', 'max', '', '$R^2$ $\\uparrow$'),
  ]

  aiono_clf_metrics = [m for m in aiono_metrics if m.name in ('auroc', 'auprc')]
  aiono_reg_metrics = [m for m in aiono_metrics if m.name not in ('auroc', 'auprc')]
  if len(aiono_clf_metrics) != 2:
    raise RuntimeError(f'Expected exactly 2 Aionoscope classification metrics, got: {[m.name for m in aiono_clf_metrics]}')
  if not aiono_reg_metrics:
    raise RuntimeError('Expected at least one Aionoscope regression metric.')

  def _y_scale_for_panel(
    *,
    dataset: str,
    metrics: list[MetricSpec],
    ymin_at_zero: bool,
  ) -> tuple[tuple[float, float], float]:
    values: list[float] = []
    for cfg in cfg_labels:
      for metric in metrics:
        v = _delta(cfg, dataset, metric.name)
        s = _delta_std(cfg, dataset, metric.name)
        values.extend([float(v) - float(s), float(v) + float(s)])
    if not values or not np.all(np.isfinite(np.array(values, dtype=float))):
      raise ValueError(
        'Non-finite values found while computing y-limits for summary delta bar charts. '
        f'dataset={dataset!r}'
      )
    y_min_data = float(np.min(values))
    y_max = float(np.max(values))
    if bool(ymin_at_zero) and y_min_data < 0.0:
      raise ValueError(
        'Cannot force ymin=0 for delta plot because some values would be clipped below 0. '
        f'dataset={dataset!r} y_min={y_min_data:.6g} metrics={[m.name for m in metrics]}'
      )
    y_min = 0.0 if bool(ymin_at_zero) else float(y_min_data)
    y_range = max(1e-12, (y_max - y_min))
    y_text_offset = float(0.012 * y_range)
    pad = float(0.06 * y_range + y_text_offset)
    y_low = 0.0 if bool(ymin_at_zero) else (y_min - pad)
    return (y_low, y_max + pad), y_text_offset

  y_lim_ptbxl, y_text_offset_ptbxl = _y_scale_for_panel(dataset='PTBXL', metrics=ptbxl_metrics, ymin_at_zero=True)
  y_lim_aiono_clf, y_text_offset_aiono_clf = _y_scale_for_panel(
    dataset='AIONO',
    metrics=aiono_clf_metrics,
    ymin_at_zero=True,
  )
  y_lim_aiono_reg, y_text_offset_aiono_reg = _y_scale_for_panel(
    dataset='AIONO',
    metrics=aiono_reg_metrics,
    ymin_at_zero=False,
  )

  fig, (ax_ptbxl, ax_aiono_clf, ax_aiono_reg) = plt.subplots(
    1,
    3,
    figsize=(14.5, 3.8),
    sharey=False,
    gridspec_kw={'width_ratios': [len(ptbxl_metrics), len(aiono_clf_metrics), len(aiono_reg_metrics)]},
  )
  _plot_on_axis(ax=ax_ptbxl, dataset='PTBXL', metrics=ptbxl_metrics, y_text_offset=y_text_offset_ptbxl)
  _plot_on_axis(ax=ax_aiono_clf, dataset='AIONO', metrics=aiono_clf_metrics, y_text_offset=y_text_offset_aiono_clf)
  _plot_on_axis(ax=ax_aiono_reg, dataset='AIONO', metrics=aiono_reg_metrics, y_text_offset=y_text_offset_aiono_reg)
  ax_ptbxl.set_ylim(*y_lim_ptbxl)
  ax_aiono_clf.set_ylim(*y_lim_aiono_clf)
  ax_aiono_reg.set_ylim(*y_lim_aiono_reg)
  ax_ptbxl.set_title('PTB-XL')
  ax_aiono_clf.set_title('Aionoscope (AUROC/AUPRC)')
  ax_aiono_reg.set_title('Aionoscope (regression)')

  handles, labels = ax_ptbxl.get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'summary_last_step_delta_from_step0_bars.png')
  fig.savefig(out_dir / 'summary_last_step_delta_from_step0_bars.pdf')
  plt.close(fig)


def _projector_flag_and_base_label(config_label: str) -> tuple[str, bool] | None:
  """Parse a config label and return (base_label, projector_flag) when applicable.

  Expected formats:
    - `NEPA +Proj` / `NEPA`
    - `LeNEPA (<variant>, +Proj)` / `LeNEPA (<variant>, -Proj)`

  Returns:
    None for configs without a projector toggle (e.g. JEPA).
  """
  if config_label.endswith(' +Proj'):
    base = config_label.removesuffix(' +Proj').strip()
    if not base:
      raise ValueError(f'Failed to parse base label from config_label={config_label!r}')
    return base, True
  if config_label.strip() == 'NEPA':
    return 'NEPA', False

  if config_label.endswith(', +Proj)'):
    base = config_label.removesuffix(', +Proj)') + ')'
    return base, True
  if config_label.endswith(', -Proj)'):
    base = config_label.removesuffix(', -Proj)') + ')'
    return base, False
  return None


def _discover_projector_ablation_pairs(
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
) -> list[tuple[str, str, str]]:
  """Return (base_label, noproj_cfg, proj_cfg) for every projector ablation pair."""
  if not config_groups:
    raise ValueError('config_groups must be non-empty')

  buckets: dict[str, dict[bool, str]] = {}
  for cfg, by_dataset in config_groups.items():
    meta_any = next(iter(next(iter(by_dataset.values())).values()))
    family = _method_family_pretty(meta_any.method_family)
    if family not in ('NEPA', 'LeNEPA'):
      continue
    parsed = _projector_flag_and_base_label(cfg)
    if parsed is None:
      raise ValueError(f'Expected projector toggle in config label, got: {cfg!r}')
    base, proj = parsed
    buckets.setdefault(base, {})[proj] = cfg

  pairs: list[tuple[str, str, str]] = []
  for base, by_flag in buckets.items():
    if True not in by_flag or False not in by_flag:
      raise ValueError(f'Projector pair is incomplete for base={base!r}: {by_flag}')
    pairs.append((base, by_flag[False], by_flag[True]))

  def _pair_sort_key(pair: tuple[str, str, str]) -> tuple:
    _base, _noproj, proj = pair
    seed0 = min(config_groups[proj]['PTBXL'].keys())
    run_name = config_groups[proj]['PTBXL'][int(seed0)].run_name
    return _run_sort_key(run_name)

  return sorted(pairs, key=_pair_sort_key)


def _tick_label_from_base_label(base_label: str) -> str:
  """Convert a base label like `LeNEPA (T20 L0-8 PD0)` to a multi-line tick label."""
  model, detail = _split_model_and_config_from_label(base_label)
  if detail:
    return f'{model}\n{detail}'
  return model


def _plot_projector_ablation_bar_charts(
  *,
  pairs: list[tuple[str, str, str]],
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  seed_pool: tuple[int, ...],
  out_dir: Path,
) -> None:
  """Plot projector ablation bars using a common eval step across seeds and toggles."""
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402

  if not pairs:
    raise ValueError('pairs must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  _set_matplotlib_paper_style()

  xs = np.arange(len(pairs), dtype=float)
  width = 0.36
  tick_labels = [_tick_label_from_base_label(base) for base, _n, _p in pairs]
  colors = {'noproj': '#7f7f7f', 'proj': '#1f77b4'}

  def _paired_seeds_for_pair_dataset(*, noproj_cfg: str, proj_cfg: str, dataset: str) -> list[int]:
    noproj_by_seed = config_groups[noproj_cfg][dataset]
    proj_by_seed = config_groups[proj_cfg][dataset]
    paired = [int(s) for s in seed_pool if int(s) in noproj_by_seed and int(s) in proj_by_seed]
    if len(paired) < 2:
      raise ValueError(
        'Need at least 2 paired seeds for projector ablation aggregation. '
        f'dataset={dataset!r} noproj_cfg={noproj_cfg!r} proj_cfg={proj_cfg!r} seed_pool={list(seed_pool)} paired={paired}'
      )
    return paired

  def _common_step_for_pair_dataset(*, noproj_cfg: str, proj_cfg: str, dataset: str, seeds: list[int]) -> int:
    metrics_all = _metric_specs_for_dataset(dataset)

    common: set[int] | None = None
    for cfg in (noproj_cfg, proj_cfg):
      for seed in seeds:
        meta = config_groups[cfg][dataset].get(int(seed))
        if meta is None:
          raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
        df = histories.get(meta.run_name)
        if df is None:
          raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')

        steps_for_run: set[int] | None = None
        for metric in metrics_all:
          best = _best_layer_per_step(df, metric=metric, layers=layers)
          steps_metric = set(int(s) for s in best['_step'].tolist())
          steps_for_run = steps_metric if steps_for_run is None else (steps_for_run & steps_metric)
        if steps_for_run is None:
          raise RuntimeError(f'Internal error: steps_for_run is None for cfg={cfg!r} dataset={dataset!r}')
        common = steps_for_run if common is None else (common & steps_for_run)

    if not common:
      raise RuntimeError(
        'No common offline-probe eval step across seeds and projector toggles. '
        f'dataset={dataset!r} noproj_cfg={noproj_cfg!r} proj_cfg={proj_cfg!r}'
      )
    return int(max(common))

  def _seed_stats_best_layer_at_step(
    *,
    cfg: str,
    dataset: str,
    metric: MetricSpec,
    step: int,
    seeds: list[int],
  ) -> SeedStats:
    values: list[float] = []
    for seed in seeds:
      meta = config_groups[cfg][dataset].get(int(seed))
      if meta is None:
        raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
      df = histories.get(meta.run_name)
      if df is None:
        raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')
      best = _best_layer_at_step(df, metric=metric, layers=layers, step=int(step))
      values.append(float(best.best_value))
    return _seed_stats(values, context=f'proj_ablation cfg={cfg} dataset={dataset} metric={metric.name} step={int(step)}')

  # Step 1: classification ablation (PTB-XL + Aionoscope, AUROC/AUPRC).
  fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
  datasets = [('PTBXL', 'PTB-XL'), ('AIONO', 'Aionoscope')]
  for i, (dataset, dataset_pretty) in enumerate(datasets):
    metric_specs = {m.name: m for m in _metric_specs_for_dataset(dataset) if m.name in ('auroc', 'auprc')}
    steps_by_pair: dict[str, int] = {}
    seeds_by_pair: dict[str, list[int]] = {}
    for j, metric_name in enumerate(('auroc', 'auprc')):
      metric = metric_specs[metric_name]
      ax = axes[i, j]
      vals_noproj: list[float] = []
      errs_noproj: list[float] = []
      vals_proj: list[float] = []
      errs_proj: list[float] = []
      for base, noproj, proj in pairs:
        seeds = seeds_by_pair.get(base)
        if seeds is None:
          seeds = _paired_seeds_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset=dataset)
          seeds_by_pair[base] = seeds

        step_common = steps_by_pair.get(base)
        if step_common is None:
          step_common = _common_step_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset=dataset, seeds=seeds)
          steps_by_pair[base] = int(step_common)
          if int(step_common) != 20000:
            print(
              f'[warn] projector_ablation: using common_step={int(step_common)} for base={base!r} dataset={dataset!r} '
              '(some runs are missing later offline-probe evals).'
            )

        s_n = _seed_stats_best_layer_at_step(
          cfg=noproj, dataset=dataset, metric=metric, step=int(step_common), seeds=seeds
        )
        s_p = _seed_stats_best_layer_at_step(cfg=proj, dataset=dataset, metric=metric, step=int(step_common), seeds=seeds)
        vals_noproj.append(float(s_n.median))
        errs_noproj.append(float(s_n.std))
        vals_proj.append(float(s_p.median))
        errs_proj.append(float(s_p.std))

      ax.bar(
        xs - width / 2.0,
        vals_noproj,
        yerr=errs_noproj,
        capsize=2.0,
        error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        width=width,
        color=colors['noproj'],
        label='-Proj',
      )
      ax.bar(
        xs + width / 2.0,
        vals_proj,
        yerr=errs_proj,
        capsize=2.0,
        error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        width=width,
        color=colors['proj'],
        label='+Proj',
      )
      ax.set_title(f'{dataset_pretty}: {metric.pretty}')
      ax.grid(True, axis='y')
      if i == len(datasets) - 1:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)

  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
  fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / 'projector_ablation_last_step_classification_bars.png')
  fig.savefig(out_dir / 'projector_ablation_last_step_classification_bars.pdf')
  plt.close(fig)

  # Step 2: Aionoscope dense ablation (MSE/MAE/Pearson/R2).
  dense_specs = {m.name: m for m in _metric_specs_for_dataset('AIONO') if m.name in ('mse', 'mae', 'pearson', 'r2')}
  fig2, axes2 = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
  axes2_list = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
  dense_titles = {
    'mse': 'MSE $\\downarrow$',
    'mae': 'MAE $\\downarrow$',
    'pearson': 'Pearson $\\uparrow$',
    'r2': '$R^2$ $\\uparrow$',
  }
  steps_by_pair_aiono: dict[str, int] = {}
  seeds_by_pair_aiono: dict[str, list[int]] = {}
  for ax, metric_name in zip(axes2_list, ('mse', 'mae', 'pearson', 'r2'), strict=True):
    metric = dense_specs[metric_name]
    vals_noproj: list[float] = []
    errs_noproj: list[float] = []
    vals_proj: list[float] = []
    errs_proj: list[float] = []

    for base, noproj, proj in pairs:
      seeds = seeds_by_pair_aiono.get(base)
      if seeds is None:
        seeds = _paired_seeds_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='AIONO')
        seeds_by_pair_aiono[base] = seeds

      step_common = steps_by_pair_aiono.get(base)
      if step_common is None:
        step_common = _common_step_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='AIONO', seeds=seeds)
        steps_by_pair_aiono[base] = int(step_common)
        if int(step_common) != 20000:
          print(
            f'[warn] projector_ablation: using common_step={int(step_common)} for base={base!r} dataset=\"AIONO\" '
            '(some runs are missing later offline-probe evals).'
          )

      s_n = _seed_stats_best_layer_at_step(cfg=noproj, dataset='AIONO', metric=metric, step=int(step_common), seeds=seeds)
      s_p = _seed_stats_best_layer_at_step(cfg=proj, dataset='AIONO', metric=metric, step=int(step_common), seeds=seeds)
      vals_noproj.append(float(s_n.median))
      errs_noproj.append(float(s_n.std))
      vals_proj.append(float(s_p.median))
      errs_proj.append(float(s_p.std))

    ax.bar(
      xs - width / 2.0,
      vals_noproj,
      yerr=errs_noproj,
      capsize=2.0,
      error_kw={'elinewidth': 0.8, 'capthick': 0.8},
      width=width,
      color=colors['noproj'],
      label='-Proj',
    )
    ax.bar(
      xs + width / 2.0,
      vals_proj,
      yerr=errs_proj,
      capsize=2.0,
      error_kw={'elinewidth': 0.8, 'capthick': 0.8},
      width=width,
      color=colors['proj'],
      label='+Proj',
    )
    ax.set_title(dense_titles.get(metric.name, metric.pretty))
    ax.grid(True, axis='y')

  for ax in axes2_list[-2:]:
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)

  handles2, labels2 = axes2_list[0].get_legend_handles_labels()
  fig2.legend(handles2, labels2, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
  fig2.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
  fig2.savefig(out_dir / 'projector_ablation_last_step_dense_bars.png')
  fig2.savefig(out_dir / 'projector_ablation_last_step_dense_bars.pdf')
  plt.close(fig2)


def _export_projector_ablation_table(
  *,
  pairs: list[tuple[str, str, str]],
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  histories: dict[str, pd.DataFrame],
  layers: list[int],
  seed_pool: tuple[int, ...],
  out_csv_path: Path,
  out_tex_path: Path,
) -> None:
  """Export projector ablation table using a common eval step across seeds and toggles."""
  if not pairs:
    raise ValueError('pairs must be non-empty')
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')

  def _paired_seeds_for_pair_dataset(*, noproj_cfg: str, proj_cfg: str, dataset: str) -> list[int]:
    noproj_by_seed = config_groups[noproj_cfg][dataset]
    proj_by_seed = config_groups[proj_cfg][dataset]
    paired = [int(s) for s in seed_pool if int(s) in noproj_by_seed and int(s) in proj_by_seed]
    if len(paired) < 2:
      raise ValueError(
        'Need at least 2 paired seeds for projector ablation aggregation. '
        f'dataset={dataset!r} noproj_cfg={noproj_cfg!r} proj_cfg={proj_cfg!r} seed_pool={list(seed_pool)} paired={paired}'
      )
    return paired

  def _common_step_for_pair_dataset(*, noproj_cfg: str, proj_cfg: str, dataset: str, seeds: list[int]) -> int:
    metrics_all = _metric_specs_for_dataset(dataset)

    common: set[int] | None = None
    for cfg in (noproj_cfg, proj_cfg):
      for seed in seeds:
        meta = config_groups[cfg][dataset].get(int(seed))
        if meta is None:
          raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
        df = histories.get(meta.run_name)
        if df is None:
          raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')

        steps_for_run: set[int] | None = None
        for metric in metrics_all:
          best = _best_layer_per_step(df, metric=metric, layers=layers)
          steps_metric = set(int(s) for s in best['_step'].tolist())
          steps_for_run = steps_metric if steps_for_run is None else (steps_for_run & steps_metric)
        if steps_for_run is None:
          raise RuntimeError(f'Internal error: steps_for_run is None for cfg={cfg!r} dataset={dataset!r}')
        common = steps_for_run if common is None else (common & steps_for_run)

    if not common:
      raise RuntimeError(
        'No common offline-probe eval step across seeds and projector toggles. '
        f'dataset={dataset!r} noproj_cfg={noproj_cfg!r} proj_cfg={proj_cfg!r}'
      )
    return int(max(common))

  def _seed_stats_best_layer_at_step(
    *,
    cfg: str,
    dataset: str,
    metric: MetricSpec,
    step: int,
    seeds: list[int],
  ) -> SeedStats:
    values: list[float] = []
    for seed in seeds:
      meta = config_groups[cfg][dataset].get(int(seed))
      if meta is None:
        raise KeyError(f'Missing required seed={int(seed)} for cfg={cfg!r} dataset={dataset!r}')
      df = histories.get(meta.run_name)
      if df is None:
        raise KeyError(f'Missing history for run_name={meta.run_name!r} cfg={cfg!r} dataset={dataset!r}')
      best = _best_layer_at_step(df, metric=metric, layers=layers, step=int(step))
      values.append(float(best.best_value))
    return _seed_stats(values, context=f'proj_ablation cfg={cfg} dataset={dataset} metric={metric.name} step={int(step)}')

  def _improvement(v_noproj: float, v_proj: float, *, direction: str) -> float:
    if direction == 'max':
      return float(v_proj) - float(v_noproj)
    if direction == 'min':
      return float(v_noproj) - float(v_proj)
    raise ValueError(f'Unexpected direction: {direction!r}')

  def _format_pair_cells(
    stats_noproj: SeedStats,
    stats_proj: SeedStats,
    *,
    direction: str,
    digits: int,
  ) -> tuple[str, str]:
    s_n = _format_seed_stats_latex(stats_noproj, digits=digits)
    s_p = _format_seed_stats_latex(stats_proj, digits=digits)
    if direction == 'max':
      if float(stats_proj.median) >= float(stats_noproj.median):
        s_p = f'\\textbf{{{s_p}}}'
      else:
        s_n = f'\\textbf{{{s_n}}}'
    elif direction == 'min':
      if float(stats_proj.median) <= float(stats_noproj.median):
        s_p = f'\\textbf{{{s_p}}}'
      else:
        s_n = f'\\textbf{{{s_n}}}'
    else:
      raise ValueError(f'Unexpected direction: {direction!r}')
    return s_n, s_p

  # Step 1: build a machine-readable CSV (includes direction-aligned improvements).
  records: list[dict[str, object]] = []
  stats_cache: dict[tuple[str, str, str, str], SeedStats] = {}
  step_by_base_and_dataset: dict[tuple[str, str], int] = {}
  seeds_by_base_and_dataset: dict[tuple[str, str], list[int]] = {}
  for base, noproj, proj in pairs:
    model, detail = _split_model_and_config_from_label(base)
    ptbxl_seeds = _paired_seeds_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='PTBXL')
    aiono_seeds = _paired_seeds_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='AIONO')
    ptbxl_step = _common_step_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='PTBXL', seeds=ptbxl_seeds)
    aiono_step = _common_step_for_pair_dataset(noproj_cfg=noproj, proj_cfg=proj, dataset='AIONO', seeds=aiono_seeds)
    if int(ptbxl_step) != 20000 or int(aiono_step) != 20000:
      print(
        '[warn] projector_ablation: using non-20k common step for some pair(s). '
        f'base={base!r} ptbxl_step={int(ptbxl_step)} aiono_step={int(aiono_step)}'
      )

    record: dict[str, object] = {
      'model': model,
      'config': detail,
      'ptbxl_step': int(ptbxl_step),
      'aiono_step': int(aiono_step),
    }
    step_by_base_and_dataset[(base, 'PTBXL')] = int(ptbxl_step)
    step_by_base_and_dataset[(base, 'AIONO')] = int(aiono_step)
    seeds_by_base_and_dataset[(base, 'PTBXL')] = ptbxl_seeds
    seeds_by_base_and_dataset[(base, 'AIONO')] = aiono_seeds

    for dataset in ('PTBXL', 'AIONO'):
      step_common = int(ptbxl_step) if dataset == 'PTBXL' else int(aiono_step)
      seeds = seeds_by_base_and_dataset[(base, dataset)]
      for metric in _metric_specs_for_dataset(dataset):
        key_n = (base, dataset, metric.name, 'noproj')
        key_p = (base, dataset, metric.name, 'proj')
        s_n = stats_cache.get(key_n)
        s_p = stats_cache.get(key_p)
        if s_n is None:
          s_n = _seed_stats_best_layer_at_step(cfg=noproj, dataset=dataset, metric=metric, step=step_common, seeds=seeds)
          stats_cache[key_n] = s_n
        if s_p is None:
          s_p = _seed_stats_best_layer_at_step(cfg=proj, dataset=dataset, metric=metric, step=step_common, seeds=seeds)
          stats_cache[key_p] = s_p
        prefix = f'{dataset.lower()}_{metric.name}'
        record[f'{prefix}_noproj_median'] = float(s_n.median)
        record[f'{prefix}_noproj_std'] = float(s_n.std)
        record[f'{prefix}_proj_median'] = float(s_p.median)
        record[f'{prefix}_proj_std'] = float(s_p.std)
        record[f'{prefix}_improvement_median'] = _improvement(
          float(s_n.median), float(s_p.median), direction=metric.direction
        )

    records.append(record)

  df = pd.DataFrame.from_records(records)
  out_csv_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_csv_path, index=False)

  # Step 2: paper table (match `paper/tables/projector_ablation_last_step.tex` exactly).
  digits = 3
  ptb_specs = {m.name: m for m in _metric_specs_for_dataset('PTBXL')}
  aiono_specs = {m.name: m for m in _metric_specs_for_dataset('AIONO')}

  def _format_pair_cell_stats(stats_noproj: SeedStats, stats_proj: SeedStats, *, direction: str, digits: int) -> str:
    """Format one cell as `-Proj/+Proj`, bolding the better value by median within the pair."""
    s_n, s_p = _format_pair_cells(stats_noproj, stats_proj, direction=direction, digits=digits)
    return f'{s_n} / {s_p}'

  pairs_by_base: dict[str, tuple[str, str]] = {}
  for base, noproj, proj in pairs:
    if base in pairs_by_base:
      raise ValueError(f'Duplicate projector ablation base label: {base!r}')
    pairs_by_base[base] = (noproj, proj)

  expected_bases = [
    'NEPA',
    'LeNEPA (T20 L0-8)',
    'LeNEPA (BRI20 L0-8)',
  ]
  missing = [b for b in expected_bases if b not in pairs_by_base]
  extra = sorted(set(pairs_by_base) - set(expected_bases))
  if missing:
    raise ValueError(
      'Missing required projector ablation pairs for paper table. '
      f'missing={missing} available={sorted(pairs_by_base)}'
    )
  if extra:
    print(f'[filter] Excluding {len(extra)} projector_ablation pair(s) from paper table: {extra}')

  caption = (
    'Projector ablation (NEPA / LeNEPA): best-layer value at the last \\emph{common} offline-probe eval step across seeds and '
    'projector toggles, aggregated across seeds as $\\mathrm{median}(\\mathrm{std})$ (ddof=1; std shown in $10^{-3}$ units). '
    'Each cell reports $-\\mathrm{Proj} / +\\mathrm{Proj}$; within each pair, the better value by median is bold '
    '(higher is better for AUROC / AUPRC / Pearson / $R^2$; lower is better for MSE / MAE).'
  )

  lines: list[str] = []
  lines.append('\\begin{table}[t]')
  lines.append('\\centering')
  lines.append('\\scriptsize')
  lines.append(f'\\caption{{{caption}}}')
  lines.append('\\label{tab:projector_ablation_last_step}')
  lines.append('')
  lines.append('\\setlength{\\tabcolsep}{3pt}')
  lines.append('\\renewcommand{\\arraystretch}{1.15}')
  lines.append('\\resizebox{\\linewidth}{!}{%')
  lines.append('\\begin{tabular}{lllrrrrrr}')
  lines.append('\\toprule')
  lines.append('& & & \\multicolumn{2}{c}{Classification} & \\multicolumn{4}{c}{Dense regression} \\\\')
  lines.append('\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}')
  lines.append('dataset & model & config & AUROC & AUPRC & MSE & MAE & Pearson & $R^2$ \\\\')
  lines.append('\\midrule')

  def _cached_pair_stats(*, base: str, dataset: str, metric_name: str) -> tuple[SeedStats, SeedStats, str]:
    """Return (noproj, proj, direction) for a cached (base, dataset, metric) entry."""
    if dataset == 'PTBXL':
      metric = ptb_specs[metric_name]
    elif dataset == 'AIONO':
      metric = aiono_specs[metric_name]
    else:
      raise ValueError(f'Unexpected dataset key: {dataset!r}')

    stats_n = stats_cache.get((base, dataset, metric.name, 'noproj'))
    stats_p = stats_cache.get((base, dataset, metric.name, 'proj'))
    if stats_n is None or stats_p is None:
      raise KeyError(f'Missing cached projector-ablation stats: base={base!r} dataset={dataset!r} metric={metric.name!r}')
    return stats_n, stats_p, metric.direction

  def _row_cells(*, base: str, dataset: str) -> list[str]:
    """Build metric cells for the output table (paper formatting)."""
    if dataset == 'PTBXL':
      cells: list[str] = []
      for metric_name in ('auroc', 'auprc'):
        s_n, s_p, direction = _cached_pair_stats(base=base, dataset=dataset, metric_name=metric_name)
        cells.append(_format_pair_cell_stats(s_n, s_p, direction=direction, digits=digits))
      cells.append('\\multicolumn{4}{c}{\\textemdash}')
      return cells

    cells = []
    for metric_name in ('auroc', 'auprc', 'mse', 'mae', 'pearson', 'r2'):
      s_n, s_p, direction = _cached_pair_stats(base=base, dataset=dataset, metric_name=metric_name)
      cells.append(_format_pair_cell_stats(s_n, s_p, direction=direction, digits=digits))
    return cells

  dataset_groups: list[tuple[str, str]] = [
    ('PTBXL', 'PTB-XL'),
    ('AIONO', 'Aionoscope'),
  ]

  for group_idx, (dataset, dataset_pretty) in enumerate(dataset_groups):
    for i, base in enumerate(expected_bases):
      model, detail = _split_model_and_config_from_label(base)
      _noproj_cfg, _proj_cfg = pairs_by_base[base]
      _step_common = step_by_base_and_dataset.get((base, dataset))
      if _step_common is None:
        raise KeyError(f'Missing common eval step for base={base!r} dataset={dataset!r}')

      cells = _row_cells(base=base, dataset=dataset)
      if i == 0:
        prefix = f'\\multirow{{3}}{{*}}{{{dataset_pretty}}}'
        lines.append(
          ' & '.join([prefix, _latex_escape(model), _latex_escape(detail), *cells]) + ' \\\\'
        )
      else:
        lines.append(' & '.join(['', _latex_escape(model), _latex_escape(detail), *cells]).lstrip() + ' \\\\')

    if group_idx != len(dataset_groups) - 1:
      lines.append('\\midrule')

  lines.append('\\bottomrule')
  lines.append('\\end{tabular}')
  lines.append('} % resizebox')
  lines.append('\\end{table}')

  out_tex_path.parent.mkdir(parents=True, exist_ok=True)
  out_tex_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _export_summary_last_step_table(
  *,
  config_groups: dict[str, dict[str, dict[int, RunMeta]]],
  last_step_best: dict[str, dict[str, dict[str, SeedStats]]],
  delta_last_minus_step0: dict[str, dict[str, dict[str, SeedStats]]],
  seed_pool: tuple[int, ...],
  out_csv_path: Path,
  out_tex_path: Path,
  out_tex_delta_path: Path,
) -> None:
  """Export the summary last-step table mirroring `summary_last_step_bars`."""
  if not seed_pool:
    raise ValueError('seed_pool must be non-empty')
  cfg_labels = _summary_config_labels(config_groups)

  rows: list[dict[str, object]] = []
  for cfg in cfg_labels:
    family, config = _split_model_and_config_from_label(cfg)
    rows.append(
      {
        'family': family,
        'config': config,
        'config_label': cfg,
        'ptbxl_auroc_median': float(last_step_best[cfg]['PTBXL']['auroc'].median),
        'ptbxl_auroc_std': float(last_step_best[cfg]['PTBXL']['auroc'].std),
        'ptbxl_delta_auroc_median': float(delta_last_minus_step0[cfg]['PTBXL']['auroc'].median),
        'ptbxl_delta_auroc_std': float(delta_last_minus_step0[cfg]['PTBXL']['auroc'].std),
        'ptbxl_auprc_median': float(last_step_best[cfg]['PTBXL']['auprc'].median),
        'ptbxl_auprc_std': float(last_step_best[cfg]['PTBXL']['auprc'].std),
        'ptbxl_delta_auprc_median': float(delta_last_minus_step0[cfg]['PTBXL']['auprc'].median),
        'ptbxl_delta_auprc_std': float(delta_last_minus_step0[cfg]['PTBXL']['auprc'].std),
        'aiono_auroc_median': float(last_step_best[cfg]['AIONO']['auroc'].median),
        'aiono_auroc_std': float(last_step_best[cfg]['AIONO']['auroc'].std),
        'aiono_delta_auroc_median': float(delta_last_minus_step0[cfg]['AIONO']['auroc'].median),
        'aiono_delta_auroc_std': float(delta_last_minus_step0[cfg]['AIONO']['auroc'].std),
        'aiono_auprc_median': float(last_step_best[cfg]['AIONO']['auprc'].median),
        'aiono_auprc_std': float(last_step_best[cfg]['AIONO']['auprc'].std),
        'aiono_delta_auprc_median': float(delta_last_minus_step0[cfg]['AIONO']['auprc'].median),
        'aiono_delta_auprc_std': float(delta_last_minus_step0[cfg]['AIONO']['auprc'].std),
        'aiono_mse_median': float(last_step_best[cfg]['AIONO']['mse'].median),
        'aiono_mse_std': float(last_step_best[cfg]['AIONO']['mse'].std),
        'aiono_delta_mse_median': float(delta_last_minus_step0[cfg]['AIONO']['mse'].median),
        'aiono_delta_mse_std': float(delta_last_minus_step0[cfg]['AIONO']['mse'].std),
        'aiono_mae_median': float(last_step_best[cfg]['AIONO']['mae'].median),
        'aiono_mae_std': float(last_step_best[cfg]['AIONO']['mae'].std),
        'aiono_delta_mae_median': float(delta_last_minus_step0[cfg]['AIONO']['mae'].median),
        'aiono_delta_mae_std': float(delta_last_minus_step0[cfg]['AIONO']['mae'].std),
        'aiono_pearson_median': float(last_step_best[cfg]['AIONO']['pearson'].median),
        'aiono_pearson_std': float(last_step_best[cfg]['AIONO']['pearson'].std),
        'aiono_delta_pearson_median': float(delta_last_minus_step0[cfg]['AIONO']['pearson'].median),
        'aiono_delta_pearson_std': float(delta_last_minus_step0[cfg]['AIONO']['pearson'].std),
        'aiono_r2_median': float(last_step_best[cfg]['AIONO']['r2'].median),
        'aiono_r2_std': float(last_step_best[cfg]['AIONO']['r2'].std),
        'aiono_delta_r2_median': float(delta_last_minus_step0[cfg]['AIONO']['r2'].median),
        'aiono_delta_r2_std': float(delta_last_minus_step0[cfg]['AIONO']['r2'].std),
      }
    )

  df = pd.DataFrame.from_records(rows)
  out_csv_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_csv_path, index=False)

  def _format_seed_stats_median_std_compact(stats: SeedStats, *, digits: int) -> str:
    """Format `median(std)` in compact `0.893(3)` notation.

    Interpretation (for digits=3):
      `0.893(3)` means `0.893 ± 0.003`.
    """
    med = _format_float(float(stats.median), digits=int(digits))
    std = float(stats.std)
    if not np.isfinite(std):
      raise ValueError(f'Expected finite std, got: {std}')
    if std < 0:
      raise ValueError(f'Expected non-negative std, got: {std}')
    scale = 10 ** int(digits)
    std_int = int(std * float(scale))
    return f'{med}({std_int})'

  tex_rows_abs: list[list[str]] = []
  tex_rows_delta: list[list[str]] = []
  for record in rows:
    tex_rows_abs.append(
      [
        _latex_escape(str(record['family'])),
        _latex_escape(str(record['config'])),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['ptbxl_auroc_median']), std=float(record['ptbxl_auroc_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['ptbxl_auprc_median']), std=float(record['ptbxl_auprc_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_auroc_median']), std=float(record['aiono_auroc_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_auprc_median']), std=float(record['aiono_auprc_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_mse_median']), std=float(record['aiono_mse_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_mae_median']), std=float(record['aiono_mae_std']), n=0), digits=3
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_pearson_median']), std=float(record['aiono_pearson_std']), n=0),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_r2_median']), std=float(record['aiono_r2_std']), n=0), digits=3
        ),
      ]
    )
    tex_rows_delta.append(
      [
        _latex_escape(str(record['family'])),
        _latex_escape(str(record['config'])),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['ptbxl_delta_auroc_median']), std=float(record['ptbxl_delta_auroc_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['ptbxl_delta_auprc_median']), std=float(record['ptbxl_delta_auprc_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['aiono_delta_auroc_median']), std=float(record['aiono_delta_auroc_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['aiono_delta_auprc_median']), std=float(record['aiono_delta_auprc_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['aiono_delta_mse_median']), std=float(record['aiono_delta_mse_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['aiono_delta_mae_median']), std=float(record['aiono_delta_mae_std']), n=0
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(
            median=float(record['aiono_delta_pearson_median']),
            std=float(record['aiono_delta_pearson_std']),
            n=0,
          ),
          digits=3,
        ),
        _format_seed_stats_median_std_compact(
          SeedStats(median=float(record['aiono_delta_r2_median']), std=float(record['aiono_delta_r2_std']), n=0),
          digits=3,
        ),
      ]
    )

  def _mark_best_and_second_best(
    *,
    records: list[dict[str, object]],
    tex_rows: list[list[str]],
    median_keys: list[str],
    directions: list[str],
    tol: float,
    wrap_in_multicolumn: bool,
  ) -> None:
    """Bold best and underline second-best per metric column.

    Ranking is computed on the median values. We treat values within `tol` of
    each other (direction-aware) as ties:
      - All rows within `tol` of the best value are bold.
      - Among the remaining rows, all rows within `tol` of the second-best value
        are underlined.

    If `wrap_in_multicolumn` is True, the formatting is wrapped with
    `\\multicolumn{1}{c}{...}` (needed for siunitx `S` columns). For plain `l`
    columns, it should be False.
    """
    if len(median_keys) != len(directions):
      raise ValueError(f'median_keys and directions length mismatch: {len(median_keys)} vs {len(directions)}')
    if not records:
      raise ValueError('records must be non-empty')
    if not tex_rows:
      raise ValueError('tex_rows must be non-empty')
    if len(records) != len(tex_rows):
      raise ValueError(f'records/tex_rows length mismatch: {len(records)} vs {len(tex_rows)}')
    if float(tol) < 0:
      raise ValueError(f'tol must be >= 0, got: {tol}')

    n_rows = len(records)
    first_value_col = 2  # [model, config, <values...>]
    for j, (median_key, direction) in enumerate(zip(median_keys, directions, strict=True)):
      values = [float(r[median_key]) for r in records]
      if any(not np.isfinite(v) for v in values):
        bad = [(i, v) for i, v in enumerate(values) if not np.isfinite(v)]
        raise ValueError(f'Non-finite median values for key={median_key!r}: {bad}')

      if direction == 'max':
        best_value = float(max(values))
        best_idxs = [i for i, v in enumerate(values) if (best_value - float(v)) <= float(tol)]
        remaining = [i for i in range(n_rows) if i not in set(best_idxs)]
        if remaining:
          second_value = float(max(values[i] for i in remaining))
          second_idxs = [i for i in remaining if (second_value - float(values[i])) <= float(tol)]
        else:
          second_idxs = []
      elif direction == 'min':
        best_value = float(min(values))
        best_idxs = [i for i, v in enumerate(values) if (float(v) - best_value) <= float(tol)]
        remaining = [i for i in range(n_rows) if i not in set(best_idxs)]
        if remaining:
          second_value = float(min(values[i] for i in remaining))
          second_idxs = [i for i in remaining if (float(values[i]) - second_value) <= float(tol)]
        else:
          second_idxs = []
      else:
        raise ValueError(f'Unexpected direction: {direction!r} (expected \"max\" or \"min\")')

      col_idx = first_value_col + int(j)
      for i in best_idxs:
        value = tex_rows[i][col_idx]
        cell = f'\\textbf{{{value}}}'
        if wrap_in_multicolumn:
          cell = f'\\multicolumn{{1}}{{c}}{{{cell}}}'
        tex_rows[i][col_idx] = cell
      for i in second_idxs:
        value = tex_rows[i][col_idx]
        cell = f'\\underline{{{value}}}'
        if wrap_in_multicolumn:
          cell = f'\\multicolumn{{1}}{{c}}{{{cell}}}'
        tex_rows[i][col_idx] = cell

  tol = 0.005
  _mark_best_and_second_best(
    records=rows,
    tex_rows=tex_rows_abs,
    median_keys=[
      'ptbxl_auroc_median',
      'ptbxl_auprc_median',
      'aiono_auroc_median',
      'aiono_auprc_median',
      'aiono_mse_median',
      'aiono_mae_median',
      'aiono_pearson_median',
      'aiono_r2_median',
    ],
    directions=['max', 'max', 'max', 'max', 'min', 'min', 'max', 'max'],
    tol=tol,
    wrap_in_multicolumn=False,
  )
  _mark_best_and_second_best(
    records=rows,
    tex_rows=tex_rows_delta,
    median_keys=[
      'ptbxl_delta_auroc_median',
      'ptbxl_delta_auprc_median',
      'aiono_delta_auroc_median',
      'aiono_delta_auprc_median',
      'aiono_delta_mse_median',
      'aiono_delta_mae_median',
      'aiono_delta_pearson_median',
      'aiono_delta_r2_median',
    ],
    directions=['max', 'max', 'max', 'max', 'min', 'min', 'max', 'max'],
    tol=tol,
    wrap_in_multicolumn=False,
  )

  def _write_summary_l_table(
    *,
    out_path: Path,
    caption: str,
    label: str,
    ptbxl_headers: list[str],
    aiono_headers: list[str],
    rows: list[list[str]],
  ) -> None:
    """Write a summary table matching `paper/tables/summary_selected_best_last_step.tex`."""
    n_metrics = int(len(ptbxl_headers) + len(aiono_headers))

    lines: list[str] = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\scriptsize')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')

    col_spec = 'll ' + ' '.join(['l'] * int(n_metrics))
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')

    lines.append('\\toprule')
    lines.append('\\multicolumn{2}{c}{} & \\multicolumn{2}{c}{PTB-XL} & \\multicolumn{6}{c}{Aionoscope} \\\\')
    lines.append('\\cmidrule(lr){3-4} \\cmidrule(lr){5-10}')

    headers = [
      'model',
      'config',
      *[f'\\multicolumn{{1}}{{c}}{{{h}}}' for h in [*ptbxl_headers, *aiono_headers]],
    ]
    lines.append(' & '.join(headers) + ' \\\\')
    lines.append('\\midrule')

    expected_cols = 2 + int(n_metrics)
    for row in rows:
      if len(row) != expected_cols:
        raise ValueError(f'Row length mismatch: expected {expected_cols}, got {len(row)} row={row}')
      lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

  _write_summary_l_table(
    out_path=out_tex_path,
    caption=(
      'Main evaluated configurations: last-step best-layer probe metrics, '
      'aggregated across seeds as $\\mathrm{median}(\\mathrm{std})$ with std shown in $10^{-3}$ units.'
    ),
    label='tab:summary_selected_best_last_step',
    ptbxl_headers=['AUROC $\\uparrow$', 'AUPRC $\\uparrow$'],
    aiono_headers=['AUROC $\\uparrow$', 'AUPRC $\\uparrow$', 'MSE $\\downarrow$', 'MAE $\\downarrow$', 'Pearson $\\uparrow$', '$R^2$ $\\uparrow$'],
    rows=tex_rows_abs,
  )
  _write_summary_l_table(
    out_path=out_tex_delta_path,
    caption=(
      'Main evaluated configurations: training gains (last-step minus step 0), aggregated across seeds '
      'as $\\mathrm{median}(\\mathrm{std})$ with std shown in $10^{-3}$ units.'
    ),
    label='tab:summary_selected_best_last_step_delta',
    ptbxl_headers=['$\\Delta$AUROC $\\uparrow$', '$\\Delta$AUPRC $\\uparrow$'],
    aiono_headers=[
      '$\\Delta$AUROC $\\uparrow$',
      '$\\Delta$AUPRC $\\uparrow$',
      '$\\Delta$MSE $\\downarrow$',
      '$\\Delta$MAE $\\downarrow$',
      '$\\Delta$Pearson $\\uparrow$',
      '$\\Delta R^2$ $\\uparrow$',
    ],
    rows=tex_rows_delta,
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--wandb-entity',
    type=str,
    default='langotime',
    help='W&B entity/team name (default: langotime).',
  )
  parser.add_argument(
    '--wandb-project',
    type=str,
    default='ECG-LeJEPA',
    help='W&B project name (default: ECG-LeJEPA).',
  )
  parser.add_argument(
    '--queue-order',
    type=Path,
    default=_repo_root() / 'experiments/legacy_job_scripts/plan110/queue_order.txt',
    help='Path to the plan-110 non-Annex queue order file.',
  )
  parser.add_argument(
    '--queue-order-extra',
    type=Path,
    default=_repo_root() / 'experiments/legacy_job_scripts/plan110/queue_seed_std_s1-4.txt',
    help='Extra queue-order file (e.g., additional seeds for variability).',
  )
  parser.add_argument(
    '--queue-order-extra2',
    type=Path,
    default=_repo_root() / 'experiments/legacy_job_scripts/plan110/queue_seed_std_s5-9.txt',
    help='Second extra queue-order file (e.g., more variability seeds).',
  )
  parser.add_argument(
    '--logs-dir',
    type=Path,
    default=_repo_root() / 'experiments/legacy_job_scripts',
    help='Directory with VM job logs (used to discover W&B run ids).',
  )
  parser.add_argument(
    '--export-dir',
    type=Path,
    default=_repo_root() / 'results/wandb_export',
    help='Output directory for W&B history exports.',
  )
  parser.add_argument(
    '--fig-dir',
    type=Path,
    default=_repo_root() / 'paper/figures',
    help='Output directory for paper-ready figures.',
  )
  parser.add_argument(
    '--tables-tex-dir',
    type=Path,
    default=_repo_root() / 'paper/tables',
    help='Output directory for LaTeX tables.',
  )
  parser.add_argument(
    '--tables-csv-dir',
    type=Path,
    default=_repo_root() / 'results/tables',
    help='Output directory for machine-readable CSV tables.',
  )
  parser.add_argument(
    '--layers',
    type=int,
    nargs='+',
    default=list(range(9)),
    help='Layers to include (default: 0..8).',
  )
  parser.add_argument(
    '--max-step',
    type=int,
    default=20000,
    help='Max step to request from W&B history (default: 20000).',
  )
  parser.add_argument(
    '--page-size',
    type=int,
    default=2000,
    help='W&B scan_history page size (default: 2000).',
  )
  parser.add_argument(
    '--skip-existing',
    action='store_true',
    help='Skip W&B download when history.csv.gz already exists.',
  )
  parser.add_argument(
    '--no-download',
    action='store_true',
    help=(
      'Never call the W&B API (offline mode). '
      'Uses existing exports under --export-dir and skips runs with missing files.'
    ),
  )
  return parser.parse_args()


def main() -> None:
  """Entry point."""
  args = _parse_args()
  layers = list(dict.fromkeys(args.layers))
  if any(layer < 0 for layer in layers):
    raise ValueError(f'Layers must be >= 0, got: {layers}')

  if args.no_download and not args.skip_existing:
    raise ValueError('--no-download requires --skip-existing (offline mode uses only local exports).')

  # Step 1: discover run names (non-Annex experiment matrix).
  queue_paths = [args.queue_order, args.queue_order_extra]
  if args.queue_order_extra2.is_file():
    queue_paths.append(args.queue_order_extra2)
  else:
    print(f'[info] Extra queue file not found; skipping: {args.queue_order_extra2}')
  queue_run_names: list[str] = []
  seen_queue_run_names: set[str] = set()
  for path in queue_paths:
    names = _load_queue_order(path)
    added = 0
    for name in names:
      if name in seen_queue_run_names:
        continue
      seen_queue_run_names.add(name)
      queue_run_names.append(name)
      added += 1
    print(f'[info] Loaded queue file: {path} (lines={len(names)} added_unique={added})')
  # NOTE: The paper run set is a strict subset of the plan-110 matrix.
  excluded_lenepa_variants = {
    'SIGREGT5_L8_PD0',  # SIGReg_time on layer [8], scale 5, pred_depth=0
    'SIGREGT20_L0-8_PD2',  # SIGReg_time on layers [0,8], scale 20, pred_depth=2
  }
  excluded_run_names: list[str] = []
  filtered_run_names: list[str] = []
  for run_name in queue_run_names:
    _dataset, method_family, variant, _projector, _seed = _parse_run_meta_stub(run_name)
    if method_family == 'LENEPA' and variant in excluded_lenepa_variants:
      excluded_run_names.append(run_name)
    else:
      filtered_run_names.append(run_name)

  if excluded_run_names:
    print(
      f'[filter] Excluding {len(excluded_run_names)} run(s) from plots/tables (LeNEPA variants): '
      f'{sorted(excluded_lenepa_variants)}'
    )

  run_names = sorted(filtered_run_names, key=_run_sort_key)
  if not run_names:
    raise ValueError(
      'After filtering, no runs remain to process. '
      f'queue_order={args.queue_order} excluded_variants={sorted(excluded_lenepa_variants)}'
    )

  # Step 2: map run names -> W&B run ids (from VM job logs).
  run_metas: list[RunMeta] = []
  for run_name in run_names:
    run_metas.append(
      _discover_wandb_run_meta(
        run_name,
        logs_dir=args.logs_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
      )
    )

  # Step 2a: discover appendix-only ablation runs (A1/A2/A4; seed0 only).
  annex_run_names = [
    # A1) JEPA masking fragility (keep ratio).
    'PTBXL_JEPA_CLS_KR05-10_s0',
    'PTBXL_JEPA_CLS_KR30-40_s0',
    # A2) NEPA Conv tokenizer sensitivity (patch_size).
    'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_PS10_s0',
    'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_PS50_s0',
    # A4) SIGReg component ablations (single component; scale=20, PROJ only).
    'PTBXL_LENEPA_SIGREGB20_L0-8_PD0_PROJ_s0',
    'PTBXL_LENEPA_SIGREGR20_L8_PD0_PROJ_s0',
    'PTBXL_LENEPA_SIGREGI20_L0-8_PD0_PROJ_s0',
    'PTBXL_LENEPA_SIGREG_BT20_L0-8_PD0_PROJ_s0',
    'AIONO_LENEPA_SIGREGB20_L0-8_PD0_PROJ_s0',
    'AIONO_LENEPA_SIGREGR20_L8_PD0_PROJ_s0',
    'AIONO_LENEPA_SIGREGI20_L0-8_PD0_PROJ_s0',
    'AIONO_LENEPA_SIGREG_BT20_L0-8_PD0_PROJ_s0',
    # A7.1) Balanced basic-components (uniform component sampling).
    'AIONO_JEPA_CLS_BAL_s0',
    'AIONO_JEPA_MEAN_BAL_s0',
    'AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0',
    'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_BAL_s0',
    'AIONO_LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_BAL_s0',
    # A7.2) ECG-like Aiono preset.
    'AIONO_JEPA_CLS_ECG_s0',  # note: known to crash at step 0 offline probe (no offline-probe rows).
    'AIONO_JEPA_MEAN_ECG_s0',
    'AIONO_NEPA_STOPGRAD_MEAN_PROJ_ECG_s0',
    'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_ECG_s0',
    'AIONO_LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_ECG_s0',
    # A8) Target-layer shift ablation (target layer = 1).
    'AIONO_NEPA_STOPGRAD_MEAN_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREGT20_L1-8_PD0_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREG_BRI20_L1-8_PD0_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREGB20_L1-8_PD0_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREGR20_L8_PD0_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREGI20_L1-8_PD0_PROJ_TL1_s0',
    'AIONO_LENEPA_SIGREG_BT20_L1-8_PD0_PROJ_TL1_s0',
  ]
  run_metas_annex: list[RunMeta] = []
  for run_name in annex_run_names:
    run_metas_annex.append(
      _discover_wandb_run_meta(
        run_name,
        logs_dir=args.logs_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
      )
    )

  # Step 2b: main paper outputs exclude LeNEPA "-Proj" runs, but keep both NEPA
  # variants (NEPA baseline vs NEPA +Proj).
  run_metas_paper = [
    m
    for m in run_metas
    if m.projector is None or m.projector is True or m.method_family == 'NEPA_STOPGRAD'
  ]
  if not run_metas_paper:
    raise ValueError('After filtering out LeNEPA "-Proj" runs, no runs remain to plot.')
  if len(run_metas_paper) != len(run_metas):
    print(
      '[filter] Excluding LeNEPA "-Proj" runs from main plots/tables '
      f'(kept {len(run_metas_paper)}/{len(run_metas)}; projector ablation outputs still use full set).'
    )

  # Step 2c: group paper run metas by config+seed (seed stats use the subset of the seed pool that exists per group).
  seed_pool = tuple(range(10))
  config_groups_paper = _group_run_metas_by_config_and_seed(
    run_metas_paper,
    required_datasets=('PTBXL', 'AIONO'),
    seed_pool=seed_pool,
    include_projector_suffix=False,
  )
  missing_seed_coverage: dict[tuple[str, str], list[int]] = {}
  for cfg, by_dataset in config_groups_paper.items():
    for dataset in ('PTBXL', 'AIONO'):
      by_seed = by_dataset.get(dataset)
      if by_seed is None:
        continue
      _present, missing = _seed_pool_present_and_missing(by_seed, seed_pool=seed_pool)
      if missing:
        missing_seed_coverage[(cfg, dataset)] = missing
  if missing_seed_coverage:
    print('[warn] Some (config, dataset) groups are missing seeds from seed_pool=0..9 (will aggregate over available seeds):')
    for (cfg, dataset), missing in sorted(missing_seed_coverage.items()):
      print(f'  - {dataset} {cfg}: missing={missing}')
  else:
    print('[info] Seed coverage: all (config, dataset) groups have full seed_pool=0..9.')

  # Step 2d: discover 5k-horizon counterparts (separate trainings, `*_S5K_*`).
  # NOTE: we keep 5k-vs-20k outputs seed0-only unless S5K seeds exist.
  run_metas_paper_seed0 = [m for m in run_metas_paper if int(m.seed) == 0]
  if not run_metas_paper_seed0:
    raise ValueError('Expected at least one seed0 run in run_metas_paper for 5k-vs-20k outputs.')
  s5k_run_names = sorted({_s5k_run_name_from_run_name(m.run_name) for m in run_metas_paper_seed0}, key=_run_sort_key)
  run_metas_s5k: list[RunMeta] = []
  missing_s5k: list[str] = []
  for s5k_run_name in s5k_run_names:
    try:
      run_metas_s5k.append(
        _discover_wandb_run_meta(
          s5k_run_name,
          logs_dir=args.logs_dir,
          wandb_entity=args.wandb_entity,
          wandb_project=args.wandb_project,
        )
      )
    except FileNotFoundError:
      missing_s5k.append(s5k_run_name)

  if missing_s5k:
    print(f'[filter] Missing {len(missing_s5k)} S5K run(s) (will skip 5k-vs-20k plots/tables for them): {missing_s5k}')

  # Step 3: export W&B histories (offline probes + train step_time).
  api: wandb.Api | None = None

  def _get_api() -> wandb.Api:
    """Create a W&B API handle lazily (only when downloads are required)."""
    nonlocal api
    if api is None:
      api = wandb.Api(timeout=180)
    return api

  export_dir: Path = args.export_dir
  export_dir.mkdir(parents=True, exist_ok=True)

  histories: dict[str, pd.DataFrame] = {}
  step_time_by_run_name: dict[str, TrainStepTimeStats] = {}

  metas_to_export: list[RunMeta] = []
  seen_run_names: set[str] = set()
  for meta in [*run_metas, *run_metas_s5k, *run_metas_annex]:
    if meta.run_name in seen_run_names:
      continue
    seen_run_names.add(meta.run_name)
    metas_to_export.append(meta)

  for meta in metas_to_export:
    out_run_dir = export_dir / meta.wandb_run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_run_dir / 'history.csv.gz'
    step_time_path = out_run_dir / 'train_step_time.csv.gz'
    meta_path = out_run_dir / 'run_meta.json'

    optional_offline_probe = meta in run_metas_annex
    if optional_offline_probe and args.skip_existing and not history_path.is_file():
      print(
        '[warn] Skipping offline-probe export for optional appendix run (missing existing export in --skip-existing). '
        f'run={meta.run_name!r} expected_file={history_path}'
      )
      continue
    if args.no_download and not history_path.is_file():
      print(
        '[warn] Missing offline-probe export (skipping run in --no-download). '
        f'run={meta.run_name!r} expected_file={history_path}'
      )
      continue
    try:
      if args.skip_existing and history_path.is_file():
        print(f'[skip] Using existing export: {meta.wandb_run_id} ({meta.run_name})')
        df = pd.read_csv(history_path, compression='gzip')
      else:
        df = _download_offline_probe_history(
          _get_api(), run_meta=meta, layers=layers, max_step=args.max_step, page_size=args.page_size
        )
        df.to_csv(history_path, index=False, compression='gzip')
      histories[meta.run_name] = df
    except Exception as e:  # noqa: BLE001
      if optional_offline_probe:
        print(
          '[warn] Skipping offline-probe export for optional appendix run (no usable offline-probe rows). '
          f'run={meta.run_name!r} err={type(e).__name__}: {e}'
        )
        continue
      raise

    if args.no_download and not step_time_path.is_file():
      print(
        '[warn] Missing train/step_time export (skipping step_time in --no-download). '
        f'run={meta.run_name!r} expected_file={step_time_path}'
      )
      continue
    if args.skip_existing and step_time_path.is_file():
      df_step_time = pd.read_csv(step_time_path, compression='gzip')
    else:
      if optional_offline_probe and args.skip_existing:
        print(
          '[warn] Skipping train/step_time export for optional appendix run (missing existing export in --skip-existing). '
          f'run={meta.run_name!r} expected_file={step_time_path}'
        )
        continue
      df_step_time = _download_train_step_time_history(
        _get_api(),
        run_meta=meta,
        min_step=1000,
        max_step=args.max_step,
        page_size=args.page_size,
      )
      df_step_time.to_csv(step_time_path, index=False, compression='gzip')

    step_times = df_step_time['train/step_time'].to_numpy(dtype=float)
    median_s = float(np.median(step_times))
    if not np.isfinite(median_s):
      raise ValueError(f'Computed non-finite median train/step_time for run={meta.run_name!r}: {median_s}')
    step_time_by_run_name[meta.run_name] = TrainStepTimeStats(
      min_step=1000,
      max_step=int(args.max_step),
      median_s=median_s,
      n=int(step_times.size),
    )

    meta_payload = {
      'run_name': meta.run_name,
      'dataset': meta.dataset,
      'method_family': meta.method_family,
      'variant': meta.variant,
      'projector': meta.projector,
      'seed': meta.seed,
      'wandb_entity': meta.wandb_entity,
      'wandb_project': meta.wandb_project,
      'wandb_run_id': meta.wandb_run_id,
      'wandb_path': meta.wandb_path,
      'job_log_path': str(meta.job_log_path),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding='utf-8')

  run_metas_available = [m for m in run_metas if m.run_name in histories]
  if len(run_metas_available) != len(run_metas):
    missing = sorted({m.run_name for m in run_metas if m.run_name not in histories})
    print(f'[warn] Missing offline-probe exports for {len(missing)} main-matrix run(s); will ignore them.')

  # Step 4: generate figures (paper output dir).
  fig_dir: Path = args.fig_dir
  metrics_ptbxl = _metric_specs_for_dataset('PTBXL')
  metrics_aiono = _metric_specs_for_dataset('AIONO')

  run_metas_paper_available = [m for m in run_metas_paper if m.run_name in histories]
  if not run_metas_paper_available:
    raise ValueError('No paper runs have offline-probe exports; cannot generate figures/tables.')
  if len(run_metas_paper_available) != len(run_metas_paper):
    missing = sorted({m.run_name for m in run_metas_paper if m.run_name not in histories})
    print(
      '[warn] Missing offline-probe exports for some paper runs; seed-aggregated outputs will use only available seeds '
      f'(missing_runs={len(missing)}).'
    )

  config_groups_paper = _group_run_metas_by_config_and_seed(
    run_metas_paper_available,
    required_datasets=('PTBXL', 'AIONO'),
    seed_pool=seed_pool,
    include_projector_suffix=False,
  )
  run_metas_paper_seed0 = [m for m in run_metas_paper_available if int(m.seed) == 0]
  run_metas_s5k = [m for m in run_metas_s5k if m.run_name in histories]

  # Paper decision: remove last-step heatmaps (hard to read); keep line plots + tables.
  obsolete_figures = [
    fig_dir / 'ptbxl_last_step_by_layer_classification.png',
    fig_dir / 'ptbxl_last_step_by_layer_classification.pdf',
    fig_dir / 'aiono_last_step_by_layer_classification.png',
    fig_dir / 'aiono_last_step_by_layer_classification.pdf',
    fig_dir / 'aiono_last_step_by_layer_dense.png',
    fig_dir / 'aiono_last_step_by_layer_dense.pdf',
    # Merge Aionoscope classification+dense by-layer line plots into one figure.
    fig_dir / 'aiono_last_step_by_layer_classification_lines.png',
    fig_dir / 'aiono_last_step_by_layer_classification_lines.pdf',
    fig_dir / 'aiono_last_step_by_layer_dense_lines.png',
    fig_dir / 'aiono_last_step_by_layer_dense_lines.pdf',
    # Merge PTB-XL + Aionoscope by-layer line plots into one figure.
    fig_dir / 'ptbxl_last_step_by_layer_classification_lines.png',
    fig_dir / 'ptbxl_last_step_by_layer_classification_lines.pdf',
    fig_dir / 'aiono_last_step_by_layer_lines.png',
    fig_dir / 'aiono_last_step_by_layer_lines.pdf',
    # Transfer dynamics: replace split dataset plots with a single side-by-side figure.
    fig_dir / 'ptbxl_dynamics_best_layer_classification.png',
    fig_dir / 'ptbxl_dynamics_best_layer_classification.pdf',
    fig_dir / 'aiono_dynamics_best_layer_classification.png',
    fig_dir / 'aiono_dynamics_best_layer_classification.pdf',
    # Remove old backward-compat outputs for the summary bar charts.
    fig_dir / 'summary_selected_best_last_step_classification_bars.png',
    fig_dir / 'summary_selected_best_last_step_classification_bars.pdf',
    fig_dir / 'summary_selected_best_last_step_dense_bars.png',
    fig_dir / 'summary_selected_best_last_step_dense_bars.pdf',
    # Remove old split-dataset summary charts (superseded by one combined figure).
    fig_dir / 'summary_selected_best_last_step_ptbxl_bars.png',
    fig_dir / 'summary_selected_best_last_step_ptbxl_bars.pdf',
    fig_dir / 'summary_selected_best_last_step_aiono_bars.png',
    fig_dir / 'summary_selected_best_last_step_aiono_bars.pdf',
    # Merge 5k-vs-20k dynamics into a single stacked PTB-XL + Aionoscope figure.
    fig_dir / 'ptbxl_dynamics_best_layer_classification_5k.png',
    fig_dir / 'ptbxl_dynamics_best_layer_classification_5k.pdf',
    fig_dir / 'aiono_dynamics_best_layer_classification_5k.png',
    fig_dir / 'aiono_dynamics_best_layer_classification_5k.pdf',
    fig_dir / 'aiono_dynamics_best_layer_dense_5k.png',
    fig_dir / 'aiono_dynamics_best_layer_dense_5k.pdf',
  ]
  for path in obsolete_figures:
    if path.is_file():
      path.unlink()
      print(f'[clean] Removed obsolete figure: {path}')

  _plot_transfer_best_layer_dynamics_classification(
    config_groups=config_groups_paper,
    histories=histories,
    metrics=metrics_ptbxl,
    layers=layers,
    seed_pool=seed_pool,
    out_dir=fig_dir,
  )
  if run_metas_s5k:
    _plot_best_layer_dynamics_5k_vs_20k_combined(
      runs_20k=run_metas_paper_seed0,
      runs_5k=run_metas_s5k,
      histories=histories,
      layers=layers,
      include_projector_suffix=False,
      out_dir=fig_dir,
    )
  else:
    print('[warn] No S5K offline-probe exports; skipping dynamics_5k_combined figure.')
  _plot_best_layer_dynamics(
    config_groups=config_groups_paper,
    histories=histories,
    dataset='AIONO',
    metrics=metrics_aiono,
    layers=layers,
    seed_pool=seed_pool,
    plot_classification=False,
    plot_dense=True,
    out_dir=fig_dir,
  )

  _plot_last_step_layer_lineplots_ptbxl_aiono_combined(
    config_groups=config_groups_paper,
    histories=histories,
    layers=layers,
    seed_pool=seed_pool,
    out_dir=fig_dir,
  )

  # Step 4b: appendix-only PTB-XL ablations (seed0 only; no error bars).
  a1_runs = ['PTBXL_JEPA_CLS_s0', 'PTBXL_JEPA_CLS_KR05-10_s0', 'PTBXL_JEPA_CLS_KR30-40_s0']
  a1_labels = [
    f'KR {_a1_jepa_keep_ratio_range(a1_runs[0])[0]:.2f}–{_a1_jepa_keep_ratio_range(a1_runs[0])[1]:.2f} (baseline)',
    f'KR {_a1_jepa_keep_ratio_range(a1_runs[1])[0]:.2f}–{_a1_jepa_keep_ratio_range(a1_runs[1])[1]:.2f}',
    f'KR {_a1_jepa_keep_ratio_range(a1_runs[2])[0]:.2f}–{_a1_jepa_keep_ratio_range(a1_runs[2])[1]:.2f}',
  ]
  _plot_ptbxl_single_seed_best_layer_dynamics_classification(
    histories=histories,
    run_names=a1_runs,
    legend_labels=a1_labels,
    layers=layers,
    out_dir=fig_dir,
    out_basename='ptbxl_jepa_masking_fragility_dynamics',
  )

  a2_runs = [
    'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_s0',
    'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_PS10_s0',
    'PTBXL_NEPA_STOPGRAD_MEAN_PROJ_PS50_s0',
  ]
  a2_labels = [
    f'P={_a2_nepa_patch_size(a2_runs[0])} (baseline)',
    f'P={_a2_nepa_patch_size(a2_runs[1])}',
    f'P={_a2_nepa_patch_size(a2_runs[2])}',
  ]
  _plot_ptbxl_single_seed_best_layer_dynamics_classification(
    histories=histories,
    run_names=a2_runs,
    legend_labels=a2_labels,
    layers=layers,
    out_dir=fig_dir,
    out_basename='ptbxl_nepa_patch_size_sensitivity_dynamics',
  )

  _plot_sigreg_component_ablations_dynamics(
    histories=histories,
    layers=layers,
    out_dir=fig_dir,
    out_basename='sigreg_component_ablations_dynamics',
  )

  _plot_aiono_single_seed_experiment_dynamics(
    histories=histories,
    run_names=[
      'AIONO_JEPA_CLS_BAL_s0',
      'AIONO_JEPA_MEAN_BAL_s0',
      'AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0',
      'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_BAL_s0',
      'AIONO_LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_BAL_s0',
    ],
    layers=layers,
    experiment_label='Aionoscope: basic-components (balanced sampling)',
    include_projector_suffix=False,
    out_dir=fig_dir,
    out_basename='aiono_balanced_basic_components_dynamics',
  )
  _plot_aiono_single_seed_experiment_dynamics(
    histories=histories,
    run_names=[
      'AIONO_JEPA_CLS_ECG_s0',
      'AIONO_JEPA_MEAN_ECG_s0',
      'AIONO_NEPA_STOPGRAD_MEAN_PROJ_ECG_s0',
      'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_ECG_s0',
      'AIONO_LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_ECG_s0',
    ],
    layers=layers,
    experiment_label='Aionoscope: Aiono (ECG-like)',
    include_projector_suffix=False,
    out_dir=fig_dir,
    out_basename='aiono_aiono_ecg_like_dynamics',
  )
  _plot_aiono_single_seed_experiment_dynamics(
    histories=histories,
    run_names=[
      'AIONO_NEPA_STOPGRAD_MEAN_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREGT20_L1-8_PD0_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREG_BRI20_L1-8_PD0_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREGB20_L1-8_PD0_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREGR20_L8_PD0_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREGI20_L1-8_PD0_PROJ_TL1_s0',
      'AIONO_LENEPA_SIGREG_BT20_L1-8_PD0_PROJ_TL1_s0',
    ],
    layers=layers,
    experiment_label='Aionoscope: target-layer shift (target layer = 1)',
    include_projector_suffix=False,
    out_dir=fig_dir,
    out_basename='aiono_target_layer_shift_tl1_dynamics',
  )

  # Step 5: export paper tables.
  csv_dir: Path = args.tables_csv_dir
  tex_dir: Path = args.tables_tex_dir
  tex_dir.mkdir(parents=True, exist_ok=True)

  _export_transfer_best_layer_dynamics_tables(
    config_groups=config_groups_paper,
    histories=histories,
    layers=layers,
    steps=[0, 1000, 2000, 5000, 10000, 15000, 20000],
    seed_pool=seed_pool,
    csv_dir=csv_dir,
    tex_dir=tex_dir,
  )

  if run_metas_s5k:
    _export_steps_5k_vs_20k_best_layer_table(
      runs_20k=run_metas_paper_seed0,
      runs_5k=run_metas_s5k,
      histories=histories,
      dataset='PTBXL',
      metrics=[m for m in metrics_ptbxl if m.name in ('auroc', 'auprc')],
      layers=layers,
      steps=(5000, 20000),
      include_projector_suffix=False,
      digits=3,
      csv_dir=csv_dir,
      tex_dir=tex_dir,
      out_basename='ptbxl_steps_5k_vs_20k_best_layer_classification',
      caption=(
        'PTB-XL: best-layer probe metrics at step=5000 vs step=20000 '
        '(oracle over layers 0--8; best layer chosen separately for each metric and step).'
      ),
      label='tab:ptbxl_steps_5k_vs_20k_best_layer_classification',
    )
    _export_steps_5k_vs_20k_best_layer_table(
      runs_20k=run_metas_paper_seed0,
      runs_5k=run_metas_s5k,
      histories=histories,
      dataset='AIONO',
      metrics=[m for m in metrics_aiono if m.name in ('auroc', 'auprc')],
      layers=layers,
      steps=(5000, 20000),
      include_projector_suffix=False,
      digits=3,
      csv_dir=csv_dir,
      tex_dir=tex_dir,
      out_basename='aiono_steps_5k_vs_20k_best_layer_classification',
      caption=(
        'Aionoscope (\\emph{basic-components}): best-layer classification probe metrics at step=5000 vs step=20000 '
        '(oracle over layers 0--8; best layer chosen separately for each metric and step).'
      ),
      label='tab:aiono_steps_5k_vs_20k_best_layer_classification',
    )
    _export_steps_5k_vs_20k_best_layer_table(
      runs_20k=run_metas_paper_seed0,
      runs_5k=run_metas_s5k,
      histories=histories,
      dataset='AIONO',
      metrics=[m for m in metrics_aiono if m.name in ('mse', 'mae', 'pearson', 'r2')],
      layers=layers,
      steps=(5000, 20000),
      include_projector_suffix=False,
      digits=3,
      csv_dir=csv_dir,
      tex_dir=tex_dir,
      out_basename='aiono_steps_5k_vs_20k_best_layer_dense',
      caption=(
        'Aionoscope (\\emph{basic-components}): best-layer dense probe metrics at step=5000 vs step=20000 '
        '(oracle over layers 0--8; best layer chosen separately for each metric and step).'
      ),
      label='tab:aiono_steps_5k_vs_20k_best_layer_dense',
    )
  else:
    print('[warn] No S5K offline-probe exports; skipping 5k-vs-20k tables.')

  digits_by_metric = {
    'auroc': 3,
    'auprc': 3,
    'mse': 3,
    'mae': 3,
    'pearson': 3,
    'r2': 3,
  }
  _export_last_step_by_layer_tables(
    config_groups=config_groups_paper,
    histories=histories,
    dataset='PTBXL',
    metrics=metrics_ptbxl,
    layers=layers,
    seed_pool=seed_pool,
    near_frac=0.90,
    digits_by_metric=digits_by_metric,
    csv_dir=csv_dir,
    tex_dir=tex_dir,
  )
  _export_last_step_by_layer_tables(
    config_groups=config_groups_paper,
    histories=histories,
    dataset='AIONO',
    metrics=metrics_aiono,
    layers=layers,
    seed_pool=seed_pool,
    near_frac=0.90,
    digits_by_metric=digits_by_metric,
    csv_dir=csv_dir,
    tex_dir=tex_dir,
  )

  # Step 5a: export Aionoscope dense targets (per target, oracle over layers) for blind-spot analysis.
  dense_target_run_names = [
    'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_s0',
    'AIONO_NEPA_STOPGRAD_MEAN_PROJ_s0',
    'AIONO_JEPA_CLS_s0',
  ]
  targets_config_path = _repo_root() / 'configs/pretrain/dataset/aiono_basic_components_balanced.yaml'
  for run_name in dense_target_run_names:
    meta = next((m for m in run_metas_paper_seed0 if m.run_name == run_name), None)
    if meta is None:
      available = sorted({m.run_name for m in run_metas_paper_seed0 if m.dataset == 'AIONO'})
      raise ValueError(
        'Missing required run meta for Aionoscope dense-target export. '
        f'expected={run_name!r} available_aiono_seed0={available}'
      )
    dense_targets_out = csv_dir / f'{meta.run_name.lower()}_dense_targets_step0_vs_20k.csv'
    if args.skip_existing and dense_targets_out.is_file():
      print(f'[skip] Using existing dense-target table: {dense_targets_out}')
      continue
    if args.no_download:
      print(
        '[warn] Missing dense-target table export in --no-download; skipping. '
        f'run={meta.run_name!r} expected_file={dense_targets_out}'
      )
      continue
    _export_aiono_dense_targets_step0_vs_20k_table(
      _get_api(),
      run_meta_step0=meta,
      run_meta_step1=meta,
      layers=layers,
      steps=(0, 20000),
      targets_config_path=targets_config_path,
      page_size=int(args.page_size),
      digits=3,
      out_csv_path=dense_targets_out,
    )

  # Step 5b: export Aionoscope categorical signals (per signal, oracle over layers) for blind-spot analysis.
  categorical_signal_run_names = list(dense_target_run_names)
  for run_name in categorical_signal_run_names:
    meta = next((m for m in run_metas_paper_seed0 if m.run_name == run_name), None)
    if meta is None:
      available = sorted({m.run_name for m in run_metas_paper_seed0 if m.dataset == 'AIONO'})
      raise ValueError(
        'Missing required run meta for Aionoscope categorical-per-signal export. '
        f'expected={run_name!r} available_aiono_seed0={available}'
      )
    cat_signals_out = csv_dir / f'{meta.run_name.lower()}_categorical_signals_step0_vs_20k.csv'
    if args.skip_existing and cat_signals_out.is_file():
      print(f'[skip] Using existing categorical-per-signal table: {cat_signals_out}')
      continue
    if args.no_download:
      print(
        '[warn] Missing categorical-per-signal table export in --no-download; skipping. '
        f'run={meta.run_name!r} expected_file={cat_signals_out}'
      )
      continue
    _export_aiono_categorical_signals_step0_vs_20k_table(
      _get_api(),
      run_meta_step0=meta,
      run_meta_step1=meta,
      layers=layers,
      steps=(0, 20000),
      signals_config_path=targets_config_path,
      page_size=int(args.page_size),
      digits=3,
      out_csv_path=cat_signals_out,
    )

  # Step 6: projector ablation (uses full PROJ+NOPROJ set).
  config_groups_full = _group_run_metas_by_config_and_seed(
    run_metas_available,
    required_datasets=('PTBXL', 'AIONO'),
    seed_pool=seed_pool,
    include_projector_suffix=True,
  )
  projector_pairs = _discover_projector_ablation_pairs(config_groups_full)
  _plot_projector_ablation_bar_charts(
    pairs=projector_pairs,
    config_groups=config_groups_full,
    histories=histories,
    layers=layers,
    seed_pool=seed_pool,
    out_dir=fig_dir,
  )
  _export_projector_ablation_table(
    pairs=projector_pairs,
    config_groups=config_groups_full,
    histories=histories,
    layers=layers,
    seed_pool=seed_pool,
    out_csv_path=csv_dir / 'projector_ablation_last_step.csv',
    out_tex_path=tex_dir / 'projector_ablation_last_step.tex',
  )

  # Step 7: select best config per family and export summary bars + table (paper set, PROJ-only).
  last_step_best_paper = _compute_last_step_best_layer_seed_stats(
    config_groups_paper,
    histories=histories,
    layers=layers,
    seed_pool=seed_pool,
  )
  delta_last_minus_step0_paper = _compute_best_layer_delta_last_minus_step_seed_stats(
    config_groups_paper,
    histories=histories,
    layers=layers,
    step=0,
    seed_pool=seed_pool,
  )
  _selected_by_family, df_selection_scores = _select_best_config_per_family(
    config_groups_paper, last_step_best=last_step_best_paper
  )

  df_selection_scores.to_csv(csv_dir / 'selection_scores.csv', index=False)
  _export_train_step_time_table(
    config_groups=config_groups_paper,
    step_time_by_run_name=step_time_by_run_name,
    seed_pool=seed_pool,
    out_csv_path=csv_dir / 'train_step_time_median.csv',
    out_tex_path=tex_dir / 'train_step_time_median.tex',
  )
  _plot_summary_last_step_bar_charts(
    config_groups=config_groups_paper,
    last_step_best=last_step_best_paper,
    seed_pool=seed_pool,
    out_dir=fig_dir,
  )
  _plot_summary_last_step_delta_from_step0_bar_charts(
    config_groups=config_groups_paper,
    delta_last_minus_step0=delta_last_minus_step0_paper,
    seed_pool=seed_pool,
    out_dir=fig_dir,
  )
  _export_summary_last_step_table(
    config_groups=config_groups_paper,
    last_step_best=last_step_best_paper,
    delta_last_minus_step0=delta_last_minus_step0_paper,
    seed_pool=seed_pool,
    out_csv_path=csv_dir / 'summary_selected_best_last_step.csv',
    out_tex_path=tex_dir / 'summary_selected_best_last_step.tex',
    out_tex_delta_path=tex_dir / 'summary_selected_best_last_step_delta.tex',
  )

  print(f'[done] Wrote figures to: {fig_dir}')
  print(f'[done] Wrote W&B exports to: {export_dir}')
  print(f'[done] Wrote tables (tex) to: {tex_dir}')
  print(f'[done] Wrote tables (csv) to: {csv_dir}')


if __name__ == '__main__':
  main()
