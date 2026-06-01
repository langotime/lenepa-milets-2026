"""Compute "time-to-%-improvement" for LeNEPA paper runs from exported W&B histories.

This utility follows the exact "percent-of-improvement from step 0 to step 20,000"
definition used in `utils/wandb_lenepa_paper_plots.py`:

For a metric to *maximize* (AUROC/AUPRC here), for each seed independently:

  pct(step) = 100 * (v(step) - v(0)) / (v(20k) - v(0)),

where v(step) is the **best-layer (oracle over layers 0..8)** offline-probe value
at that step. We then clamp pct(0)=0 and pct(20k)=100.

For each (model, dataset, metric, threshold), we compute the earliest offline-probe
eval step where pct(step) >= threshold, per seed, and then aggregate those steps
as `median ± std` across seeds (sample std, ddof=1).

The script is **offline**: it reads `history.csv.gz` exports produced by
`wandb_lenepa_paper_plots.py` under `results/wandb_export/<run_id>/`.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python -m utils.wandb_lenepa_improvement_thresholds
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExportedRun:
  """One exported W&B run (history + metadata) on disk."""

  run_name: str
  dataset: str
  seed: int
  wandb_run_id: str
  run_dir: Path

  @property
  def history_path(self) -> Path:
    """Return `history.csv.gz` path for this run."""
    return self.run_dir / 'history.csv.gz'


def _repo_root() -> Path:
  """Return the repo root path regardless of current working directory."""
  return Path(__file__).resolve().parents[1]


def _dedupe_wandb_history(df: pd.DataFrame) -> pd.DataFrame:
  """Collapse W&B history rows into one row per `_step`.

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


def _load_exported_runs(export_dir: Path) -> dict[str, ExportedRun]:
  """Index exported runs by `run_name` from `<export_dir>/<run_id>/run_meta.json`."""
  if not export_dir.is_dir():
    raise FileNotFoundError(f'Export dir not found: {export_dir}')

  # Step 1: scan all run subdirectories for run_meta.json.
  runs: dict[str, ExportedRun] = {}
  for run_dir in sorted(export_dir.iterdir()):
    if not run_dir.is_dir():
      continue
    meta_path = run_dir / 'run_meta.json'
    if not meta_path.is_file():
      continue

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    run_name = str(meta.get('run_name', '')).strip()
    if not run_name:
      raise ValueError(f'Invalid run_meta.json (missing run_name): {meta_path}')

    if run_name in runs:
      raise ValueError(f'Duplicate exported run_name={run_name!r} under export_dir={export_dir}')

    dataset = str(meta.get('dataset', '')).strip()
    if not dataset:
      raise ValueError(f'Invalid run_meta.json (missing dataset): {meta_path}')

    seed = meta.get('seed')
    if not isinstance(seed, int):
      raise ValueError(f'Invalid run_meta.json (seed must be int): {meta_path}')

    wandb_run_id = str(meta.get('wandb_run_id', '')).strip()
    if not wandb_run_id:
      raise ValueError(f'Invalid run_meta.json (missing wandb_run_id): {meta_path}')

    runs[run_name] = ExportedRun(
      run_name=run_name,
      dataset=dataset,
      seed=int(seed),
      wandb_run_id=wandb_run_id,
      run_dir=run_dir,
    )

  if not runs:
    raise ValueError(f'No exported runs found under: {export_dir}')

  return runs


def _metric_key_template(*, metric: str) -> str:
  """Return W&B history key template for a metric."""
  if metric == 'AUROC':
    return 'offline_probe/layer_{layer}/best_macro_auc'
  if metric == 'AUPRC':
    return 'offline_probe/layer_{layer}/best_macro_auprc'
  raise ValueError(f'Unexpected metric: {metric!r}')


def _best_layer_values_by_step(
  df_history: pd.DataFrame,
  *,
  metric_key_template: str,
  layers: list[int],
) -> dict[int, float]:
  """Compute best-layer (oracle) metric value per eval step.

  Returns:
    Dict mapping `step -> best_value`.
  """
  if not layers:
    raise ValueError('layers must be non-empty')

  # Step 1: select required columns and normalize `_step` to int.
  cols = [metric_key_template.format(layer=int(layer)) for layer in layers]
  missing = [c for c in cols if c not in df_history.columns]
  if missing:
    raise KeyError(f'Missing required metric columns (example): {missing[:3]}...')

  df = df_history[['_step', *cols]].copy()
  df['_step'] = pd.to_numeric(df['_step'], errors='coerce')
  df = df.dropna(subset=['_step']).copy()
  df['_step'] = df['_step'].astype(int)

  # Step 2: coerce per-layer metric values to float and drop all-NaN rows.
  values = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)  # [n_steps, n_layers]
  mask_all_nan = np.all(~np.isfinite(values), axis=1)
  if np.any(mask_all_nan):
    df = df.loc[~mask_all_nan].reset_index(drop=True)
    values = values[~mask_all_nan]
  if values.size == 0:
    raise RuntimeError('All rows are NaN for the requested metric/layers.')

  # Step 3: row-wise best layer (nanargmax ignores NaNs).
  best_idx = np.nanargmax(values, axis=1)
  best_values = values[np.arange(values.shape[0]), best_idx].astype(float)

  steps = df['_step'].to_numpy(dtype=int)
  if len(set(steps.tolist())) != len(steps):
    raise ValueError('Duplicate steps found after history dedupe; expected unique `_step` values.')

  out = {int(step): float(val) for step, val in zip(steps.tolist(), best_values.tolist(), strict=True)}
  if not out:
    raise RuntimeError('Internal error: empty best-layer output.')
  return out


def _percent_of_improvement_by_step(
  best_by_step: dict[int, float],
  *,
  final_step: int,
) -> dict[int, float]:
  """Compute percent-of-improvement series (max-metric convention)."""
  if 0 not in best_by_step:
    raise KeyError('Missing step 0 in best-layer series (required for percent-of-improvement).')
  if int(final_step) not in best_by_step:
    raise KeyError(f'Missing final_step={int(final_step)} in best-layer series (required for percent-of-improvement).')

  # Step 1: compute direction-aware denominator for (v_final - v0).
  v0 = float(best_by_step[0])
  v_final = float(best_by_step[int(final_step)])
  if not np.isfinite(v0):
    raise ValueError(f'Non-finite v0: {v0}')
  if not np.isfinite(v_final):
    raise ValueError(f'Non-finite v_final: {v_final}')
  denom = float(v_final) - float(v0)
  if float(denom) == 0.0:
    raise ValueError(f'No net change between step 0 and step {int(final_step)} (denom=0). v0={v0} v_final={v_final}')

  # Step 2: compute pct series and clamp endpoints to 0/100.
  pct_by_step: dict[int, float] = {}
  for step, v in best_by_step.items():
    vv = float(v)
    if not np.isfinite(vv):
      raise ValueError(f'Non-finite value at step={int(step)}: {vv}')
    pct_by_step[int(step)] = 100.0 * (float(vv) - float(v0)) / float(denom)
  pct_by_step[0] = 0.0
  pct_by_step[int(final_step)] = 100.0
  return pct_by_step


def _first_step_at_or_above_threshold(pct_by_step: dict[int, float], *, threshold_pct: float) -> int:
  """Return the earliest step where `pct(step) >= threshold_pct`."""
  if not pct_by_step:
    raise ValueError('pct_by_step must be non-empty')
  if not np.isfinite(float(threshold_pct)):
    raise ValueError(f'threshold_pct must be finite, got: {threshold_pct}')

  # Step 1: scan steps in ascending order and pick first threshold crossing.
  for step in sorted(pct_by_step.keys()):
    if float(pct_by_step[int(step)]) >= float(threshold_pct):
      return int(step)
  raise ValueError(f'Threshold {float(threshold_pct)} was never reached (unexpected; final step is clamped to 100%).')


def _seed_stats(values: list[int]) -> tuple[float, float, int]:
  """Return (median, sample_std(ddof=1), n) for a list of ints."""
  if not values:
    raise ValueError('values must be non-empty')
  arr = np.asarray(values, dtype=float)
  if not np.all(np.isfinite(arr)):
    raise ValueError(f'Non-finite values in seed stats: {values}')
  if int(arr.size) < 2:
    raise ValueError('Need at least 2 seeds to compute sample std (ddof=1).')
  return float(np.median(arr)), float(np.std(arr, ddof=1)), int(arr.size)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--export-dir',
    type=Path,
    default=_repo_root() / 'results/wandb_export',
    help='Directory with exported W&B histories (default: results/wandb_export).',
  )
  parser.add_argument(
    '--out-csv',
    type=Path,
    default=_repo_root() / 'results/tables/time_to_pct_improvement.csv',
    help='Output CSV path (default: results/tables/time_to_pct_improvement.csv).',
  )
  parser.add_argument(
    '--final-step',
    type=int,
    default=20000,
    help='Final training step used as 100% improvement (default: 20000).',
  )
  parser.add_argument(
    '--eval-interval',
    type=int,
    default=1000,
    help='Offline-probe eval interval (default: 1000). Used for step-grid validation.',
  )
  parser.add_argument(
    '--layers',
    type=int,
    nargs='+',
    default=list(range(9)),
    help='Layers to consider for oracle best-layer (default: 0..8).',
  )
  parser.add_argument(
    '--seeds',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4],
    help='Seeds to aggregate (default: 0 1 2 3 4).',
  )
  parser.add_argument(
    '--thresholds',
    type=float,
    nargs='+',
    default=[80.0, 85.0, 90.0, 95.0, 98.0],
    help='Percent-of-improvement thresholds (default: 80 85 90 95 98).',
  )
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = _parse_args()
  export_dir: Path = args.export_dir
  out_csv: Path = args.out_csv

  layers = [int(x) for x in args.layers]
  if not layers:
    raise ValueError('layers must be non-empty')
  if len(set(layers)) != len(layers):
    raise ValueError(f'layers must not contain duplicates, got: {layers}')
  if any(int(layer) < 0 for layer in layers):
    raise ValueError(f'layers must be >= 0, got: {layers}')

  seeds = [int(s) for s in args.seeds]
  if not seeds:
    raise ValueError('seeds must be non-empty')
  if len(set(seeds)) != len(seeds):
    raise ValueError(f'seeds must not contain duplicates, got: {seeds}')

  thresholds = [float(t) for t in args.thresholds]
  if not thresholds:
    raise ValueError('thresholds must be non-empty')

  final_step = int(args.final_step)
  if final_step <= 0:
    raise ValueError(f'final_step must be > 0, got: {final_step}')
  eval_interval = int(args.eval_interval)
  if eval_interval <= 0:
    raise ValueError(f'eval_interval must be > 0, got: {eval_interval}')

  # Step 1: index all exported runs by run_name.
  exported = _load_exported_runs(export_dir)

  # Step 2: define the exact paper run-name templates requested by the user.
  methods: list[tuple[str, str]] = [
    ('LeNEPA T20', '{dataset}_LENEPA_SIGREGT20_L0-8_PD0_PROJ_s{seed}'),
    ('JEPA CLS', '{dataset}_JEPA_CLS_s{seed}'),
    ('JEPA Mean', '{dataset}_JEPA_MEAN_s{seed}'),
  ]
  datasets: list[tuple[str, str]] = [
    ('PTBXL', 'PTB-XL'),
    ('AIONO', 'Aionoscope'),
  ]
  metrics = ['AUROC', 'AUPRC']

  # Step 3: compute per-seed "time-to-threshold" and aggregate across seeds.
  records: list[dict[str, object]] = []
  expected_steps = set(range(0, int(final_step) + 1, int(eval_interval)))

  for dataset_id, dataset_pretty in datasets:
    for model_label, run_template in methods:
      for metric in metrics:
        key_template = _metric_key_template(metric=metric)

        per_seed_threshold_steps: dict[float, list[int]] = {float(t): [] for t in thresholds}
        per_seed_values: dict[int, dict[float, int]] = {}

        for seed in seeds:
          run_name = run_template.format(dataset=dataset_id, seed=int(seed))
          run = exported.get(run_name)
          if run is None:
            raise KeyError(f'Missing exported run for run_name={run_name!r} under export_dir={export_dir}')
          if not run.history_path.is_file():
            raise FileNotFoundError(f'Missing history export for run_name={run_name!r}: {run.history_path}')

          df = pd.read_csv(run.history_path, compression='gzip')
          df = _dedupe_wandb_history(df)

          best_by_step = _best_layer_values_by_step(df, metric_key_template=key_template, layers=layers)
          pct_by_step = _percent_of_improvement_by_step(best_by_step, final_step=final_step)

          available_steps = set(best_by_step.keys())
          missing_steps = sorted(expected_steps - available_steps)
          if missing_steps:
            raise KeyError(
              'Missing offline-probe eval steps in best-layer series. '
              f'run_name={run_name!r} metric={metric} missing_steps={missing_steps[:10]}...'
            )

          per_seed_values[int(seed)] = {}
          for thr in thresholds:
            step_thr = _first_step_at_or_above_threshold(pct_by_step, threshold_pct=float(thr))
            per_seed_threshold_steps[float(thr)].append(int(step_thr))
            per_seed_values[int(seed)][float(thr)] = int(step_thr)

        for thr in thresholds:
          median_step, std_step, n = _seed_stats(per_seed_threshold_steps[float(thr)])
          record: dict[str, object] = {
            'model': model_label,
            'dataset': dataset_pretty,
            'metric': metric,
            'threshold_pct': float(thr),
            'step_median': int(median_step),
            'step_std': float(std_step),
            'n_seeds': int(n),
          }
          for seed in seeds:
            record[f'seed_{int(seed)}_step'] = int(per_seed_values[int(seed)][float(thr)])
          records.append(record)

  df_out = pd.DataFrame.from_records(records)
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  df_out.to_csv(out_csv, index=False)
  print(f'[done] Wrote CSV: {out_csv}')


if __name__ == '__main__':
  main()
