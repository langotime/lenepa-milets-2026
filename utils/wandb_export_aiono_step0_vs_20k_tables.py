"""Export Aionoscope step0-vs-step1 per-target/per-signal probe tables from W&B.

This script is a small, focused wrapper around the export helpers in
`utils.wandb_lenepa_paper_plots`. It exists to support additional
run groups (e.g. Plan-110 A7 `*_BAL_s0`) without re-running the full paper
plot pipeline.

Outputs (one pair per run, written to `--csv-dir`):
  - `<run_name_lower>_dense_targets_step0_vs_<step1>.csv` (suffix uses `k` for multiples of 1000)
  - `<run_name_lower>_categorical_signals_step0_vs_<step1>.csv` (same suffix rule)

These CSVs are consumed by the radar plotting scripts:
  - `utils.plot_aiono_dense_radar`
  - `utils.plot_aiono_categorical_signal_radar`
  - `utils.plot_aiono_combined_radar_grid`

Usage (example):
  uv run python -m utils.wandb_export_aiono_step0_vs_20k_tables \\
    --run-name AIONO_JEPA_CLS_BAL_s0 AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0 \\
    --step1 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from utils.wandb_lenepa_paper_plots import (
  RunMeta,
  _discover_wandb_run_meta,
  _export_aiono_categorical_signals_step0_vs_20k_table,
  _export_aiono_dense_targets_step0_vs_20k_table,
  _format_step_suffix,
  _parse_run_meta_stub,
  _repo_root,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--run-name',
    type=str,
    nargs='+',
    required=True,
    help='W&B run name(s) to export (e.g. AIONO_JEPA_CLS_BAL_s0).',
  )
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
    '--wandb-run-id',
    type=str,
    nargs='+',
    default=None,
    help=(
      'Optional explicit W&B run id(s), one per --run-name (used for the step1 snapshot). When provided, skips '
      '--logs-dir discovery (useful for local runs without VM job logs).'
    ),
  )
  parser.add_argument(
    '--wandb-run-id-step0',
    type=str,
    nargs='+',
    default=None,
    help=(
      'Optional explicit W&B run id(s) for the step0 snapshot, one per --run-name. '
      'Use this for resumed runs where step0 was logged under a different W&B run id.'
    ),
  )
  parser.add_argument(
    '--logs-dir',
    type=Path,
    default=_repo_root() / 'experiments/legacy_job_scripts',
    help='Directory with VM job logs (used to discover W&B run ids).',
  )
  parser.add_argument(
    '--csv-dir',
    type=Path,
    default=_repo_root() / 'results/tables',
    help='Output directory for machine-readable CSV tables.',
  )
  parser.add_argument(
    '--step1',
    type=int,
    default=20000,
    help='Second training step to compare against step0 (default: 20000).',
  )
  parser.add_argument(
    '--layers',
    type=int,
    nargs='+',
    default=list(range(9)),
    help='Offline-probe layers to include (default: 0..8).',
  )
  parser.add_argument(
    '--page-size',
    type=int,
    default=2000,
    help='W&B scan_history page size (default: 2000).',
  )
  parser.add_argument(
    '--wandb-timeout',
    type=int,
    default=60,
    help='W&B API timeout in seconds for GraphQL requests (default: 60).',
  )
  parser.add_argument(
    '--digits',
    type=int,
    default=3,
    help='Float precision for exported CSVs (default: 3).',
  )
  parser.add_argument(
    '--skip-existing',
    action='store_true',
    help='Skip exporting a table if its CSV already exists.',
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()

  run_names = [str(x) for x in args.run_name if str(x).strip()]
  if not run_names:
    raise ValueError('run-name must be non-empty')

  layers = list(dict.fromkeys(int(x) for x in args.layers))
  if not layers:
    raise ValueError('layers must be non-empty')
  if any(layer < 0 for layer in layers):
    raise ValueError(f'layers must be >= 0, got: {layers}')

  if int(args.page_size) <= 0:
    raise ValueError(f'page-size must be > 0, got: {args.page_size}')
  if int(args.wandb_timeout) <= 0:
    raise ValueError(f'wandb-timeout must be > 0, got: {args.wandb_timeout}')
  if int(args.digits) < 0:
    raise ValueError(f'digits must be >= 0, got: {args.digits}')

  step1 = int(args.step1)
  if step1 <= 0:
    raise ValueError(f'step1 must be > 0, got: {step1}')
  step_suffix = _format_step_suffix(step1)

  # Step 1: instantiate W&B API once.
  api = wandb.Api(timeout=int(args.wandb_timeout))

  # Step 2: use the canonical Aionoscope basic-components config for row ordering and metric massaging.
  config_path = _repo_root() / 'configs/pretrain/dataset/aiono_basic_components_balanced.yaml'
  if not config_path.is_file():
    raise FileNotFoundError(f'Missing Aionoscope basic-components config: {config_path}')

  csv_dir: Path = args.csv_dir
  csv_dir.mkdir(parents=True, exist_ok=True)

  run_ids_step1: list[str] | None = None
  if args.wandb_run_id is not None:
    run_ids_step1 = [str(x) for x in args.wandb_run_id if str(x).strip()]
    if len(run_ids_step1) != len(run_names):
      raise ValueError(
        f'wandb-run-id must have exactly one id per run-name: got={len(run_ids_step1)} expected={len(run_names)}'
      )

  run_ids_step0: list[str] | None = None
  if args.wandb_run_id_step0 is not None:
    run_ids_step0 = [str(x) for x in args.wandb_run_id_step0 if str(x).strip()]
    if len(run_ids_step0) != len(run_names):
      raise ValueError(
        f'wandb-run-id-step0 must have exactly one id per run-name: got={len(run_ids_step0)} expected={len(run_names)}'
      )

  # Step 3: export per-run tables.
  for idx, run_name in enumerate(run_names):
    if run_ids_step1 is None:
      meta_step1 = _discover_wandb_run_meta(
        run_name,
        logs_dir=Path(args.logs_dir),
        wandb_entity=str(args.wandb_entity),
        wandb_project=str(args.wandb_project),
      )
    else:
      dataset, method_family, variant, projector, seed = _parse_run_meta_stub(run_name)
      meta_step1 = RunMeta(
        run_name=run_name,
        dataset=dataset,
        method_family=method_family,
        variant=variant,
        projector=projector,
        seed=seed,
        wandb_entity=str(args.wandb_entity),
        wandb_project=str(args.wandb_project),
        wandb_run_id=str(run_ids_step1[idx]),
        job_log_path=Path('<wandb_api>'),
      )

    meta_step0 = meta_step1
    if run_ids_step0 is not None:
      step0_run_id = str(run_ids_step0[idx])
      step0_run = api.run(f'{str(args.wandb_entity)}/{str(args.wandb_project)}/{step0_run_id}')
      dataset0, method_family0, variant0, projector0, seed0 = _parse_run_meta_stub(step0_run.name)
      meta_step0 = RunMeta(
        run_name=str(step0_run.name),
        dataset=dataset0,
        method_family=method_family0,
        variant=variant0,
        projector=projector0,
        seed=seed0,
        wandb_entity=str(args.wandb_entity),
        wandb_project=str(args.wandb_project),
        wandb_run_id=step0_run_id,
        job_log_path=Path('<wandb_api>'),
      )

    dense_out = csv_dir / f'{meta_step1.run_name.lower()}_dense_targets_step0_vs_{step_suffix}.csv'
    if not (args.skip_existing and dense_out.is_file()):
      _export_aiono_dense_targets_step0_vs_20k_table(
        api,
        run_meta_step0=meta_step0,
        run_meta_step1=meta_step1,
        layers=layers,
        steps=(0, step1),
        targets_config_path=config_path,
        page_size=int(args.page_size),
        digits=int(args.digits),
        out_csv_path=dense_out,
      )
    else:
      print(f'[skip] Dense targets: {dense_out}')

    cat_out = csv_dir / f'{meta_step1.run_name.lower()}_categorical_signals_step0_vs_{step_suffix}.csv'
    if not (args.skip_existing and cat_out.is_file()):
      _export_aiono_categorical_signals_step0_vs_20k_table(
        api,
        run_meta_step0=meta_step0,
        run_meta_step1=meta_step1,
        layers=layers,
        steps=(0, step1),
        signals_config_path=config_path,
        page_size=int(args.page_size),
        digits=int(args.digits),
        out_csv_path=cat_out,
      )
    else:
      print(f'[skip] Categorical signals: {cat_out}')


if __name__ == '__main__':
  main()
