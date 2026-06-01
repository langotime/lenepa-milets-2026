"""Plot a balanced-only Aionoscope combined radar grid with the CAUKER2M run added.

This script is a focused sibling of `utils.plot_aiono_combined_radar_grid`.
It keeps the balanced Aionoscope comparison, adds the CAUKER2M LeNEPA run, and writes
to a separate output basename so the paper figure is not overwritten.

Inputs
  This script consumes CSV tables exported into `results/tables`:
    - `<run_name_lower>_categorical_signals_step0_vs_<step1>.csv`
    - `<run_name_lower>_dense_targets_step0_vs_<step1>.csv`

Outputs (default `--out-root-dir=paper`)
  - `<root>/figures/aiono_balanced_basic_components_combined_radar_grid_cauker2m.png`
  - `<root>/figures/aiono_balanced_basic_components_combined_radar_grid_cauker2m.pdf`
  - `<root>/figures/aiono_balanced_basic_components_combined_radar_grid_cauker2m_latest.png`
  - `<root>/figures/aiono_balanced_basic_components_combined_radar_grid_cauker2m_latest.pdf`

Usage
  uv run python -m utils.plot_aiono_combined_radar_grid_balanced_cauker2m
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_aiono_combined_radar_grid import (
  _SIGNAL_ORDER,
  _add_row_label,
  _add_vertical_separator,
  _ordered_categories,
  _plot_radar,
)
from utils.plot_aiono_dense_radar import (
  ModelTable,
  RadarAggregates,
  _TARGET_METRIC_ORDER,
  _TARGET_SIGNAL_ORDER,
  _compute_radar_aggregates,
)
from utils.wandb_lenepa_paper_plots import _format_step_suffix, _paper_colors_for_runs


_BALANCED_CAUKER_RUN_NAMES = (
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_BAL_s0',
  'CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_OPAIONOBCBAL_s0_UCRinterp5000_CONT200K_LOCAL',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0',
)

_RUN_LABELS = {
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_BAL_s0': 'LeNEPA (SIGREGT20)',
  'CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_OPAIONOBCBAL_s0_UCRinterp5000_CONT200K_LOCAL': 'LeNEPA (SIGREGT2p5, CAUKER2M)',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0': 'NEPA',
  'AIONO_JEPA_CLS_BAL_s0': 'JEPA (CLS)',
  'AIONO_JEPA_MEAN_BAL_s0': 'JEPA (MEAN)',
}

_DEFAULT_DENSE_METRICS = ('mse', 'pearson')
_RADAR_FONTSIZE_BASE = 16
_DEFAULT_OUT_BASENAME = 'aiono_balanced_basic_components_combined_radar_grid_cauker2m'
_DEFAULT_LATEST_OUT_BASENAME = 'aiono_balanced_basic_components_combined_radar_grid_cauker2m_latest'
_DEFAULT_ABSOLUTE_OUT_BASENAME = 'aiono_balanced_basic_components_combined_radar_grid_cauker2m_absolute'
_DEFAULT_LATEST_ABSOLUTE_OUT_BASENAME = 'aiono_balanced_basic_components_combined_radar_grid_cauker2m_latest_absolute'
_DEFAULT_CAUKER_LATEST_STEP = 88000
_AUROC_ABSOLUTE_RADIAL_MIN = 0.75
_CAUKER_OVERRIDE_COLOR = (0.22, 0.22, 0.22, 1.0)
_CAUKER_CONT_RUN_NAME = (
  'CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_OPAIONOBCBAL_s0_UCRinterp5000_CONT200K_LOCAL'
)
_CAUKER_PRECURSOR_RUN_NAME = (
  'CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_OPAIONOBCBAL_s0_UCRinterp5000'
)


def _parse_args() -> argparse.Namespace:
  """Parse CLI arguments for the balanced CAUKER2M radar plot."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--csv-dir',
    type=Path,
    default=Path('results/tables'),
    help='Directory containing the CSV exports.',
  )
  parser.add_argument(
    '--mode',
    type=str,
    default='20k',
    choices=('20k', 'latest'),
    help='Comparison mode: common 20k snapshot or per-run latest snapshot.',
  )
  parser.add_argument(
    '--score-mode',
    type=str,
    default='delta',
    choices=('delta', 'absolute'),
    help='Whether to plot normalized deltas or absolute best-layer values at the comparison step.',
  )
  parser.add_argument(
    '--step1',
    type=int,
    default=20000,
    help='Baseline step used for Aionoscope balanced runs (default: 20000 -> "20k").',
  )
  parser.add_argument(
    '--cauker-latest-step',
    type=int,
    default=_DEFAULT_CAUKER_LATEST_STEP,
    help='Latest verified Aionoscope probe step for the CAUKER continuation run.',
  )
  parser.add_argument(
    '--dense-metrics',
    type=str,
    nargs='+',
    default=list(_DEFAULT_DENSE_METRICS),
    choices=('mse', 'mae', 'pearson', 'r2'),
    help='Dense metrics to plot on the right side (must be exactly 2 for the 2x2 grid).',
  )
  parser.add_argument(
    '--exclude-dense-target-signal',
    type=str,
    nargs='*',
    default=[],
    help='Dense `target_signal` values to exclude (default: none).',
  )
  parser.add_argument(
    '--out-root-dir',
    '--out-dir',
    dest='out_root_dir',
    type=Path,
    default=Path('paper'),
    help='Output root directory. Writes figures to <root>/figures.',
  )
  parser.add_argument(
    '--out-basename',
    type=str,
    default='',
    help='Optional output base name. When omitted, it depends on --mode.',
  )
  return parser.parse_args()


def _label_from_run_name(run_name: str) -> str:
  """Return the fixed legend label for the balanced comparison runs."""
  label = _RUN_LABELS.get(run_name)
  if label is None:
    raise KeyError(f'Unexpected run name for balanced CAUKER2M radar plot: {run_name!r}')
  return label


def _step1_by_run_name(*, mode: str, baseline_step1: int, cauker_latest_step: int) -> dict[str, int]:
  """Return the CSV step suffix to use for each plotted run."""
  if int(baseline_step1) <= 0:
    raise ValueError(f'baseline_step1 must be > 0, got: {baseline_step1}')
  if int(cauker_latest_step) <= 0:
    raise ValueError(f'cauker_latest_step must be > 0, got: {cauker_latest_step}')

  if mode == '20k':
    return {run_name: int(baseline_step1) for run_name in _BALANCED_CAUKER_RUN_NAMES}
  if mode == 'latest':
    step1_by_run_name = {run_name: int(baseline_step1) for run_name in _BALANCED_CAUKER_RUN_NAMES}
    step1_by_run_name[_CAUKER_CONT_RUN_NAME] = int(cauker_latest_step)
    return step1_by_run_name

  raise ValueError(f'Unexpected mode: {mode!r}')


def _csv_source_run_name(*, plot_run_name: str, step1: int) -> str:
  """Return the CSV-source run name used for a plotted run at the requested step."""
  if plot_run_name == _CAUKER_CONT_RUN_NAME and int(step1) == 20000:
    # The continuation run starts after step 0, so the step0-vs-20k snapshot comes from
    # the precursor run with the same configuration before local continuation.
    return _CAUKER_PRECURSOR_RUN_NAME
  return str(plot_run_name)


def _load_labeled_dense_tables(*, csv_dir: Path, step1_by_run_name: dict[str, int]) -> list[ModelTable]:
  """Load dense CSV tables and attach the fixed plot labels/run names."""
  # Step 1: load each CSV from its declared source run.
  tables: list[ModelTable] = []
  for plot_run_name in _BALANCED_CAUKER_RUN_NAMES:
    step1 = int(step1_by_run_name[plot_run_name])
    step_suffix = _format_step_suffix(step1)
    csv_run_name = _csv_source_run_name(plot_run_name=plot_run_name, step1=step1)
    path = csv_dir / f'{csv_run_name.lower()}_dense_targets_step0_vs_{step_suffix}.csv'
    if not path.is_file():
      raise FileNotFoundError(
        'Missing per-target dense table CSV for the balanced CAUKER2M radar plot. '
        f'plot_run_name={plot_run_name!r} step1={step1} csv_run_name={csv_run_name!r} path={path}'
      )
    tables.append(
      ModelTable(
        label=_label_from_run_name(plot_run_name),
        run_name=plot_run_name,
        df=pd.read_csv(path),
      )
    )

  return tables


def _label_to_color_map(tables: list[ModelTable]) -> dict[str, tuple[float, float, float, float]]:
  """Return plot colors with an explicit dark-grey override for the CAUKER run."""
  if not tables:
    raise ValueError('tables must be non-empty')

  colors_by_run_name = _paper_colors_for_runs([table.run_name for table in tables])
  colors_by_run_name[_CAUKER_CONT_RUN_NAME] = _CAUKER_OVERRIDE_COLOR
  return {table.label: colors_by_run_name[table.run_name] for table in tables}


def _compute_radar_aggregates_mixed_cols(
  tables: list[ModelTable],
  *,
  score_col_by_run_name: dict[str, str],
  exclude_target_signals: set[str],
  clip_to_unit: bool,
  clip_min: float | None = None,
) -> RadarAggregates:
  """Compute dense radar aggregates when each run can use a different source column."""
  if clip_min is not None and not math.isfinite(float(clip_min)):
    raise ValueError(f'clip_min must be finite when provided, got: {clip_min}')

  filtered_tables: list[ModelTable] = []
  for table in tables:
    score_col = score_col_by_run_name.get(table.run_name)
    if score_col is None:
      raise KeyError(f'Missing score column mapping for run={table.run_name!r}')

    required = ('target', 'target_signal', 'target_metric', score_col)
    missing = [col for col in required if col not in table.df.columns]
    if missing:
      raise ValueError(
        f'Missing required columns in dense table for run={table.run_name!r}: '
        f'missing={missing} available={list(table.df.columns)}'
      )

    filtered = table.df[~table.df['target_signal'].isin(exclude_target_signals)].copy()
    if filtered.empty:
      raise ValueError(f'After filtering target_signal, dense table is empty for run={table.run_name!r}.')

    score = pd.to_numeric(filtered[score_col], errors='coerce')
    if not np.all(np.isfinite(score.to_numpy(dtype=float))):
      bad = filtered[~np.isfinite(score)].loc[:, ['target', 'target_signal', 'target_metric', score_col]]
      raise ValueError(f'Non-finite score values found in {score_col!r}: rows=\n{bad}')

    values = score.to_numpy(dtype=float)
    if clip_to_unit:
      filtered['score'] = np.clip(values, a_min=0.0, a_max=1.0)
    elif clip_min is not None:
      filtered['score'] = np.maximum(values, float(clip_min))
    else:
      filtered['score'] = values
    filtered_tables.append(ModelTable(label=table.label, run_name=table.run_name, df=filtered))

  signals = _ordered_categories(
    categories=sorted({str(x) for x in filtered_tables[0].df['target_signal'].tolist()}),
    preferred_order=_TARGET_SIGNAL_ORDER,
  )
  for table in filtered_tables[1:]:
    other = sorted({str(x) for x in table.df['target_signal'].tolist()})
    if set(other) != set(signals):
      raise ValueError(
        'target_signal axis mismatch between models after filtering. '
        f'expected={signals} got={other} model={table.run_name!r}'
      )
  model_to_signal_values = {
    table.label: [float(table.df.groupby('target_signal', sort=False)['score'].median().loc[signal]) for signal in signals]
    for table in filtered_tables
  }

  metric_counts = filtered_tables[0].df.groupby('target_metric', sort=False).size().to_dict()
  metrics_multi = [metric for metric, count in metric_counts.items() if int(count) > 1]
  metrics_multi = _ordered_categories(
    categories=[str(metric) for metric in metrics_multi],
    preferred_order=_TARGET_METRIC_ORDER,
  )
  for table in filtered_tables[1:]:
    other_counts = table.df.groupby('target_metric', sort=False).size().to_dict()
    other_multi = [metric for metric, count in other_counts.items() if int(count) > 1]
    if set(other_multi) != set(metrics_multi):
      raise ValueError(
        'target_metric>1 axis mismatch between models after filtering. '
        f'expected={sorted(metrics_multi)} got={sorted(other_multi)} model={table.run_name!r}'
      )

  model_to_metric_values = {
    table.label: [float(table.df.groupby('target_metric', sort=False)['score'].median().loc[metric]) for metric in metrics_multi]
    for table in filtered_tables
  }

  return RadarAggregates(
    score_col='mixed',
    signals=signals,
    model_to_signal_values=model_to_signal_values,
    metrics_multi=metrics_multi,
    model_to_metric_values=model_to_metric_values,
  )


def _absolute_column_name(*, metric: str, step1: int) -> str:
  """Return the absolute-value CSV column for a metric at the comparison step."""
  return f'{metric}_step{int(step1)}_best'


def _nice_upper_bound(value: float) -> float:
  """Round a positive maximum value up to a compact plotting bound."""
  if not math.isfinite(value) or value <= 0.0:
    return 1.0
  if value <= 1.0:
    return 1.0

  magnitude = 10.0 ** math.floor(math.log10(value))
  for factor in (1.0, 1.5, 2.0, 2.5, 5.0, 10.0):
    candidate = float(factor * magnitude)
    if candidate >= value:
      return candidate
  return float(10.0 * magnitude)


def _format_tick(value: float, *, radial_max: float) -> str:
  """Format a radar tick label compactly."""
  if abs(value - round(value)) < 1e-9:
    return str(int(round(value)))
  digits = 2 if radial_max <= 2.0 else 1
  return f'{value:.{digits}f}'


def _radial_max_for_values(model_to_values: dict[str, list[float]]) -> float:
  """Return a panel-specific upper bound from plotted values."""
  maxima = [max(values) for values in model_to_values.values() if values]
  if not maxima:
    return 1.0
  return _nice_upper_bound(max(maxima))


def _plot_radar_scaled(
  ax: plt.Axes,
  *,
  categories: list[str],
  model_to_values: dict[str, list[float]],
  label_to_color: dict[str, tuple[float, float, float, float]],
  radial_min: float = 0.0,
  radial_max: float,
) -> None:
  """Plot a radar chart with a configurable radial scale."""
  if radial_min < 0.0:
    raise ValueError(f'radial_min must be >= 0, got: {radial_min}')
  if radial_max <= 0.0:
    raise ValueError(f'radial_max must be > 0, got: {radial_max}')
  if radial_max <= radial_min:
    raise ValueError(f'radial_max must be > radial_min, got: radial_min={radial_min} radial_max={radial_max}')

  if abs(float(radial_min)) < 1e-9 and abs(float(radial_max) - 1.0) < 1e-9:
    _plot_radar(
      ax,
      categories=categories,
      model_to_values=model_to_values,
      label_to_color=label_to_color,
    )
    return

  if not categories:
    raise ValueError('categories must be non-empty')
  if not model_to_values:
    raise ValueError('model_to_values must be non-empty')

  n = len(categories)
  angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).tolist()
  angles += angles[:1]

  ax.set_theta_offset(np.pi / 2.0)
  ax.set_theta_direction(-1)
  ax.set_xticks(angles[:-1])
  ax.set_xticklabels(categories)
  ax.tick_params(axis='x', pad=12)

  ax.set_ylim(float(radial_min), float(radial_max))
  ticks = np.linspace(float(radial_min), float(radial_max), 5)
  ax.set_yticks(ticks.tolist())
  ax.set_yticklabels([_format_tick(float(tick), radial_max=float(radial_max)) for tick in ticks])
  ax.tick_params(axis='y', pad=6)

  for label, values in model_to_values.items():
    if len(values) != n:
      raise ValueError(f'Value length mismatch for model={label!r}: got {len(values)} expected {n}')
    values_closed = list(values) + [values[0]]
    color = label_to_color.get(label)
    if color is None:
      raise KeyError(f'Missing required color for model label={label!r}. Known={sorted(label_to_color)[:10]}...')
    ax.plot(angles, values_closed, linewidth=2, label=label, color=color)
    ax.fill(angles, values_closed, alpha=0.12, color=color)


def main() -> None:
  """Generate the balanced combined radar plot with the extra CAUKER2M run."""
  args = _parse_args()
  plt.rcParams.update(
    {
      'font.size': _RADAR_FONTSIZE_BASE,
      'axes.titlesize': _RADAR_FONTSIZE_BASE + 2,
      'axes.labelsize': _RADAR_FONTSIZE_BASE,
      'xtick.labelsize': _RADAR_FONTSIZE_BASE,
      'ytick.labelsize': _RADAR_FONTSIZE_BASE - 1,
      'legend.fontsize': _RADAR_FONTSIZE_BASE,
    }
  )

  step1_by_run_name = _step1_by_run_name(
    mode=str(args.mode),
    baseline_step1=int(args.step1),
    cauker_latest_step=int(args.cauker_latest_step),
  )

  if str(args.score_mode) == 'delta':
    out_basename_default = _DEFAULT_OUT_BASENAME if str(args.mode) == '20k' else _DEFAULT_LATEST_OUT_BASENAME
  else:
    out_basename_default = (
      _DEFAULT_ABSOLUTE_OUT_BASENAME if str(args.mode) == '20k' else _DEFAULT_LATEST_ABSOLUTE_OUT_BASENAME
    )
  out_basename = str(args.out_basename).strip() if str(args.out_basename).strip() else out_basename_default
  if not out_basename:
    raise ValueError('out_basename must be non-empty')

  dense_metrics = [str(x) for x in args.dense_metrics]
  if len(dense_metrics) != 2:
    raise ValueError(f'dense-metrics must have exactly 2 items for the 2x2 grid, got: {dense_metrics}')

  exclude_dense_signals = {str(x) for x in args.exclude_dense_target_signal if str(x).strip()}
  csv_dir: Path = args.csv_dir

  # Step 1: load dense per-target tables and derive the fixed labels/colors.
  dense_tables = _load_labeled_dense_tables(csv_dir=csv_dir, step1_by_run_name=step1_by_run_name)
  label_by_run_name = {table.run_name: table.label for table in dense_tables}
  label_to_color = _label_to_color_map(dense_tables)

  # Step 2: load categorical-per-signal tables for the same balanced run set.
  cat_tables: list[tuple[str, pd.DataFrame]] = []
  for run_name in _BALANCED_CAUKER_RUN_NAMES:
    step1 = int(step1_by_run_name[run_name])
    step_suffix = _format_step_suffix(step1)
    csv_run_name = _csv_source_run_name(plot_run_name=run_name, step1=step1)
    csv_path = csv_dir / f'{csv_run_name.lower()}_categorical_signals_step0_vs_{step_suffix}.csv'
    if not csv_path.is_file():
      raise FileNotFoundError(
        'Missing categorical-per-signal CSV. '
        f'Fix: export the Aionoscope categorical table first. '
        f'plot_run_name={run_name!r} step1={step1} csv_run_name={csv_run_name!r} path={csv_path}'
      )
    df = pd.read_csv(csv_path)
    required = ('signal',)
    if str(args.score_mode) == 'delta':
      required += ('auroc_delta_norm0', 'auprc_delta_norm0')
    else:
      required += (
        _absolute_column_name(metric='auroc', step1=step1),
        _absolute_column_name(metric='auprc', step1=step1),
      )
    missing = [col for col in required if col not in df.columns]
    if missing:
      raise ValueError(f'Missing required columns in {csv_path}: missing={missing} available={list(df.columns)}')
    cat_tables.append((run_name, df))

  # Step 3: align the categorical axis across all runs.
  cat_categories = _ordered_categories(
    categories=sorted({str(x) for x in cat_tables[0][1]['signal'].tolist()}),
    preferred_order=_SIGNAL_ORDER,
  )
  cat_categories = [category for category in cat_categories if category != 'constant']
  if not cat_categories:
    raise ValueError('After excluding "constant", no categorical signals remain to plot.')
  for run_name, df in cat_tables[1:]:
    other = sorted({str(x) for x in df['signal'].tolist()})
    other = [category for category in other if category != 'constant']
    if set(other) != set(cat_categories):
      raise ValueError(f'Categorical signal axis mismatch for run={run_name!r}: expected={cat_categories} got={other}')

  # Step 4: convert categorical tables into per-model radar vectors.
  model_to_cat_auroc: dict[str, list[float]] = {}
  model_to_cat_auprc: dict[str, list[float]] = {}
  for run_name, df in cat_tables:
    label = label_by_run_name[run_name]
    by_signal = df.set_index('signal', drop=False)
    step1 = int(step1_by_run_name[run_name])
    auroc_col = 'auroc_delta_norm0' if str(args.score_mode) == 'delta' else _absolute_column_name(metric='auroc', step1=step1)
    auprc_col = 'auprc_delta_norm0' if str(args.score_mode) == 'delta' else _absolute_column_name(metric='auprc', step1=step1)
    auroc = pd.to_numeric(by_signal[auroc_col], errors='coerce')
    auprc = pd.to_numeric(by_signal[auprc_col], errors='coerce')
    if auroc.isna().any():
      bad = list(auroc[auroc.isna()].index)
      raise ValueError(f'Non-numeric {auroc_col} for run={run_name!r}: signals={bad}')
    if auprc.isna().any():
      bad = list(auprc[auprc.isna()].index)
      raise ValueError(f'Non-numeric {auprc_col} for run={run_name!r}: signals={bad}')

    if str(args.score_mode) == 'delta':
      model_to_cat_auroc[label] = [float(np.clip(float(auroc.loc[signal]), 0.0, 1.0)) for signal in cat_categories]
      model_to_cat_auprc[label] = [float(np.clip(float(auprc.loc[signal]), 0.0, 1.0)) for signal in cat_categories]
    else:
      model_to_cat_auroc[label] = [float(auroc.loc[signal]) for signal in cat_categories]
      model_to_cat_auprc[label] = [float(auprc.loc[signal]) for signal in cat_categories]

  # Step 5: compute dense radar aggregates from the shared helper.
  if str(args.score_mode) == 'delta':
    dense_aggs = [
      _compute_radar_aggregates(
        dense_tables,
        score_col=f'{metric}_delta_norm0',
        exclude_target_signals=exclude_dense_signals,
      )
      for metric in dense_metrics
    ]
  else:
    dense_aggs = [
      _compute_radar_aggregates_mixed_cols(
        dense_tables,
        score_col_by_run_name={
          run_name: _absolute_column_name(metric=metric, step1=int(step1_by_run_name[run_name]))
          for run_name in _BALANCED_CAUKER_RUN_NAMES
        },
        exclude_target_signals=exclude_dense_signals,
        clip_to_unit=False,
        clip_min=0.0 if metric == 'r2' else None,
      )
      for metric in dense_metrics
    ]
  for metric, agg in zip(dense_metrics[1:], dense_aggs[1:]):
    if agg.signals != dense_aggs[0].signals:
      raise ValueError(
        'Dense target_signal axis mismatch between metrics. '
        f'first={dense_aggs[0].signals} metric={metric!r} got={agg.signals}'
      )
    if agg.metrics_multi != dense_aggs[0].metrics_multi:
      raise ValueError(
        'Dense target_metric>1 axis mismatch between metrics. '
        f'first={dense_aggs[0].metrics_multi} metric={metric!r} got={agg.metrics_multi}'
      )

  # Step 6: assemble the shared 2x3 combined layout.
  fig = plt.figure(figsize=(22.0, 13.5))
  gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    width_ratios=[1.0, 1.25, 1.25],
    wspace=0.35,
    hspace=0.30,
    left=0.07,
    right=0.99,
    top=0.97,
    bottom=0.12,
  )

  ax_cat_auroc = fig.add_subplot(gs[0, 0], projection='polar')
  ax_cat_auprc = fig.add_subplot(gs[1, 0], projection='polar')
  ax_dense_r0_signal = fig.add_subplot(gs[0, 1], projection='polar')
  ax_dense_r0_metric = fig.add_subplot(gs[0, 2], projection='polar')
  ax_dense_r1_signal = fig.add_subplot(gs[1, 1], projection='polar')
  ax_dense_r1_metric = fig.add_subplot(gs[1, 2], projection='polar')

  _plot_radar_scaled(
    ax_cat_auroc,
    categories=cat_categories,
    model_to_values=model_to_cat_auroc,
    label_to_color=label_to_color,
    radial_min=_AUROC_ABSOLUTE_RADIAL_MIN if str(args.score_mode) == 'absolute' else 0.0,
    radial_max=_radial_max_for_values(model_to_cat_auroc),
  )
  _plot_radar_scaled(
    ax_cat_auprc,
    categories=cat_categories,
    model_to_values=model_to_cat_auprc,
    label_to_color=label_to_color,
    radial_max=_radial_max_for_values(model_to_cat_auprc),
  )
  _plot_radar_scaled(
    ax_dense_r0_signal,
    categories=dense_aggs[0].signals,
    model_to_values=dense_aggs[0].model_to_signal_values,
    label_to_color=label_to_color,
    radial_max=_radial_max_for_values(dense_aggs[0].model_to_signal_values),
  )
  _plot_radar_scaled(
    ax_dense_r0_metric,
    categories=dense_aggs[0].metrics_multi,
    model_to_values=dense_aggs[0].model_to_metric_values,
    label_to_color=label_to_color,
    radial_max=_radial_max_for_values(dense_aggs[0].model_to_metric_values),
  )
  _plot_radar_scaled(
    ax_dense_r1_signal,
    categories=dense_aggs[1].signals,
    model_to_values=dense_aggs[1].model_to_signal_values,
    label_to_color=label_to_color,
    radial_max=_radial_max_for_values(dense_aggs[1].model_to_signal_values),
  )
  _plot_radar_scaled(
    ax_dense_r1_metric,
    categories=dense_aggs[1].metrics_multi,
    model_to_values=dense_aggs[1].model_to_metric_values,
    label_to_color=label_to_color,
    radial_max=_radial_max_for_values(dense_aggs[1].model_to_metric_values),
  )

  # Step 7: add figure-level labels and separators.
  _add_row_label(fig=fig, ax=ax_cat_auroc, text='AUROC', x_offset=-0.06)
  _add_row_label(fig=fig, ax=ax_cat_auprc, text='AUPRC', x_offset=-0.06)
  dense_r0_label = str(dense_metrics[0]).upper()
  if str(args.score_mode) == 'absolute' and dense_metrics[0] in ('mse', 'mae'):
    dense_r0_label = f'{dense_r0_label} (lower)'
  _add_row_label(fig=fig, ax=ax_dense_r0_signal, text=dense_r0_label, x_offset=-0.04)
  _add_row_label(fig=fig, ax=ax_dense_r1_signal, text=str(dense_metrics[1]).upper(), x_offset=-0.04)
  _add_vertical_separator(
    fig=fig,
    left_axes=[ax_cat_auroc, ax_cat_auprc],
    right_axes=[ax_dense_r0_signal, ax_dense_r0_metric, ax_dense_r1_signal, ax_dense_r1_metric],
  )

  # Step 8: write outputs without overwriting the paper figure basename.
  handles, labels = ax_dense_r0_metric.get_legend_handles_labels()
  if not handles:
    handles, labels = ax_cat_auroc.get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)))

  out_root_dir: Path = args.out_root_dir
  out_fig_dir = out_root_dir / 'figures'
  out_fig_dir.mkdir(parents=True, exist_ok=True)

  out_png = out_fig_dir / f'{out_basename}.png'
  out_pdf = out_fig_dir / f'{out_basename}.pdf'
  fig.savefig(out_png, dpi=200)
  fig.savefig(out_pdf)
  plt.close(fig)

  print(f'[ok] Wrote: {out_png}')
  print(f'[ok] Wrote: {out_pdf}')


if __name__ == '__main__':
  main()
