"""Plot a combined Aionoscope radar grid: categorical (left) + dense (right).

Inputs
  This script consumes CSV tables exported by `utils.wandb_lenepa_paper_plots`:
    - Per-signal categorical deltas:
        `results/tables/<run_name_lower>_categorical_signals_step0_vs_<step1>.csv`
    - Per-target dense deltas:
        `results/tables/<run_name_lower>_dense_targets_step0_vs_<step1>.csv`

Figure layout (requested)
  - Left: categorical radars stacked vertically (AUROC over AUPRC).
  - Separator: vertical line.
  - Right: dense radars as a 2x2 grid (columns=target_signal/target_metric; rows=dense metrics).

Score definition
  Uses step0-normalized deltas:
    - Categorical: `auroc_delta_norm0`, `auprc_delta_norm0`
    - Dense: `<metric>_delta_norm0` for each dense metric

  For radar readability we clamp scores to [0, 1], i.e. we plot only non-negative
  improvements (regressions are shown as 0 on the chart).

Outputs (default `--out-root-dir=paper`)
  - `<root>/figures/<basename>.png`
  - `<root>/figures/<basename>.pdf`

Usage
  uv run python -m utils.plot_aiono_combined_radar_grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils.plot_aiono_dense_radar import _compute_radar_aggregates, _load_model_tables
from utils.wandb_lenepa_paper_plots import _format_step_suffix, _paper_colors_for_runs


_DEFAULT_RUN_NAMES = (
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_s0',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_s0',
  'AIONO_JEPA_CLS_s0',
)

_BALANCED_BASIC_COMPONENTS_RUN_NAMES = (
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_BAL_s0',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_BAL_s0',
  'AIONO_JEPA_CLS_BAL_s0',
  'AIONO_JEPA_MEAN_BAL_s0',
)


_DEFAULT_DENSE_METRICS = ('mse', 'pearson')

_RADAR_FONTSIZE_BASE = 16


_SIGNAL_ORDER = (
  # noise
  'gaussian_noise',
  'uniform_noise',
  'random_walk_noise',
  # trends
  'linear_trend',
  'quadratic_trend',
  'log_trend',
  'sigmoid_trend',
  # events
  'level_change',
  'spike',
  'gaussian',
  # periodic
  'sine',
  'sawtooth',
  'square',
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--csv-dir',
    type=Path,
    default=Path('results/tables'),
    help='Directory containing the CSV exports.',
  )
  parser.add_argument(
    '--step1',
    type=int,
    default=20000,
    help='Second training step used in the CSV suffix (default: 20000 -> "20k").',
  )
  parser.add_argument(
    '--balanced',
    action='store_true',
    help=(
      'Use the Plan-110 A7.1 balanced basic-components run set. '
      'This selects a fixed set of 4 runs (JEPA CLS/MEAN, NEPA StopGrad, LeNEPA SIGRegT20) '
      'and defaults --out-basename to avoid overwriting the main paper figure.'
    ),
  )
  parser.add_argument(
    '--run-name',
    type=str,
    nargs='+',
    default=None,
    help='Run name(s) to include (must exist in --csv-dir).',
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
    default=None,
    help='Output base name (writes <name>.png and <name>.pdf).',
  )
  return parser.parse_args()


def _ordered_categories(categories: list[str], *, preferred_order: tuple[str, ...]) -> list[str]:
  """Return categories in preferred order, then any remaining in sorted order."""
  present = set(categories)
  ordered = [c for c in preferred_order if c in present]
  ordered += sorted(present - set(ordered))
  if not ordered:
    raise ValueError('No categories to plot (empty after ordering).')
  return ordered


def _plot_radar(
  ax: plt.Axes,
  *,
  categories: list[str],
  model_to_values: dict[str, list[float]],
  label_to_color: dict[str, tuple[float, float, float, float]],
) -> None:
  """Plot a radar chart with one polygon per model."""
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

  ax.set_ylim(0.0, 1.0)
  ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
  ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
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


def _add_vertical_separator(*, fig: plt.Figure, left_axes: list[plt.Axes], right_axes: list[plt.Axes]) -> None:
  """Draw a vertical separator line between two axis groups."""
  if not left_axes or not right_axes:
    raise ValueError('left_axes and right_axes must be non-empty')

  fig.canvas.draw()
  left_right = max(ax.get_position().x1 for ax in left_axes)
  right_left = min(ax.get_position().x0 for ax in right_axes)
  x = 0.5 * (float(left_right) + float(right_left))

  y0 = min(ax.get_position().y0 for ax in [*left_axes, *right_axes])
  y1 = max(ax.get_position().y1 for ax in [*left_axes, *right_axes])

  fig.add_artist(
    Line2D(
      [x, x],
      [float(y0), float(y1)],
      transform=fig.transFigure,
      color='black',
      linewidth=1.0,
      alpha=0.9,
    )
  )


def _add_row_label(*, fig: plt.Figure, ax: plt.Axes, text: str, x_offset: float) -> None:
  """Add a rotated row label to the left of an axis."""
  fig.canvas.draw()
  pos = ax.get_position()
  x = float(pos.x0) + float(x_offset)
  y = 0.5 * (float(pos.y0) + float(pos.y1))
  fig.text(
    x,
    y,
    str(text),
    rotation=90,
    va='center',
    ha='left',
    fontsize=_RADAR_FONTSIZE_BASE + 1,
  )


def main() -> None:
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

  if args.run_name is not None and args.balanced:
    raise ValueError('Use either --balanced or --run-name, not both.')

  if args.run_name is None:
    run_names = list(_BALANCED_BASIC_COMPONENTS_RUN_NAMES) if args.balanced else list(_DEFAULT_RUN_NAMES)
  else:
    run_names = [str(x) for x in args.run_name]
  if not run_names:
    raise ValueError('run_name must be non-empty')

  if int(args.step1) <= 0:
    raise ValueError(f'step1 must be > 0, got: {args.step1}')
  step_suffix = _format_step_suffix(int(args.step1))

  out_basename = (
    str(args.out_basename).strip()
    if args.out_basename is not None
    else ('aiono_balanced_basic_components_combined_radar_grid' if args.balanced else 'aiono_combined_radar_grid')
  )
  if not out_basename:
    raise ValueError('out_basename must be non-empty')

  dense_metrics = [str(x) for x in args.dense_metrics]
  if len(dense_metrics) != 2:
    raise ValueError(f'dense-metrics must have exactly 2 items for the 2x2 grid, got: {dense_metrics}')

  exclude_dense_signals = {str(x) for x in args.exclude_dense_target_signal if str(x).strip()}

  csv_dir: Path = args.csv_dir

  # Step 0: load dense per-target tables (also provides stable, paper-friendly labels).
  dense_tables = _load_model_tables(csv_dir=csv_dir, run_names=run_names, step1=int(args.step1))
  label_by_run_name = {t.run_name: t.label for t in dense_tables}
  colors_by_run_name = _paper_colors_for_runs([t.run_name for t in dense_tables])
  label_to_color = {t.label: colors_by_run_name[t.run_name] for t in dense_tables}

  # Step 1: load categorical-per-signal tables.
  cat_tables: list[tuple[str, pd.DataFrame]] = []
  for run_name in run_names:
    csv_path = csv_dir / f'{run_name.lower()}_categorical_signals_step0_vs_{step_suffix}.csv'
    if not csv_path.is_file():
      raise FileNotFoundError(
        'Missing categorical-per-signal CSV. '
        f'Fix: run wandb_lenepa_paper_plots to export it. path={csv_path}'
      )
    df = pd.read_csv(csv_path)
    required = ('signal', 'auroc_delta_norm0', 'auprc_delta_norm0')
    missing = [c for c in required if c not in df.columns]
    if missing:
      raise ValueError(f'Missing required columns in {csv_path}: missing={missing} available={list(df.columns)}')
    cat_tables.append((run_name, df))

  cat_categories = _ordered_categories(
    categories=sorted({str(x) for x in cat_tables[0][1]['signal'].tolist()}),
    preferred_order=_SIGNAL_ORDER,
  )
  cat_categories = [c for c in cat_categories if c != 'constant']
  if not cat_categories:
    raise ValueError('After excluding "constant", no categorical signals remain to plot.')
  for run_name, df in cat_tables[1:]:
    other = sorted({str(x) for x in df['signal'].tolist()})
    other = [c for c in other if c != 'constant']
    if set(other) != set(cat_categories):
      raise ValueError(f'Categorical signal axis mismatch for run={run_name!r}: expected={cat_categories} got={other}')

  model_to_cat_auroc: dict[str, list[float]] = {}
  model_to_cat_auprc: dict[str, list[float]] = {}
  for run_name, df in cat_tables:
    label = label_by_run_name.get(run_name)
    if label is None:
      raise KeyError(f'Missing label for run_name={run_name!r}. Known={sorted(label_by_run_name)[:10]}...')
    by_signal = df.set_index('signal', drop=False)
    auroc = pd.to_numeric(by_signal['auroc_delta_norm0'], errors='coerce')
    auprc = pd.to_numeric(by_signal['auprc_delta_norm0'], errors='coerce')
    if auroc.isna().any():
      bad = list(auroc[auroc.isna()].index)
      raise ValueError(f'Non-numeric auroc_delta_norm0 for run={run_name!r}: signals={bad}')
    if auprc.isna().any():
      bad = list(auprc[auprc.isna()].index)
      raise ValueError(f'Non-numeric auprc_delta_norm0 for run={run_name!r}: signals={bad}')

    model_to_cat_auroc[label] = [float(np.clip(float(auroc.loc[s]), 0.0, 1.0)) for s in cat_categories]
    model_to_cat_auprc[label] = [float(np.clip(float(auprc.loc[s]), 0.0, 1.0)) for s in cat_categories]

  # Step 2: load dense per-target tables and compute aggregates.
  dense_aggs = [
    _compute_radar_aggregates(
      dense_tables,
      score_col=f'{metric}_delta_norm0',
      exclude_target_signals=exclude_dense_signals,
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

  # Step 3: plot combined layout.
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

  _plot_radar(ax_cat_auroc, categories=cat_categories, model_to_values=model_to_cat_auroc, label_to_color=label_to_color)
  _plot_radar(ax_cat_auprc, categories=cat_categories, model_to_values=model_to_cat_auprc, label_to_color=label_to_color)

  _plot_radar(
    ax_dense_r0_signal,
    categories=dense_aggs[0].signals,
    model_to_values=dense_aggs[0].model_to_signal_values,
    label_to_color=label_to_color,
  )
  _plot_radar(
    ax_dense_r0_metric,
    categories=dense_aggs[0].metrics_multi,
    model_to_values=dense_aggs[0].model_to_metric_values,
    label_to_color=label_to_color,
  )
  _plot_radar(
    ax_dense_r1_signal,
    categories=dense_aggs[1].signals,
    model_to_values=dense_aggs[1].model_to_signal_values,
    label_to_color=label_to_color,
  )
  _plot_radar(
    ax_dense_r1_metric,
    categories=dense_aggs[1].metrics_multi,
    model_to_values=dense_aggs[1].model_to_metric_values,
    label_to_color=label_to_color,
  )

  # Step 4: labels and separators (in figure coordinates).
  _add_row_label(fig=fig, ax=ax_cat_auroc, text='AUROC', x_offset=-0.06)
  _add_row_label(fig=fig, ax=ax_cat_auprc, text='AUPRC', x_offset=-0.06)

  _add_row_label(fig=fig, ax=ax_dense_r0_signal, text=str(dense_metrics[0]).upper(), x_offset=-0.04)
  _add_row_label(fig=fig, ax=ax_dense_r1_signal, text=str(dense_metrics[1]).upper(), x_offset=-0.04)

  _add_vertical_separator(
    fig=fig,
    left_axes=[ax_cat_auroc, ax_cat_auprc],
    right_axes=[ax_dense_r0_signal, ax_dense_r0_metric, ax_dense_r1_signal, ax_dense_r1_metric],
  )

  # Step 5: legend and write outputs.
  handles, labels = ax_dense_r0_metric.get_legend_handles_labels()
  if not handles:
    handles, labels = ax_cat_auroc.get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)))

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
