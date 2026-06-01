"""Plot Aionoscope dense-target improvement radar charts for multiple models.

This script consumes the per-target dense tables exported by
`utils.wandb_lenepa_paper_plots`:
  `results/tables/<run_name_lower>_dense_targets_step0_vs_<step1>.csv`

It produces a combined grid of radar ("star") charts with:
  - columns: `target_signal` and `target_metric` (only metrics with >1 target)
  - rows: multiple dense metrics (default: MSE, Pearson)

Score definition (default):
  For each dense target we use `<metric>_delta_norm0`, which is a step0-normalized
  improvement. For radar readability we clamp the score to [0, 1], i.e. we plot
  only non-negative improvements (regressions are shown as 0 on the chart).

Usage:
  uv run python -m utils.plot_aiono_dense_radar
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.wandb_lenepa_paper_plots import _format_step_suffix, _paper_colors_for_runs


_DEFAULT_RUN_NAMES = (
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_s0',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_s0',
  'AIONO_JEPA_CLS_s0',
)

_RADAR_FONTSIZE_BASE = 12


_DEFAULT_EXCLUDE_TARGET_SIGNALS: tuple[str, ...] = ()


_TARGET_SIGNAL_ORDER = (
  # noise
  'gaussian_noise',
  'uniform_noise',
  'random_walk_noise',
  # trends
  'linear_trend',
  'quadratic_trend',
  'log_trend',
  'sigmoid_trend',
  # events (singular)
  'level_change',
  'spike',
  'gaussian',
  # periodic
  'sine',
  'sawtooth',
  'square',
)


_TARGET_METRIC_ORDER = (
  'frequency_hz',
  'std',
  'amplitude',
  'slope',
  'intercept',
  'phase',
  'offset',
  'time_frac',
  'magnitude',
)


@dataclass(frozen=True)
class ModelTable:
  """Loaded per-target dense table for one model."""

  label: str
  run_name: str
  df: pd.DataFrame


@dataclass(frozen=True)
class RadarAggregates:
  """Aggregated radar-plot data for one metric."""

  score_col: str
  signals: list[str]
  model_to_signal_values: dict[str, list[float]]
  metrics_multi: list[str]
  model_to_metric_values: dict[str, list[float]]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--csv-dir',
    type=Path,
    default=Path('results/tables'),
    help='Directory containing per-target dense CSV exports.',
  )
  parser.add_argument(
    '--run-name',
    type=str,
    nargs='+',
    default=list(_DEFAULT_RUN_NAMES),
    help='Run name(s) to include. Each must have a corresponding CSV in --csv-dir.',
  )
  parser.add_argument(
    '--step1',
    type=int,
    default=20000,
    help='Second training step used in the CSV suffix (default: 20000 -> "20k").',
  )
  parser.add_argument(
    '--metrics',
    type=str,
    nargs='+',
    default=['mse', 'pearson'],
    choices=('mse', 'mae', 'pearson', 'r2'),
    help='Metric order for the grid (one row per metric).',
  )
  parser.add_argument(
    '--exclude-target-signal',
    type=str,
    nargs='*',
    default=list(_DEFAULT_EXCLUDE_TARGET_SIGNALS),
    help='target_signal values to exclude (default: none).',
  )
  parser.add_argument(
    '--out-root-dir',
    '--out-dir',
    dest='out_root_dir',
    type=Path,
    default=Path('paper'),
    help='Output root directory. Writes figures to <root>/figures and tables to <root>/tables.',
  )
  parser.add_argument(
    '--out-basename',
    type=str,
    default='aiono_dense_radar',
    help='Output base name (writes <name>.png and <name>.pdf).',
  )
  return parser.parse_args()


def _latex_escape(text: str) -> str:
  """Escape a string for safe use in LaTeX (minimal, deterministic)."""
  replacements = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
  }
  out = str(text)
  for src, dst in replacements.items():
    out = out.replace(src, dst)
  return out


def _latex_tt(text: str) -> str:
  """Format a short code-like token as monospace LaTeX."""
  return r'\texttt{' + _latex_escape(text) + '}'


def _format_float(value: float, *, digits: int) -> str:
  """Format a finite float for tables."""
  if not math.isfinite(value):
    raise ValueError(f'Expected finite float, got: {value}')
  return f'{float(value):.{int(digits)}f}'


def _write_latex_table(
  *,
  out_path: Path,
  caption: str,
  label: str,
  header: list[str],
  rows: list[list[str]],
  resize_to_linewidth: bool,
) -> None:
  """Write a standalone LaTeX table environment (booktabs)."""
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    _render_latex_table(
      caption=caption,
      label=label,
      header=header,
      rows=rows,
      resize_to_linewidth=resize_to_linewidth,
    ),
    encoding='utf-8',
  )


def _render_latex_table(
  *,
  caption: str,
  label: str,
  header: list[str],
  rows: list[list[str]],
  resize_to_linewidth: bool,
) -> str:
  """Render a standalone LaTeX table environment (booktabs)."""
  if not caption.strip():
    raise ValueError('caption must be non-empty')
  if not label.strip():
    raise ValueError('label must be non-empty')
  if not header:
    raise ValueError('header must be non-empty')
  if not rows:
    raise ValueError('rows must be non-empty')
  for row in rows:
    if len(row) != len(header):
      raise ValueError(f'Row length mismatch: expected {len(header)}, got {len(row)}')

  col_spec = 'l' + ('r' * (len(header) - 1))

  lines: list[str] = []
  lines.append(r'\begin{table}[t]')
  lines.append(r'\centering')
  lines.append(r'\scriptsize')
  lines.append(rf'\caption{{{caption}}}')
  lines.append(rf'\label{{{label}}}')
  if resize_to_linewidth:
    lines.append(r'\resizebox{\linewidth}{!}{%')
  lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
  lines.append(r'\toprule')
  lines.append(' & '.join(header) + r' \\')
  lines.append(r'\midrule')
  for row in rows:
    lines.append(' & '.join(row) + r' \\')
  lines.append(r'\bottomrule')
  lines.append(r'\end{tabular}')
  if resize_to_linewidth:
    lines.append('}')
  lines.append(r'\end{table}')
  lines.append('')

  return '\n'.join(lines)


def _label_from_run_name(run_name: str) -> str:
  """Build a compact legend label for a run name."""
  if '_LENEPA_' in run_name:
    # Keep the main SIGREG variant for differentiation.
    parts = run_name.split('_')
    sigreg = next((p for p in parts if p.startswith('SIGREG')), 'LENEPA')
    return f'LeNEPA ({sigreg})'
  if '_NEPA_' in run_name:
    return 'NEPA'
  if '_JEPA_' in run_name:
    variant = next((p for p in run_name.split('_') if p in ('CLS', 'MEAN')), None)
    return f'JEPA ({variant})' if variant is not None else 'JEPA'
  return run_name


def _load_model_tables(*, csv_dir: Path, run_names: list[str], step1: int) -> list[ModelTable]:
  """Load per-target dense CSV tables for multiple runs."""
  if not run_names:
    raise ValueError('run_names must be non-empty')

  if int(step1) <= 0:
    raise ValueError(f'step1 must be > 0, got: {step1}')
  step_suffix = _format_step_suffix(int(step1))

  out: list[ModelTable] = []
  for run_name in run_names:
    path = csv_dir / f'{run_name.lower()}_dense_targets_step0_vs_{step_suffix}.csv'
    if not path.is_file():
      raise FileNotFoundError(
        'Missing per-target dense table CSV. '
        f'Fix: generate it via wandb_lenepa_paper_plots or provide a correct --csv-dir. path={path}'
      )
    df = pd.read_csv(path)
    out.append(ModelTable(label=_label_from_run_name(run_name), run_name=run_name, df=df))
  return out


def _validate_table(df: pd.DataFrame, *, score_col: str) -> None:
  """Fail fast if required columns are missing or non-numeric."""
  required = ('target', 'target_signal', 'target_metric', score_col)
  missing = [c for c in required if c not in df.columns]
  if missing:
    raise ValueError(f'Missing required columns in table: missing={missing} available={list(df.columns)}')

  scores = pd.to_numeric(df[score_col], errors='coerce')
  if scores.isna().all():
    raise ValueError(f'All values in {score_col!r} are NaN after numeric coercion.')


def _filter_and_score(
  df: pd.DataFrame,
  *,
  score_col: str,
  exclude_target_signals: set[str],
) -> pd.DataFrame:
  """Return a filtered table with a non-negative radar score column."""
  _validate_table(df, score_col=score_col)

  # Step 1: filter signals (e.g., remove singular events).
  filtered = df[~df['target_signal'].isin(exclude_target_signals)].copy()
  if filtered.empty:
    raise ValueError('After filtering target_signal, table is empty.')

  # Step 2: numeric score + clamp for radar plotting.
  score = pd.to_numeric(filtered[score_col], errors='coerce')
  if not np.all(np.isfinite(score.to_numpy(dtype=float))):
    bad = filtered[~np.isfinite(score)].loc[:, ['target', 'target_signal', 'target_metric', score_col]]
    raise ValueError(f'Non-finite score values found in {score_col!r}: rows=\n{bad}')

  # Clamp to [0, 1] for polar-axis readability.
  filtered['score'] = np.clip(score.to_numpy(dtype=float), a_min=0.0, a_max=1.0)
  return filtered


def _ordered_categories(categories: list[str], *, preferred_order: tuple[str, ...]) -> list[str]:
  """Return categories in preferred order, then any remaining in sorted order."""
  present = set(categories)
  ordered = [c for c in preferred_order if c in present]
  ordered += sorted(present - set(ordered))
  if not ordered:
    raise ValueError('No categories to plot (empty after ordering).')
  return ordered


def _aggregate_group_scores(
  df: pd.DataFrame,
  *,
  group_col: str,
  categories: list[str],
) -> list[float]:
  """Aggregate per-target scores to one score per category (median across targets)."""
  grouped = df.groupby(group_col, sort=False)['score'].median()
  values: list[float] = []
  for cat in categories:
    if cat not in grouped.index:
      raise ValueError(f'Missing required category={cat!r} in group_col={group_col!r}.')
    value = float(grouped.loc[cat])
    if not math.isfinite(value):
      raise ValueError(f'Non-finite aggregated value for category={cat!r}: {value}')
    values.append(value)
  return values


def _compute_radar_aggregates(
  tables: list[ModelTable],
  *,
  score_col: str,
  exclude_target_signals: set[str],
) -> RadarAggregates:
  """Compute aggregated radar values for a given score column."""
  filtered_tables: list[ModelTable] = []
  for table in tables:
    filtered_tables.append(
      ModelTable(
        label=table.label,
        run_name=table.run_name,
        df=_filter_and_score(table.df, score_col=score_col, exclude_target_signals=exclude_target_signals),
      )
    )

  # Chart 1: models vs target_signal.
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
    t.label: _aggregate_group_scores(t.df, group_col='target_signal', categories=signals) for t in filtered_tables
  }

  # Chart 2: models vs target_metric (only metrics with >1 target).
  metric_counts = filtered_tables[0].df.groupby('target_metric', sort=False).size().to_dict()
  metrics_multi = [m for m, n in metric_counts.items() if int(n) > 1]
  metrics_multi = _ordered_categories(categories=[str(x) for x in metrics_multi], preferred_order=_TARGET_METRIC_ORDER)

  for table in filtered_tables[1:]:
    other_counts = table.df.groupby('target_metric', sort=False).size().to_dict()
    other_multi = [m for m, n in other_counts.items() if int(n) > 1]
    if set(other_multi) != set(metrics_multi):
      raise ValueError(
        'target_metric>1 axis mismatch between models after filtering. '
        f'expected={sorted(metrics_multi)} got={sorted(other_multi)} model={table.run_name!r}'
      )

  model_to_metric_values = {
    t.label: _aggregate_group_scores(t.df, group_col='target_metric', categories=metrics_multi) for t in filtered_tables
  }

  return RadarAggregates(
    score_col=score_col,
    signals=signals,
    model_to_signal_values=model_to_signal_values,
    metrics_multi=metrics_multi,
    model_to_metric_values=model_to_metric_values,
  )


def _plot_radar(
  ax: plt.Axes,
  *,
  title: str | None,
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

  ax.set_ylim(0.0, 1.0)
  ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
  ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
  if title is not None and str(title).strip():
    ax.set_title(str(title), pad=16)

  for label, values in model_to_values.items():
    if len(values) != n:
      raise ValueError(f'Value length mismatch for model={label!r}: got {len(values)} expected {n}')
    values_closed = list(values) + [values[0]]
    color = label_to_color.get(label)
    if color is None:
      raise KeyError(f'Missing required color for model label={label!r}. Known={sorted(label_to_color)[:10]}...')
    ax.plot(angles, values_closed, linewidth=2, label=label, color=color)
    ax.fill(angles, values_closed, alpha=0.12, color=color)


def _write_radar_grid_data_tex(
  *,
  out_path: Path,
  metrics: list[str],
  aggregates: list[RadarAggregates],
) -> None:
  """Write the aggregated radar data used for the grid plot as LaTeX tables."""
  if not metrics:
    raise ValueError('metrics must be non-empty')
  if len(metrics) != len(aggregates):
    raise ValueError(f'metrics/aggregates length mismatch: {len(metrics)} vs {len(aggregates)}')

  parts: list[str] = []
  for metric, agg in zip(metrics, aggregates):
    rows_signal = [
      [_latex_escape(model_label), *[_format_float(v, digits=3) for v in values]]
      for model_label, values in agg.model_to_signal_values.items()
    ]
    parts.append(
      _render_latex_table(
        caption=(
          f'Aionoscope dense radar grid data aggregated by target\\_signal '
          f'(score={_latex_tt(agg.score_col)}, clamp to [0,1], median across targets; metric={_latex_tt(metric)}).'
        ),
        label=f'tab:aiono_dense_radar_grid_signal_{metric}',
        header=['model', *[_latex_tt(x) for x in agg.signals]],
        rows=rows_signal,
        resize_to_linewidth=True,
      )
    )

    rows_metric = [
      [_latex_escape(model_label), *[_format_float(v, digits=3) for v in values]]
      for model_label, values in agg.model_to_metric_values.items()
    ]
    parts.append(
      _render_latex_table(
        caption=(
          f'Aionoscope dense radar grid data aggregated by target\\_metric (only metrics with $>1$ targets) '
          f'(score={_latex_tt(agg.score_col)}, clamp to [0,1], median across targets; metric={_latex_tt(metric)}).'
        ),
        label=f'tab:aiono_dense_radar_grid_metric_{metric}',
        header=['model', *[_latex_tt(x) for x in agg.metrics_multi]],
        rows=rows_metric,
        resize_to_linewidth=True,
      )
    )
    parts.append('')

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text('\n'.join(parts).rstrip() + '\n', encoding='utf-8')


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
  exclude_signals = {str(x) for x in args.exclude_target_signal if str(x).strip()}

  metrics = [str(x) for x in args.metrics]
  if not metrics:
    raise ValueError('metrics must be non-empty')

  tables = _load_model_tables(csv_dir=args.csv_dir, run_names=[str(x) for x in args.run_name], step1=int(args.step1))
  colors_by_run_name = _paper_colors_for_runs([t.run_name for t in tables])
  label_to_color = {t.label: colors_by_run_name[t.run_name] for t in tables}

  out_root_dir: Path = args.out_root_dir
  out_fig_dir = out_root_dir / 'figures'
  out_table_dir = out_root_dir / 'tables'
  out_fig_dir.mkdir(parents=True, exist_ok=True)
  out_table_dir.mkdir(parents=True, exist_ok=True)

  aggregates: list[RadarAggregates] = []
  for metric in metrics:
    aggregates.append(
      _compute_radar_aggregates(
        tables,
        score_col=f'{metric}_delta_norm0',
        exclude_target_signals=exclude_signals,
      )
    )

  # Ensure axis categories match across metrics.
  for metric, agg in zip(metrics[1:], aggregates[1:]):
    if agg.signals != aggregates[0].signals:
      raise ValueError(
        'target_signal axis mismatch between metrics. '
        f'first={aggregates[0].signals} metric={metric!r} got={agg.signals}'
      )
    if agg.metrics_multi != aggregates[0].metrics_multi:
      raise ValueError(
        'target_metric>1 axis mismatch between metrics. '
        f'first={aggregates[0].metrics_multi} metric={metric!r} got={agg.metrics_multi}'
      )

  fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=2,
    subplot_kw={'projection': 'polar'},
    figsize=(15, 6.5 * len(metrics)),
    constrained_layout=True,
  )
  axes_arr = np.asarray(axes)
  if axes_arr.ndim == 1:
    axes_arr = axes_arr.reshape(1, -1)
  if axes_arr.shape != (len(metrics), 2):
    raise ValueError(f'Unexpected axes shape for grid: got={axes_arr.shape} expected={(len(metrics), 2)}')

  for row_idx, (metric, agg) in enumerate(zip(metrics, aggregates)):
    ax_signal = axes_arr[row_idx, 0]
    ax_metric = axes_arr[row_idx, 1]

    _plot_radar(
      ax_signal,
      title=None,
      categories=agg.signals,
      model_to_values=agg.model_to_signal_values,
      label_to_color=label_to_color,
    )
    _plot_radar(
      ax_metric,
      title=None,
      categories=agg.metrics_multi,
      model_to_values=agg.model_to_metric_values,
      label_to_color=label_to_color,
    )

    # Row label (metric name).
    fig.text(
      0.01,
      1.0 - (row_idx + 0.5) / len(metrics),
      str(metric).upper(),
      rotation=90,
      va='center',
      ha='left',
      fontsize=_RADAR_FONTSIZE_BASE + 1,
    )

  handles, labels = axes_arr[0, 1].get_legend_handles_labels()
  if not handles:
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)))

  out_png = out_fig_dir / f'{args.out_basename}_grid.png'
  out_pdf = out_fig_dir / f'{args.out_basename}_grid.pdf'
  out_tex = out_table_dir / f'{args.out_basename}_grid_data.tex'
  fig.savefig(out_png, dpi=200)
  fig.savefig(out_pdf)
  plt.close(fig)

  _write_radar_grid_data_tex(out_path=out_tex, metrics=metrics, aggregates=aggregates)

  print(f'[ok] Wrote: {out_png}')
  print(f'[ok] Wrote: {out_pdf}')
  print(f'[ok] Wrote: {out_tex}')


if __name__ == '__main__':
  main()
