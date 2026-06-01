"""Plot Aionoscope categorical (AUROC/AUPRC) per-signal improvement radar charts.

This script consumes the per-signal categorical tables exported by
`utils.wandb_lenepa_paper_plots`:
  `results/tables/<run_name_lower>_categorical_signals_step0_vs_<step1>.csv`

It produces a single figure with two radar ("star") charts:
  1) AUROC improvement per signal (step0-normalized delta)
  2) AUPRC improvement per signal (step0-normalized delta)

Score definition:
  For each signal we use `<metric>_delta_norm0`, which is a step0-normalized
  improvement (headroom-to-1 captured). For radar readability we clamp the
  score to [0, 1], i.e. we plot only non-negative improvements (regressions are
  shown as 0 on the chart).

Outputs (default `--out-root-dir=paper`):
  - figures: `<root>/figures/<basename>.png|.pdf`
  - tables:  `<root>/tables/<basename>_data.tex` (two tables: AUROC and AUPRC)

Usage:
  uv run python -m utils.plot_aiono_categorical_signal_radar
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

from utils.wandb_lenepa_paper_plots import _format_step_suffix, _paper_colors_for_runs


_DEFAULT_RUN_NAMES = (
  'AIONO_LENEPA_SIGREGT20_L0-8_PD0_PROJ_s0',
  'AIONO_NEPA_STOPGRAD_MEAN_PROJ_s0',
  'AIONO_JEPA_CLS_s0',
)

_RADAR_FONTSIZE_BASE = 12


_SIGNAL_ORDER = (
  'constant',
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
    help='Directory containing per-signal categorical CSV exports.',
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
    help='Second training step used in the CSV suffix (default: 20000 -> \"20k\").',
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
    default='aiono_categorical_signal_radar',
    help='Output base name (writes <name>.png/.pdf and <name>_data.tex).',
  )
  return parser.parse_args()


def _label_from_run_name(run_name: str) -> str:
  """Build a compact legend label for a run name."""
  if '_LENEPA_' in run_name:
    parts = run_name.split('_')
    sigreg = next((p for p in parts if p.startswith('SIGREG')), 'LENEPA')
    return f'LeNEPA ({sigreg})'
  if '_NEPA_' in run_name:
    return 'NEPA'
  if '_JEPA_' in run_name:
    variant = next((p for p in run_name.split('_') if p in ('CLS', 'MEAN')), None)
    return f'JEPA ({variant})' if variant is not None else 'JEPA'
  return run_name


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

  ax.set_ylim(0.0, 1.0)
  ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
  ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

  for label, values in model_to_values.items():
    if len(values) != n:
      raise ValueError(f'Value length mismatch for model={label!r}: got {len(values)} expected {n}')
    values_closed = list(values) + [values[0]]
    color = label_to_color.get(label)
    if color is None:
      raise KeyError(f'Missing required color for model label={label!r}. Known={sorted(label_to_color)[:10]}...')
    ax.plot(angles, values_closed, linewidth=2, label=label, color=color)
    ax.fill(angles, values_closed, alpha=0.12, color=color)


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


def _write_radar_data_tex(
  *,
  out_path: Path,
  categories: list[str],
  model_to_auroc: dict[str, list[float]],
  model_to_auprc: dict[str, list[float]],
) -> None:
  """Write the aggregated radar data used for plotting as LaTeX tables."""
  digits = 3
  header = ['model', *[_latex_tt(x) for x in categories]]

  def _rows(model_to_values: dict[str, list[float]]) -> list[list[str]]:
    out_rows: list[list[str]] = []
    for model_label, values in model_to_values.items():
      out_rows.append([_latex_escape(model_label), *[_format_float(v, digits=digits) for v in values]])
    return out_rows

  parts: list[str] = []
  parts.append(
    _render_latex_table(
      caption=(
        f'Aionoscope categorical radar data by signal (score={_latex_tt("auroc_delta_norm0")}, '
        f'clamp to [0,1]; metric={_latex_tt("auroc")}).'
      ),
      label='tab:aiono_categorical_signal_radar_auroc',
      header=header,
      rows=_rows(model_to_auroc),
      resize_to_linewidth=True,
    )
  )
  parts.append(
    _render_latex_table(
      caption=(
        f'Aionoscope categorical radar data by signal (score={_latex_tt("auprc_delta_norm0")}, '
        f'clamp to [0,1]; metric={_latex_tt("auprc")}).'
      ),
      label='tab:aiono_categorical_signal_radar_auprc',
      header=header,
      rows=_rows(model_to_auprc),
      resize_to_linewidth=True,
    )
  )

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text('\n'.join(parts), encoding='utf-8')


def _load_signal_table(csv_path: Path) -> pd.DataFrame:
  """Load and validate one per-signal categorical CSV table."""
  if not csv_path.is_file():
    raise FileNotFoundError(f'Missing categorical-per-signal CSV: {csv_path}')
  df = pd.read_csv(csv_path)
  required = ('signal', 'auroc_delta_norm0', 'auprc_delta_norm0')
  missing = [c for c in required if c not in df.columns]
  if missing:
    raise ValueError(f'Missing required columns in {csv_path}: missing={missing} available={list(df.columns)}')
  return df


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

  # Step 1: load tables.
  csv_dir: Path = args.csv_dir
  run_names = [str(x) for x in args.run_name]
  if not run_names:
    raise ValueError('run_names must be non-empty')

  if int(args.step1) <= 0:
    raise ValueError(f'step1 must be > 0, got: {args.step1}')
  step_suffix = _format_step_suffix(int(args.step1))

  tables: list[tuple[str, str, pd.DataFrame]] = []
  for run_name in run_names:
    path = csv_dir / f'{run_name.lower()}_categorical_signals_step0_vs_{step_suffix}.csv'
    tables.append((run_name, _label_from_run_name(run_name), _load_signal_table(path)))

  colors_by_run_name = _paper_colors_for_runs(run_names)
  label_to_color = {label: colors_by_run_name[run_name] for run_name, label, _df in tables}

  # Step 2: build a shared ordered axis.
  categories = _ordered_categories(
    categories=sorted({str(x) for x in tables[0][2]['signal'].tolist()}),
    preferred_order=_SIGNAL_ORDER,
  )
  for _run_name, label, df in tables[1:]:
    other = sorted({str(x) for x in df['signal'].tolist()})
    if set(other) != set(categories):
      raise ValueError(
        'signal axis mismatch between models. '
        f'expected={categories} got={other} model={label!r}'
      )

  # Step 3: extract clamped scores for plotting.
  model_to_auroc: dict[str, list[float]] = {}
  model_to_auprc: dict[str, list[float]] = {}
  for _run_name, label, df in tables:
    by_signal = df.set_index('signal', drop=False)
    auroc = pd.to_numeric(by_signal['auroc_delta_norm0'], errors='coerce')
    auprc = pd.to_numeric(by_signal['auprc_delta_norm0'], errors='coerce')
    if auroc.isna().any():
      bad = auroc[auroc.isna()]
      raise ValueError(f'Non-numeric auroc_delta_norm0 for model={label!r}: signals={list(bad.index)}')
    if auprc.isna().any():
      bad = auprc[auprc.isna()]
      raise ValueError(f'Non-numeric auprc_delta_norm0 for model={label!r}: signals={list(bad.index)}')

    model_to_auroc[label] = [float(np.clip(float(auroc.loc[s]), 0.0, 1.0)) for s in categories]
    model_to_auprc[label] = [float(np.clip(float(auprc.loc[s]), 0.0, 1.0)) for s in categories]

  # Step 4: plot (two panels: AUROC and AUPRC).
  fig, (ax_auroc, ax_auprc) = plt.subplots(
    nrows=1,
    ncols=2,
    subplot_kw={'projection': 'polar'},
    figsize=(15, 6.5),
    constrained_layout=True,
  )
  _plot_radar(ax_auroc, categories=categories, model_to_values=model_to_auroc, label_to_color=label_to_color)
  _plot_radar(ax_auprc, categories=categories, model_to_values=model_to_auprc, label_to_color=label_to_color)
  ax_auroc.set_title('AUROC', pad=16)
  ax_auprc.set_title('AUPRC', pad=16)

  handles, labels = ax_auprc.get_legend_handles_labels()
  if not handles:
    handles, labels = ax_auroc.get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)))

  # Step 5: write outputs.
  out_root_dir: Path = args.out_root_dir
  out_fig_dir = out_root_dir / 'figures'
  out_table_dir = out_root_dir / 'tables'
  out_fig_dir.mkdir(parents=True, exist_ok=True)
  out_table_dir.mkdir(parents=True, exist_ok=True)

  out_png = out_fig_dir / f'{args.out_basename}_grid.png'
  out_pdf = out_fig_dir / f'{args.out_basename}_grid.pdf'
  out_tex = out_table_dir / f'{args.out_basename}_grid_data.tex'
  fig.savefig(out_png, dpi=200)
  fig.savefig(out_pdf)
  plt.close(fig)

  _write_radar_data_tex(
    out_path=out_tex,
    categories=categories,
    model_to_auroc=model_to_auroc,
    model_to_auprc=model_to_auprc,
  )

  print(f'[ok] Wrote: {out_png}')
  print(f'[ok] Wrote: {out_pdf}')
  print(f'[ok] Wrote: {out_tex}')


if __name__ == '__main__':
  main()
