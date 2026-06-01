from __future__ import annotations

"""W&B logging utilities for offline probes.

These helpers keep `pretrain.py` focused on training / evaluation orchestration,
while sharing consistent logging schemas between legacy (single-layer) and
layered offline probe modes.
"""

from time import perf_counter

import wandb

from utils.probe_metrics import probe_build_wandb_val_metric_logs


def offline_probe_build_prefix(*, root: str, source: str, layer: int | None, multi_source: bool) -> str:
  """Build the W&B log prefix for offline probing.

  Args:
    root: Root prefix (e.g., `offline_probe`, `offline_dense_probe`).
    source: Offline probe dataset source name.
    layer: Optional encoder layer index. `None` means legacy (no layer component).
    multi_source: Whether multiple offline probe sources are being logged.

  Returns:
    W&B prefix string.
  """
  if layer is None:
    return root if not multi_source else f'{root}/{source}'
  return f'{root}/layer_{layer}' if not multi_source else f'{root}/layer_{layer}/{source}'


def offline_probe_build_best_across_layers_categorical_logs(
  *,
  probe_prefix: str,
  categorical_by_layer: dict[int, dict[str, object]],
  group_to_classes: dict[str, list[str]],
  groups: tuple[str, ...],
) -> dict[str, object]:
  """Build W&B logs for best-across-layer categorical probe metrics.

  This is used in offline-probe layer mode to emit a small set of compatibility
  metrics under the legacy prefix (no explicit layer component). The logs include:
  - best macro AUROC/AUPRC across candidate layers
  - which layer achieved each best metric
  - high-level group macro AUROC/AUPRC (e.g., Aiono: events/noise/periodic/trend)

  Args:
    probe_prefix: W&B prefix to write compatibility logs under (e.g., `offline_probe`).
    categorical_by_layer: Layer-indexed metrics produced by
      `offline_probe_run_linear_multihead_by_layer(...).get("categorical")`.
    group_to_classes: Mapping from group name to class names.
    groups: Group names to log (only groups present in group_to_classes are logged).

  Returns:
    Dict of W&B key/value pairs ready to merge into a global log dict.
  """
  # Step 1: validate inputs.
  if not isinstance(probe_prefix, str) or not probe_prefix.strip():
    raise ValueError(f'probe_prefix must be a non-empty string, got {probe_prefix!r}')
  if not categorical_by_layer:
    raise ValueError('categorical_by_layer must be a non-empty dict[int, dict].')
  if not group_to_classes:
    raise ValueError('group_to_classes must be a non-empty dict[str, list[str]].')
  if not groups:
    raise ValueError('groups must be a non-empty tuple[str, ...].')

  def _require_metric_dict(layer: int, *, key: str) -> dict[str, object]:
    metrics = categorical_by_layer.get(layer)
    if metrics is None:
      raise ValueError(f'Missing categorical metrics for layer={layer}.')
    value = metrics.get(key)
    if not isinstance(value, dict):
      raise ValueError(
        f'categorical_by_layer[{layer}][{key!r}] must be a dict, got {type(value).__name__}')
    return value

  def _require_float(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float)):
      raise ValueError(f'{context} must be a number, got {type(value).__name__}')
    return float(value)

  def _require_int(value: object, *, context: str) -> int:
    if not isinstance(value, int):
      raise ValueError(f'{context} must be an int, got {type(value).__name__}')
    return int(value)

  def _group_mean(
    per_class_metric: dict[str, object],
    class_names: list[str],
    *,
    context: str,
  ) -> float:
    """Compute mean metric over class_names (fail-fast on missing class keys)."""
    if not class_names:
      raise ValueError(f'{context} group has no classes.')
    total = 0.0
    for class_name in class_names:
      if class_name not in per_class_metric:
        raise ValueError(
          f'{context} is missing per-class metric for {class_name!r}. '
          f'Known classes: {sorted(per_class_metric)[:10]}...')
      total += _require_float(per_class_metric[class_name], context=f'{context}[{class_name}]')
    return total / float(len(class_names))

  # Step 2: pick best layers for macro AUROC and macro AUPRC.
  best_auc_layer = max(
    categorical_by_layer,
    key=lambda layer: _require_float(_require_metric_dict(layer, key='best_auc').get('macro_auc'),
                                     context=f'layer={layer}.best_auc.macro_auc'),
  )
  best_auprc_layer = max(
    categorical_by_layer,
    key=lambda layer: _require_float(
      _require_metric_dict(layer, key='best_auprc').get('macro_auprc'),
      context=f'layer={layer}.best_auprc.macro_auprc'),
  )

  best_auc = _require_metric_dict(best_auc_layer, key='best_auc')
  best_auprc = _require_metric_dict(best_auprc_layer, key='best_auprc')

  best_macro_auc = _require_float(best_auc.get('macro_auc'), context='best_auc.macro_auc')
  best_macro_auprc = _require_float(best_auprc.get('macro_auprc'), context='best_auprc.macro_auprc')
  best_probe_step = _require_int(best_auc.get('best_probe_step'), context='best_auc.best_probe_step')
  best_macro_auprc_step = _require_int(
    best_auprc.get('best_probe_step'),
    context='best_auprc.best_probe_step',
  )

  # Step 3: log macro metrics and which layer achieved them.
  logs: dict[str, object] = {
    f'{probe_prefix}/best_macro_auc': best_macro_auc,
    f'{probe_prefix}/best_macro_auc_layer': int(best_auc_layer),
    f'{probe_prefix}/best_probe_step': best_probe_step,
    f'{probe_prefix}/best_macro_auprc': best_macro_auprc,
    f'{probe_prefix}/best_macro_auprc_layer': int(best_auprc_layer),
    f'{probe_prefix}/best_macro_auprc_step': best_macro_auprc_step,
  }

  # Step 4: log high-level grouped metrics for the selected best layers.
  if not isinstance(best_auc.get('per_class_auc'), dict):
    raise ValueError('best_auc.per_class_auc must be a dict[str, float].')
  if not isinstance(best_auc.get('per_class_auprc'), dict):
    raise ValueError('best_auc.per_class_auprc must be a dict[str, float].')
  if not isinstance(best_auprc.get('per_class_auc'), dict):
    raise ValueError('best_auprc.per_class_auc must be a dict[str, float].')
  if not isinstance(best_auprc.get('per_class_auprc'), dict):
    raise ValueError('best_auprc.per_class_auprc must be a dict[str, float].')

  best_auc_per_class_auc = best_auc['per_class_auc']
  best_auc_per_class_auprc = best_auc['per_class_auprc']
  best_auprc_per_class_auc = best_auprc['per_class_auc']
  best_auprc_per_class_auprc = best_auprc['per_class_auprc']

  for group_name in groups:
    group_classes = group_to_classes.get(group_name)
    if not group_classes:
      continue
    logs[f'{probe_prefix}/best_macro_auc_group/auc/{group_name}'] = _group_mean(
      best_auc_per_class_auc,
      group_classes,
      context=f'best_macro_auc_group_auc/{group_name}',
    )
    logs[f'{probe_prefix}/best_macro_auc_group/auprc/{group_name}'] = _group_mean(
      best_auc_per_class_auprc,
      group_classes,
      context=f'best_macro_auc_group_auprc/{group_name}',
    )
    logs[f'{probe_prefix}/best_macro_auprc_group/auc/{group_name}'] = _group_mean(
      best_auprc_per_class_auc,
      group_classes,
      context=f'best_macro_auprc_group_auc/{group_name}',
    )
    logs[f'{probe_prefix}/best_macro_auprc_group/auprc/{group_name}'] = _group_mean(
      best_auprc_per_class_auprc,
      group_classes,
      context=f'best_macro_auprc_group_auprc/{group_name}',
    )

  return logs


def offline_probe_build_classification_logs(
  *,
  probe_prefix: str,
  best_auprc_prefix: str,
  best_auc: dict[str, object],
  best_auprc: dict[str, object],
  group_to_classes: dict[str, list[str]],
  feature_dim: int,
  train_size: int,
  val_size: int,
  val_num_crops: int,
) -> tuple[dict[str, object], dict[str, object]]:
  """Build W&B logs for offline multi-label classification probing.

  Returns:
    (best_auc_logs, best_auprc_logs) where each dict is ready to be merged into
    the global log dict.
  """
  def _add_seed_stats(
    *,
    logs: dict[str, object],
    prefix: str,
    metrics: dict[str, object],
  ) -> None:
    """Append optional validation-seed spread metrics when present."""
    macro_auc_std = metrics.get('macro_auc_std')
    if isinstance(macro_auc_std, (int, float)):
      logs[f'{prefix}/val/probe_auc_std'] = float(macro_auc_std)
    macro_auprc_std = metrics.get('macro_auprc_std')
    if isinstance(macro_auprc_std, (int, float)):
      logs[f'{prefix}/val/probe_auprc_std'] = float(macro_auprc_std)

    per_class_auc_std = metrics.get('per_class_auc_std')
    if isinstance(per_class_auc_std, dict):
      for class_name, value in per_class_auc_std.items():
        logs[f'{prefix}/val/probe_auc_std/{class_name}'] = float(value)
    per_class_auprc_std = metrics.get('per_class_auprc_std')
    if isinstance(per_class_auprc_std, dict):
      for class_name, value in per_class_auprc_std.items():
        logs[f'{prefix}/val/probe_auprc_std/{class_name}'] = float(value)

    validation_seed_count = metrics.get('validation_seed_count')
    if isinstance(validation_seed_count, int):
      logs[f'{prefix}/validation_seed_count'] = int(validation_seed_count)

  # Step 1: build best-AUC logs for this prefix.
  best_auc_logs = probe_build_wandb_val_metric_logs(
    macro_auc=best_auc['macro_auc'],
    per_class_auc=best_auc['per_class_auc'],
    macro_auprc=best_auc['macro_auprc'],
    per_class_auprc=best_auc['per_class_auprc'],
    group_to_classes=group_to_classes)
  best_auc_logs = {
    f'{probe_prefix}/{key}': value for key, value in best_auc_logs.items()
  }
  best_auc_logs[f'{probe_prefix}/best_probe_step'] = best_auc['best_probe_step']
  best_auc_logs[f'{probe_prefix}/best_macro_auc'] = best_auc['macro_auc']
  best_auc_logs[f'{probe_prefix}/best_macro_auprc'] = best_auprc['macro_auprc']
  best_auc_logs[f'{probe_prefix}/best_macro_auprc_step'] = best_auprc['best_probe_step']
  if isinstance(best_auc.get('macro_auc_std'), (int, float)):
    best_auc_logs[f'{probe_prefix}/best_macro_auc_std'] = float(best_auc['macro_auc_std'])
  if isinstance(best_auprc.get('macro_auprc_std'), (int, float)):
    best_auc_logs[f'{probe_prefix}/best_macro_auprc_std'] = float(best_auprc['macro_auprc_std'])
  best_auc_logs[f'{probe_prefix}/feature_dim'] = int(feature_dim)
  best_auc_logs[f'{probe_prefix}/train_size'] = int(train_size)
  best_auc_logs[f'{probe_prefix}/val_size'] = int(val_size)
  best_auc_logs[f'{probe_prefix}/val_num_crops'] = int(val_num_crops)
  _add_seed_stats(logs=best_auc_logs, prefix=probe_prefix, metrics=best_auc)

  # Step 2: build best-AUPRC logs for this prefix.
  best_auprc_logs = probe_build_wandb_val_metric_logs(
    macro_auc=best_auprc['macro_auc'],
    per_class_auc=best_auprc['per_class_auc'],
    macro_auprc=best_auprc['macro_auprc'],
    per_class_auprc=best_auprc['per_class_auprc'],
    group_to_classes=group_to_classes)
  best_auprc_logs = {
    f'{best_auprc_prefix}/{key}': value for key, value in best_auprc_logs.items()
  }
  best_auprc_logs[f'{best_auprc_prefix}/best_probe_step'] = best_auprc['best_probe_step']
  best_auprc_logs[f'{best_auprc_prefix}/feature_dim'] = int(feature_dim)
  best_auprc_logs[f'{best_auprc_prefix}/train_size'] = int(train_size)
  best_auprc_logs[f'{best_auprc_prefix}/val_size'] = int(val_size)
  best_auprc_logs[f'{best_auprc_prefix}/val_num_crops'] = int(val_num_crops)
  _add_seed_stats(logs=best_auprc_logs, prefix=best_auprc_prefix, metrics=best_auprc)
  return best_auc_logs, best_auprc_logs


def offline_probe_build_dense_logs(
  *,
  dense_prefix: str,
  dense_best: dict[str, object],
  feature_dim: int,
  train_size: int,
  val_size: int,
  val_num_crops: int,
) -> dict[str, object]:
  """Build W&B logs for offline dense regression probing."""
  # Step 1: log macro regression metrics and dataset metadata.
  dense_logs = {
    f'{dense_prefix}/mse': dense_best['macro_mse'],
    f'{dense_prefix}/mae': dense_best['macro_mae'],
    f'{dense_prefix}/r2': dense_best['macro_r2'],
    f'{dense_prefix}/pearson': dense_best['macro_pearson'],
    f'{dense_prefix}/feature_dim': int(feature_dim),
    f'{dense_prefix}/train_size': int(train_size),
    f'{dense_prefix}/val_size': int(val_size),
    f'{dense_prefix}/val_num_crops': int(val_num_crops),
  }
  for metric_name in ('mse', 'mae', 'r2', 'pearson'):
    metric_std = dense_best.get(f'macro_{metric_name}_std')
    if isinstance(metric_std, (int, float)):
      dense_logs[f'{dense_prefix}/{metric_name}_std'] = float(metric_std)
  best_probe_step = dense_best.get('best_probe_step')
  if best_probe_step is not None:
    dense_logs[f'{dense_prefix}/best_probe_step'] = best_probe_step
  validation_seed_count = dense_best.get('validation_seed_count')
  if isinstance(validation_seed_count, int):
    dense_logs[f'{dense_prefix}/validation_seed_count'] = int(validation_seed_count)

  # Step 2: optionally log per-target regression metrics.
  per_target_mse = dense_best.get('per_target_mse')
  per_target_mae = dense_best.get('per_target_mae')
  per_target_r2 = dense_best.get('per_target_r2')
  per_target_pearson = dense_best.get('per_target_pearson')
  per_target_best_step = dense_best.get('per_target_best_step')
  if per_target_mse is not None:
    for target_name, value in per_target_mse.items():
      dense_logs[f'{dense_prefix}/target_mse/{target_name}'] = value
  if per_target_mae is not None:
    for target_name, value in per_target_mae.items():
      dense_logs[f'{dense_prefix}/target_mae/{target_name}'] = value
  if per_target_r2 is not None:
    for target_name, value in per_target_r2.items():
      dense_logs[f'{dense_prefix}/target_r2/{target_name}'] = value
  if per_target_pearson is not None:
    for target_name, value in per_target_pearson.items():
      dense_logs[f'{dense_prefix}/target_pearson/{target_name}'] = value
  for metric_name in ('mse', 'mae', 'r2', 'pearson'):
    metric_values = dense_best.get(f'per_target_{metric_name}_std')
    if isinstance(metric_values, dict):
      for target_name, value in metric_values.items():
        dense_logs[f'{dense_prefix}/target_{metric_name}_std/{target_name}'] = value
  if per_target_best_step is not None:
    for target_name, value in per_target_best_step.items():
      dense_logs[f'{dense_prefix}/target_best_step/{target_name}'] = value
  return dense_logs


def wandb_build_pairwise_confusion_artifacts(
  *,
  pairwise_confusion: dict[str, object],
  title: str,
  timings: dict[str, float | int] | None = None,
) -> tuple[wandb.Image, wandb.Table]:
  """Build W&B artifacts for threshold-free multi-label pairwise confusion.

  The expected `pairwise_confusion` schema is produced by
  `utils.probe_metrics.probe_compute_pairwise_confusion(...)`:
    - mean_pred[i][j] = mean(p_hat[j] | y[i]=1 AND y[j]=0) for i!=j
    - mean_pred[i][i] = mean(p_hat[i] | y[i]=1)

  Returns:
    (wandb.Image, wandb.Table): a labeled heatmap image and a long-form table.
  """
  total_start = perf_counter()
  if not isinstance(pairwise_confusion, dict):
    raise ValueError(
      'pairwise_confusion must be a dict, '
      f'got {type(pairwise_confusion).__name__}')
  class_names = pairwise_confusion.get('class_names')
  mean_pred = pairwise_confusion.get('mean_pred')
  counts = pairwise_confusion.get('count')
  if not isinstance(class_names, list) or not class_names:
    raise ValueError('pairwise_confusion.class_names must be a non-empty list[str].')
  if not isinstance(mean_pred, list) or not mean_pred:
    raise ValueError('pairwise_confusion.mean_pred must be a non-empty list[list[float]].')
  if not isinstance(counts, list) or not counts:
    raise ValueError('pairwise_confusion.count must be a non-empty list[list[int]].')
  num_classes = len(class_names)
  if len(mean_pred) != num_classes or len(counts) != num_classes:
    raise ValueError(
      'pairwise_confusion size mismatch. '
      f'Got len(mean_pred)={len(mean_pred)}, len(count)={len(counts)}, '
      f'len(class_names)={num_classes}')

  # Local imports keep the training hot path lightweight.
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # noqa: E402
  import numpy as np  # noqa: E402

  mean_pred_np = np.asarray(mean_pred, dtype=np.float32)  # [C, C]
  count_np = np.asarray(counts, dtype=np.int64)  # [C, C]
  if mean_pred_np.shape != (num_classes, num_classes):
    raise ValueError(
      'pairwise_confusion.mean_pred must be a square matrix. '
      f'Got shape {mean_pred_np.shape}, expected {(num_classes, num_classes)}')
  if count_np.shape != (num_classes, num_classes):
    raise ValueError(
      'pairwise_confusion.count must be a square matrix. '
      f'Got shape {count_np.shape}, expected {(num_classes, num_classes)}')

  # Step 1: build a labeled heatmap image.
  heatmap_start = perf_counter()
  fig_size = max(7.0, 0.25 * num_classes)
  fig, ax = plt.subplots(figsize=(fig_size, fig_size))
  cmap = plt.get_cmap('magma').copy()
  cmap.set_bad(color='#dddddd')
  img = ax.imshow(mean_pred_np, vmin=0.0, vmax=1.0, cmap=cmap, interpolation='nearest')
  ax.set_title(title)
  ax.set_xlabel('predicted class')
  ax.set_ylabel('true class (conditioned on y_true=1)')
  ax.set_xticks(range(num_classes))
  ax.set_yticks(range(num_classes))
  ax.set_xticklabels(class_names, rotation=90)
  ax.set_yticklabels(class_names)
  ax.tick_params(axis='both', which='major', labelsize=6)
  fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
  fig.tight_layout()
  heatmap = wandb.Image(fig)
  plt.close(fig)
  heatmap_s = perf_counter() - heatmap_start

  # Step 2: build a long-form table for sorting/filtering in W&B.
  table_start = perf_counter()
  rows = []
  for true_index, true_name in enumerate(class_names):
    for pred_index, pred_name in enumerate(class_names):
      value = float(mean_pred_np[true_index, pred_index])
      if not np.isfinite(value):
        value = None
      rows.append([
        true_name,
        pred_name,
        value,
        int(count_np[true_index, pred_index]),
      ])
  table = wandb.Table(
    data=rows,
    columns=['true_class', 'pred_class', 'mean_pred', 'count'])
  table_s = perf_counter() - table_start
  total_s = perf_counter() - total_start
  if timings is not None:
    timings.update({
      'total_s': float(total_s),
      'heatmap_s': float(heatmap_s),
      'table_s': float(table_s),
      'num_classes': int(num_classes),
      'rows': int(len(rows)),
    })
  return heatmap, table


def offline_probe_build_confusion_logs(
  *,
  probe_prefix: str,
  best_auprc_prefix: str,
  best_auc: dict[str, object],
  best_auprc: dict[str, object],
  title_suffix: str,
  error_context: str,
  timing_logs: dict[str, float | int] | None,
) -> dict[str, object]:
  """Build W&B pairwise confusion artifacts for offline probes.

  Args:
    probe_prefix: Base prefix for best-AUC logs.
    best_auprc_prefix: Base prefix for best-AUPRC logs.
    best_auc: Metrics dict for the best-AUC checkpoint (may contain pairwise_confusion).
    best_auprc: Metrics dict for the best-AUPRC checkpoint (may contain pairwise_confusion).
    title_suffix: Suffix appended to the artifact title (e.g., ` source=...`).
    error_context: Human-readable identifier used in error messages.
    timing_logs: Optional dict updated with artifact build timings.

  Returns:
    Dict containing W&B artifacts under the relevant prefixes.
  """
  confusion_logs: dict[str, object] = {}

  # Step 1: build best-AUC pairwise confusion artifacts.
  pairwise_confusion_auc = best_auc.get('pairwise_confusion')
  if pairwise_confusion_auc is None:
    raise ValueError(
      f'Offline probe best_auc is missing pairwise_confusion for {error_context}.')
  auc_confusion_timings = {} if timing_logs is not None else None
  auc_heatmap, auc_table = wandb_build_pairwise_confusion_artifacts(
    pairwise_confusion=pairwise_confusion_auc,
    title=(
      'Offline probe pairwise confusion (mean p(pred_j) | y_i=1, y_j=0) '
      f'[best_auc probe_step={best_auc["best_probe_step"]}]{title_suffix}'),
    timings=auc_confusion_timings)
  confusion_logs[f'{probe_prefix}/val/pairwise_confusion_heatmap'] = auc_heatmap
  confusion_logs[f'{probe_prefix}/val/pairwise_confusion_table'] = auc_table
  if timing_logs is not None and auc_confusion_timings:
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auc_total_s'] = float(
      auc_confusion_timings.get('total_s', 0.0))
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auc_heatmap_s'] = float(
      auc_confusion_timings.get('heatmap_s', 0.0))
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auc_table_s'] = float(
      auc_confusion_timings.get('table_s', 0.0))

  # Step 2: build best-AUPRC pairwise confusion artifacts.
  pairwise_confusion_auprc = best_auprc.get('pairwise_confusion')
  if pairwise_confusion_auprc is None:
    raise ValueError(
      f'Offline probe best_auprc is missing pairwise_confusion for {error_context}.')
  auprc_confusion_timings = {} if timing_logs is not None else None
  auprc_heatmap, auprc_table = wandb_build_pairwise_confusion_artifacts(
    pairwise_confusion=pairwise_confusion_auprc,
    title=(
      'Offline probe pairwise confusion (mean p(pred_j) | y_i=1, y_j=0) '
      f'[best_auprc probe_step={best_auprc["best_probe_step"]}]{title_suffix}'),
    timings=auprc_confusion_timings)
  confusion_logs[f'{best_auprc_prefix}/val/pairwise_confusion_heatmap'] = auprc_heatmap
  confusion_logs[f'{best_auprc_prefix}/val/pairwise_confusion_table'] = auprc_table
  if timing_logs is not None and auprc_confusion_timings:
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auprc_total_s'] = float(
      auprc_confusion_timings.get('total_s', 0.0))
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auprc_heatmap_s'] = float(
      auprc_confusion_timings.get('heatmap_s', 0.0))
    timing_logs[f'{probe_prefix}/time/wandb_pairwise_confusion_best_auprc_table_s'] = float(
      auprc_confusion_timings.get('table_s', 0.0))
  return confusion_logs
