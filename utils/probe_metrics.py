from __future__ import annotations

from collections import defaultdict
from os import path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def probe_compute_metrics(
  *, targets: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> tuple[float, dict[str, float], float, dict[str, float]]:
  if targets.shape != predictions.shape:
    raise ValueError(
      f'Probe targets/predictions shape mismatch: targets={targets.shape}, '
      f'predictions={predictions.shape}')
  if targets.ndim != 2:
    raise ValueError(f'Probe targets must be 2D (num_samples, num_classes), got: {targets.shape}')
  if not class_names:
    raise ValueError('probe_class_names is required to compute per-class AUC metrics')

  num_classes = targets.shape[1]
  if len(class_names) != num_classes:
    raise ValueError(
      f'Expected {num_classes} class names for probe metrics, got {len(class_names)}')

  invalid_classes = []
  for class_index, class_name in enumerate(class_names):
    class_targets = targets[:, class_index]
    positive_count = int(np.count_nonzero(class_targets))
    negative_count = class_targets.size - positive_count
    if positive_count == 0 or negative_count == 0:
      invalid_classes.append(f'{class_name} (pos={positive_count}, neg={negative_count})')

  if invalid_classes:
    invalid_detail = ', '.join(invalid_classes)
    raise ValueError(
      'AUC is undefined for classes with only one label in the validation set (fold 9): '
      f'{invalid_detail}. Ensure each class has both positive and negative examples.')

  per_class_auc = {}
  per_class_auprc = {}
  for class_index, class_name in enumerate(class_names):
    class_targets = targets[:, class_index]
    class_predictions = predictions[:, class_index]
    per_class_auc[class_name] = float(
      roc_auc_score(y_true=class_targets, y_score=class_predictions))
    per_class_auprc[class_name] = float(
      average_precision_score(y_true=class_targets, y_score=class_predictions))

  macro_auc = float(np.mean(list(per_class_auc.values())))
  macro_auprc = float(np.mean(list(per_class_auprc.values())))
  return macro_auc, per_class_auc, macro_auprc, per_class_auprc


def probe_compute_pairwise_confusion(
  *,
  targets: np.ndarray,
  predictions: np.ndarray,
  class_names: list[str],
) -> dict[str, object]:
  """Compute a threshold-free pairwise confusion matrix for multi-label probes.

  For multi-label probes, each class is a binary task (no single multi-class confusion matrix).
  This function computes an *ordered* pairwise matrix that answers:

    "When true class i is present, how strongly does the probe predict class j?"

  The returned matrix is:
    - Off-diagonal (i != j):
        mean_pred[i, j] = mean(p_hat[j] | y[i]=1 AND y[j]=0)
      i.e. average predicted probability for class j on samples where i is present
      and j is explicitly absent (helps measure confusion independent of co-occurrence).
    - Diagonal (i == i):
        mean_pred[i, i] = mean(p_hat[i] | y[i]=1)

  The function also returns the sample counts used for each cell:
    - count[i, j] = number of samples in the condition above.
      Cells with count==0 yield mean_pred==NaN.

  Args:
    targets: Binary targets array with shape [N, C].
    predictions: Predicted probabilities array with shape [N, C].
    class_names: List of length C, used for validation and downstream visualization.

  Returns:
    Dict with:
      - class_names: list[str]
      - mean_pred: list[list[float]]  (C x C; may contain NaN)
      - count: list[list[int]]        (C x C)
  """
  if targets.shape != predictions.shape:
    raise ValueError(
      f'Probe targets/predictions shape mismatch: targets={targets.shape}, '
      f'predictions={predictions.shape}')
  if targets.ndim != 2:
    raise ValueError(
      f'Probe targets must be 2D (num_samples, num_classes), got: {targets.shape}')
  if not class_names:
    raise ValueError('probe_class_names is required to compute pairwise confusion')

  num_classes = targets.shape[1]
  if len(class_names) != num_classes:
    raise ValueError(
      f'Expected {num_classes} class names for pairwise confusion, got {len(class_names)}')

  targets_f = targets.astype(np.float32)  # [N, C]
  unique_targets = np.unique(targets_f)
  if not np.all(np.isin(unique_targets, [0.0, 1.0])):
    raise ValueError(
      'Pairwise confusion requires binary targets in {0,1}. '
      f'Got unique values: {unique_targets!r}')

  predictions_f = predictions.astype(np.float32)  # [N, C]
  if np.any(predictions_f < 0.0) or np.any(predictions_f > 1.0):
    raise ValueError('Pairwise confusion requires predicted probabilities in [0, 1].')

  present = targets_f  # [N, C]
  absent = 1.0 - targets_f  # [N, C]
  # counts_offdiag[i, j] = sum_n y[n,i] * (1 - y[n,j])  # [C, C]
  counts_offdiag = present.T @ absent  # [C, C]
  # numerator_offdiag[i, j] = sum_n y[n,i] * (1 - y[n,j]) * p_hat[n,j]  # [C, C]
  numerator_offdiag = present.T @ (absent * predictions_f)  # [C, C]

  with np.errstate(divide='ignore', invalid='ignore'):
    mean_pred = numerator_offdiag / counts_offdiag  # [C, C] (NaN where count==0)
  counts = counts_offdiag.astype(np.int64)  # [C, C]

  diag_counts = np.sum(present, axis=0).astype(np.int64)  # [C]
  diag_numerator = np.sum(present * predictions_f, axis=0)  # [C]
  with np.errstate(divide='ignore', invalid='ignore'):
    diag_mean = diag_numerator / diag_counts  # [C] (NaN if class missing positives)

  diag_idx = np.arange(num_classes)
  mean_pred[diag_idx, diag_idx] = diag_mean
  counts[diag_idx, diag_idx] = diag_counts

  return {
    'class_names': list(class_names),
    'mean_pred': mean_pred.tolist(),
    'count': counts.tolist(),
  }


def probe_compute_pairwise_confusion_torch(
  *,
  targets: torch.Tensor,
  predictions: torch.Tensor,
  class_names: list[str],
) -> dict[str, object]:
  """Compute a threshold-free pairwise confusion matrix with torch tensors.

  This mirrors `probe_compute_pairwise_confusion` but allows GPU execution for
  the core matrix multiplications while preserving float32 math.
  """
  # Step 1: validate shapes, devices, and class metadata.
  if targets.shape != predictions.shape:
    raise ValueError(
      'Probe targets/predictions shape mismatch: '
      f'targets={tuple(targets.shape)}, predictions={tuple(predictions.shape)}')
  if targets.dim() != 2:
    raise ValueError(
      f'Probe targets must be 2D (num_samples, num_classes), got: {tuple(targets.shape)}')
  if targets.device != predictions.device:
    raise ValueError(
      'Probe targets/predictions device mismatch: '
      f'targets={targets.device}, predictions={predictions.device}')
  if not class_names:
    raise ValueError('probe_class_names is required to compute pairwise confusion')

  num_classes = int(targets.size(1))
  if len(class_names) != num_classes:
    raise ValueError(
      f'Expected {num_classes} class names for pairwise confusion, got {len(class_names)}')

  # Step 2: validate target/prediction values and cast to float32.
  targets_f = targets.to(dtype=torch.float32)  # [N, C]
  unique_targets = torch.unique(targets_f)  # [U]
  invalid_targets = torch.any((unique_targets != 0.0) & (unique_targets != 1.0))  # []
  if bool(invalid_targets):
    raise ValueError(
      'Pairwise confusion requires binary targets in {0,1}. '
      f'Got unique values: {unique_targets.tolist()!r}')

  predictions_f = predictions.to(dtype=torch.float32)  # [N, C]
  if torch.any(predictions_f < 0.0) or torch.any(predictions_f > 1.0):
    raise ValueError('Pairwise confusion requires predicted probabilities in [0, 1].')

  # Step 3: compute off-diagonal counts and numerators.
  present = targets_f  # [N, C]
  absent = 1.0 - targets_f  # [N, C]
  counts_offdiag = torch.matmul(present.transpose(0, 1), absent)  # [C, C]
  numerator_offdiag = torch.matmul(
    present.transpose(0, 1), absent * predictions_f)  # [C, C]
  mean_pred = numerator_offdiag / counts_offdiag  # [C, C]
  counts = counts_offdiag.to(torch.int64)  # [C, C]

  # Step 4: fill diagonals and move outputs to CPU lists.
  diag_counts = torch.sum(present, dim=0).to(torch.int64)  # [C]
  diag_numerator = torch.sum(present * predictions_f, dim=0)  # [C]
  diag_mean = diag_numerator / diag_counts  # [C]
  diag_idx = torch.arange(num_classes, device=targets.device)  # [C]
  mean_pred[diag_idx, diag_idx] = diag_mean
  counts[diag_idx, diag_idx] = diag_counts

  return {
    'class_names': list(class_names),
    'mean_pred': mean_pred.cpu().tolist(),
    'count': counts.cpu().tolist(),
  }


def probe_compute_regression_metrics(
  *,
  targets: np.ndarray,
  predictions: np.ndarray,
  target_names: list[str],
  return_per_target: bool = True,
) -> tuple[dict[str, float], dict[str, dict[str, float]] | None]:
  """Compute macro and optional per-target regression metrics for dense probes."""
  # Step 1: validate shapes and target names.
  if targets.shape != predictions.shape:
    raise ValueError(
      'Dense probe targets/predictions shape mismatch: '
      f'targets={targets.shape}, predictions={predictions.shape}')
  if targets.ndim != 2:
    raise ValueError(
      'Dense probe targets must be 2D (num_samples, num_targets), '
      f'got: {targets.shape}')
  if not target_names:
    raise ValueError('dense target_names is required to compute regression metrics')
  num_targets = targets.shape[1]
  if len(target_names) != num_targets:
    raise ValueError(
      f'Expected {num_targets} dense target names, got {len(target_names)}')

  # Step 2: compute per-target error metrics.
  errors = predictions - targets  # [N, D]
  mse_per_target = np.mean(np.square(errors), axis=0)  # [D]
  mae_per_target = np.mean(np.abs(errors), axis=0)  # [D]

  # Step 3: compute per-target R2 and Pearson correlation.
  target_mean = np.mean(targets, axis=0)  # [D]
  pred_mean = np.mean(predictions, axis=0)  # [D]
  target_centered = targets - target_mean  # [N, D]
  pred_centered = predictions - pred_mean  # [N, D]

  ss_res = np.sum(np.square(errors), axis=0)  # [D]
  ss_tot = np.sum(np.square(target_centered), axis=0)  # [D]
  if np.any(ss_tot == 0):
    zero_var = [name for name, var in zip(target_names, ss_tot, strict=True) if var == 0]
    raise ValueError(
      'Dense targets have zero variance; R2 undefined for: '
      f'{", ".join(zero_var)}')
  r2_per_target = 1.0 - (ss_res / ss_tot)  # [D]

  denom = np.sqrt(np.sum(np.square(target_centered), axis=0)) * np.sqrt(
    np.sum(np.square(pred_centered), axis=0))  # [D]
  if np.any(denom == 0):
    zero_std = [name for name, value in zip(target_names, denom, strict=True) if value == 0]
    raise ValueError(
      'Dense targets/predictions have zero std; Pearson undefined for: '
      f'{", ".join(zero_std)}')
  pearson_per_target = np.sum(target_centered * pred_centered, axis=0) / denom  # [D]

  # Step 4: package macro and per-target metrics.
  macro_metrics = {
    'mse': float(np.mean(mse_per_target)),
    'mae': float(np.mean(mae_per_target)),
    'r2': float(np.mean(r2_per_target)),
    'pearson': float(np.mean(pearson_per_target)),
  }
  if not return_per_target:
    return macro_metrics, None

  per_target_metrics = {}
  for name, mse, mae, r2, pearson in zip(
    target_names,
    mse_per_target,
    mae_per_target,
    r2_per_target,
    pearson_per_target,
    strict=True,
  ):
    per_target_metrics[name] = {
      'mse': float(mse),
      'mae': float(mae),
      'r2': float(r2),
      'pearson': float(pearson),
    }
  return macro_metrics, per_target_metrics


def probe_build_wandb_val_metric_logs(
  *,
  macro_auc: float,
  per_class_auc: dict[str, float],
  macro_auprc: float,
  per_class_auprc: dict[str, float],
  group_to_classes: dict[str, list[str]],
) -> dict[str, float]:
  if not group_to_classes:
    raise ValueError('probe_group_to_classes is required to log grouped probe metrics')

  per_class_log = {f'val/probe_auc/{class_name}': auc for class_name, auc in per_class_auc.items()}
  per_class_auprc_log = {
    f'val/probe_auprc/{class_name}': auprc for class_name, auprc in per_class_auprc.items()}

  group_auc_log = {}
  group_auprc_log = {}
  per_class_by_group_log = {}
  per_class_by_group_auprc_log = {}
  for group_name, group_classes in group_to_classes.items():
    group_auc = float(np.mean([per_class_auc[class_name] for class_name in group_classes]))
    group_auprc = float(np.mean([per_class_auprc[class_name] for class_name in group_classes]))
    group_auc_log[f'val/probe_auc_group/{group_name}'] = group_auc
    group_auprc_log[f'val/probe_auprc_group/{group_name}'] = group_auprc
    for class_name in group_classes:
      per_class_by_group_log[f'val/probe_auc_by_group/{group_name}/{class_name}'] = per_class_auc[class_name]
      per_class_by_group_auprc_log[f'val/probe_auprc_by_group/{group_name}/{class_name}'] = (
        per_class_auprc[class_name])

  return {
    'val/probe_auc': macro_auc,
    'val/probe_auprc': macro_auprc,
    **per_class_log,
    **per_class_auprc_log,
    **group_auc_log,
    **group_auprc_log,
    **per_class_by_group_log,
    **per_class_by_group_auprc_log,
  }


def _probe_sanitize_wandb_key_component(component: str) -> str:
  component = component.strip()
  if not component:
    raise ValueError('W&B key component must be a non-empty string')
  return component.replace(' ', '_').replace('/', '-')


def _ptbxl_load_scp_statements(ptb_xl_data_dir: str) -> pd.DataFrame:
  scp_path = path.join(ptb_xl_data_dir, 'scp_statements.csv')
  if not path.isfile(scp_path):
    raise ValueError(
      f'Missing PTB-XL file: {scp_path}. Ensure ptb_xl_data_dir points to the raw PTB-XL directory.')
  return pd.read_csv(scp_path, index_col=0)


def probe_build_class_groups_from_scp_statements(
  *, scp_statements: pd.DataFrame, class_names: list[str]
) -> dict[str, list[str]]:
  missing_classes = [class_name for class_name in class_names if class_name not in scp_statements.index]
  if missing_classes:
    raise ValueError(
      'SCP statement metadata is missing for the following probe classes: '
      f'{", ".join(missing_classes)}. Ensure scp_statements.csv matches the PTB-XL label set.')

  class_to_groups: dict[str, list[str]] = {}
  classes_without_groups: list[str] = []

  for class_name in class_names:
    row = scp_statements.loc[class_name]
    groups: list[str] = []
    if float(row.get('rhythm', 0.0)) == 1.0:
      groups.append('rhythm')
    if float(row.get('form', 0.0)) == 1.0:
      groups.append('form')
    if float(row.get('diagnostic', 0.0)) == 1.0:
      groups.append('diagnostic')

    diagnostic_class = row.get('diagnostic_class')
    if isinstance(diagnostic_class, str) and diagnostic_class:
      groups.append(f'diagnostic_class/{_probe_sanitize_wandb_key_component(diagnostic_class)}')

    diagnostic_subclass = row.get('diagnostic_subclass')
    if isinstance(diagnostic_subclass, str) and diagnostic_subclass:
      groups.append(f'diagnostic_subclass/{_probe_sanitize_wandb_key_component(diagnostic_subclass)}')

    if not groups:
      classes_without_groups.append(class_name)
      continue

    class_to_groups[class_name] = sorted(set(groups))

  if classes_without_groups:
    raise ValueError(
      'No grouping metadata found in scp_statements.csv for the following probe classes: '
      f'{", ".join(classes_without_groups)}. Expected at least one of: rhythm/form/diagnostic/diagnostic_class/diagnostic_subclass.')

  return class_to_groups


def probe_build_auc_groups(
  *, ptb_xl_data_dir: str, class_names: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
  scp_statements = _ptbxl_load_scp_statements(ptb_xl_data_dir)
  class_to_groups = probe_build_class_groups_from_scp_statements(
    scp_statements=scp_statements,
    class_names=class_names)

  group_to_classes: dict[str, list[str]] = defaultdict(list)
  for class_name, groups in class_to_groups.items():
    for group_name in groups:
      group_to_classes[group_name].append(class_name)

  group_to_classes_sorted = {group: sorted(classes) for group, classes in group_to_classes.items()}
  if not group_to_classes_sorted:
    raise ValueError('Failed to build probe AUC groups from scp_statements.csv')

  return class_to_groups, group_to_classes_sorted
