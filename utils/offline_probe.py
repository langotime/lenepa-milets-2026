from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset as TorchTensorDataset

from utils.label_exposure import UniqueSampleLabelExposureTracker
from utils.probe_metrics import (
  probe_compute_metrics,
  probe_compute_pairwise_confusion_torch,
)
from utils.schedules import cosine_schedule, update_learning_rate_
from utils.aiono_benchmark import aiono_stat_summary


@dataclass(frozen=True)
class OfflineProbeConfig:
  """Configuration for offline linear probes."""
  steps: int
  batch_size: int
  learning_rate: float
  final_learning_rate: float
  learning_rate_warmup_steps: int
  weight_decay: float
  opt_betas: tuple[float, float]
  gradient_clip: float
  checkpoint_interval: int


def _validate_offline_probe_config(*, eval_config: OfflineProbeConfig) -> None:
  """Validate offline probe hyperparameters."""
  # Step 1: validate required numeric hyperparameters.
  if eval_config.batch_size <= 0:
    raise ValueError(f'eval_config.batch_size must be > 0, got {eval_config.batch_size}')
  if eval_config.steps <= 0:
    raise ValueError(f'eval_config.steps must be > 0, got {eval_config.steps}')
  if eval_config.checkpoint_interval <= 0:
    raise ValueError(
      f'eval_config.checkpoint_interval must be > 0, got {eval_config.checkpoint_interval}')
  if eval_config.gradient_clip < 0:
    raise ValueError(
      f'eval_config.gradient_clip must be >= 0, got {eval_config.gradient_clip}')


def _extract_representation_tensor(encoder_output: object) -> torch.Tensor:
  """Select a 2D representation tensor from encoder outputs."""
  # Step 1: return direct tensor outputs.
  if isinstance(encoder_output, torch.Tensor):
    return encoder_output
  # Step 2: search tuple/list outputs for a 2D or 3D tensor.
  if isinstance(encoder_output, (tuple, list)):
    rep_candidate = None
    token_candidate = None
    for item in encoder_output:
      if not isinstance(item, torch.Tensor):
        continue
      if item.dim() == 2:
        rep_candidate = item
      elif item.dim() == 3 and token_candidate is None:
        token_candidate = item
    if rep_candidate is not None:
      return rep_candidate
    if token_candidate is not None:
      return token_candidate
  # Step 3: fail fast on unknown formats.
  raise ValueError(
    'Failed to extract a representation tensor from encoder outputs. '
    f'Got type={type(encoder_output).__name__}.')


def _split_offline_probe_batch(
  batch: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Split an offline probe batch into (x, y_cls, y_dense)."""
  # Step 1: validate batch structure and extract tensors.
  if not isinstance(batch, (tuple, list)):
    raise ValueError(
      f'Offline probe batches must be tuples, got {type(batch).__name__}.')
  if len(batch) == 2:
    x, y_cls = batch
    y_dense = None
  elif len(batch) == 3:
    x, y_cls, y_dense = batch
  else:
    raise ValueError(
      f'Offline probe batches must have 2 or 3 items, got {len(batch)}.')
  if not isinstance(x, torch.Tensor):
    raise ValueError(f'Offline probe inputs must be tensors, got {type(x).__name__}.')
  if not isinstance(y_cls, torch.Tensor):
    raise ValueError(
      f'Offline probe class targets must be tensors, got {type(y_cls).__name__}.')
  if y_dense is not None and not isinstance(y_dense, torch.Tensor):
    raise ValueError(
      f'Offline probe dense targets must be tensors, got {type(y_dense).__name__}.')
  return x, y_cls, y_dense


def _collect_probe_features(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], torch.Tensor] | None,
  loader: Iterable[tuple[torch.Tensor, ...]],
  device: torch.device,
  auto_mixed_precision,
  allow_crops: bool,
  timings: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
  """Collect cached features and targets for offline probing.

  When `timings` is provided, it is populated with coarse wall-time breakdowns
  for feature extraction (no CUDA synchronization).
  """
  # Step 1: initialize accumulation buffers.
  total_start = perf_counter()
  feature_batches: list[torch.Tensor] = []  # each [B, D] or [B, V, D]
  class_target_batches: list[torch.Tensor] = []  # each [B, C]
  dense_target_batches: list[torch.Tensor] = []  # each [B, D_dense]
  has_crops = False
  saw_dense_targets = None
  batches = 0
  samples = 0
  to_device_s = 0.0
  forward_s = 0.0
  cpu_copy_s = 0.0
  cat_s = 0.0

  # Step 2: freeze the encoder and iterate the loader once.
  encoder_training = encoder.training
  encoder.eval()
  # NOTE: use `no_grad()` instead of `inference_mode()`.
  # Some encoders cache tensors during forward (e.g., RoPE cos/sin tables). If those
  # tensors are created under inference mode, later training backprop can fail with:
  # "Inference tensors cannot be saved for backward".
  with torch.no_grad():
    for batch in loader:
      batches += 1
      x, y_cls, y_dense = _split_offline_probe_batch(batch)
      batch_size_hint = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim > 0 else 0
      samples += batch_size_hint
      if y_dense is None:
        if saw_dense_targets is True:
          raise ValueError('Offline probe loader mixes dense and non-dense batches.')
        saw_dense_targets = False
      else:
        if saw_dense_targets is False:
          raise ValueError('Offline probe loader mixes dense and non-dense batches.')
        saw_dense_targets = True

      if x.dim() == 4:
        if not allow_crops:
          raise ValueError(f'Expected 3D inputs without crops, got {tuple(x.shape)}')
        has_crops = True
        batch_size, num_crops, num_channels, channel_size = x.size()
        x = x.reshape(batch_size * num_crops, num_channels, channel_size)  # [B*V, C, L]
        batch_to_device_start = perf_counter()
        x = x.to(device, non_blocking=True)
        to_device_s += perf_counter() - batch_to_device_start
        forward_start = perf_counter()
        with auto_mixed_precision:
          if representation_fn is None:
            encoder_output = encoder(x)
            rep = _extract_representation_tensor(encoder_output)  # [B*V, D] or [B*V, T, D]
          else:
            rep = representation_fn(x)  # [B*V, D]
        forward_s += perf_counter() - forward_start
        if rep.dim() == 3:
          rep = rep[:, 0]  # [B*V, D]
        elif rep.dim() != 2:
          raise ValueError(
            f'Expected encoder representation to be 2D or 3D, got {tuple(rep.shape)}')
        rep = rep.reshape(batch_size, num_crops, -1)  # [B, V, D]
      elif x.dim() == 3:
        if has_crops:
          raise ValueError('Mixed cropped and non-cropped batches in offline probe features.')
        batch_to_device_start = perf_counter()
        x = x.to(device, non_blocking=True)
        to_device_s += perf_counter() - batch_to_device_start
        forward_start = perf_counter()
        with auto_mixed_precision:
          if representation_fn is None:
            encoder_output = encoder(x)
            rep = _extract_representation_tensor(encoder_output)  # [B, D] or [B, T, D]
          else:
            rep = representation_fn(x)  # [B, D]
        forward_s += perf_counter() - forward_start
        if rep.dim() == 3:
          rep = rep[:, 0]  # [B, D]
        elif rep.dim() != 2:
          raise ValueError(
            f'Expected encoder representation to be 2D or 3D, got {tuple(rep.shape)}')
      else:
        raise ValueError(f'Expected inputs with 3 or 4 dims, got {tuple(x.shape)}')

      cpu_copy_start = perf_counter()
      rep = rep.float()  # [B, D] or [B, V, D]
      y_cls = y_cls.float().to(device, non_blocking=True)  # [B, C]
      feature_batches.append(rep)
      class_target_batches.append(y_cls)
      if y_dense is not None:
        y_dense = y_dense.float().to(device, non_blocking=True)  # [B, D_dense]
        dense_target_batches.append(y_dense)
      cpu_copy_s += perf_counter() - cpu_copy_start

  encoder.train(encoder_training)

  # Step 3: concatenate accumulated tensors and validate presence.
  if not feature_batches:
    raise ValueError('Offline probe feature extraction produced no batches.')

  cat_start = perf_counter()
  features = torch.cat(feature_batches, dim=0)  # [N, D] or [N, V, D]
  class_targets = torch.cat(class_target_batches, dim=0)  # [N, C]
  dense_targets = None
  if dense_target_batches:
    dense_targets = torch.cat(dense_target_batches, dim=0)  # [N, D_dense]
  cat_s = perf_counter() - cat_start
  total_s = perf_counter() - total_start
  if timings is not None:
    timings.update({
      'total_s': float(total_s),
      'to_device_s': float(to_device_s),
      'forward_s': float(forward_s),
      'cpu_copy_s': float(cpu_copy_s),
      'cat_s': float(cat_s),
      'batches': int(batches),
      'samples': int(samples),
    })
  return features, class_targets, dense_targets, has_crops


def _collect_probe_features_by_layer(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
  layers: tuple[int, ...],
  loader: Iterable[tuple[torch.Tensor, ...]],
  device: torch.device,
  auto_mixed_precision,
  allow_crops: bool,
  timings: dict[str, float | int] | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor | None, bool]:
  """Collect cached features/targets for multiple representation layers.

  The `representation_fn` must return a mapping from each requested `layer` to a pooled
  representation tensor `[B, D]` for the provided batch inputs.
  """
  # Step 1: validate requested layers.
  if not layers:
    raise ValueError('layers must be non-empty when collecting layer features.')
  if len(set(layers)) != len(layers):
    raise ValueError(f'layers must be unique, got {layers}')
  for idx, layer in enumerate(layers):
    if not isinstance(layer, int):
      raise ValueError(f'layers[{idx}] must be an int, got {layer!r}')

  # Step 2: initialize accumulation buffers.
  total_start = perf_counter()
  feature_batches: dict[int, list[torch.Tensor]] = {
    layer: [] for layer in layers
  }  # each list contains [B, D] or [B, V, D]
  class_target_batches: list[torch.Tensor] = []  # each [B, C]
  dense_target_batches: list[torch.Tensor] = []  # each [B, D_dense]
  has_crops = False
  saw_dense_targets = None
  batches = 0
  samples = 0
  to_device_s = 0.0
  forward_s = 0.0
  cpu_copy_s = 0.0

  # Step 3: freeze the encoder and iterate the loader once.
  encoder_training = encoder.training
  encoder.eval()
  # NOTE: use `no_grad()` instead of `inference_mode()` for the same reason as
  # `_collect_probe_features`: some encoders cache tensors during forward.
  with torch.no_grad():
    for batch in loader:
      batches += 1
      x, y_cls, y_dense = _split_offline_probe_batch(batch)
      batch_size_hint = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim > 0 else 0
      samples += batch_size_hint

      if y_dense is None:
        if saw_dense_targets is True:
          raise ValueError('Offline probe loader mixes dense and non-dense batches.')
        saw_dense_targets = False
      else:
        if saw_dense_targets is False:
          raise ValueError('Offline probe loader mixes dense and non-dense batches.')
        saw_dense_targets = True

      if x.dim() == 4:
        if not allow_crops:
          raise ValueError(f'Expected 3D inputs without crops, got {tuple(x.shape)}')
        has_crops = True
        batch_size, num_crops, num_channels, channel_size = x.size()
        x = x.reshape(batch_size * num_crops, num_channels, channel_size)  # [B*V, C, L]
        batch_to_device_start = perf_counter()
        x = x.to(device, non_blocking=True)
        to_device_s += perf_counter() - batch_to_device_start
        forward_start = perf_counter()
        with auto_mixed_precision:
          layer_reps = representation_fn(x)
        forward_s += perf_counter() - forward_start
        if not isinstance(layer_reps, dict):
          raise ValueError(
            'representation_fn must return a dict[int, torch.Tensor], '
            f'got {type(layer_reps).__name__}')

        cpu_copy_start = perf_counter()
        for layer in layers:
          rep = layer_reps.get(layer)
          if rep is None:
            raise ValueError(
              f'representation_fn missing requested layer={layer}. '
              f'Returned keys={sorted(layer_reps.keys())}')
          if not isinstance(rep, torch.Tensor):
            raise ValueError(
              f'representation_fn output for layer={layer} must be a tensor, '
              f'got {type(rep).__name__}')
          if rep.dim() != 2:
            raise ValueError(
              f'representation_fn output for layer={layer} must be 2D [B, D], got {tuple(rep.shape)}')
          if rep.size(0) != batch_size * num_crops:
            raise ValueError(
              'representation_fn output batch mismatch for crops: '
              f'layer={layer}, rep.shape={tuple(rep.shape)}, '
              f'expected B*V={batch_size * num_crops}')
          rep = rep.float().reshape(batch_size, num_crops, -1)  # [B, V, D]
          feature_batches[layer].append(rep)

        y_cls = y_cls.float().to(device, non_blocking=True)  # [B, C]
        class_target_batches.append(y_cls)
        if y_dense is not None:
          y_dense = y_dense.float().to(device, non_blocking=True)  # [B, D_dense]
          dense_target_batches.append(y_dense)
        cpu_copy_s += perf_counter() - cpu_copy_start
      elif x.dim() == 3:
        if has_crops:
          raise ValueError('Mixed cropped and non-cropped batches in offline probe features.')
        batch_to_device_start = perf_counter()
        x = x.to(device, non_blocking=True)
        to_device_s += perf_counter() - batch_to_device_start
        forward_start = perf_counter()
        with auto_mixed_precision:
          layer_reps = representation_fn(x)
        forward_s += perf_counter() - forward_start
        if not isinstance(layer_reps, dict):
          raise ValueError(
            'representation_fn must return a dict[int, torch.Tensor], '
            f'got {type(layer_reps).__name__}')

        cpu_copy_start = perf_counter()
        for layer in layers:
          rep = layer_reps.get(layer)
          if rep is None:
            raise ValueError(
              f'representation_fn missing requested layer={layer}. '
              f'Returned keys={sorted(layer_reps.keys())}')
          if not isinstance(rep, torch.Tensor):
            raise ValueError(
              f'representation_fn output for layer={layer} must be a tensor, '
              f'got {type(rep).__name__}')
          if rep.dim() != 2:
            raise ValueError(
              f'representation_fn output for layer={layer} must be 2D [B, D], got {tuple(rep.shape)}')
          rep = rep.float()  # [B, D]
          feature_batches[layer].append(rep)

        y_cls = y_cls.float().to(device, non_blocking=True)  # [B, C]
        class_target_batches.append(y_cls)
        if y_dense is not None:
          y_dense = y_dense.float().to(device, non_blocking=True)  # [B, D_dense]
          dense_target_batches.append(y_dense)
        cpu_copy_s += perf_counter() - cpu_copy_start
      else:
        raise ValueError(f'Expected inputs with 3 or 4 dims, got {tuple(x.shape)}')

  encoder.train(encoder_training)

  # Step 4: concatenate accumulated tensors and validate presence.
  if not class_target_batches:
    raise ValueError('Offline probe feature extraction produced no batches.')

  cat_start = perf_counter()
  features_by_layer: dict[int, torch.Tensor] = {}
  for layer, batches_for_layer in feature_batches.items():
    if not batches_for_layer:
      raise ValueError(f'Offline probe feature extraction produced no features for layer={layer}.')
    features_by_layer[layer] = torch.cat(batches_for_layer, dim=0)  # [N, D] or [N, V, D]
  class_targets = torch.cat(class_target_batches, dim=0)  # [N, C]
  dense_targets = None
  if dense_target_batches:
    dense_targets = torch.cat(dense_target_batches, dim=0)  # [N, D_dense]
  cat_s = perf_counter() - cat_start

  total_s = perf_counter() - total_start
  if timings is not None:
    timings.update({
      'total_s': float(total_s),
      'to_device_s': float(to_device_s),
      'forward_s': float(forward_s),
      'cpu_copy_s': float(cpu_copy_s),
      'cat_s': float(cat_s),
      'batches': int(batches),
      'samples': int(samples),
    })
  return features_by_layer, class_targets, dense_targets, has_crops


def _evaluate_probe_features(
  *,
  probe: nn.Module,
  features: torch.Tensor,
  targets: torch.Tensor,
  device: torch.device,
  batch_size: int,
  class_names: list[str],
  compute_pairwise_confusion: bool = True,
  timings: dict[str, float | int] | None = None,
) -> tuple[
  tuple[float, dict[str, float], float, dict[str, float]],
  dict[str, object] | None,
]:
  """Evaluate a classification probe on cached features and optionally compute confusion stats.

  Logits/targets stay on the probe device until a single CPU transfer for sklearn
  metrics. Pairwise confusion (when enabled) is computed via torch on the probe device.
  """
  # Step 1: run the probe head over cached features.
  total_start = perf_counter()
  probe.eval()
  logits_batches: list[torch.Tensor] = []  # each [B, C]
  target_batches: list[torch.Tensor] = []  # each [B, C]
  forward_s = 0.0

  feature_loader = DataLoader(
    dataset=TorchTensorDataset(features, targets),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False)

  forward_start = perf_counter()
  with torch.inference_mode():
    for batch in feature_loader:
      batch_features, batch_targets = batch  # [B, D] or [B, V, D], [B, C]
      if batch_features.dim() == 3:
        batch_size, num_crops, feature_dim = batch_features.size()
        flat_features = batch_features.reshape(batch_size * num_crops, feature_dim)  # [B*V, D]
        flat_features = flat_features.to(device, non_blocking=True)
        logits = probe(flat_features)  # [B*V, C]
        logits = logits.reshape(batch_size, num_crops, -1).mean(dim=1)  # [B, C]
      elif batch_features.dim() == 2:
        batch_features = batch_features.to(device, non_blocking=True)
        logits = probe(batch_features)  # [B, C]
      else:
        raise ValueError(
          f'Expected offline probe features to be 2D or 3D, got {tuple(batch_features.shape)}')
      logits_batches.append(logits)
      target_batches.append(batch_targets)
  forward_s = perf_counter() - forward_start

  # Step 2: aggregate logits and compute metrics on CPU.
  probe.train()
  numpy_start = perf_counter()
  logits_all = torch.cat(logits_batches, dim=0)  # [N, C]
  predictions_t = logits_all.float().sigmoid()  # [N, C]
  targets_t = torch.cat(target_batches, dim=0).float()  # [N, C]
  predictions = predictions_t.cpu().numpy()  # [N, C]
  targets_np = targets_t.cpu().numpy()  # [N, C]
  numpy_s = perf_counter() - numpy_start
  metrics_start = perf_counter()
  metrics = probe_compute_metrics(
    targets=targets_np,
    predictions=predictions,
    class_names=class_names)
  metrics_s = perf_counter() - metrics_start
  pairwise_confusion = None
  confusion_s = 0.0
  if compute_pairwise_confusion:
    confusion_start = perf_counter()
    pairwise_confusion = probe_compute_pairwise_confusion_torch(
      targets=targets_t,
      predictions=predictions_t,
      class_names=class_names)
    confusion_s = perf_counter() - confusion_start
  total_s = perf_counter() - total_start
  if timings is not None:
    timings.update({
      'total_s': float(total_s),
      'forward_s': float(forward_s),
      'numpy_s': float(numpy_s),
      'metrics_s': float(metrics_s),
      'pairwise_confusion_s': float(confusion_s),
    })
  return metrics, pairwise_confusion


def _evaluate_regression_features_streaming(
  *,
  probe: nn.Module,
  features: torch.Tensor,
  targets: torch.Tensor,
  device: torch.device,
  batch_size: int,
  target_names: list[str],
  timings: dict[str, float | int] | None = None,
) -> dict[str, torch.Tensor]:
  """Evaluate a regression probe with streaming accumulators.

  Targets may include NaNs to indicate missing/undefined values; those entries
  are ignored per target dimension.
  Accumulators remain on the probe device and final metrics are transferred to CPU.
  """
  # Step 1: initialize running sums for per-target metrics.
  total_start = perf_counter()
  probe.eval()
  target_dim = int(targets.size(1))
  sum_sq_error = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_abs_error = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_targets = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_target_sq = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_predictions = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_prediction_sq = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  sum_target_prediction = torch.zeros(
    target_dim, dtype=torch.float64, device=device)  # [D_dense]
  valid_count = torch.zeros(
    target_dim, dtype=torch.int64, device=device)  # [D_dense]

  feature_loader = DataLoader(
    dataset=TorchTensorDataset(features, targets),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False)

  # Step 2: stream predictions and accumulate error statistics.
  forward_start = perf_counter()
  with torch.inference_mode():
    for batch in feature_loader:
      batch_features, batch_targets = batch  # [B, D] or [B, V, D], [B, D_dense]
      if batch_features.dim() == 3:
        batch_size, num_crops, feature_dim = batch_features.size()
        flat_features = batch_features.reshape(batch_size * num_crops, feature_dim)  # [B*V, D]
        flat_features = flat_features.to(device, non_blocking=True)
        predictions = probe(flat_features)  # [B*V, D_dense]
        predictions = predictions.reshape(batch_size, num_crops, -1).mean(dim=1)  # [B, D_dense]
      elif batch_features.dim() == 2:
        batch_features = batch_features.to(device, non_blocking=True)
        predictions = probe(batch_features)  # [B, D_dense]
      else:
        raise ValueError(
          f'Expected offline probe features to be 2D or 3D, got {tuple(batch_features.shape)}')

      predictions = predictions.float()  # [B, D_dense]
      batch_targets = batch_targets.float()  # [B, D_dense]
      valid = torch.isfinite(batch_targets)  # [B, D_dense]
      batch_targets = torch.nan_to_num(batch_targets, nan=0.0)  # [B, D_dense]

      predictions64 = predictions.to(torch.float64)  # [B, D_dense]
      targets64 = batch_targets.to(torch.float64)  # [B, D_dense]
      valid64 = valid.to(torch.float64)  # [B, D_dense]

      errors64 = (predictions64 - targets64) * valid64  # [B, D_dense]
      sum_sq_error += torch.sum(errors64 * errors64, dim=0)  # [D_dense]
      sum_abs_error += torch.sum(errors64.abs(), dim=0)  # [D_dense]
      sum_targets += torch.sum(targets64 * valid64, dim=0)  # [D_dense]
      sum_target_sq += torch.sum(targets64 * targets64 * valid64, dim=0)  # [D_dense]
      sum_predictions += torch.sum(predictions64 * valid64, dim=0)  # [D_dense]
      sum_prediction_sq += torch.sum(predictions64 * predictions64 * valid64, dim=0)  # [D_dense]
      sum_target_prediction += torch.sum(predictions64 * targets64 * valid64, dim=0)  # [D_dense]
      valid_count += valid.sum(dim=0).to(torch.int64)  # [D_dense]
  forward_s = perf_counter() - forward_start

  probe.train()
  if int(valid_count.sum().item()) < 1:
    raise ValueError('Offline dense probe validation split has no finite targets.')

  if torch.any(valid_count == 0):
    missing = [
      name
      for name, count in zip(target_names, valid_count.tolist(), strict=True)
      if count == 0
    ]
    raise ValueError(
      'Offline dense probe validation has no finite samples for targets: '
      f'{", ".join(missing)}')

  # Step 3: compute per-target regression metrics from accumulated sums.
  finalize_start = perf_counter()
  count = valid_count.to(torch.float64)  # [D_dense]
  mean_targets = sum_targets / count  # [D_dense]
  mean_predictions = sum_predictions / count  # [D_dense]
  ss_tot = sum_target_sq - count * (mean_targets * mean_targets)  # [D_dense]
  ss_pred = sum_prediction_sq - count * (mean_predictions * mean_predictions)  # [D_dense]
  denom = torch.sqrt(ss_tot * ss_pred)  # [D_dense]

  mse = sum_sq_error / count  # [D_dense]
  mae = sum_abs_error / count  # [D_dense]
  r2 = torch.full_like(mse, float('nan'))  # [D_dense]
  r2_defined = ss_tot > 0  # [D_dense]
  r2[r2_defined] = 1.0 - (sum_sq_error[r2_defined] / ss_tot[r2_defined])  # [D_dense]

  pearson = torch.full_like(mse, float('nan'))  # [D_dense]
  pearson_defined = denom > 0  # [D_dense]
  covariance = sum_target_prediction - count * mean_targets * mean_predictions  # [D_dense]
  pearson[pearson_defined] = covariance[pearson_defined] / denom[pearson_defined]  # [D_dense]
  finalize_s = perf_counter() - finalize_start
  total_s = perf_counter() - total_start
  # Step 4: move metrics to CPU for downstream aggregation.
  mse = mse.cpu()  # [D_dense]
  mae = mae.cpu()  # [D_dense]
  r2 = r2.cpu()  # [D_dense]
  pearson = pearson.cpu()  # [D_dense]
  if timings is not None:
    timings.update({
      'total_s': float(total_s),
      'forward_s': float(forward_s),
      'finalize_s': float(finalize_s),
    })
  return {
    'mse': mse,
    'mae': mae,
    'r2': r2,
    'pearson': pearson,
  }


def _validation_seed_values_ordered(
  *,
  val_features_by_seed: dict[int, torch.Tensor],
  val_targets_by_seed: dict[int, torch.Tensor],
  val_has_crops_by_seed: dict[int, bool],
  context: str,
) -> list[int]:
  """Validate multi-seed feature dictionaries and return a stable seed order."""
  if not val_features_by_seed:
    raise ValueError(f'{context}: val_features_by_seed must be non-empty.')
  ordered_feature_seeds = sorted(int(seed_value) for seed_value in val_features_by_seed)
  ordered_target_seeds = sorted(int(seed_value) for seed_value in val_targets_by_seed)
  ordered_crop_seeds = sorted(int(seed_value) for seed_value in val_has_crops_by_seed)
  if ordered_feature_seeds != ordered_target_seeds:
    raise ValueError(
      f'{context}: validation feature/target seed mismatch. '
      f'features={ordered_feature_seeds} targets={ordered_target_seeds}')
  if ordered_feature_seeds != ordered_crop_seeds:
    raise ValueError(
      f'{context}: validation feature/has_crops seed mismatch. '
      f'features={ordered_feature_seeds} has_crops={ordered_crop_seeds}')
  return ordered_feature_seeds


def _aggregate_pairwise_confusion_payloads(
  *,
  payloads: list[dict[str, object]],
) -> dict[str, object]:
  """Aggregate pairwise-confusion payloads across validation seeds."""
  if not payloads:
    raise ValueError('payloads must be non-empty.')
  class_names = payloads[0].get('class_names')
  if not isinstance(class_names, list) or not class_names:
    raise ValueError('pairwise_confusion.class_names must be a non-empty list[str].')
  weighted_sum = None
  total_count = None
  for payload in payloads:
    if payload.get('class_names') != class_names:
      raise ValueError('pairwise_confusion.class_names must match across validation seeds.')
    mean_pred_raw = payload.get('mean_pred')
    count_raw = payload.get('count')
    mean_pred = np.asarray(mean_pred_raw, dtype=float)
    count = np.asarray(count_raw, dtype=np.int64)
    if mean_pred.shape != count.shape:
      raise ValueError(
        'pairwise_confusion mean_pred/count shape mismatch. '
        f'mean_pred.shape={mean_pred.shape} count.shape={count.shape}')
    numerator = np.where(np.isfinite(mean_pred), mean_pred * count.astype(float), 0.0)
    weighted_sum = numerator if weighted_sum is None else weighted_sum + numerator
    total_count = count if total_count is None else total_count + count
  assert weighted_sum is not None
  assert total_count is not None
  aggregated_mean = np.full_like(weighted_sum, np.nan, dtype=float)
  valid = total_count > 0
  aggregated_mean[valid] = weighted_sum[valid] / total_count[valid].astype(float)
  return {
    'class_names': list(class_names),
    'mean_pred': aggregated_mean.tolist(),
    'count': total_count.astype(int).tolist(),
  }


def _aggregate_per_class_metric_dicts(
  metric_dicts: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
  """Aggregate per-class metrics into median/std dicts."""
  if not metric_dicts:
    raise ValueError('metric_dicts must be non-empty.')
  class_names = list(metric_dicts[0].keys())
  if not class_names:
    raise ValueError('metric_dicts entries must be non-empty.')
  median_by_class: dict[str, float] = {}
  std_by_class: dict[str, float] = {}
  for class_name in class_names:
    values = []
    for metric_dict in metric_dicts:
      if class_name not in metric_dict:
        raise ValueError(
          f'Per-class metric aggregation is missing class={class_name!r}. '
          f'Available classes={sorted(metric_dict)}')
      values.append(float(metric_dict[class_name]))
    stats = aiono_stat_summary(values)
    median = stats['median']
    std = stats['std']
    if median is None or std is None:
      raise ValueError(
        'Per-class metric aggregation produced no finite values. '
        f'class_name={class_name!r} values={values}')
    median_by_class[class_name] = float(median)
    std_by_class[class_name] = float(std)
  return median_by_class, std_by_class


def _train_linear_classification_probe(
  *,
  train_features: torch.Tensor,
  train_targets: torch.Tensor,
  val_features: torch.Tensor,
  val_targets: torch.Tensor,
  num_classes: int,
  class_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  val_has_crops: bool,
  compute_pairwise_confusion: bool = True,
  label_exposure: bool = False,
) -> dict[str, object]:
  """Train and evaluate a linear classification probe from cached features."""
  total_start = perf_counter()
  # Step 1: validate feature/target shapes.
  if train_features.dim() != 2:
    raise ValueError(f'Expected train features to be 2D, got {tuple(train_features.shape)}')
  if val_features.dim() not in (2, 3):
    raise ValueError(f'Expected val features to be 2D or 3D, got {tuple(val_features.shape)}')
  if train_targets.dim() != 2:
    raise ValueError(f'Expected train targets to be 2D, got {tuple(train_targets.shape)}')
  if val_targets.dim() != 2:
    raise ValueError(f'Expected val targets to be 2D, got {tuple(val_targets.shape)}')
  if train_targets.size(1) != num_classes:
    raise ValueError(
      f'Offline probe train target classes mismatch: expected {num_classes}, '
      f'got {train_targets.size(1)}')
  if val_targets.size(1) != num_classes:
    raise ValueError(
      f'Offline probe val target classes mismatch: expected {num_classes}, '
      f'got {val_targets.size(1)}')

  # Step 2: validate dataset sizes.
  feature_dim = train_features.size(1)
  train_size = train_features.size(0)
  val_size = val_targets.size(0)
  if train_size < eval_config.batch_size:
    raise ValueError(
      f'Offline probe train split too small for batch_size={eval_config.batch_size}: '
      f'{train_size} samples')
  if val_size < 1:
    raise ValueError('Offline probe validation split is empty.')

  # Step 3: build probe head and optimizer.
  probe = nn.Sequential(
    nn.LayerNorm(feature_dim),
    nn.Linear(feature_dim, num_classes)).to(device)
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=eval_config.learning_rate,
    betas=eval_config.opt_betas,
    weight_decay=eval_config.weight_decay)

  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  train_feature_loader = DataLoader(
    dataset=(
      TorchTensorDataset(
        torch.arange(train_size, dtype=torch.int64),  # [N]
        train_features,
        train_targets)
      if label_exposure else
      TorchTensorDataset(train_features, train_targets)
    ),
    batch_size=eval_config.batch_size,
    shuffle=True,
    drop_last=True)

  def cycle(dataloader):
    """Yield batches indefinitely from a dataloader."""
    while True:
      yield from dataloader

  # Step 4: optimize and track best checkpoints.
  train_iterator = cycle(train_feature_loader)
  best_auc_metrics = None
  best_auc_step = None
  best_auc_pairwise_confusion = None
  best_auprc_metrics = None
  best_auprc_step = None
  best_auprc_pairwise_confusion = None
  eval_calls = 0
  eval_total_s = 0.0
  eval_forward_s = 0.0
  eval_numpy_s = 0.0
  eval_metrics_s = 0.0
  eval_pairwise_confusion_s = 0.0
  exposure_tracker = None
  if label_exposure:
    exposure_tracker = UniqueSampleLabelExposureTracker(
      class_names=class_names,
      num_samples=int(train_size))
  for step in range(eval_config.steps):
    learning_rate = next(lr_schedule)
    update_learning_rate_(optimizer, learning_rate)
    if exposure_tracker is None:
      batch_features, batch_targets = next(train_iterator)  # [B, D], [B, C]
    else:
      batch_indices, batch_features, batch_targets = next(train_iterator)  # [B], [B, D], [B, C]
      exposure_tracker.update_unique(batch_indices, batch_targets, step=step + 1)
    batch_features = batch_features.to(device, non_blocking=True)  # [B, D]
    batch_targets = batch_targets.to(device, non_blocking=True)  # [B, C]
    logits = probe(batch_features)  # [B, C]
    loss = F.binary_cross_entropy_with_logits(logits, batch_targets)  # []
    loss.backward()
    max_norm = eval_config.gradient_clip if eval_config.gradient_clip > 0 else float('inf')
    torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if (step + 1) % eval_config.checkpoint_interval == 0:
      eval_call_timings: dict[str, float | int] = {}
      eval_call_start = perf_counter()
      metrics, pairwise_confusion = _evaluate_probe_features(
        probe=probe,
        features=val_features,
        targets=val_targets,
        device=device,
        batch_size=eval_config.batch_size,
        class_names=class_names,
        compute_pairwise_confusion=compute_pairwise_confusion,
        timings=eval_call_timings)
      eval_total_s += perf_counter() - eval_call_start
      eval_calls += 1
      eval_forward_s += float(eval_call_timings.get('forward_s', 0.0))
      eval_numpy_s += float(eval_call_timings.get('numpy_s', 0.0))
      eval_metrics_s += float(eval_call_timings.get('metrics_s', 0.0))
      eval_pairwise_confusion_s += float(eval_call_timings.get('pairwise_confusion_s', 0.0))
      macro_auc = metrics[0]
      macro_auprc = metrics[2]
      if best_auc_metrics is None or macro_auc > best_auc_metrics[0]:
        best_auc_metrics = metrics
        best_auc_step = step + 1
        best_auc_pairwise_confusion = pairwise_confusion
      if best_auprc_metrics is None or macro_auprc > best_auprc_metrics[2]:
        best_auprc_metrics = metrics
        best_auprc_step = step + 1
        best_auprc_pairwise_confusion = pairwise_confusion

  if (best_auc_metrics is None
      or best_auc_step is None
      or best_auprc_metrics is None
      or best_auprc_step is None):
    raise ValueError('Offline probe did not evaluate any checkpoints.')
  if compute_pairwise_confusion and (
    best_auc_pairwise_confusion is None or best_auprc_pairwise_confusion is None
  ):
    raise ValueError('Offline probe expected pairwise confusion but none was computed.')

  def _pack_best(
    metrics: tuple[float, dict[str, float], float, dict[str, float]],
    pairwise_confusion: dict[str, object] | None,
    *,
    best_step: int,
  ) -> dict[str, object]:
    """Package best checkpoint metrics into a dict."""
    macro_auc, per_class_auc, macro_auprc, per_class_auprc = metrics
    packed = {
      'macro_auc': float(macro_auc),
      'per_class_auc': per_class_auc,
      'macro_auprc': float(macro_auprc),
      'per_class_auprc': per_class_auprc,
      'best_probe_step': int(best_step),
    }
    if pairwise_confusion is not None:
      packed['pairwise_confusion'] = pairwise_confusion
    return packed

  # Step 5: return packed metrics and metadata.
  val_num_crops = int(val_features.size(1)) if val_has_crops else 1
  total_s = perf_counter() - total_start
  timings = {
    'total_s': float(total_s),
    'train_steps_s': float(total_s - eval_total_s),
    'eval_total_s': float(eval_total_s),
    'eval_calls': int(eval_calls),
    'eval_forward_s': float(eval_forward_s),
    'eval_numpy_s': float(eval_numpy_s),
    'eval_metrics_s': float(eval_metrics_s),
    'eval_pairwise_confusion_s': float(eval_pairwise_confusion_s),
  }
  packed: dict[str, object] = {
    'best_auc': _pack_best(
      best_auc_metrics,
      best_auc_pairwise_confusion,
      best_step=best_auc_step),
    'best_auprc': _pack_best(
      best_auprc_metrics,
      best_auprc_pairwise_confusion,
      best_step=best_auprc_step),
    'feature_dim': int(feature_dim),
    'train_size': int(train_size),
    'val_size': int(val_size),
    'val_num_crops': int(val_num_crops),
    'timings': {'classification': timings},
  }
  if exposure_tracker is not None:
    packed['label_exposure'] = {
      'mode': 'unique',
      'steps': int(eval_config.steps),
      'batch_size': int(eval_config.batch_size),
      'num_samples': int(exposure_tracker.num_samples),
      'unique_samples_seen_by_step': list(exposure_tracker.unique_samples_seen_by_step),
      'coverage_count_by_step': list(exposure_tracker.coverage_count_by_step),
      'cumulative_positives': [int(v) for v in exposure_tracker.cumulative_positives.tolist()],
      'first_positive_step': list(exposure_tracker.first_positive_step),
      'class_names': list(class_names),
    }
  return packed


def _train_linear_classification_probe_multi_val(
  *,
  train_features: torch.Tensor,
  train_targets: torch.Tensor,
  val_features_by_seed: dict[int, torch.Tensor],
  val_targets_by_seed: dict[int, torch.Tensor],
  val_has_crops_by_seed: dict[int, bool],
  num_classes: int,
  class_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  compute_pairwise_confusion: bool = True,
  label_exposure: bool = False,
) -> dict[str, object]:
  """Train a classification probe and select checkpoints by median validation score."""
  total_start = perf_counter()
  if train_features.dim() != 2:
    raise ValueError(f'Expected train features to be 2D, got {tuple(train_features.shape)}')
  if train_targets.dim() != 2:
    raise ValueError(f'Expected train targets to be 2D, got {tuple(train_targets.shape)}')
  if train_targets.size(1) != num_classes:
    raise ValueError(
      f'Offline probe train target classes mismatch: expected {num_classes}, '
      f'got {train_targets.size(1)}')

  validation_seed_values = _validation_seed_values_ordered(
    val_features_by_seed=val_features_by_seed,
    val_targets_by_seed=val_targets_by_seed,
    val_has_crops_by_seed=val_has_crops_by_seed,
    context='classification',
  )
  for seed_value in validation_seed_values:
    val_features = val_features_by_seed[seed_value]
    val_targets = val_targets_by_seed[seed_value]
    if val_features.dim() not in (2, 3):
      raise ValueError(
        f'Expected val features to be 2D or 3D for seed={seed_value}, '
        f'got {tuple(val_features.shape)}')
    if val_targets.dim() != 2:
      raise ValueError(
        f'Expected val targets to be 2D for seed={seed_value}, got {tuple(val_targets.shape)}')
    if val_targets.size(1) != num_classes:
      raise ValueError(
        f'Offline probe val target classes mismatch for seed={seed_value}: expected {num_classes}, '
        f'got {val_targets.size(1)}')

  feature_dim = train_features.size(1)
  train_size = train_features.size(0)
  if train_size < eval_config.batch_size:
    raise ValueError(
      f'Offline probe train split too small for batch_size={eval_config.batch_size}: '
      f'{train_size} samples')
  for seed_value in validation_seed_values:
    val_size = int(val_targets_by_seed[seed_value].size(0))
    if val_size < 1:
      raise ValueError(f'Offline probe validation split is empty for validation seed={seed_value}.')

  probe = nn.Sequential(
    nn.LayerNorm(feature_dim),
    nn.Linear(feature_dim, num_classes)).to(device)
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=eval_config.learning_rate,
    betas=eval_config.opt_betas,
    weight_decay=eval_config.weight_decay)

  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  train_feature_loader = DataLoader(
    dataset=(
      TorchTensorDataset(
        torch.arange(train_size, dtype=torch.int64),  # [N]
        train_features,
        train_targets)
      if label_exposure else
      TorchTensorDataset(train_features, train_targets)
    ),
    batch_size=eval_config.batch_size,
    shuffle=True,
    drop_last=True)

  def cycle(dataloader):
    """Yield batches indefinitely from a dataloader."""
    while True:
      yield from dataloader

  def _pack_checkpoint(
    *,
    metrics_by_seed: dict[int, tuple[float, dict[str, float], float, dict[str, float]]],
    pairwise_confusion_by_seed: dict[int, dict[str, object] | None],
    best_step: int,
  ) -> dict[str, object]:
    """Aggregate one evaluated checkpoint across validation seeds."""
    auc_stats = aiono_stat_summary([
      float(metrics_by_seed[seed_value][0]) for seed_value in validation_seed_values
    ])
    auprc_stats = aiono_stat_summary([
      float(metrics_by_seed[seed_value][2]) for seed_value in validation_seed_values
    ])
    if auc_stats['median'] is None or auc_stats['std'] is None:
      raise ValueError('Multi-seed AUROC aggregation produced no finite values.')
    if auprc_stats['median'] is None or auprc_stats['std'] is None:
      raise ValueError('Multi-seed AUPRC aggregation produced no finite values.')
    per_class_auc, per_class_auc_std = _aggregate_per_class_metric_dicts([
      metrics_by_seed[seed_value][1] for seed_value in validation_seed_values
    ])
    per_class_auprc, per_class_auprc_std = _aggregate_per_class_metric_dicts([
      metrics_by_seed[seed_value][3] for seed_value in validation_seed_values
    ])
    packed: dict[str, object] = {
      'macro_auc': float(auc_stats['median']),
      'macro_auc_std': float(auc_stats['std']),
      'per_class_auc': per_class_auc,
      'per_class_auc_std': per_class_auc_std,
      'macro_auprc': float(auprc_stats['median']),
      'macro_auprc_std': float(auprc_stats['std']),
      'per_class_auprc': per_class_auprc,
      'per_class_auprc_std': per_class_auprc_std,
      'best_probe_step': int(best_step),
      'validation_seed_values': list(validation_seed_values),
      'validation_seed_count': int(len(validation_seed_values)),
    }
    if compute_pairwise_confusion:
      confusion_payloads = [
        pairwise_confusion_by_seed[seed_value]
        for seed_value in validation_seed_values
      ]
      if any(payload is None for payload in confusion_payloads):
        raise ValueError('Offline probe expected pairwise confusion for every validation seed.')
      packed['pairwise_confusion'] = _aggregate_pairwise_confusion_payloads(
        payloads=[payload for payload in confusion_payloads if payload is not None])
    return packed

  train_iterator = cycle(train_feature_loader)
  best_auc = None
  best_auprc = None
  best_auc_value = None
  best_auprc_value = None
  eval_calls = 0
  eval_total_s = 0.0
  eval_forward_s = 0.0
  eval_numpy_s = 0.0
  eval_metrics_s = 0.0
  eval_pairwise_confusion_s = 0.0
  exposure_tracker = None
  if label_exposure:
    exposure_tracker = UniqueSampleLabelExposureTracker(
      class_names=class_names,
      num_samples=int(train_size))

  for step in range(eval_config.steps):
    learning_rate = next(lr_schedule)
    update_learning_rate_(optimizer, learning_rate)
    if exposure_tracker is None:
      batch_features, batch_targets = next(train_iterator)  # [B, D], [B, C]
    else:
      batch_indices, batch_features, batch_targets = next(train_iterator)  # [B], [B, D], [B, C]
      exposure_tracker.update_unique(batch_indices, batch_targets, step=step + 1)
    batch_features = batch_features.to(device, non_blocking=True)  # [B, D]
    batch_targets = batch_targets.to(device, non_blocking=True)  # [B, C]
    logits = probe(batch_features)  # [B, C]
    loss = F.binary_cross_entropy_with_logits(logits, batch_targets)  # []
    loss.backward()
    max_norm = eval_config.gradient_clip if eval_config.gradient_clip > 0 else float('inf')
    torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if (step + 1) % eval_config.checkpoint_interval == 0:
      metrics_by_seed: dict[int, tuple[float, dict[str, float], float, dict[str, float]]] = {}
      pairwise_confusion_by_seed: dict[int, dict[str, object] | None] = {}
      auc_values: list[float] = []
      auprc_values: list[float] = []
      for seed_value in validation_seed_values:
        eval_call_timings: dict[str, float | int] = {}
        eval_call_start = perf_counter()
        metrics, pairwise_confusion = _evaluate_probe_features(
          probe=probe,
          features=val_features_by_seed[seed_value],
          targets=val_targets_by_seed[seed_value],
          device=device,
          batch_size=eval_config.batch_size,
          class_names=class_names,
          compute_pairwise_confusion=compute_pairwise_confusion,
          timings=eval_call_timings)
        eval_total_s += perf_counter() - eval_call_start
        eval_calls += 1
        eval_forward_s += float(eval_call_timings.get('forward_s', 0.0))
        eval_numpy_s += float(eval_call_timings.get('numpy_s', 0.0))
        eval_metrics_s += float(eval_call_timings.get('metrics_s', 0.0))
        eval_pairwise_confusion_s += float(eval_call_timings.get('pairwise_confusion_s', 0.0))
        metrics_by_seed[seed_value] = metrics
        pairwise_confusion_by_seed[seed_value] = pairwise_confusion
        auc_values.append(float(metrics[0]))
        auprc_values.append(float(metrics[2]))
      auc_stats = aiono_stat_summary(auc_values)
      auprc_stats = aiono_stat_summary(auprc_values)
      if auc_stats['median'] is None or auprc_stats['median'] is None:
        raise ValueError(
          'Offline probe produced no finite validation metrics across validation seeds.')
      if best_auc_value is None or float(auc_stats['median']) > best_auc_value:
        best_auc_value = float(auc_stats['median'])
        best_auc = _pack_checkpoint(
          metrics_by_seed=metrics_by_seed,
          pairwise_confusion_by_seed=pairwise_confusion_by_seed,
          best_step=step + 1)
      if best_auprc_value is None or float(auprc_stats['median']) > best_auprc_value:
        best_auprc_value = float(auprc_stats['median'])
        best_auprc = _pack_checkpoint(
          metrics_by_seed=metrics_by_seed,
          pairwise_confusion_by_seed=pairwise_confusion_by_seed,
          best_step=step + 1)

  if best_auc is None or best_auprc is None:
    raise ValueError('Offline probe did not evaluate any checkpoints.')

  first_seed_value = validation_seed_values[0]
  first_val_features = val_features_by_seed[first_seed_value]
  first_val_targets = val_targets_by_seed[first_seed_value]
  first_val_has_crops = bool(val_has_crops_by_seed[first_seed_value])
  val_num_crops = int(first_val_features.size(1)) if first_val_has_crops else 1
  total_s = perf_counter() - total_start
  timings = {
    'total_s': float(total_s),
    'train_steps_s': float(total_s - eval_total_s),
    'eval_total_s': float(eval_total_s),
    'eval_calls': int(eval_calls),
    'eval_forward_s': float(eval_forward_s),
    'eval_numpy_s': float(eval_numpy_s),
    'eval_metrics_s': float(eval_metrics_s),
    'eval_pairwise_confusion_s': float(eval_pairwise_confusion_s),
  }
  packed: dict[str, object] = {
    'best_auc': best_auc,
    'best_auprc': best_auprc,
    'feature_dim': int(feature_dim),
    'train_size': int(train_size),
    'val_size': int(first_val_targets.size(0)),
    'val_num_crops': int(val_num_crops),
    'validation_seed_values': list(validation_seed_values),
    'validation_seed_count': int(len(validation_seed_values)),
    'timings': {'classification': timings},
  }
  if exposure_tracker is not None:
    packed['label_exposure'] = {
      'mode': 'unique',
      'steps': int(eval_config.steps),
      'batch_size': int(eval_config.batch_size),
      'num_samples': int(exposure_tracker.num_samples),
      'unique_samples_seen_by_step': list(exposure_tracker.unique_samples_seen_by_step),
      'coverage_count_by_step': list(exposure_tracker.coverage_count_by_step),
      'cumulative_positives': [int(v) for v in exposure_tracker.cumulative_positives.tolist()],
      'first_positive_step': list(exposure_tracker.first_positive_step),
      'class_names': list(class_names),
    }
  return packed


def _clip_linear_per_target_(linear: nn.Linear, max_norm: float) -> None:
  """Clip per-target gradient norms for a linear regression head."""
  # Step 1: validate gradients and compute per-target norms.
  if linear.weight.grad is None:
    raise ValueError('Linear weight gradients missing during dense probe clipping.')
  if linear.bias is None or linear.bias.grad is None:
    raise ValueError('Linear bias gradients missing during dense probe clipping.')
  if max_norm <= 0:
    return

  weight_grad = linear.weight.grad  # [D_dense, D]
  bias_grad = linear.bias.grad  # [D_dense]
  row_norm_sq = torch.sum(weight_grad * weight_grad, dim=1) + bias_grad * bias_grad  # [D_dense]
  row_norm = torch.sqrt(row_norm_sq)  # [D_dense]
  clip_coef = torch.ones_like(row_norm)  # [D_dense]
  over_limit = row_norm > max_norm  # [D_dense]
  clip_coef[over_limit] = max_norm / row_norm[over_limit]

  # Step 2: scale row-wise gradients in-place.
  linear.weight.grad.mul_(clip_coef.unsqueeze(1))  # [D_dense, D]
  linear.bias.grad.mul_(clip_coef)  # [D_dense]


def _train_linear_regression_probe(
  *,
  train_features: torch.Tensor,
  train_targets: torch.Tensor,
  val_features: torch.Tensor,
  val_targets: torch.Tensor,
  target_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  val_has_crops: bool,
  log_per_target: bool,
) -> dict[str, object]:
  """Train and evaluate a linear regression probe from cached features."""
  total_start = perf_counter()
  # Step 1: validate feature/target shapes.
  if train_features.dim() != 2:
    raise ValueError(f'Expected train features to be 2D, got {tuple(train_features.shape)}')
  if val_features.dim() not in (2, 3):
    raise ValueError(f'Expected val features to be 2D or 3D, got {tuple(val_features.shape)}')
  if train_targets.dim() != 2:
    raise ValueError(f'Expected train regression targets to be 2D, got {tuple(train_targets.shape)}')
  if val_targets.dim() != 2:
    raise ValueError(f'Expected val regression targets to be 2D, got {tuple(val_targets.shape)}')

  # Step 2: validate dataset sizes and target names.
  feature_dim = train_features.size(1)
  train_size = train_features.size(0)
  val_size = val_targets.size(0)
  if train_size < eval_config.batch_size:
    raise ValueError(
      f'Offline probe train split too small for batch_size={eval_config.batch_size}: '
      f'{train_size} samples')
  if val_size < 1:
    raise ValueError('Offline probe validation split is empty.')
  if not target_names:
    raise ValueError('dense target_names must be non-empty when running regression probe.')
  if train_targets.size(1) != len(target_names):
    raise ValueError(
      'Dense probe train target size mismatch: '
      f'expected {len(target_names)}, got {train_targets.size(1)}')
  if val_targets.size(1) != len(target_names):
    raise ValueError(
      'Dense probe val target size mismatch: '
      f'expected {len(target_names)}, got {val_targets.size(1)}')

  # Step 3: validate finite target support per dimension.
  train_valid = torch.isfinite(train_targets)  # [N, D_dense]
  val_valid = torch.isfinite(val_targets)  # [N, D_dense]
  train_counts = train_valid.sum(dim=0)  # [D_dense]
  val_counts = val_valid.sum(dim=0)  # [D_dense]
  missing_train = [
    name
    for name, count in zip(target_names, train_counts.tolist(), strict=True)
    if count == 0
  ]
  if missing_train:
    raise ValueError(
      'Offline dense probe train split has no finite samples for targets: '
      f'{", ".join(missing_train)}')
  missing_val = [
    name
    for name, count in zip(target_names, val_counts.tolist(), strict=True)
    if count == 0
  ]
  if missing_val:
    raise ValueError(
      'Offline dense probe validation split has no finite samples for targets: '
      f'{", ".join(missing_val)}')

  # Step 4: build probe head and optimizer (no shared learnable normalization).
  dense_dim = int(train_targets.size(1))
  probe = nn.Sequential(
    nn.LayerNorm(feature_dim, elementwise_affine=False),
    nn.Linear(feature_dim, dense_dim)).to(device)
  linear_head = probe[1]
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=eval_config.learning_rate,
    betas=eval_config.opt_betas,
    weight_decay=eval_config.weight_decay)

  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  train_feature_loader = DataLoader(
    dataset=TorchTensorDataset(train_features, train_targets),
    batch_size=eval_config.batch_size,
    shuffle=True,
    drop_last=True)

  def cycle(dataloader):
    """Yield batches indefinitely from a dataloader."""
    while True:
      yield from dataloader

  # Step 5: optimize and track best checkpoint per target by MSE.
  train_iterator = cycle(train_feature_loader)
  best_metrics = None
  best_steps = None
  eval_calls = 0
  eval_total_s = 0.0
  eval_forward_s = 0.0
  eval_finalize_s = 0.0
  for step in range(eval_config.steps):
    learning_rate = next(lr_schedule)
    update_learning_rate_(optimizer, learning_rate)
    batch_features, batch_targets = next(train_iterator)  # [B, D], [B, D_dense]
    batch_features = batch_features.to(device, non_blocking=True)  # [B, D]
    batch_targets = batch_targets.to(device, non_blocking=True)  # [B, D_dense]
    predictions = probe(batch_features)  # [B, D_dense]
    valid = torch.isfinite(batch_targets)  # [B, D_dense]
    if not torch.any(valid):
      raise ValueError(
        'Offline dense probe training batch has no finite targets. '
        'Increase offline_probe_batch_size/offline_probe_train_batches or adjust target gating.')
    batch_targets = torch.nan_to_num(batch_targets, nan=0.0)  # [B, D_dense]
    errors = (predictions - batch_targets) * valid.to(dtype=predictions.dtype)  # [B, D_dense]
    sq_error = errors * errors  # [B, D_dense]
    sum_sq_error = torch.sum(sq_error, dim=0)  # [D_dense]
    count = torch.sum(valid, dim=0).to(dtype=predictions.dtype)  # [D_dense]
    valid_targets = count > 0  # [D_dense]
    if not torch.any(valid_targets):
      raise ValueError(
        'Offline dense probe training batch has no valid targets after masking. '
        'Increase offline_probe_batch_size/offline_probe_train_batches or adjust target gating.')
    mse = sum_sq_error[valid_targets] / count[valid_targets]  # [D_valid]
    loss = mse.mean()  # []
    loss.backward()
    _clip_linear_per_target_(linear_head, float(eval_config.gradient_clip))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if (step + 1) % eval_config.checkpoint_interval == 0:
      eval_call_timings: dict[str, float | int] = {}
      eval_call_start = perf_counter()
      per_target_metrics = _evaluate_regression_features_streaming(  # each metric: [D_dense]
        probe=probe,
        features=val_features,
        targets=val_targets,
        device=device,
        batch_size=eval_config.batch_size,
        target_names=target_names,
        timings=eval_call_timings,
      )
      eval_total_s += perf_counter() - eval_call_start
      eval_calls += 1
      eval_forward_s += float(eval_call_timings.get('forward_s', 0.0))
      eval_finalize_s += float(eval_call_timings.get('finalize_s', 0.0))
      per_target_mse = per_target_metrics['mse']  # [D_dense]
      if best_metrics is None:
        best_metrics = {  # each [D_dense]
          key: value.clone() for key, value in per_target_metrics.items()
        }
        best_steps = torch.full((dense_dim,), step + 1, dtype=torch.int64)  # [D_dense]
      else:
        improved = per_target_mse < best_metrics['mse']  # [D_dense]
        step_tensor = torch.full((dense_dim,), step + 1, dtype=torch.int64)  # [D_dense]
        for key, values in per_target_metrics.items():
          best_metrics[key] = torch.where(improved, values, best_metrics[key])  # [D_dense]
        best_steps = torch.where(improved, step_tensor, best_steps)  # [D_dense]

  if best_metrics is None or best_steps is None:
    raise ValueError('Offline dense probe did not evaluate any checkpoints.')

  # Step 6: pack regression metrics.
  total_s = perf_counter() - total_start
  timings = {
    'total_s': float(total_s),
    'train_steps_s': float(total_s - eval_total_s),
    'eval_total_s': float(eval_total_s),
    'eval_calls': int(eval_calls),
    'eval_forward_s': float(eval_forward_s),
    'eval_finalize_s': float(eval_finalize_s),
  }
  result = {
    'macro_mse': float(best_metrics['mse'].mean()),
    'macro_mae': float(best_metrics['mae'].mean()),
    'macro_r2': float(torch.nanmean(best_metrics['r2'])),
    'macro_pearson': float(torch.nanmean(best_metrics['pearson'])),
    'timings': {'regression': timings},
  }
  if log_per_target:
    result['per_target_mse'] = {
      name: float(value)
      for name, value in zip(target_names, best_metrics['mse'].tolist(), strict=True)
    }
    result['per_target_mae'] = {
      name: float(value)
      for name, value in zip(target_names, best_metrics['mae'].tolist(), strict=True)
    }
    result['per_target_r2'] = {
      name: float(value)
      for name, value in zip(target_names, best_metrics['r2'].tolist(), strict=True)
    }
    result['per_target_pearson'] = {
      name: float(value)
      for name, value in zip(target_names, best_metrics['pearson'].tolist(), strict=True)
    }
    result['per_target_best_step'] = {
      name: int(value)
      for name, value in zip(target_names, best_steps.tolist(), strict=True)
    }
  return result


def _train_linear_regression_probe_multi_val(
  *,
  train_features: torch.Tensor,
  train_targets: torch.Tensor,
  val_features_by_seed: dict[int, torch.Tensor],
  val_targets_by_seed: dict[int, torch.Tensor],
  val_has_crops_by_seed: dict[int, bool],
  target_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  log_per_target: bool,
) -> dict[str, object]:
  """Train a dense probe and select checkpoints by median validation macro-MSE."""
  total_start = perf_counter()
  if train_features.dim() != 2:
    raise ValueError(f'Expected train features to be 2D, got {tuple(train_features.shape)}')
  if train_targets.dim() != 2:
    raise ValueError(f'Expected train targets to be 2D, got {tuple(train_targets.shape)}')

  validation_seed_values = _validation_seed_values_ordered(
    val_features_by_seed=val_features_by_seed,
    val_targets_by_seed=val_targets_by_seed,
    val_has_crops_by_seed=val_has_crops_by_seed,
    context='dense',
  )
  for seed_value in validation_seed_values:
    val_features = val_features_by_seed[seed_value]
    val_targets = val_targets_by_seed[seed_value]
    if val_features.dim() not in (2, 3):
      raise ValueError(
        f'Expected val features to be 2D or 3D for seed={seed_value}, got {tuple(val_features.shape)}')
    if val_targets.dim() != 2:
      raise ValueError(
        f'Expected val targets to be 2D for seed={seed_value}, got {tuple(val_targets.shape)}')

  feature_dim = int(train_features.size(1))
  train_size = int(train_features.size(0))
  if train_size < eval_config.batch_size:
    raise ValueError(
      f'Offline probe train split too small for batch_size={eval_config.batch_size}: '
      f'{train_size} samples')

  train_valid = torch.isfinite(train_targets)
  missing_train = [
    name
    for name, count in zip(target_names, train_valid.sum(dim=0).tolist(), strict=True)
    if count == 0
  ]
  if missing_train:
    raise ValueError(
      'Train split has no finite dense samples for targets: '
      f'{", ".join(missing_train)}')
  for seed_value in validation_seed_values:
    val_valid = torch.isfinite(val_targets_by_seed[seed_value])
    missing_val = [
      name
      for name, count in zip(target_names, val_valid.sum(dim=0).tolist(), strict=True)
      if count == 0
    ]
    if missing_val:
      raise ValueError(
        f'Validation split has no finite dense samples for seed={seed_value}: '
        f'{", ".join(missing_val)}')

  dense_dim = int(train_targets.size(1))
  probe = nn.Sequential(
    nn.LayerNorm(feature_dim, elementwise_affine=False),
    nn.Linear(feature_dim, dense_dim),
  ).to(device)
  linear_head = probe[1]
  optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=eval_config.learning_rate,
    betas=eval_config.opt_betas,
    weight_decay=eval_config.weight_decay)
  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  train_feature_loader = DataLoader(
    dataset=TorchTensorDataset(train_features, train_targets),
    batch_size=eval_config.batch_size,
    shuffle=True,
    drop_last=True)

  def cycle(dataloader):
    """Yield batches indefinitely from a dataloader."""
    while True:
      yield from dataloader

  def _pack_checkpoint(
    *,
    metrics_by_seed: dict[int, dict[str, torch.Tensor]],
    best_step: int,
  ) -> dict[str, object]:
    """Aggregate one dense checkpoint across validation seeds."""
    macro_metric_values: dict[str, list[float]] = {
      'mse': [],
      'mae': [],
      'r2': [],
      'pearson': [],
    }
    per_target_metric_values: dict[str, list[list[float]]] = {
      'mse': [],
      'mae': [],
      'r2': [],
      'pearson': [],
    }
    for seed_value in validation_seed_values:
      per_target_metrics = metrics_by_seed[seed_value]
      macro_metric_values['mse'].append(float(per_target_metrics['mse'].mean()))
      macro_metric_values['mae'].append(float(per_target_metrics['mae'].mean()))
      macro_metric_values['r2'].append(float(torch.nanmean(per_target_metrics['r2'])))
      macro_metric_values['pearson'].append(float(torch.nanmean(per_target_metrics['pearson'])))
      for metric_name in per_target_metric_values:
        per_target_metric_values[metric_name].append(
          [float(value) for value in per_target_metrics[metric_name].tolist()])

    packed: dict[str, object] = {
      'best_probe_step': int(best_step),
      'validation_seed_values': list(validation_seed_values),
      'validation_seed_count': int(len(validation_seed_values)),
    }
    for metric_name, values in macro_metric_values.items():
      stats = aiono_stat_summary(values)
      if stats['median'] is None or stats['std'] is None:
        raise ValueError(
          'Dense macro-metric aggregation produced no finite values. '
          f'metric_name={metric_name!r} values={values}')
      packed[f'macro_{metric_name}'] = float(stats['median'])
      packed[f'macro_{metric_name}_std'] = float(stats['std'])

    if log_per_target:
      per_target_best_step = {name: int(best_step) for name in target_names}
      for metric_name, rows in per_target_metric_values.items():
        median_by_target: dict[str, float] = {}
        std_by_target: dict[str, float] = {}
        for target_index, target_name in enumerate(target_names):
          values = [float(row[target_index]) for row in rows]
          stats = aiono_stat_summary(values)
          if stats['median'] is None or stats['std'] is None:
            raise ValueError(
              'Dense per-target aggregation produced no finite values. '
              f'metric_name={metric_name!r} target_name={target_name!r} values={values}')
          median_by_target[target_name] = float(stats['median'])
          std_by_target[target_name] = float(stats['std'])
        packed[f'per_target_{metric_name}'] = median_by_target
        packed[f'per_target_{metric_name}_std'] = std_by_target
      packed['per_target_best_step'] = per_target_best_step
    return packed

  train_iterator = cycle(train_feature_loader)
  best_dense = None
  best_macro_mse = None
  eval_calls = 0
  eval_total_s = 0.0
  eval_forward_s = 0.0
  eval_finalize_s = 0.0

  for step in range(eval_config.steps):
    learning_rate = next(lr_schedule)
    update_learning_rate_(optimizer, learning_rate)
    batch_features, batch_targets = next(train_iterator)  # [B, D], [B, D_dense]
    batch_features = batch_features.to(device, non_blocking=True)  # [B, D]
    batch_targets = batch_targets.to(device, non_blocking=True)  # [B, D_dense]
    predictions = probe(batch_features)  # [B, D_dense]
    valid = torch.isfinite(batch_targets)  # [B, D_dense]
    if not torch.any(valid):
      raise ValueError('Training batch has no finite dense targets.')
    batch_targets = torch.nan_to_num(batch_targets, nan=0.0)  # [B, D_dense]
    errors = (predictions - batch_targets) * valid.to(dtype=predictions.dtype)  # [B, D_dense]
    sq_error = errors * errors  # [B, D_dense]
    sum_sq_error = torch.sum(sq_error, dim=0)  # [D_dense]
    count = torch.sum(valid, dim=0).to(dtype=predictions.dtype)  # [D_dense]
    valid_targets = count > 0  # [D_dense]
    if not torch.any(valid_targets):
      raise ValueError('Training batch has no valid dense targets after masking.')
    mse = sum_sq_error[valid_targets] / count[valid_targets]  # [D_valid]
    loss = mse.mean()  # []
    loss.backward()
    _clip_linear_per_target_(linear_head, float(eval_config.gradient_clip))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if (step + 1) % eval_config.checkpoint_interval == 0:
      metrics_by_seed: dict[int, dict[str, torch.Tensor]] = {}
      macro_mse_values: list[float] = []
      for seed_value in validation_seed_values:
        eval_call_timings: dict[str, float | int] = {}
        eval_call_start = perf_counter()
        per_target_metrics = _evaluate_regression_features_streaming(
          probe=probe,
          features=val_features_by_seed[seed_value],
          targets=val_targets_by_seed[seed_value],
          device=device,
          batch_size=eval_config.batch_size,
          target_names=target_names,
          timings=eval_call_timings,
        )
        eval_total_s += perf_counter() - eval_call_start
        eval_calls += 1
        eval_forward_s += float(eval_call_timings.get('forward_s', 0.0))
        eval_finalize_s += float(eval_call_timings.get('finalize_s', 0.0))
        metrics_by_seed[seed_value] = per_target_metrics
        macro_mse_values.append(float(per_target_metrics['mse'].mean()))
      macro_mse_stats = aiono_stat_summary(macro_mse_values)
      if macro_mse_stats['median'] is None:
        raise ValueError('Dense multi-seed checkpoint selection produced no finite macro-MSE values.')
      if best_macro_mse is None or float(macro_mse_stats['median']) < best_macro_mse:
        best_macro_mse = float(macro_mse_stats['median'])
        best_dense = _pack_checkpoint(metrics_by_seed=metrics_by_seed, best_step=step + 1)

  if best_dense is None:
    raise ValueError('Offline dense probe did not evaluate any checkpoints.')

  total_s = perf_counter() - total_start
  best_dense['timings'] = {
    'regression': {
      'total_s': float(total_s),
      'train_steps_s': float(total_s - eval_total_s),
      'eval_total_s': float(eval_total_s),
      'eval_calls': int(eval_calls),
      'eval_forward_s': float(eval_forward_s),
      'eval_finalize_s': float(eval_finalize_s),
    }
  }
  return best_dense


def offline_probe_run_linear(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
  train_loader: Iterable[tuple[torch.Tensor, ...]],
  val_loader: Iterable[tuple[torch.Tensor, ...]],
  num_classes: int,
  class_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  auto_mixed_precision,
  compute_pairwise_confusion: bool = True,
) -> dict[str, object]:
  """Run an offline probe on cached features from the frozen encoder.

  Uses the CLS token as representation and ignores eval_config pooling settings.
  Cached features/targets stay on the probe device. Classification metrics are
  computed with sklearn (CPU) after a single transfer of logits/targets.

  Returns both the best checkpoint by macro AUC and the best checkpoint by macro AUPRC.
  """
  # Step 1: validate config and collect cached features.
  _validate_offline_probe_config(eval_config=eval_config)

  total_start = perf_counter()
  train_collect_timings: dict[str, float | int] = {}
  train_features, train_targets, _, train_has_crops = _collect_probe_features(  # [N, D], [N, C]
    encoder=encoder,
    representation_fn=representation_fn,
    loader=train_loader,
    device=device,
    auto_mixed_precision=auto_mixed_precision,
    allow_crops=False,
    timings=train_collect_timings,
  )
  if train_has_crops:
    raise ValueError('Offline probe training does not support multi-crop train features.')

  val_collect_timings: dict[str, float | int] = {}
  val_features, val_targets, _, val_has_crops = _collect_probe_features(  # [N, D] or [N, V, D], [N, C]
    encoder=encoder,
    representation_fn=representation_fn,
    loader=val_loader,
    device=device,
    auto_mixed_precision=auto_mixed_precision,
    allow_crops=True,
    timings=val_collect_timings,
  )

  # Step 2: train classification head from cached features.
  metrics = _train_linear_classification_probe(
    train_features=train_features,
    train_targets=train_targets,
    val_features=val_features,
    val_targets=val_targets,
    num_classes=num_classes,
    class_names=class_names,
    eval_config=eval_config,
    device=device,
    val_has_crops=val_has_crops,
    compute_pairwise_confusion=compute_pairwise_confusion,
  )
  timings = metrics.get('timings')
  if not isinstance(timings, dict):
    raise ValueError('Offline probe expected timings dict from classification probe.')
  timings['collect_train'] = train_collect_timings
  timings['collect_val'] = val_collect_timings
  timings['total_s'] = float(perf_counter() - total_start)
  return metrics


def offline_probe_run_linear_multihead(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
  train_loader: Iterable[tuple[torch.Tensor, ...]],
  val_loader: Iterable[tuple[torch.Tensor, ...]] | None = None,
  val_loaders_by_seed: dict[int, Iterable[tuple[torch.Tensor, ...]]] | None = None,
  num_classes: int,
  class_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  auto_mixed_precision,
  dense_target_names: list[str] | None = None,
  dense_log_per_target: bool | None = None,
  compute_pairwise_confusion: bool = True,
  run_dense: bool = True,
  label_exposure: bool = False,
) -> dict[str, object]:
  """Run offline classification and dense regression probes from a shared feature cache.

  The runner supports either:
  - one validation loader (`val_loader`) for single-split probing, or
  - multiple validation loaders keyed by logical seed (`val_loaders_by_seed`) for
    benchmark-style median/std aggregation.
  """
  # Step 1: validate config and dense probe settings.
  _validate_offline_probe_config(eval_config=eval_config)
  if dense_target_names is None:
    if dense_log_per_target is not None:
      raise ValueError('dense_log_per_target must be null when dense_target_names is null.')
  else:
    if not dense_target_names:
      raise ValueError('dense_target_names must be non-empty when provided.')
    if len(set(dense_target_names)) != len(dense_target_names):
      raise ValueError(f'dense_target_names must be unique, got {dense_target_names}')
    if dense_log_per_target is None:
      raise ValueError('dense_log_per_target must be set when dense_target_names is provided.')
    if not isinstance(dense_log_per_target, bool):
      raise ValueError(
        f'dense_log_per_target must be a bool, got {type(dense_log_per_target).__name__}')

  if val_loader is None and not val_loaders_by_seed:
    raise ValueError('Provide val_loader or val_loaders_by_seed for offline probing.')
  if val_loader is not None and val_loaders_by_seed:
    raise ValueError('Provide only one of val_loader or val_loaders_by_seed.')

  # Step 2: collect cached features once per split.
  total_start = perf_counter()
  train_collect_timings: dict[str, float | int] = {}
  train_features, train_targets, train_dense_targets, train_has_crops = _collect_probe_features(
    encoder=encoder,
    representation_fn=representation_fn,
    loader=train_loader,
    device=device,
    auto_mixed_precision=auto_mixed_precision,
    allow_crops=False,
    timings=train_collect_timings)
  if train_has_crops:
    raise ValueError('Offline probe training does not support multi-crop train features.')

  # Step 2b: collect validation features for either one split or multiple benchmark seeds.
  val_collect_timings: dict[str, object] = {}
  val_features = None
  val_targets = None
  val_dense_targets = None
  val_has_crops = None
  val_features_by_seed = None
  val_targets_by_seed = None
  val_dense_targets_by_seed = None
  val_has_crops_by_seed = None
  validation_seed_values = None

  if val_loaders_by_seed is None:
    assert val_loader is not None
    val_collect_single: dict[str, float | int] = {}
    val_features, val_targets, val_dense_targets, val_has_crops = _collect_probe_features(
      encoder=encoder,
      representation_fn=representation_fn,
      loader=val_loader,
      device=device,
      auto_mixed_precision=auto_mixed_precision,
      allow_crops=True,
      timings=val_collect_single)
    val_collect_timings = val_collect_single
    if run_dense and dense_target_names is not None and train_dense_targets is None and val_dense_targets is None:
      raise ValueError('dense_target_names provided but offline probe loader has no dense targets.')
    metrics = _train_linear_classification_probe(
      train_features=train_features,
      train_targets=train_targets,
      val_features=val_features,
      val_targets=val_targets,
      num_classes=num_classes,
      class_names=class_names,
      eval_config=eval_config,
      device=device,
      val_has_crops=bool(val_has_crops),
      compute_pairwise_confusion=compute_pairwise_confusion,
      label_exposure=label_exposure,
    )
  else:
    val_features_by_seed = {}
    val_targets_by_seed = {}
    val_dense_targets_by_seed = {}
    val_has_crops_by_seed = {}
    validation_seed_values = sorted(int(seed_value) for seed_value in val_loaders_by_seed)
    for seed_value in validation_seed_values:
      seed_timings: dict[str, float | int] = {}
      current_features, current_targets, current_dense_targets, current_has_crops = _collect_probe_features(
        encoder=encoder,
        representation_fn=representation_fn,
        loader=val_loaders_by_seed[seed_value],
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        allow_crops=True,
        timings=seed_timings)
      val_collect_timings[int(seed_value)] = seed_timings
      val_features_by_seed[int(seed_value)] = current_features.cpu()
      val_targets_by_seed[int(seed_value)] = current_targets.cpu()
      val_dense_targets_by_seed[int(seed_value)] = (
        current_dense_targets.cpu() if current_dense_targets is not None else None)
      val_has_crops_by_seed[int(seed_value)] = bool(current_has_crops)
    train_features = train_features.cpu()
    train_targets = train_targets.cpu()
    train_dense_targets = train_dense_targets.cpu() if train_dense_targets is not None else None
    if run_dense and dense_target_names is not None:
      if train_dense_targets is None:
        raise ValueError('dense_target_names provided but offline probe train loader has no dense targets.')
      if all(value is None for value in val_dense_targets_by_seed.values()):
        raise ValueError('dense_target_names provided but offline probe validation loaders have no dense targets.')
    metrics = _train_linear_classification_probe_multi_val(
      train_features=train_features,
      train_targets=train_targets,
      val_features_by_seed=val_features_by_seed,
      val_targets_by_seed=val_targets_by_seed,
      val_has_crops_by_seed=val_has_crops_by_seed,
      num_classes=num_classes,
      class_names=class_names,
      eval_config=eval_config,
      device=device,
      compute_pairwise_confusion=compute_pairwise_confusion,
      label_exposure=label_exposure,
    )

  timings = metrics.get('timings')
  if not isinstance(timings, dict):
    raise ValueError('Offline probe expected timings dict from classification probe.')
  timings['collect_train'] = train_collect_timings
  timings['collect_val'] = val_collect_timings

  # Step 4: optionally train dense regression head from the same cache.
  if val_loaders_by_seed is None:
    if run_dense and (train_dense_targets is not None or val_dense_targets is not None):
      if train_dense_targets is None or val_dense_targets is None:
        raise ValueError('Dense targets must be present in both train and val splits.')
      if dense_target_names is None or dense_log_per_target is None:
        raise ValueError('Dense targets present but dense_target_names/log_per_target not provided.')
      dense_metrics = _train_linear_regression_probe(
        train_features=train_features,
        train_targets=train_dense_targets,
        val_features=val_features,
        val_targets=val_dense_targets,
        target_names=dense_target_names,
        eval_config=eval_config,
        device=device,
        val_has_crops=bool(val_has_crops),
        log_per_target=dense_log_per_target,
      )
      metrics['dense_best'] = dense_metrics
      dense_timings = dense_metrics.get('timings')
      if isinstance(dense_timings, dict):
        timings['regression'] = dense_timings.get('regression', dense_timings)
  elif run_dense and (train_dense_targets is not None or any(
    value is not None for value in val_dense_targets_by_seed.values())):
    if train_dense_targets is None:
      raise ValueError('Dense targets must be present in the train split for dense probing.')
    if dense_target_names is None or dense_log_per_target is None:
      raise ValueError('Dense targets present but dense_target_names/log_per_target not provided.')
    if any(value is None for value in val_dense_targets_by_seed.values()):
      raise ValueError('Dense targets must be present in every validation split.')
    dense_metrics = _train_linear_regression_probe_multi_val(
      train_features=train_features,
      train_targets=train_dense_targets,
      val_features_by_seed=val_features_by_seed,
      val_targets_by_seed={seed_value: value for seed_value, value in val_dense_targets_by_seed.items() if value is not None},
      val_has_crops_by_seed=val_has_crops_by_seed,
      target_names=dense_target_names,
      eval_config=eval_config,
      device=device,
      log_per_target=dense_log_per_target,
    )
    metrics['dense_best'] = dense_metrics
    dense_timings = dense_metrics.get('timings')
    if isinstance(dense_timings, dict):
      timings['regression'] = dense_timings.get('regression', dense_timings)

  if validation_seed_values is not None:
    metrics['validation_seed_values'] = list(validation_seed_values)
    metrics['validation_seed_count'] = int(len(validation_seed_values))
  timings['total_s'] = float(perf_counter() - total_start)
  return metrics


def offline_probe_run_linear_multihead_by_layer(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
  train_loader: Iterable[tuple[torch.Tensor, ...]],
  val_loader: Iterable[tuple[torch.Tensor, ...]] | None = None,
  val_loaders_by_seed: dict[int, Iterable[tuple[torch.Tensor, ...]]] | None = None,
  num_classes: int,
  class_names: list[str],
  eval_config: OfflineProbeConfig,
  device: torch.device,
  auto_mixed_precision,
  layers_categorical: tuple[int, ...],
  layers_dense: tuple[int, ...],
  layers_confusion: tuple[int, ...],
  dense_target_names: list[str] | None = None,
  dense_log_per_target: bool | None = None,
  label_exposure: bool = False,
) -> dict[str, object]:
  """Run offline probes for multiple representation layers from a shared feature cache.

  This function collects cached features for the union of requested layers in a single
  pass over the train and val loaders, then trains:
    - one classification probe per `layers_categorical`,
    - one regression probe per `layers_dense` (only when dense targets are present).

  Pairwise confusion is computed only for `layers_confusion` (and included in the
  returned best-checkpoint payloads for those layers).
  """
  # Step 1: validate probe config and requested layer sets.
  _validate_offline_probe_config(eval_config=eval_config)
  if len(set(layers_categorical)) != len(layers_categorical):
    raise ValueError(f'layers_categorical must be unique, got {layers_categorical}')
  if len(set(layers_dense)) != len(layers_dense):
    raise ValueError(f'layers_dense must be unique, got {layers_dense}')
  if len(set(layers_confusion)) != len(layers_confusion):
    raise ValueError(f'layers_confusion must be unique, got {layers_confusion}')
  categorical_set = set(layers_categorical)
  confusion_set = set(layers_confusion)
  if not confusion_set.issubset(categorical_set):
    missing = sorted(confusion_set - categorical_set)
    raise ValueError(
      'layers_confusion must be a subset of layers_categorical. '
      f'Missing categorical layers: {missing}')

  layers_all = tuple(sorted(categorical_set | set(layers_dense) | confusion_set))
  if not layers_all:
    raise ValueError(
      'At least one layer must be requested across categorical/dense/confusion probes.')
  if val_loader is None and not val_loaders_by_seed:
    raise ValueError('Provide val_loader or val_loaders_by_seed for offline probing.')
  if val_loader is not None and val_loaders_by_seed:
    raise ValueError('Provide only one of val_loader or val_loaders_by_seed.')

  # Step 2: collect cached features once per split for all requested layers.
  total_start = perf_counter()
  train_collect_timings: dict[str, float | int] = {}
  train_features_by_layer, train_targets, train_dense_targets, train_has_crops = _collect_probe_features_by_layer(
    encoder=encoder,
    representation_fn=representation_fn,
    layers=layers_all,
    loader=train_loader,
    device=device,
    auto_mixed_precision=auto_mixed_precision,
    allow_crops=False,
    timings=train_collect_timings)
  if train_has_crops:
    raise ValueError('Offline probe training does not support multi-crop train features.')

  val_collect_timings: dict[str, object] = {}
  val_features_by_layer = None
  val_targets = None
  val_dense_targets = None
  val_has_crops = None
  val_features_by_layer_by_seed = None
  val_targets_by_seed = None
  val_dense_targets_by_seed = None
  val_has_crops_by_seed = None
  validation_seed_values = None

  if val_loaders_by_seed is None:
    assert val_loader is not None
    val_collect_single: dict[str, float | int] = {}
    val_features_by_layer, val_targets, val_dense_targets, val_has_crops = _collect_probe_features_by_layer(
      encoder=encoder,
      representation_fn=representation_fn,
      layers=layers_all,
      loader=val_loader,
      device=device,
      auto_mixed_precision=auto_mixed_precision,
      allow_crops=True,
      timings=val_collect_single)
    val_collect_timings = val_collect_single
  else:
    val_features_by_layer_by_seed = {}
    val_targets_by_seed = {}
    val_dense_targets_by_seed = {}
    val_has_crops_by_seed = {}
    validation_seed_values = sorted(int(seed_value) for seed_value in val_loaders_by_seed)
    for seed_value in validation_seed_values:
      seed_timings: dict[str, float | int] = {}
      current_features_by_layer, current_targets, current_dense_targets, current_has_crops = _collect_probe_features_by_layer(
        encoder=encoder,
        representation_fn=representation_fn,
        layers=layers_all,
        loader=val_loaders_by_seed[seed_value],
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        allow_crops=True,
        timings=seed_timings)
      val_collect_timings[int(seed_value)] = seed_timings
      val_features_by_layer_by_seed[int(seed_value)] = {
        layer: value.cpu() for layer, value in current_features_by_layer.items()
      }
      val_targets_by_seed[int(seed_value)] = current_targets.cpu()
      val_dense_targets_by_seed[int(seed_value)] = (
        current_dense_targets.cpu() if current_dense_targets is not None else None)
      val_has_crops_by_seed[int(seed_value)] = bool(current_has_crops)
    train_features_by_layer = {
      layer: value.cpu() for layer, value in train_features_by_layer.items()
    }
    train_targets = train_targets.cpu()
    train_dense_targets = train_dense_targets.cpu() if train_dense_targets is not None else None

  # Step 3: train classification probes per requested layer.
  categorical: dict[int, dict[str, object]] = {}
  for layer in layers_categorical:
    if val_loaders_by_seed is None:
      categorical[layer] = _train_linear_classification_probe(
        train_features=train_features_by_layer[layer],
        train_targets=train_targets,
        val_features=val_features_by_layer[layer],
        val_targets=val_targets,
        num_classes=num_classes,
        class_names=class_names,
        eval_config=eval_config,
        device=device,
        val_has_crops=bool(val_has_crops),
        compute_pairwise_confusion=layer in confusion_set,
        label_exposure=label_exposure,
      )
    else:
      categorical[layer] = _train_linear_classification_probe_multi_val(
        train_features=train_features_by_layer[layer],
        train_targets=train_targets,
        val_features_by_seed={
          seed_value: val_features_by_layer_by_seed[seed_value][layer]
          for seed_value in validation_seed_values
        },
        val_targets_by_seed=val_targets_by_seed,
        val_has_crops_by_seed=val_has_crops_by_seed,
        num_classes=num_classes,
        class_names=class_names,
        eval_config=eval_config,
        device=device,
        compute_pairwise_confusion=layer in confusion_set,
        label_exposure=label_exposure,
      )

  # Step 4: optionally train dense probes per requested layer.
  dense: dict[int, dict[str, object]] = {}
  dense_targets_available = (
    train_dense_targets is not None
    or (
      val_dense_targets is not None
      if val_loaders_by_seed is None else
      any(value is not None for value in val_dense_targets_by_seed.values())
    )
  )
  if layers_dense and dense_targets_available:
    if dense_target_names is None:
      raise ValueError('dense_target_names must be provided when running dense probes.')
    if dense_log_per_target is None:
      raise ValueError('dense_log_per_target must be provided when running dense probes.')
    if not dense_target_names:
      raise ValueError('dense_target_names must be non-empty when running dense probes.')
    if len(set(dense_target_names)) != len(dense_target_names):
      raise ValueError(f'dense_target_names must be unique, got {dense_target_names}')
    if not isinstance(dense_log_per_target, bool):
      raise ValueError(
        f'dense_log_per_target must be a bool, got {type(dense_log_per_target).__name__}')

    for layer in layers_dense:
      if val_loaders_by_seed is None:
        if train_dense_targets is None or val_dense_targets is None:
          raise ValueError('Dense targets must be present in both train and val splits.')
        dense[layer] = _train_linear_regression_probe(
          train_features=train_features_by_layer[layer],
          train_targets=train_dense_targets,
          val_features=val_features_by_layer[layer],
          val_targets=val_dense_targets,
          target_names=dense_target_names,
          eval_config=eval_config,
          device=device,
          val_has_crops=bool(val_has_crops),
          log_per_target=dense_log_per_target,
        )
      else:
        if train_dense_targets is None:
          raise ValueError('Dense targets must be present in the train split for dense probing.')
        if any(value is None for value in val_dense_targets_by_seed.values()):
          raise ValueError('Dense targets must be present in every validation split.')
        dense[layer] = _train_linear_regression_probe_multi_val(
          train_features=train_features_by_layer[layer],
          train_targets=train_dense_targets,
          val_features_by_seed={
            seed_value: val_features_by_layer_by_seed[seed_value][layer]
            for seed_value in validation_seed_values
          },
          val_targets_by_seed={
            seed_value: value
            for seed_value, value in val_dense_targets_by_seed.items()
            if value is not None
          },
          val_has_crops_by_seed=val_has_crops_by_seed,
          target_names=dense_target_names,
          eval_config=eval_config,
          device=device,
          log_per_target=dense_log_per_target,
        )

  # Step 5: return packed results and shared metadata.
  if val_loaders_by_seed is None:
    example_val_features = val_features_by_layer[layers_all[0]]
    example_val_targets = val_targets
    val_num_crops = int(example_val_features.size(1)) if bool(val_has_crops) else 1
  else:
    first_seed_value = validation_seed_values[0]
    example_val_features = val_features_by_layer_by_seed[first_seed_value][layers_all[0]]
    example_val_targets = val_targets_by_seed[first_seed_value]
    val_num_crops = int(example_val_features.size(1)) if val_has_crops_by_seed[first_seed_value] else 1
  feature_dim = int(train_features_by_layer[layers_all[0]].size(-1))
  timings = {
    'total_s': float(perf_counter() - total_start),
    'collect_train': train_collect_timings,
    'collect_val': val_collect_timings,
  }
  packed = {
    'categorical': categorical,
    'dense': dense,
    'dense_targets_available': bool(
      train_dense_targets is not None
      and (
        val_dense_targets is not None
        if val_loaders_by_seed is None else
        all(value is not None for value in val_dense_targets_by_seed.values())
      )
    ),
    'feature_dim': int(feature_dim),
    'train_size': int(train_targets.size(0)),
    'val_size': int(example_val_targets.size(0)),
    'val_num_crops': int(val_num_crops),
    'timings': timings,
  }
  if validation_seed_values is not None:
    packed['validation_seed_values'] = list(validation_seed_values)
    packed['validation_seed_count'] = int(len(validation_seed_values))
  return packed
