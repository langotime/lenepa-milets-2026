"""UCR-128 offline benchmark: frozen encoder → embeddings → classifier → test accuracy.

This implements the evaluation protocol used in the CauKer / Mantis style setup:
- For each UCR dataset:
  1) Embed TRAIN and TEST split examples with a frozen encoder.
  2) Fit a classifier on TRAIN embeddings.
  3) Report TEST accuracy.
- Aggregate: mean accuracy over datasets.

Supported classifiers:
- classifier="rf": RandomForest (scikit-learn)
- classifier="logreg": StandardScaler + LogisticRegression (scikit-learn)

The caller controls evaluation cadence externally (in our codebase, via
`offline_probe_eval_interval` in pretrain.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset as TorchTensorDataset

from data.ucr import ucr_list_datasets, ucr_load_splits
from models.vit_layer_outputs import ViTLayerOutputs

__all__ = [
  'UCRBenchmarkConfig',
  'ucr_benchmark_eval',
]


@dataclass(frozen=True)
class UCRBenchmarkConfig:
  """Configuration for running the UCR offline benchmark."""

  archive_root: str
  resize_mode: str  # "interp512" | "none"
  resize_target_length: int
  eval_batch_size: int

  classifier: str  # "rf" | "logreg"
  seed: int | None = None  # Optional: seeds classifier randomness; default leaves estimator RNG unset.
  rf_n_estimators: int | None = None
  rf_n_jobs: int | None = None
  logreg_max_iter: int | None = None

  dataset_names: tuple[str, ...] | None = None


def _validate_ucr_benchmark_config(*, cfg: UCRBenchmarkConfig) -> None:
  """Validate UCR benchmark config (no silent defaults)."""
  # Step 1: validate required scalar hyperparameters.
  if not isinstance(cfg.archive_root, str) or not cfg.archive_root:
    raise ValueError('UCRBenchmarkConfig.archive_root must be a non-empty string')
  if cfg.resize_mode not in ('interp512', 'none'):
    raise ValueError(
      'UCRBenchmarkConfig.resize_mode must be "interp512" or "none". '
      f'Got resize_mode={cfg.resize_mode!r}')
  if cfg.resize_mode == 'interp512' and cfg.resize_target_length <= 0:
    raise ValueError(
      'UCRBenchmarkConfig.resize_target_length must be > 0 when resize_mode="interp512". '
      f'Got resize_target_length={cfg.resize_target_length}')
  if cfg.eval_batch_size <= 0:
    raise ValueError(f'UCRBenchmarkConfig.eval_batch_size must be > 0, got {cfg.eval_batch_size}')

  if cfg.seed is not None:
    if not isinstance(cfg.seed, int):
      raise ValueError(
        'UCRBenchmarkConfig.seed must be an int or null. '
        f'Got type={type(cfg.seed).__name__}')
    if cfg.seed < 0:
      raise ValueError(f'UCRBenchmarkConfig.seed must be >= 0, got {cfg.seed}')

  # Step 2: validate classifier settings.
  if cfg.classifier not in ('rf', 'logreg'):
    raise ValueError(
      'UCRBenchmarkConfig.classifier must be "rf" or "logreg". '
      f'Got classifier={cfg.classifier!r}')

  if cfg.rf_n_estimators is not None and cfg.rf_n_estimators <= 0:
    raise ValueError(
      'UCRBenchmarkConfig.rf_n_estimators must be > 0 when provided. '
      f'Got {cfg.rf_n_estimators}')
  if cfg.rf_n_jobs is not None and not isinstance(cfg.rf_n_jobs, int):
    raise ValueError(
      'UCRBenchmarkConfig.rf_n_jobs must be an int when provided (e.g., -1 for all cores). '
      f'Got type={type(cfg.rf_n_jobs).__name__}')

  if cfg.logreg_max_iter is not None and cfg.logreg_max_iter <= 0:
    raise ValueError(
      'UCRBenchmarkConfig.logreg_max_iter must be > 0 when provided. '
      f'Got {cfg.logreg_max_iter}')

  if cfg.classifier == 'rf':
    if cfg.rf_n_estimators is None:
      raise ValueError(
        'UCRBenchmarkConfig.rf_n_estimators is required when classifier="rf" (no silent defaults).')
    if cfg.rf_n_jobs is None:
      raise ValueError(
        'UCRBenchmarkConfig.rf_n_jobs is required when classifier="rf" (no silent defaults).')
  else:
    if cfg.logreg_max_iter is None:
      raise ValueError(
        'UCRBenchmarkConfig.logreg_max_iter is required when classifier="logreg" (no silent defaults).')

  # Step 3: validate optional dataset allowlist.
  if cfg.dataset_names is None:
    return
  if not cfg.dataset_names:
    raise ValueError('UCRBenchmarkConfig.dataset_names must be non-empty when provided')
  if len(set(cfg.dataset_names)) != len(cfg.dataset_names):
    raise ValueError(f'UCRBenchmarkConfig.dataset_names must be unique, got {cfg.dataset_names}')
  for name in cfg.dataset_names:
    if not isinstance(name, str) or not name:
      raise ValueError(f'UCRBenchmarkConfig.dataset_names must contain non-empty strings, got {name!r}')


def _extract_representation_tensor(encoder_output: object) -> torch.Tensor:
  """Extract a [B, D] representation tensor from common encoder output formats."""
  # Step 1: handle ViTLayerOutputs containers.
  if isinstance(encoder_output, ViTLayerOutputs):
    depth = encoder_output.depth
    return encoder_output.representation(depth)  # [B, D]

  # Step 2: handle direct tensor outputs.
  if isinstance(encoder_output, torch.Tensor):
    return encoder_output

  # Step 3: handle tuple/list outputs (search for a suitable tensor or ViTLayerOutputs).
  if isinstance(encoder_output, (tuple, list)):
    for item in encoder_output:
      if isinstance(item, ViTLayerOutputs):
        depth = item.depth
        return item.representation(depth)  # [B, D]
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

  raise ValueError(
    'Failed to extract a representation tensor from encoder outputs. '
    f'Got type={type(encoder_output).__name__}.')


def _compute_embeddings(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], torch.Tensor] | None,
  x: torch.Tensor,
  batch_size: int,
  device: torch.device,
  using_cuda: bool,
  auto_mixed_precision,
) -> np.ndarray:
  """Compute encoder embeddings for a tensor dataset.

  Args:
    encoder: Frozen encoder module.
    representation_fn: Optional callable returning [B, D] given x [B, 1, L].
    x: Input tensor [N, 1, L] on CPU.

  Returns:
    z: float32 numpy array [N, D].
  """
  # Step 1: validate the input contract.
  # x: [N, 1, L]
  if not isinstance(x, torch.Tensor):
    raise ValueError(f'x must be a torch.Tensor, got {type(x).__name__}')
  if x.dim() != 3 or x.size(1) != 1:
    raise ValueError(f'Expected x to be [N, 1, L], got {tuple(x.shape)}')
  if batch_size <= 0:
    raise ValueError(f'batch_size must be > 0, got {batch_size}')

  loader = DataLoader(
    dataset=TorchTensorDataset(x),
    batch_size=int(batch_size),
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=using_cuda,
  )

  # Step 2: run the encoder in no_grad eval mode and collect embeddings on CPU.
  features: list[torch.Tensor] = []  # each [B, D] float32 on CPU
  encoder_training = encoder.training
  encoder.eval()
  # NOTE: avoid inference_mode() (see utils/offline_probe.py rationale).
  with torch.no_grad():
    for (x_b,) in loader:
      x_b = x_b.to(device, non_blocking=using_cuda)  # [B, 1, L]
      with auto_mixed_precision:
        if representation_fn is None:
          encoder_out = encoder(x_b)
          rep = _extract_representation_tensor(encoder_out)
        else:
          rep = representation_fn(x_b)
      if not isinstance(rep, torch.Tensor):
        raise ValueError(
          'representation_fn must return a torch.Tensor. '
          f'Got type={type(rep).__name__}')
      if rep.dim() == 3:
        rep = rep[:, 0]  # [B, D]
      if rep.dim() != 2:
        raise ValueError(f'Expected representation to be [B, D], got {tuple(rep.shape)}')
      features.append(rep.float().cpu())  # [B, D] float32
  encoder.train(encoder_training)

  # Step 3: concatenate and return numpy.
  if not features:
    raise ValueError('Embedding extraction produced no batches')
  z = torch.cat(features, dim=0)  # [N, D]
  return z.numpy().astype(np.float32, copy=False)


def _compute_embeddings_by_layer(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
  layers: tuple[int, ...],
  x: torch.Tensor,
  batch_size: int,
  device: torch.device,
  using_cuda: bool,
  auto_mixed_precision,
) -> dict[int, np.ndarray]:
  """Compute encoder embeddings for a tensor dataset at multiple representation layers.

  Args:
    encoder: Frozen encoder module.
    representation_fn: Callable returning a dict mapping each requested layer index to
      a pooled representation `[B, D]` for the provided batch `x_b` `[B, 1, L]`.
    layers: Unique layer indices to compute.
    x: Input tensor [N, 1, L] on CPU.

  Returns:
    Mapping layer -> float32 numpy array [N, D].
  """
  # Step 1: validate the input contract.
  # x: [N, 1, L]
  if not isinstance(x, torch.Tensor):
    raise ValueError(f'x must be a torch.Tensor, got {type(x).__name__}')
  if x.dim() != 3 or x.size(1) != 1:
    raise ValueError(f'Expected x to be [N, 1, L], got {tuple(x.shape)}')
  if batch_size <= 0:
    raise ValueError(f'batch_size must be > 0, got {batch_size}')
  if not layers:
    raise ValueError('layers must be non-empty when computing embeddings by layer.')
  if len(set(layers)) != len(layers):
    raise ValueError(f'layers must be unique, got {layers}')
  for idx, layer in enumerate(layers):
    if not isinstance(layer, int):
      raise ValueError(f'layers[{idx}] must be int, got {layer!r}')

  loader = DataLoader(
    dataset=TorchTensorDataset(x),
    batch_size=int(batch_size),
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=using_cuda,
  )

  # Step 2: run the encoder in no_grad eval mode and collect per-layer embeddings on CPU.
  features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}  # each [B, D] fp32 on CPU
  encoder_training = encoder.training
  encoder.eval()
  # NOTE: avoid inference_mode() (see utils/offline_probe.py rationale).
  with torch.no_grad():
    for (x_b,) in loader:
      x_b = x_b.to(device, non_blocking=using_cuda)  # [B, 1, L]
      with auto_mixed_precision:
        layer_reps = representation_fn(x_b)
      if not isinstance(layer_reps, dict):
        raise ValueError(
          'representation_fn must return dict[int, torch.Tensor] in multi-layer mode. '
          f'Got type={type(layer_reps).__name__}')
      for layer in layers:
        rep = layer_reps.get(layer)
        if rep is None:
          raise ValueError(
            f'representation_fn missing requested layer={layer}. '
            f'Returned keys={sorted(layer_reps.keys())}')
        if not isinstance(rep, torch.Tensor):
          raise ValueError(
            f'representation_fn output for layer={layer} must be a torch.Tensor, '
            f'got {type(rep).__name__}')
        if rep.dim() != 2:
          raise ValueError(
            f'representation_fn output for layer={layer} must be [B, D], got {tuple(rep.shape)}')
        features[layer].append(rep.float().cpu())  # [B, D] fp32
  encoder.train(encoder_training)

  # Step 3: concatenate and return numpy arrays.
  z_by_layer: dict[int, np.ndarray] = {}
  for layer in layers:
    batches = features.get(layer)
    if not batches:
      raise ValueError(f'Embedding extraction produced no batches for layer={layer}')
    z = torch.cat(batches, dim=0)  # [N, D]
    z_by_layer[layer] = z.numpy().astype(np.float32, copy=False)
  return z_by_layer


def _build_classifier(*, cfg: UCRBenchmarkConfig):
  """Build the scikit-learn classifier used in the UCR benchmark."""
  if cfg.classifier == 'rf':
    if cfg.rf_n_estimators is None or cfg.rf_n_jobs is None:
      raise ValueError(
        'RandomForest UCR classifier requires rf_n_estimators and rf_n_jobs (no silent defaults).')
    rf_random_state = 0 if cfg.seed is None else int(cfg.seed)
    return RandomForestClassifier(
      n_estimators=int(cfg.rf_n_estimators),
      n_jobs=int(cfg.rf_n_jobs),
      random_state=rf_random_state,
    )
  if cfg.classifier == 'logreg':
    if cfg.logreg_max_iter is None:
      raise ValueError(
        'LogisticRegression UCR classifier requires logreg_max_iter (no silent defaults).')
    logreg_random_state = None if cfg.seed is None else int(cfg.seed)
    return make_pipeline(
      StandardScaler(),
      LogisticRegression(
        max_iter=int(cfg.logreg_max_iter),
        random_state=logreg_random_state,
      ),
    )
  raise ValueError(f'Unsupported UCR classifier: {cfg.classifier!r}')


def ucr_benchmark_eval(
  *,
  encoder: nn.Module,
  representation_fn: Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor], dict[int, torch.Tensor]] | None,
  cfg: UCRBenchmarkConfig,
  patch_size: int,
  device: torch.device,
  using_cuda: bool,
  auto_mixed_precision,
  layers: tuple[int, ...] | None = None,
) -> dict[str, object]:
  """Run the UCR offline benchmark and return a structured result payload."""
  # Step 1: validate config and patch_size.
  _validate_ucr_benchmark_config(cfg=cfg)
  if patch_size <= 0:
    raise ValueError(f'patch_size must be > 0, got {patch_size}')
  if layers is not None:
    if not layers:
      raise ValueError('layers must be non-empty when provided.')
    if len(set(layers)) != len(layers):
      raise ValueError(f'layers must be unique, got {layers}')
    for idx, layer in enumerate(layers):
      if not isinstance(layer, int):
        raise ValueError(f'layers[{idx}] must be an int, got {layer!r}')

  total_start = perf_counter()
  dataset_names = ucr_list_datasets(archive_root=cfg.archive_root)
  if cfg.dataset_names is not None:
    missing = sorted(set(cfg.dataset_names) - set(dataset_names))
    if missing:
      raise ValueError(f'UCR dataset_names contains unknown datasets: {missing}')
    dataset_names = [name for name in dataset_names if name in set(cfg.dataset_names)]

  per_dataset: list[dict[str, object]] = []
  load_s = 0.0
  embed_s = 0.0
  clf_s = 0.0

  # Step 2: iterate datasets (TRAIN→fit, TEST→accuracy).
  for name in dataset_names:
    load_start = perf_counter()
    splits = ucr_load_splits(
      archive_root=cfg.archive_root,
      name=name,
      resize_mode=cfg.resize_mode,
      resize_target_length=cfg.resize_target_length,
      patch_size=patch_size,
    )
    load_s += perf_counter() - load_start

    y_train = splits.y_train.numpy().astype(np.int64, copy=False)  # [N_train]
    y_test = splits.y_test.numpy().astype(np.int64, copy=False)  # [N_test]

    acc_by_layer: dict[str, float] | None = None
    if layers is None:
      embed_start = perf_counter()
      z_train = _compute_embeddings(
        encoder=encoder,
        representation_fn=representation_fn if callable(representation_fn) else None,
        x=splits.x_train,
        batch_size=cfg.eval_batch_size,
        device=device,
        using_cuda=using_cuda,
        auto_mixed_precision=auto_mixed_precision,
      )  # [N_train, D] fp32
      z_test = _compute_embeddings(
        encoder=encoder,
        representation_fn=representation_fn if callable(representation_fn) else None,
        x=splits.x_test,
        batch_size=cfg.eval_batch_size,
        device=device,
        using_cuda=using_cuda,
        auto_mixed_precision=auto_mixed_precision,
      )  # [N_test, D] fp32
      embed_s += perf_counter() - embed_start

      clf_start = perf_counter()
      classifier = _build_classifier(cfg=cfg)
      classifier.fit(z_train, y_train)
      y_pred = classifier.predict(z_test)  # [N_test]
      clf_s += perf_counter() - clf_start

      acc = float(np.mean(y_pred == y_test))
    else:
      if representation_fn is None:
        raise ValueError(
          'layers is set for UCR benchmark, but representation_fn is null. '
          'Fix: pass a representation_fn that returns per-layer pooled representations.')
      if not callable(representation_fn):
        raise ValueError(
          'representation_fn must be callable when layers is set for UCR benchmark. '
          f'Got type={type(representation_fn).__name__}')
      embed_start = perf_counter()
      z_train_by_layer = _compute_embeddings_by_layer(
        encoder=encoder,
        representation_fn=representation_fn,
        layers=layers,
        x=splits.x_train,
        batch_size=cfg.eval_batch_size,
        device=device,
        using_cuda=using_cuda,
        auto_mixed_precision=auto_mixed_precision,
      )  # dict layer -> [N_train, D] fp32
      z_test_by_layer = _compute_embeddings_by_layer(
        encoder=encoder,
        representation_fn=representation_fn,
        layers=layers,
        x=splits.x_test,
        batch_size=cfg.eval_batch_size,
        device=device,
        using_cuda=using_cuda,
        auto_mixed_precision=auto_mixed_precision,
      )  # dict layer -> [N_test, D] fp32
      embed_s += perf_counter() - embed_start

      clf_start = perf_counter()
      acc_by_layer = {}
      for layer in layers:
        classifier = _build_classifier(cfg=cfg)
        classifier.fit(z_train_by_layer[layer], y_train)
        y_pred = classifier.predict(z_test_by_layer[layer])  # [N_test]
        acc_by_layer[str(layer)] = float(np.mean(y_pred == y_test))
      clf_s += perf_counter() - clf_start

      # For backward compatibility, define acc as the best layer for this dataset.
      best_layer, best_acc = max(acc_by_layer.items(), key=lambda kv: float(kv[1]))
      acc = float(best_acc)

    per_dataset.append({
      'name': splits.name,
      'acc': acc,
      'acc_by_layer': acc_by_layer,
      'num_classes': int(splits.num_classes),
      'n_train': int(splits.x_train.size(0)),
      'n_test': int(splits.x_test.size(0)),
      'orig_length': int(splits.orig_length),
      'used_length': int(splits.used_length),
    })

  # Step 3: aggregate mean accuracy (and best layer when requested).
  if not per_dataset:
    raise ValueError('UCR benchmark evaluated zero datasets')
  mean_acc_by_layer: dict[str, float] | None = None
  best_layer = None
  best_mean_acc = None
  if layers is None:
    mean_acc = float(np.mean([float(row['acc']) for row in per_dataset]))
  else:
    mean_acc_by_layer = {}
    for layer in layers:
      layer_key = str(layer)
      layer_values = []
      for row in per_dataset:
        acc_layer = row.get('acc_by_layer', {}).get(layer_key)
        if acc_layer is None:
          raise RuntimeError(
            'UCR benchmark did not return per-layer accuracy for a dataset; this is a bug. '
            f'Missing layer={layer_key} in row.name={row.get("name")}')
        layer_values.append(float(acc_layer))
      mean_acc_by_layer[layer_key] = float(np.mean(layer_values))

    best_layer, best_mean_acc = max(
      mean_acc_by_layer.items(), key=lambda kv: float(kv[1]))
    mean_acc = float(best_mean_acc)

  return {
    'mean_acc': mean_acc,
    'mean_acc_by_layer': mean_acc_by_layer,
    'best_layer': None if best_layer is None else int(best_layer),
    'best_mean_acc': best_mean_acc,
    'num_datasets': int(len(per_dataset)),
    'resize_mode': str(cfg.resize_mode),
    'classifier': str(cfg.classifier),
    'per_dataset': per_dataset,
    'timings': {
      'total_s': float(perf_counter() - total_start),
      'load_s': float(load_s),
      'embed_s': float(embed_s),
      'clf_s': float(clf_s),
      **({'rf_s': float(clf_s)} if cfg.classifier == 'rf' else {}),
    },
  }
