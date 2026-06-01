"""Base types and utilities for pretrain data backends.

This module defines the common API used by all data backend implementations:
- PretrainDataBundle: The unified return type for all backends
- Iterator utilities: cycle, map_to_device, prefetch_batch

Backend implementations (backend_ptbxl, backend_aiono, and the legacy
implementation file backend_aiono) import from this module and return
PretrainDataBundle instances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader

PROBE_TASKS = ('all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm')
"""Valid probe task types for PTB-XL dataset."""


@dataclass(frozen=True)
class PretrainDataBundle:
  """Container for all data needed during pretraining.

  This is the unified return type for all data backend implementations.
  It bundles the training iterator with optional probe/evaluation data loaders.

  Attributes:
    train_iterator: Infinite iterator yielding training batches. Yields either
      tensors [B, C, L] for unlabeled data, or (tensor, labels) tuples for labeled.
    labeled: Whether training data includes labels.
    probe_val_loader: DataLoader for online probe validation. None if probing disabled.
    num_classes: Number of classes for online probe. None if probing disabled.
    probe_class_names: List of class names for online probe. None if probing disabled.
    probe_group_to_classes: Mapping from group names to class names for grouped AUC.
      None if probing disabled.
    offline_probe_train_loader: DataLoader for offline probe training. None if disabled.
    offline_probe_val_loader: DataLoader for offline probe validation. None if disabled.
    offline_probe_val_loaders_by_seed: Optional validation loaders keyed by logical
      validation seed value for benchmark-style offline probing.
    offline_probe_num_classes: Number of classes for offline probe. None if disabled.
    offline_probe_class_names: Class names for offline probe. None if disabled.
    offline_probe_group_to_classes: Group-to-class mapping for offline probe. None if disabled.
    offline_dense_probe_target_names: Dense target names for offline regression probe. None if disabled.
    offline_dense_probe_log_per_target: Whether to log per-target dense metrics. None if disabled.
    offline_probe_source_name: Canonical source name for the offline probe bundle.
    offline_probe_benchmark_family: Optional benchmark family identifier.
    offline_probe_benchmark_version: Optional benchmark version identifier.
    offline_probe_benchmark_manifest: Optional resolved benchmark manifest.
    offline_probe_benchmark_comparable: Whether the offline probe bundle is benchmark-comparable.
    offline_probe_validation_seed_values: Logical validation seed values for benchmark probing.
    offline_probe_validation_seed_offset: Generator-seed offset for benchmark probing.
    offline_probe_validation_seed_to_generator_seed: Mapping from logical validation seed
      values to actual generator seeds.
    train_class_positive_counts: Optional per-class positive counts computed on the training split
      for the online probe label space (length `C`). Used for diagnostics like rare-label tracking.
  """
  train_iterator: Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
  labeled: bool
  probe_val_loader: DataLoader | None
  num_classes: int | None
  probe_class_names: list[str] | None
  probe_group_to_classes: dict[str, list[str]] | None
  offline_probe_train_loader: DataLoader | None
  offline_probe_val_loader: DataLoader | None
  offline_probe_num_classes: int | None
  offline_probe_class_names: list[str] | None
  offline_probe_group_to_classes: dict[str, list[str]] | None
  offline_dense_probe_target_names: list[str] | None
  offline_dense_probe_log_per_target: bool | None
  train_class_positive_counts: list[int] | None = None
  offline_probe_val_loaders_by_seed: dict[int, DataLoader] | None = None
  offline_probe_source_name: str | None = None
  offline_probe_benchmark_family: str | None = None
  offline_probe_benchmark_version: str | None = None
  offline_probe_benchmark_manifest: dict[str, object] | None = None
  offline_probe_benchmark_comparable: bool | None = None
  offline_probe_validation_seed_values: list[int] | None = None
  offline_probe_validation_seed_offset: int | None = None
  offline_probe_validation_seed_to_generator_seed: dict[int, int] | None = None


def cycle(dataloader: DataLoader) -> Iterator:
  """Create an infinite iterator that cycles through a DataLoader.

  Args:
    dataloader: The DataLoader to cycle through.

  Yields:
    Batches from the DataLoader, restarting from the beginning when exhausted.
  """
  while True:
    yield from dataloader


def map_to_device(
  *,
  data_iterator: Iterator,
  device: torch.device,
  using_cuda: bool,
  labeled: bool,
) -> Iterator[
  torch.Tensor
  | tuple[torch.Tensor, torch.Tensor]
  | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
  """Map batches from an iterator to a specific device.

  Args:
    data_iterator: Source iterator yielding batches.
    device: Target device for tensors.
    using_cuda: If True, use non_blocking transfers for better performance.
    labeled: If True, expect (x, y) tuples; otherwise expect x tensors only.

  Yields:
    Batches moved to the target device. Format matches input:
    - If labeled: `(x, y)` or `(x, y, extra)` tuples with tensors on device
    - If unlabeled: x tensors on device
  """
  for batch in data_iterator:
    if labeled:
      if not isinstance(batch, (tuple, list)):
        raise ValueError(
          'Labeled batches must be tuples/lists. '
          f'Got {type(batch).__name__}.')
      if len(batch) == 2:
        x, y = batch
        yield x.to(device, non_blocking=using_cuda), y.to(device, non_blocking=using_cuda)
      elif len(batch) == 3:
        x, y, extra = batch
        yield (
          x.to(device, non_blocking=using_cuda),
          y.to(device, non_blocking=using_cuda),
          extra.to(device, non_blocking=using_cuda),
        )
      else:
        raise ValueError(
          'Labeled batches must have length 2 or 3. '
          f'Got len(batch)={len(batch)}.')
    else:
      yield batch.to(device, non_blocking=using_cuda)


def prefetch_batch(
  data_iterator: Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
  """Prefetch one batch ahead to overlap data loading with computation.

  This creates a simple single-batch prefetch buffer. While yielding the current
  batch, the next batch is already loaded in memory.

  Args:
    data_iterator: Source iterator yielding batches.

  Yields:
    Batches from the source iterator, with one batch prefetched ahead.
  """
  prefetched_batch = next(data_iterator)
  for next_batch in data_iterator:
    yield prefetched_batch
    prefetched_batch = next_batch
  yield prefetched_batch
