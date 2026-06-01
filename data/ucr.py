"""UCR Archive 2018 loader for offline evaluation.

This module implements minimal utilities for working with the UCR Archive 2018
time-series classification benchmark (univariate datasets with TRAIN/TEST TSV files).

Expected on-disk layout:
  {ucr_root}/{NAME}/{NAME}_TRAIN.tsv
  {ucr_root}/{NAME}/{NAME}_TEST.tsv

Each TSV row is:
  y \t x_0 \t x_1 \t ... \t x_{L-1}

We return:
  x: float32 tensor with shape [N, 1, L]
  y: int64 tensor with shape [N] (labels mapped to contiguous indices 0..C-1)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from os import path

import numpy as np
import torch
from torch.nn import functional as F

__all__ = [
  'UCRDatasetSplits',
  'ucr_list_datasets',
  'ucr_load_splits',
]


@dataclass(frozen=True)
class UCRDatasetSplits:
  """Loaded UCR dataset splits and metadata."""

  name: str
  x_train: torch.Tensor  # [N_train, 1, L]
  y_train: torch.Tensor  # [N_train]
  x_test: torch.Tensor  # [N_test, 1, L]
  y_test: torch.Tensor  # [N_test]
  num_classes: int
  orig_length: int
  used_length: int


def ucr_list_datasets(*, archive_root: str) -> list[str]:
  """Discover valid UCR dataset folders under the archive root."""
  # Step 1: validate the root path.
  if not isinstance(archive_root, str) or not archive_root:
    raise ValueError('archive_root must be a non-empty string')
  if not path.isdir(archive_root):
    raise ValueError(
      f'UCR archive_root does not exist: {archive_root}. '
      'Fix: set ucr.archive_root to the extracted UCRArchive_2018 directory.')

  # Step 2: list dataset folders that include both TRAIN and TEST TSVs.
  dataset_names: list[str] = []
  for entry in sorted(os.listdir(archive_root)):
    dataset_dir = path.join(archive_root, entry)
    if not path.isdir(dataset_dir):
      continue
    train_path = path.join(dataset_dir, f'{entry}_TRAIN.tsv')
    test_path = path.join(dataset_dir, f'{entry}_TEST.tsv')
    if path.isfile(train_path) and path.isfile(test_path):
      dataset_names.append(entry)

  if not dataset_names:
    raise ValueError(
      f'No UCR datasets found under {archive_root}. '
      'Expected {NAME}/{NAME}_TRAIN.tsv and {NAME}/{NAME}_TEST.tsv for at least one dataset.')
  return dataset_names


def _load_tsv(*, tsv_path: str) -> tuple[np.ndarray, np.ndarray]:
  """Load a UCR TSV file into (x, y_raw)."""
  # Step 1: validate file path.
  if not isinstance(tsv_path, str) or not tsv_path:
    raise ValueError('tsv_path must be a non-empty string')
  if not path.isfile(tsv_path):
    raise ValueError(f'UCR TSV file does not exist: {tsv_path}')

  # Step 2: parse numeric payload.
  raw = np.loadtxt(tsv_path, delimiter='\t', dtype=np.float32)
  if raw.ndim != 2 or raw.shape[1] < 2:
    raise ValueError(
      'UCR TSV must parse to a 2D array with at least 2 columns (label + values). '
      f'Got shape={raw.shape} for {tsv_path}')

  y_raw = raw[:, 0]  # [N]
  x_np = raw[:, 1:]  # [N, L]
  if not np.isfinite(x_np).all():
    raise ValueError(f'UCR TSV contains non-finite values: {tsv_path}')
  return x_np, y_raw


def _map_labels_to_contiguous_indices(
  *,
  y_train_raw: np.ndarray,
  y_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Map raw labels to contiguous indices 0..C-1 using the TRAIN label set."""
  # Step 1: validate inputs.
  if y_train_raw.ndim != 1 or y_test_raw.ndim != 1:
    raise ValueError('Expected 1D y arrays for UCR label mapping')
  if y_train_raw.size < 1 or y_test_raw.size < 1:
    raise ValueError('UCR TRAIN/TEST splits must be non-empty')

  # Step 2: build label vocabulary from TRAIN and validate TEST labels.
  labels = np.unique(y_train_raw.astype(np.float32))
  if labels.size < 1:
    raise ValueError('UCR TRAIN labels must contain at least one class')
  test_unknown = np.setdiff1d(np.unique(y_test_raw.astype(np.float32)), labels)
  if test_unknown.size > 0:
    raise ValueError(
      'UCR TEST split contains labels not present in TRAIN split. '
      f'Unknown labels: {test_unknown.tolist()}')

  # Step 3: vectorized mapping via searchsorted.
  y_train = np.searchsorted(labels, y_train_raw.astype(np.float32))  # [N_train]
  y_test = np.searchsorted(labels, y_test_raw.astype(np.float32))  # [N_test]
  if not np.all(labels[y_train] == y_train_raw.astype(np.float32)):
    raise ValueError('Failed to map UCR TRAIN labels to contiguous indices')
  if not np.all(labels[y_test] == y_test_raw.astype(np.float32)):
    raise ValueError('Failed to map UCR TEST labels to contiguous indices')
  return y_train.astype(np.int64), y_test.astype(np.int64), labels


def _pad_to_multiple(*, x: torch.Tensor, multiple: int) -> torch.Tensor:
  """Right-pad `x` to make the time dimension divisible by `multiple`."""
  # x: [N, 1, L]
  if x.dim() != 3:
    raise ValueError(f'Expected x to be [N, 1, L], got {tuple(x.shape)}')
  if multiple <= 0:
    raise ValueError(f'multiple must be > 0, got {multiple}')
  length = int(x.size(-1))
  remainder = length % multiple
  if remainder == 0:
    return x
  pad_len = multiple - remainder
  return F.pad(x, (0, pad_len))  # [N, 1, L_pad]


def _linear_interpolate_to_length(*, x: torch.Tensor, target_length: int) -> torch.Tensor:
  """Resize a batch of univariate series to `target_length` via linear interpolation."""
  # x: [N, 1, L]
  if x.dim() != 3:
    raise ValueError(f'Expected x to be [N, 1, L], got {tuple(x.shape)}')
  if target_length <= 0:
    raise ValueError(f'target_length must be > 0, got {target_length}')
  return F.interpolate(x, size=int(target_length), mode='linear', align_corners=False)  # [N, 1, target_length]


def ucr_load_splits(
  *,
  archive_root: str,
  name: str,
  resize_mode: str,
  resize_target_length: int,
  patch_size: int,
) -> UCRDatasetSplits:
  """Load one UCR dataset into torch tensors (TRAIN/TEST splits)."""
  # Step 1: validate args and resolve expected paths.
  if not isinstance(name, str) or not name:
    raise ValueError('name must be a non-empty string')
  if resize_mode not in ('interp512', 'none'):
    raise ValueError(
      'resize_mode must be "interp512" or "none". '
      f'Got resize_mode={resize_mode!r}')
  if patch_size <= 0:
    raise ValueError(f'patch_size must be > 0, got {patch_size}')
  if resize_mode == 'interp512' and resize_target_length <= 0:
    raise ValueError(
      'resize_target_length must be > 0 when resize_mode="interp512". '
      f'Got resize_target_length={resize_target_length}')

  # Step 2: resolve dataset files.
  #
  # UCRArchive_2018 includes a folder `Missing_value_and_variable_length_datasets_adjusted/`
  # that provides fixed-length numeric TSVs for datasets that otherwise contain NaNs due to
  # missing values / variable-length padding. When available, we prefer the adjusted split
  # files to keep the benchmark deterministic and avoid ad-hoc NaN handling.
  dataset_dir = path.join(archive_root, name)
  base_train_path = path.join(dataset_dir, f'{name}_TRAIN.tsv')
  base_test_path = path.join(dataset_dir, f'{name}_TEST.tsv')

  adjusted_dir = path.join(
    archive_root,
    'Missing_value_and_variable_length_datasets_adjusted',
    name,
  )
  adjusted_train_path = path.join(adjusted_dir, f'{name}_TRAIN.tsv')
  adjusted_test_path = path.join(adjusted_dir, f'{name}_TEST.tsv')

  train_path = adjusted_train_path if path.isfile(adjusted_train_path) else base_train_path
  test_path = adjusted_test_path if path.isfile(adjusted_test_path) else base_test_path

  if not path.isfile(train_path) or not path.isfile(test_path):
    raise ValueError(
      f'UCR dataset {name!r} is missing TRAIN/TEST TSVs under {dataset_dir}. '
      f'Expected {base_train_path} and {base_test_path}.')

  # Step 3: load raw TSV arrays.
  x_train_np, y_train_raw = _load_tsv(tsv_path=train_path)  # [N_train, L_orig], [N_train]
  x_test_np, y_test_raw = _load_tsv(tsv_path=test_path)  # [N_test, L_orig], [N_test]

  orig_length = int(x_train_np.shape[1])
  if x_test_np.shape[1] != orig_length:
    raise ValueError(
      'UCR TRAIN and TEST lengths must match within a dataset. '
      f'Got train_len={orig_length}, test_len={x_test_np.shape[1]} for dataset={name!r}')

  # Step 3: map labels to contiguous indices.
  y_train_idx, y_test_idx, label_values = _map_labels_to_contiguous_indices(
    y_train_raw=y_train_raw,
    y_test_raw=y_test_raw,
  )

  # Step 4: convert to torch tensors.
  x_train = torch.from_numpy(x_train_np).unsqueeze(1)  # [N_train, 1, L_orig]
  x_test = torch.from_numpy(x_test_np).unsqueeze(1)  # [N_test, 1, L_orig]
  y_train = torch.from_numpy(y_train_idx)  # [N_train]
  y_test = torch.from_numpy(y_test_idx)  # [N_test]

  # Step 5: optional resizing to the encoder input length.
  used_length = orig_length
  if resize_mode == 'interp512':
    x_train = _linear_interpolate_to_length(x=x_train, target_length=resize_target_length)  # [N_train, 1, L_tgt]
    x_test = _linear_interpolate_to_length(x=x_test, target_length=resize_target_length)  # [N_test, 1, L_tgt]
    used_length = int(x_train.size(-1))

  # Step 6: right-pad to avoid dropping the tail in Conv1D patch embedding.
  x_train = _pad_to_multiple(x=x_train, multiple=patch_size)  # [N_train, 1, L_pad]
  x_test = _pad_to_multiple(x=x_test, multiple=patch_size)  # [N_test, 1, L_pad]
  used_length = int(x_train.size(-1))

  return UCRDatasetSplits(
    name=str(name),
    x_train=x_train.to(dtype=torch.float32),
    y_train=y_train.to(dtype=torch.int64),
    x_test=x_test.to(dtype=torch.float32),
    y_test=y_test.to(dtype=torch.int64),
    num_classes=int(label_values.size),
    orig_length=orig_length,
    used_length=used_length,
  )
