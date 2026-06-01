"""CauKer parquet backend for unlabeled pretraining streams.

This backend loads a local CauKer parquet dataset (e.g., CauKer2M) and exposes it
as an infinite iterator of batches shaped like our other backends:

  x: float32 tensor with shape [B, C, L]

The CauKer2M snapshot used in Plan 138 stores `data` as a tensor-like field with
shape [1, 512] per sample and dtype float64; we cast to float32.
"""

from __future__ import annotations

from os import path

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

import configs
from data.backend_base import (
  PretrainDataBundle,
  cycle,
  map_to_device,
  prefetch_batch,
)
from utils.seeding import derive_seed, seed_worker_init_fn, torch_generator


class CauKerParquetDataset(Dataset[torch.Tensor]):
  """Torch Dataset wrapper around a HF parquet dataset split.

  Returns:
    x: float32 tensor with shape [1, L].
  """

  def __init__(
    self,
    *,
    parquet_path: str,
    expected_num_channels: int,
    expected_channel_size: int,
  ) -> None:
    # Step 1: validate dataset contract and file path.
    if not isinstance(parquet_path, str) or not parquet_path:
      raise ValueError('parquet_path must be a non-empty string')
    if not path.isfile(parquet_path):
      raise ValueError(
        f'CauKer parquet file does not exist: {parquet_path}. '
        'Fix: set cauker.parquet_path to a local .parquet file.')
    if expected_num_channels <= 0:
      raise ValueError(
        f'expected_num_channels must be > 0, got {expected_num_channels}')
    if expected_channel_size <= 0:
      raise ValueError(
        f'expected_channel_size must be > 0, got {expected_channel_size}')

    # Step 2: load the parquet split via HF datasets.
    ds = load_dataset(
      'parquet',
      data_files={'train': parquet_path},
      split='train',
    )
    ds = ds.with_format('torch', columns=['data'])

    # Step 3: store state.
    self._ds = ds
    self._expected_num_channels = int(expected_num_channels)
    self._expected_channel_size = int(expected_channel_size)

  def __len__(self) -> int:
    return len(self._ds)

  def __getitem__(self, index: int) -> torch.Tensor:
    # Step 1: fetch the sample payload.
    sample = self._ds[int(index)]
    if not isinstance(sample, dict) or 'data' not in sample:
      raise ValueError(
        'CauKer dataset must yield dict samples with a "data" field. '
        f'Got keys={list(sample) if isinstance(sample, dict) else type(sample).__name__}.')

    x = sample['data']
    if not isinstance(x, torch.Tensor):
      raise ValueError(
        'CauKer "data" field must be a torch.Tensor after with_format("torch"). '
        f'Got type={type(x).__name__}.')

    # Step 2: validate shape and cast to float32.
    # x: [C, L]
    if x.dim() != 2:
      raise ValueError(f'Expected CauKer sample to have shape [C, L], got {tuple(x.shape)}')
    if x.size(0) != self._expected_num_channels or x.size(1) != self._expected_channel_size:
      raise ValueError(
        'Unexpected CauKer sample shape. '
        f'Expected [C, L]=[{self._expected_num_channels}, {self._expected_channel_size}], '
        f'got {tuple(x.shape)}.')

    return x.to(dtype=torch.float32)  # [C, L]


def _resolve_dataloader_base_seed(config: configs.pretrain.Config) -> int | None:
  """Return the base seed used for DataLoader order (legacy `seed` or split `seed_data`)."""
  if config.seed is not None:
    return int(config.seed)
  if config.seed_data is not None:
    return int(config.seed_data)
  return None


def build_cauker_backend(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
) -> PretrainDataBundle:
  """Build an unlabeled CauKer parquet backend.

  Args:
    cfg: Hydra config containing a `cauker.parquet_path` runtime key.
    config: Pretrain config (expects data_backend="cauker").
    device: Target device for training tensors.
    using_cuda: Whether to use pinned memory and non-blocking transfers.

  Returns:
    PretrainDataBundle with an infinite unlabeled training iterator.
  """
  # Step 1: validate config invariants (no silent defaults).
  if config.probe_source is not None:
    raise ValueError(
      'CauKer backend does not provide labels; probe_source must be null. '
      f'Got probe_source={config.probe_source!r}')
  if config.end_truncate != 0:
    raise ValueError(
      'CauKer backend currently requires end_truncate=0 (no truncation). '
      f'Got end_truncate={config.end_truncate}')
  if set(config.datasets) != {'cauker'}:
    raise ValueError(
      'CauKer backend requires datasets={cauker: 1.0}. '
      f'Got datasets={sorted(config.datasets)}')
  if config.num_channels != 1:
    raise ValueError(
      'CauKer2M is univariate; expected num_channels=1. '
      f'Got channels={config.channels}')

  # Step 2: resolve runtime dataset config.
  cauker_cfg_raw = cfg.get('cauker')
  if cauker_cfg_raw is None:
    raise ValueError(
      'Missing required runtime key: cauker.parquet_path for data_backend="cauker". '
      'Fix: set it in dataset/cauker2m.yaml or override via CLI.')
  if not isinstance(cauker_cfg_raw, (dict, DictConfig)):
    raise ValueError(
      'cauker runtime config must be a dict. '
      f'Got type={type(cauker_cfg_raw).__name__}')
  parquet_path = cauker_cfg_raw.get('parquet_path')
  if not isinstance(parquet_path, str) or not parquet_path:
    raise ValueError(
      'cauker.parquet_path must be a non-empty string. '
      f'Got {parquet_path!r}')

  dataset = CauKerParquetDataset(
    parquet_path=str(parquet_path),
    expected_num_channels=config.num_channels,
    expected_channel_size=config.channel_size,
  )

  # Step 3: build an infinite shuffled DataLoader stream.
  dataloader_generator = None
  worker_init_fn = None
  base_seed = _resolve_dataloader_base_seed(config)
  if base_seed is not None:
    loader_seed = derive_seed(seed=base_seed, salt='cauker_train_loader')
    dataloader_generator = torch_generator(loader_seed)
    worker_init_fn = seed_worker_init_fn

  train_loader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=using_cuda,
    num_workers=2,
    generator=dataloader_generator,
    worker_init_fn=worker_init_fn,
  )
  train_iterator = cycle(train_loader)
  train_iterator = map_to_device(
    data_iterator=train_iterator,
    device=device,
    using_cuda=using_cuda,
    labeled=False,
  )
  train_iterator = prefetch_batch(train_iterator)

  return PretrainDataBundle(
    train_iterator=train_iterator,
    labeled=False,
    probe_val_loader=None,
    num_classes=None,
    probe_class_names=None,
    probe_group_to_classes=None,
    offline_probe_train_loader=None,
    offline_probe_val_loader=None,
    offline_probe_num_classes=None,
    offline_probe_class_names=None,
    offline_probe_group_to_classes=None,
    offline_dense_probe_target_names=None,
    offline_dense_probe_log_per_target=None,
  )
