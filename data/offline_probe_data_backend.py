"""Offline probe data backend dispatcher.

This module decouples offline probing datasets from the training dataset/backend.

It builds *labeled* train/val dataloaders used by `utils.offline_probe` to evaluate
frozen encoder representations on a dataset that can differ from the pretraining
dataset (e.g., train on Aionoscope and probe on PTB-XL, or vice versa).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from os import path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import configs
from configs.hydra_utils import hydra_get_data_mapping
from data.backend_aiono import build_aiono_backend
from data.backend_ptbxl import build_ptbxl_probe_backend

__all__ = ['OfflineProbeDataBundle', 'offline_probe_data_build']


@dataclass(frozen=True)
class OfflineProbeDataBundle:
  """Container for offline probe datasets and metadata."""

  train_loader: DataLoader
  val_loader: DataLoader
  num_classes: int
  class_names: list[str]
  group_to_classes: dict[str, list[str]]
  dense_target_names: list[str] | None
  dense_log_per_target: bool | None
  val_loaders_by_seed: dict[int, DataLoader] | None = None
  source_name: str | None = None
  benchmark_family: str | None = None
  benchmark_version: str | None = None
  benchmark_manifest: dict[str, object] | None = None
  benchmark_comparable: bool | None = None
  validation_seed_values: list[int] | None = None
  validation_seed_offset: int | None = None
  validation_seed_to_generator_seed: dict[int, int] | None = None


def offline_probe_data_build(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
  num_cpus: int,
  offline_probe_source: str | None = None,
) -> OfflineProbeDataBundle:
  """Build the offline probe dataset independent from the training dataset.

  Args:
    cfg: Hydra config holding runtime keys (e.g., `ptb_xl_data_dir`,
      `offline_probe_data`).
    config: Pretrain config with offline probe settings.
    device: Device for dataset generation/preprocessing (Aionoscope synthetic generation uses this device).
    using_cuda: Whether to enable CUDA pinning in dataloaders where applicable.
    num_cpus: Number of CPU workers used for preprocessing dump datasets.
    offline_probe_source: Optional explicit dataset source override.

  Returns:
    OfflineProbeDataBundle with train/val loaders and label metadata.

  Raises:
    ValueError: If offline probe dataset settings are missing or incompatible.
  """
  # Step 1: validate the requested offline probe dataset.
  if offline_probe_source is None:
    offline_probe_source = config.offline_probe_source
  if offline_probe_source is None:
    raise ValueError(
      'offline_probe_source is required when offline probing is enabled. '
      'Set it via Hydra (e.g., offline_probe_dataset=ptbxl or offline_probe_dataset=aiono_basic_components_balanced).')
  if offline_probe_source not in ('ptb-xl', 'aiono'):
    raise ValueError(
      'offline_probe_source must be "ptb-xl" or "aiono". '
      f'Got offline_probe_source={offline_probe_source!r}')

  # Step 2: dispatch to the requested dataset backend.
  if offline_probe_source == 'ptb-xl':
    return _offline_probe_data_build_ptbxl(
      cfg=cfg,
      config=config,
      device=device,
      using_cuda=using_cuda,
      num_cpus=num_cpus,
    )
  return _offline_probe_data_build_aiono(
    cfg=cfg,
    config=config,
    device=device,
    using_cuda=using_cuda,
  )


def _offline_probe_data_build_ptbxl(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
  num_cpus: int,
) -> OfflineProbeDataBundle:
  """Build PTB-XL offline probe loaders from `offline_probe_data` dumps."""
  # Step 1: validate runtime keys.
  offline_probe_data_cfg = cfg.get('offline_probe_data')
  if not offline_probe_data_cfg:
    raise ValueError(
      "Missing required runtime key: 'offline_probe_data' for offline_probe_source='ptb-xl'. "
      "Provide via CLI: '+offline_probe_data.ptb-xl=/path/to/ptbxl_dump.npy'")

  dump_files = hydra_get_data_mapping(offline_probe_data_cfg)
  if 'ptb-xl' not in dump_files:
    raise ValueError(
      "Missing ptb-xl dataset in 'offline_probe_data'. "
      "Provide via CLI: '+offline_probe_data.ptb-xl=/path/to/ptbxl_dump.npy'")

  dump_file = dump_files['ptb-xl']
  if not path.isfile(dump_file):
    raise ValueError(f'Offline probe dataset does not exist: {dump_file}')

  ptb_xl_data_dir = cfg.get('ptb_xl_data_dir')
  if not ptb_xl_data_dir:
    raise ValueError(
      "Missing required runtime key: 'ptb_xl_data_dir' for offline_probe_source='ptb-xl'. "
      'Provide via CLI: +ptb_xl_data_dir=/path/to/ptbxl')

  # Step 2: adapt config/cfg to reuse the existing PTB-XL backend implementation.
  offline_task = config.offline_probe_task
  if offline_task is None:
    raise ValueError('offline_probe_task is required when offline_probe_source="ptb-xl".')

  ptbxl_config = dataclasses.replace(
    config,
    data_backend='dump',
    probe_source='ptb-xl',
    aiono=None,
    datasets={'ptb-xl': 1.0},
    probe_task=offline_task,
    offline_probe_source='ptb-xl',
    offline_probe_sources=None,
    offline_probe_aiono=None,
  )
  ptbxl_cfg = OmegaConf.create({
    'ptb_xl_data_dir': ptb_xl_data_dir,
    'data': {'ptb-xl': dump_file},
  })

  data_bundle = build_ptbxl_probe_backend(
    cfg=ptbxl_cfg,
    config=ptbxl_config,
    device=device,
    using_cuda=using_cuda,
    num_cpus=num_cpus,
  )

  train_loader = data_bundle.offline_probe_train_loader
  val_loader = data_bundle.offline_probe_val_loader
  num_classes = data_bundle.offline_probe_num_classes
  class_names = data_bundle.offline_probe_class_names
  group_to_classes = data_bundle.offline_probe_group_to_classes
  if (train_loader is None
      or val_loader is None
      or num_classes is None
      or class_names is None
      or group_to_classes is None):
    raise ValueError('PTB-XL offline probe backend failed to initialize loaders.')

  return OfflineProbeDataBundle(
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=int(num_classes),
    class_names=list(class_names),
    group_to_classes=dict(group_to_classes),
    dense_target_names=None,
    dense_log_per_target=None,
    source_name='ptb-xl',
  )


def _offline_probe_data_build_aiono(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
) -> OfflineProbeDataBundle:
  """Build Aionoscope offline probe loaders from `offline_probe_aiono` config."""
  # Step 1: validate config blocks.
  offline_probe_aiono = config.offline_probe_synthetic_config
  if offline_probe_aiono is None:
    raise ValueError(
      'offline_probe_aiono config is required when offline_probe_source="aiono". '
      'Set it via offline_probe_dataset=aiono_basic_components_balanced/'
      'aiono_basic_components_imbalanced, or provide '
      'offline_probe_aiono.{...} overrides.')

  # Step 2: adapt config/cfg to reuse the existing synthetic backend implementation.
  aiono_config = dataclasses.replace(
    config,
    data_backend='aiono',
    probe_source='aiono',
    datasets={'aiono': 1.0},
    aiono=offline_probe_aiono,
    offline_probe_source='aiono',
    offline_probe_sources=None,
    offline_probe_aiono=offline_probe_aiono,
  )
  aiono_cfg = OmegaConf.create({'data': None})

  data_bundle = build_aiono_backend(
    cfg=aiono_cfg,
    config=aiono_config,
    device=device,
    using_cuda=using_cuda,
  )

  train_loader = data_bundle.offline_probe_train_loader
  val_loader = data_bundle.offline_probe_val_loader
  num_classes = data_bundle.offline_probe_num_classes
  class_names = data_bundle.offline_probe_class_names
  group_to_classes = data_bundle.offline_probe_group_to_classes
  dense_target_names = data_bundle.offline_dense_probe_target_names
  dense_log_per_target = data_bundle.offline_dense_probe_log_per_target
  if (train_loader is None
      or val_loader is None
      or num_classes is None
      or class_names is None
      or group_to_classes is None):
    raise ValueError('Aionoscope offline probe backend failed to initialize loaders.')

  return OfflineProbeDataBundle(
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=int(num_classes),
    class_names=list(class_names),
    group_to_classes=dict(group_to_classes),
    dense_target_names=list(dense_target_names) if dense_target_names is not None else None,
    dense_log_per_target=dense_log_per_target,
    val_loaders_by_seed=data_bundle.offline_probe_val_loaders_by_seed,
    source_name=data_bundle.offline_probe_source_name,
    benchmark_family=data_bundle.offline_probe_benchmark_family,
    benchmark_version=data_bundle.offline_probe_benchmark_version,
    benchmark_manifest=data_bundle.offline_probe_benchmark_manifest,
    benchmark_comparable=data_bundle.offline_probe_benchmark_comparable,
    validation_seed_values=(
      list(data_bundle.offline_probe_validation_seed_values)
      if data_bundle.offline_probe_validation_seed_values is not None else None),
    validation_seed_offset=data_bundle.offline_probe_validation_seed_offset,
    validation_seed_to_generator_seed=(
      dict(data_bundle.offline_probe_validation_seed_to_generator_seed)
      if data_bundle.offline_probe_validation_seed_to_generator_seed is not None else None),
  )
