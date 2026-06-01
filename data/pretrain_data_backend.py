"""Pretrain data backend dispatcher.

This module provides the main entry point for loading pretraining data.
It dispatches to the appropriate backend implementation based on config:

- data_backend='dump': Load from pre-dumped .npy/.npz files (see backend_ptbxl)
- data_backend='aiono': Generate synthetic data on-the-fly (see backend_aiono)
- data_backend='cauker': Load a local CauKer parquet dataset (see backend_cauker)

Usage:
  from data.pretrain_data_backend import pretrain_data_build, PretrainDataBundle

  bundle = pretrain_data_build(cfg=cfg, config=config, device=device, ...)
  for batch in bundle.train_iterator:
      ...
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig

import configs
from data.backend_aiono import build_aiono_backend
from data.backend_base import PROBE_TASKS, PretrainDataBundle
from data.backend_cauker import build_cauker_backend
from data.backend_ptbxl import build_dump_unlabeled_backend, build_ptbxl_probe_backend

__all__ = ['PROBE_TASKS', 'PretrainDataBundle', 'pretrain_data_build']


def pretrain_data_build(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
  num_cpus: int,
  strict: bool,
) -> PretrainDataBundle:
  """Build the appropriate data backend based on configuration.

  This is the main entry point for loading pretraining data. It selects
  the appropriate backend based on data_backend and probe_source config.

  Backend selection logic:
  1. If data_backend='aiono': Use Aionoscope synthetic generation
  2. If data_backend='dump' and probe_source='ptb-xl': Use PTB-XL with probes
  3. If data_backend='dump' and probe_source=None: Use unlabeled dump loading

  Args:
    cfg: Hydra config with data paths (e.g., cfg.data, cfg.ptb_xl_data_dir).
    config: Pretrain config with data_backend, probe_source, datasets, etc.
    device: Target device for training tensors.
    using_cuda: Whether to use CUDA optimizations (non-blocking transfers).
    num_cpus: Number of CPU workers for parallel data loading.
    strict: If True, require explicit data_backend config; if False, default to 'dump'.

  Returns:
    PretrainDataBundle containing training iterator and optional probe loaders.

  Raises:
    ValueError: If data_backend is missing (in strict mode) or unsupported,
      or if probe_source is incompatible with data_backend.
  """
  data_backend = config.data_backend_resolved
  if data_backend is None:
    if strict:
      raise ValueError(
        "Missing required config key: 'data_backend'. "
        'Set it in your Hydra dataset config (e.g. dataset/ptbxl.yaml or dataset/aiono_basic_components_balanced.yaml).')
    data_backend = 'dump'

  if data_backend == 'aiono':
    return build_aiono_backend(
      cfg=cfg,
      config=config,
      device=device,
      using_cuda=using_cuda,
    )

  if data_backend == 'cauker':
    return build_cauker_backend(
      cfg=cfg,
      config=config,
      device=device,
      using_cuda=using_cuda,
    )

  if data_backend != 'dump':
    raise ValueError(f'Unsupported data_backend: {data_backend!r}')

  probe_source = config.probe_source_resolved
  if probe_source is None and (not strict) and set(config.datasets) == {'ptb-xl'}:
    probe_source = 'ptb-xl'

  if probe_source == 'ptb-xl':
    return build_ptbxl_probe_backend(
      cfg=cfg,
      config=config,
      device=device,
      using_cuda=using_cuda,
      num_cpus=num_cpus,
    )

  if probe_source is not None:
    raise ValueError(
      f'probe_source={probe_source!r} is not compatible with data_backend="dump". '
      'Use probe_source="ptb-xl" for PTB-XL probing, or null to disable probing.')

  return build_dump_unlabeled_backend(
    cfg=cfg,
    config=config,
    device=device,
    using_cuda=using_cuda,
    num_cpus=num_cpus,
  )
