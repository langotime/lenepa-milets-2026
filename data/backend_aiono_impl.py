"""Aionoscope synthetic time-series data backend for pretraining.

This module contains the concrete implementation behind the canonical
`data.backend_aiono` entrypoint.

It generates synthetic ECG-like time-series data on-the-fly using the upstream
Aionoscope generators. Unlike dump-based backends, the synthetic path
generates infinite data streams without requiring pre-dumped files.

The synthetic data can be used for:
- Debugging and rapid iteration without real data
- Controlled experiments with known ground truth
- Probe evaluation using rhythm/shape class labels (or both)
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

import configs
from data.backend_base import (
  PretrainDataBundle,
  map_to_device,
  prefetch_batch,
)
from data.aiono_dense_targets import (
  aiono_dense_targets_extract,
  aiono_dense_targets_validate_config,
)
from utils.aiono_benchmark import AIONO_BASIC_COMPONENTS_V1_COMPONENT_KEYS


class AionoBatchIterableDataset(
  IterableDataset[
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ]
):
  """Iterable dataset that generates synthetic ECG batches using Aiono pipeline.

  This dataset generates batches on-the-fly using a Aiono SynthPipeline. It supports
  both unlabeled (x only) and labeled (x, y) modes for pretraining with or without
  online probing.

  The dataset is infinite when max_batches is None, making it suitable for training
  iterators that cycle indefinitely.

  Attributes:
    pipeline: Aiono SynthPipeline module for generating synthetic data.
    batch_size: Number of samples per batch.
    device: Device where data is generated (GPU recommended for speed).
    seed: Random seed for reproducible generation.
    max_batches: Maximum batches to generate, or None for infinite.
    view_name: Name of the view to extract from pipeline output.
    num_channels: Expected number of channels (C dimension).
    channel_size: Expected sequence length after truncation (L dimension).
    end_truncate: Number of samples to remove from end of each sequence.
    labeled: Whether to yield (x, y) tuples or x only.
    label_keys: Keys for extracting labels from obs.y (['rhythm'], ['shape'], or ['rhythm', 'shape']).
    num_classes_per_key: Number of classes per label key for one-hot encoding.
    enabled_component_keys: Optional keys for extracting enabled masks from obs.meta['process']['enabled'].
    dense_targets_cfg: Optional dense target config for offline probing.
    dense_target_names: Target names derived from dense_targets_cfg.
  """

  def __init__(
    self,
    *,
    pipeline: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    seed: int,
    max_batches: int | None,
    view_name: str,
    num_channels: int,
    channel_size: int,
    end_truncate: int,
    labeled: bool,
    label_keys: list[str] | None,
    num_classes_per_key: list[int] | None,
    enabled_component_keys: list[str] | None = None,
    dense_targets_cfg: list[dict[str, object]] | None = None,
    dense_target_names: list[str] | None = None,
    emit_oracle_patch_mask: bool = False,
    patch_size: int | None = None,
  ) -> None:
    """Initialize the Aiono batch iterator dataset."""
    # Step 1: validate batch and signal configuration.
    if batch_size <= 0:
      raise ValueError(f'batch_size must be positive, got {batch_size}')
    if seed < 0:
      raise ValueError(f'seed must be non-negative, got {seed}')
    if max_batches is not None and max_batches <= 0:
      raise ValueError(f'max_batches must be positive, got {max_batches}')
    if channel_size <= 0:
      raise ValueError(f'channel_size must be positive, got {channel_size}')
    if num_channels <= 0:
      raise ValueError(f'num_channels must be positive, got {num_channels}')
    if end_truncate < 0:
      raise ValueError(f'end_truncate must be non-negative, got {end_truncate}')
    if not view_name:
      raise ValueError('view_name must be a non-empty string')

    # Step 2: validate label configuration.
    if labeled:
      if enabled_component_keys is None:
        if not label_keys:
          raise ValueError('label_keys is required when labeled=True and enabled_component_keys is null')
        if not num_classes_per_key:
          raise ValueError(
            'num_classes_per_key is required when labeled=True and enabled_component_keys is null')
        if len(label_keys) != len(num_classes_per_key):
          raise ValueError(
            'label_keys/num_classes_per_key length mismatch: '
            f'{len(label_keys)} keys vs {len(num_classes_per_key)} class counts.')
        if len(set(label_keys)) != len(label_keys):
          raise ValueError(f'label_keys must be unique, got {label_keys}')
        for label_key in label_keys:
          if not isinstance(label_key, str) or not label_key.strip():
            raise ValueError(f'label_keys must contain non-empty strings, got {label_key!r}')
        for num_classes in num_classes_per_key:
          if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(
              'num_classes_per_key must contain positive ints, '
              f'got {num_classes_per_key}')
      else:
        if label_keys is not None or num_classes_per_key is not None:
          raise ValueError(
            'label_keys/num_classes_per_key must be null when enabled_component_keys is set')
        if not enabled_component_keys:
          raise ValueError('enabled_component_keys must be non-empty when labeled=True')
        if len(set(enabled_component_keys)) != len(enabled_component_keys):
          raise ValueError(f'enabled_component_keys must be unique, got {enabled_component_keys}')
        for key in enabled_component_keys:
          if not isinstance(key, str) or not key.strip():
            raise ValueError(
              'enabled_component_keys must contain non-empty strings, '
              f'got {key!r}')
    else:
      if label_keys is not None or num_classes_per_key is not None or enabled_component_keys is not None:
        raise ValueError(
          'label_keys/num_classes_per_key/enabled_component_keys must be null when labeled=False')

    # Step 3: validate dense target config.
    if dense_targets_cfg is not None:
      if not labeled:
        raise ValueError('dense_targets_cfg requires labeled=True')
      if dense_target_names is None:
        raise ValueError('dense_target_names is required when dense_targets_cfg is set')
      if not dense_target_names:
        raise ValueError('dense_target_names must be non-empty when dense_targets_cfg is set')
      if len(set(dense_target_names)) != len(dense_target_names):
        raise ValueError(f'dense_target_names must be unique, got {dense_target_names}')
    elif dense_target_names is not None:
      raise ValueError('dense_target_names must be null when dense_targets_cfg is null')
    if emit_oracle_patch_mask:
      if dense_targets_cfg is not None:
        raise ValueError(
          'emit_oracle_patch_mask is only supported for the training iterator '
          '(dense_targets_cfg must be null).')
      if patch_size is None:
        raise ValueError('patch_size is required when emit_oracle_patch_mask=True')
      if patch_size <= 0:
        raise ValueError(f'patch_size must be positive, got {patch_size}')
      if channel_size % patch_size != 0:
        raise ValueError(
          'emit_oracle_patch_mask requires channel_size to be divisible by patch_size. '
          f'Got channel_size={channel_size}, patch_size={patch_size}.')

    # Step 4: store validated configuration.
    self.pipeline = pipeline
    self.batch_size = batch_size
    self.device = device
    self.seed = seed
    self.max_batches = max_batches
    self.view_name = view_name
    self.num_channels = num_channels
    self.channel_size = channel_size
    self.end_truncate = end_truncate
    self.labeled = labeled
    self.label_keys = list(label_keys) if label_keys is not None else None
    self.num_classes_per_key = list(num_classes_per_key) if num_classes_per_key is not None else None
    self.enabled_component_keys = (
      list(enabled_component_keys) if enabled_component_keys is not None else None)
    self.dense_targets_cfg = list(dense_targets_cfg) if dense_targets_cfg is not None else None
    self.dense_target_names = list(dense_target_names) if dense_target_names is not None else None
    self.emit_oracle_patch_mask = bool(emit_oracle_patch_mask)
    self.patch_size = patch_size

  def __iter__(
    self,
  ) -> Iterator[
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ]:
    """Yield batches from the Aiono pipeline."""
    generator = torch.Generator(device=self.device)
    generator.manual_seed(self.seed)

    batch_count = 0
    while self.max_batches is None or batch_count < self.max_batches:
      with torch.no_grad():
        # Step 1: generate a synthetic observation batch.
        views = self.pipeline(self.batch_size, self.device, rng=generator)
        if self.view_name not in views:
          raise ValueError(
            f'Unknown Aiono view_name={self.view_name!r}. Available views are {sorted(views)}')
        obs = views[self.view_name]
        x = obs.x  # [B, C, L]
        # Step 2: validate and crop the signal tensor.
        if x.ndim != 3:
          raise ValueError(f'Aiono observation x must be 3D [B, C, L], got {tuple(x.shape)}')
        if x.dtype != torch.float32:
          raise ValueError(f'Aiono observation x must be float32, got {x.dtype}')
        if int(x.shape[1]) != self.num_channels:
          raise ValueError(
            'Aiono observation channel count mismatch. '
            f'Got C={int(x.shape[1])}, expected num_channels={self.num_channels}. '
            'Fix by setting channels in the pretrain dataset config.')
        if self.end_truncate > 0:
          x = x[..., :-self.end_truncate]  # [B, C, L - end_truncate]
        if int(x.shape[-1]) != self.channel_size:
          raise ValueError(
            'Aiono observation length mismatch after end_truncate. '
            f'Got L={int(x.shape[-1])}, expected channel_size={self.channel_size}. '
            'Fix by setting channel_size/end_truncate in the pretrain dataset config.')

        if self.labeled:
          # Step 3: build multi-hot probe labels.
          if self.enabled_component_keys is None:
            if self.label_keys is None or self.num_classes_per_key is None:
              raise ValueError('label_keys/num_classes_per_key must be set when labeled=True')
            label_batches = []
            for label_key, num_classes in zip(self.label_keys, self.num_classes_per_key):
              label_index = obs.y[label_key]  # [B]
              if label_index.ndim != 1:
                raise ValueError(
                  f'Aiono label {label_key!r} must be 1D [B], got {tuple(label_index.shape)}')
              label_one_hot = F.one_hot(
                label_index.to(torch.int64),
                num_classes=num_classes).float()  # [B, K]
              label_batches.append(label_one_hot)
            if len(label_batches) == 1:
              y = label_batches[0]  # [B, K]
            else:
              y = torch.cat(label_batches, dim=1)  # [B, sum(K)]
          else:
            if not isinstance(getattr(obs, 'meta', None), dict):
              raise ValueError('Aiono observation meta must be a dict.')
            process_meta = obs.meta.get('process')
            if not isinstance(process_meta, dict):
              raise ValueError('Aiono observation meta["process"] must be a dict.')
            enabled = process_meta.get('enabled')
            if not isinstance(enabled, dict):
              raise ValueError('Aiono observation meta["process"]["enabled"] must be a dict.')
            enabled_batches: list[torch.Tensor] = []  # each [B]
            for enabled_key in self.enabled_component_keys:
              mask = enabled.get(enabled_key)
              if not isinstance(mask, torch.Tensor):
                raise ValueError(
                  f'Aiono enabled mask {enabled_key!r} must be a torch.Tensor, '
                  f'got {type(mask).__name__}.')
              if mask.dtype != torch.bool:
                raise ValueError(
                  f'Aiono enabled mask {enabled_key!r} must be bool, got {mask.dtype}.')
              if mask.ndim != 1:
                raise ValueError(
                  f'Aiono enabled mask {enabled_key!r} must be 1D [B], got {tuple(mask.shape)}')
              if int(mask.shape[0]) != int(x.shape[0]):
                raise ValueError(
                  f'Aiono enabled mask {enabled_key!r} batch mismatch. '
                  f'Got B={int(mask.shape[0])}, expected B={int(x.shape[0])}.')
              enabled_batches.append(mask.to(torch.float32))  # [B]
            y = torch.stack(enabled_batches, dim=1)  # [B, K]

          if self.emit_oracle_patch_mask:
            if self.patch_size is None:
              raise ValueError('patch_size must be set when emit_oracle_patch_mask=True')
            num_patches = self.channel_size // int(self.patch_size)
            oracle_mask = torch.zeros(
              (int(x.shape[0]), int(num_patches)),
              dtype=torch.bool,
              device=x.device,
            )  # [B, T_patches]
            if self.enabled_component_keys is not None:
              enabled_non_constant = [
                key for key in self.enabled_component_keys
                if key != 'constant'
              ]
              for batch_index in range(int(x.shape[0])):
                if any(bool(enabled[key][batch_index].item()) for key in enabled_non_constant):
                  oracle_mask[batch_index].fill_(True)
            batch = (x, y, oracle_mask)
          elif self.dense_targets_cfg is not None:
            if self.dense_target_names is None:
              raise ValueError('dense_target_names must be set when dense_targets_cfg is enabled')
            y_dense, _ = aiono_dense_targets_extract(
              obs=obs,
              targets_cfg=self.dense_targets_cfg,
              target_names=self.dense_target_names)  # [B, D]
            batch = (x, y, y_dense)
          else:
            batch = (x, y)
        else:
          batch = x

      # Yield outside no_grad so training autograd stays enabled.
      yield batch

      batch_count += 1


def _build_aiono_label_spec(
  *,
  probe_label_key: str,
  rhythm_classes: list[str],
  shape_classes: list[str],
) -> tuple[list[str], list[int], list[str], dict[str, list[str]]]:
  """Build label keys, class names, and groups for Aiono probe targets."""
  # Step 1: validate label selection and class name uniqueness.
  allowed_keys = ('rhythm', 'shape', 'both')
  if probe_label_key not in allowed_keys:
    raise ValueError(
      f'aiono.probe_label_key must be one of {allowed_keys}, got {probe_label_key!r}')
  if probe_label_key == 'both':
    overlap = sorted(set(rhythm_classes) & set(shape_classes))
    if overlap:
      raise ValueError(
        'Aiono rhythm/shape class names must be unique across label groups. '
        f'Duplicates: {overlap}')

  # Step 2: assemble label keys and class names in a fixed order.
  if probe_label_key == 'rhythm':
    label_keys = ['rhythm']
    class_names = list(rhythm_classes)
    num_classes_per_key = [len(class_names)]
    group_to_classes = {'all': list(class_names)}
  elif probe_label_key == 'shape':
    label_keys = ['shape']
    class_names = list(shape_classes)
    num_classes_per_key = [len(class_names)]
    group_to_classes = {'all': list(class_names)}
  else:
    label_keys = ['rhythm', 'shape']
    class_names = list(rhythm_classes) + list(shape_classes)
    num_classes_per_key = [len(rhythm_classes), len(shape_classes)]
    group_to_classes = {
      'rhythm': list(rhythm_classes),
      'shape': list(shape_classes),
      'all': list(class_names),
    }

  # Step 3: validate class counts for probe targets.
  if any(num_classes <= 0 for num_classes in num_classes_per_key):
    raise ValueError(
      f'Aiono num_classes must be positive, got {num_classes_per_key}')

  return label_keys, num_classes_per_key, class_names, group_to_classes


def _build_aiono_backend(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
) -> PretrainDataBundle:
  """Build a synthetic Aionoscope/Aiono-compatible backend.

  Creates an Aionoscope pipeline (selected by `aiono.kind`) and generates data on-the-fly
  on the target device, avoiding I/O bottlenecks from disk-based loading.

  Supported synthetic configs:
  - kind="pqrst_ecg": pulse-train ECG synthesis (current default)
  - kind="basic_components": Aionoscope basic-components pipeline

  Args:
    cfg: Hydra config. Must NOT contain 'data' key (the synthetic backend doesn't use dump files).
    config: Pretrain config with aiono settings, channel_size, num_channels, etc.
    device: Target device for data generation (GPU recommended).
    using_cuda: Whether to use CUDA optimizations.

  Returns:
    PretrainDataBundle with synthetic training iterator and labeled probe streams.
    When offline probing is enabled, includes offline probe train/val loaders.

  Raises:
    ValueError: If data config is provided, datasets is not {'aiono'},
      aiono config is missing, offline probe batches are missing,
      or Aionoscope cannot be imported.
  """
  # Step 1: validate backend configuration and probe source.
  if cfg.get('data') not in (None, {}):
    raise ValueError(
      'data must be null for the synthetic Aionoscope backend. '
      'Remove any +data.* overrides when using data_backend="aiono".')
  if set(config.datasets) != {'aiono'}:
    raise ValueError(
      'Synthetic Aionoscope backend requires datasets: {aiono: 1.0} '
      f'Got datasets={sorted(config.datasets)}')

  if config.synthetic_config is None:
    raise ValueError('aiono config is required when data_backend="aiono"')

  probe_source = config.probe_source_resolved
  if probe_source != 'aiono':
    raise ValueError(
      'Synthetic Aionoscope backend requires probe_source="aiono" '
      f'Got probe_source={probe_source!r}.')

  # Step 2: import the shared sampler API and parse the pipeline configuration.
  try:
    from aiono import (
      ConstantSampler,
      LogUniformSampler,
      NormalSampler,
      RandIntSampler,
      Sampler,
      UniformSampler,
      WeightedPermutationSampler,
    )
  except Exception as exc:
    raise ValueError(
      'Failed to import Aionoscope (`aiono`). '
      'Ensure the environment is synced from pyproject.toml / uv.lock. '
      f'Original error: {exc!r}') from exc

  aiono_cfg = config.synthetic_config

  def _parse_sampler_spec(
    spec: dict[str, object],
    *,
    name: str,
    expected: str,
  ) -> Sampler:
    """Parse a sampler spec dict into a Aiono Sampler instance."""
    # Step 1: validate the sampler kind field.
    kind = spec.get('kind')
    if not isinstance(kind, str) or not kind.strip():
      raise ValueError(f'{name} sampler spec requires a non-empty "kind" field.')
    kind = kind.strip()

    # Step 2: instantiate the sampler with validated parameters.
    if kind == 'constant':
      value = spec.get('value')
      if isinstance(value, bool) or value is None:
        raise ValueError(f'{name} constant sampler requires a numeric "value".')
      if expected == 'int':
        if not isinstance(value, int):
          raise ValueError(f'{name} constant sampler requires an int value, got {value!r}')
        return ConstantSampler(value=value)
      if not isinstance(value, (int, float)):
        raise ValueError(f'{name} constant sampler requires a float value, got {value!r}')
      return ConstantSampler(value=float(value))

    if kind == 'uniform':
      if expected != 'float':
        raise ValueError(f'{name} uniform sampler is only valid for float parameters.')
      low = spec.get('low')
      high = spec.get('high')
      if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f'{name} uniform sampler requires numeric low/high, got {spec!r}')
      return UniformSampler(low=float(low), high=float(high))

    if kind == 'log_uniform':
      if expected != 'float':
        raise ValueError(f'{name} log_uniform sampler is only valid for float parameters.')
      low = spec.get('low')
      high = spec.get('high')
      if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f'{name} log_uniform sampler requires numeric low/high, got {spec!r}')
      return LogUniformSampler(low=float(low), high=float(high))

    if kind == 'normal':
      if expected != 'float':
        raise ValueError(f'{name} normal sampler is only valid for float parameters.')
      mean = spec.get('mean')
      std = spec.get('std')
      if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
        raise ValueError(f'{name} normal sampler requires numeric mean/std, got {spec!r}')
      clamp = spec.get('clamp')
      clamp_tuple = None
      if clamp is not None:
        if (not isinstance(clamp, (list, tuple))) or len(clamp) != 2:
          raise ValueError(f'{name} normal sampler clamp must be a 2-item list/tuple.')
        clamp_tuple = (float(clamp[0]), float(clamp[1]))
      return NormalSampler(mean=float(mean), std=float(std), clamp=clamp_tuple)

    if kind == 'randint':
      if expected != 'int':
        raise ValueError(f'{name} randint sampler is only valid for int parameters.')
      low = spec.get('low')
      high = spec.get('high')
      if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError(f'{name} randint sampler requires int low/high, got {spec!r}')
      return RandIntSampler(low=int(low), high=int(high))

    raise ValueError(f'{name} has unsupported sampler kind {kind!r}.')

  def _require_sampler_value(
    mapping: dict[str, object],
    key: str,
    *,
    context: str,
    expected: str,
  ) -> Sampler | float | int:
    """Extract a scalar or sampler spec from config for Aiono."""
    # Step 1: validate key presence.
    if key not in mapping:
      raise ValueError(f'{context}.{key} is required for data_backend="aiono".')
    value = mapping.get(key)

    # Step 2: accept sampler instances directly.
    if isinstance(value, Sampler):
      return value

    # Step 3: parse sampler specs.
    if isinstance(value, dict):
      return _parse_sampler_spec(value, name=f'{context}.{key}', expected=expected)

    # Step 4: parse scalar values based on expected type.
    if isinstance(value, bool):
      raise ValueError(f'{context}.{key} must be a {expected}, got bool.')
    if expected == 'float':
      if not isinstance(value, (int, float)):
        raise ValueError(
          f'{context}.{key} must be a float or sampler spec, got {type(value).__name__}')
      return float(value)
    if expected == 'int':
      if not isinstance(value, int):
        raise ValueError(
          f'{context}.{key} must be an int or sampler spec, got {type(value).__name__}')
      return int(value)
    raise ValueError(f'Unsupported expected type {expected!r} for {context}.{key}')

  def _require_optional_float(
    mapping: dict[str, object],
    key: str,
    *,
    context: str,
  ) -> float | None:
    """Extract an optional float value from config."""
    # Step 1: allow missing keys.
    if key not in mapping:
      return None
    value = mapping.get(key)
    # Step 2: validate the provided value.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(
        f'{context}.{key} must be a float, got {type(value).__name__}: {value!r}')
    return float(value)

  def _spacing_from_frequency(
    *,
    frequency_hz: float,
    seq_len: int,
    sample_rate_hz: float,
    context: str,
  ) -> float:
    """Compute the kernel spacing in samples from a fixed frequency."""
    # Step 1: validate frequency.
    if frequency_hz <= 0:
      raise ValueError(f'{context} must be positive, got {frequency_hz}.')
    # Step 2: compute a rounded pulse count to match PulseTrainProcess.
    duration_sec = (seq_len - 1) / sample_rate_hz
    expected_pulses = frequency_hz * duration_sec
    num_pulses = int(math.floor(expected_pulses + 0.5))
    if num_pulses <= 0:
      raise ValueError(
        f'{context} yields fewer than 1 pulse for seq_len={seq_len}. '
        f'Got duration_sec={duration_sec:.6f}, expected_pulses={expected_pulses:.6f}.')
    # Step 3: convert to spacing in samples.
    return (seq_len - 1) / (num_pulses + 1)

  def _resolve_validation_seed_config(
    *,
    mapping: dict[str, object],
    train_seed_value: int,
  ) -> tuple[list[int], int, list[int], dict[int, int]]:
    """Resolve logical validation seeds and their generator seeds."""
    # Step 1: accept either the benchmark protocol or a single validation seed.
    if 'validation_seed_values' in mapping:
      raw_values = mapping.get('validation_seed_values')
      if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(
          'aiono.validation_seed_values must be a non-empty list[int]. '
          f'Got {raw_values!r}.')
      validation_seed_values: list[int] = []
      for raw_value in raw_values:
        if not isinstance(raw_value, int):
          raise ValueError(
            'aiono.validation_seed_values must contain only ints. '
            f'Got {raw_values!r}.')
        validation_seed_values.append(int(raw_value))
      if len(set(validation_seed_values)) != len(validation_seed_values):
        raise ValueError(
          'aiono.validation_seed_values must be unique. '
          f'Got {validation_seed_values}.')
      validation_seed_offset = _require_int(mapping, 'validation_seed_offset')
    else:
      validation_seed_values = [_require_int(mapping, 'val_seed')]
      validation_seed_offset = int(mapping.get('validation_seed_offset', 0))

    # Step 2: derive actual generator seeds used to build each validation split.
    generator_seeds = [
      int(validation_seed_offset) + int(seed_value)
      for seed_value in validation_seed_values
    ]
    if int(train_seed_value) in generator_seeds:
      raise ValueError(
        'Train seed overlaps with at least one validation generator seed. '
        'Fix: set validation_seed_offset (for example 100) to separate the ranges. '
        f'train_seed={int(train_seed_value)} generator_seeds={generator_seeds}')
    seed_mapping = {
      int(seed_value): int(generator_seed)
      for seed_value, generator_seed in zip(validation_seed_values, generator_seeds, strict=True)
    }
    return validation_seed_values, int(validation_seed_offset), generator_seeds, seed_mapping

  # Step 3: parse the synthetic config and build the requested pipeline.
  kind = _require_str(aiono_cfg, 'kind')
  allowed_kinds = ('pqrst_ecg', 'basic_components')
  if kind not in allowed_kinds:
    raise ValueError(f'aiono.kind must be one of {allowed_kinds}, got {kind!r}')

  train_seed = _require_int(aiono_cfg, 'train_seed')
  (
    validation_seed_values,
    validation_seed_offset,
    validation_generator_seeds,
    validation_seed_to_generator_seed,
  ) = _resolve_validation_seed_config(mapping=aiono_cfg, train_seed_value=train_seed)
  val_seed = int(validation_generator_seeds[0])
  val_batches = _require_int(aiono_cfg, 'val_batches')
  view_name = _require_str(aiono_cfg, 'view_name')

  seq_len = config.channel_size + config.end_truncate
  if seq_len <= 0:
    raise ValueError(
      f'Invalid Aiono seq_len derived from channel_size/end_truncate: {seq_len}. '
      f'channel_size={config.channel_size}, end_truncate={config.end_truncate}')

  sample_rate_hz = float(config.sampling_frequency)
  if sample_rate_hz <= 0:
    raise ValueError(
      'Aiono requires a positive sampling_frequency. '
      f'Got sampling_frequency={config.sampling_frequency}')

  labeled = True
  label_keys = None
  num_classes_per_key = None
  enabled_component_keys = None
  num_classes = None
  probe_class_names = None
  probe_group_to_classes = None

  dense_probe_cfg = aiono_cfg.get('dense_probe')
  dense_probe_enabled = False
  dense_targets_cfg = None
  dense_target_names = None
  dense_probe_log_per_target = None
  benchmark_family = None
  benchmark_version = None
  benchmark_manifest = None
  benchmark_comparable = False

  if kind == 'pqrst_ecg':
    try:
      from aiono import (
        BaselineWanderView,
        ECGLeadsView,
        EventImpulseView,
        GaussianNoiseView,
        KernelConvView,
        NormalizeView,
        PulseTrainProcess,
        SynthPipeline,
        ViewChain,
        make_pqrst_kernel_bank,
        pqrst_kernel_size,
      )
      from aiono.core.utils import utils_make_canonical_A0
    except Exception as exc:
      raise ValueError(
        'Failed to import the Aionoscope PQRST API. '
        'Ensure the environment contains the upstream `aiono` package. '
        f'Original error: {exc!r}') from exc

    probe_label_key = _require_str(aiono_cfg, 'probe_label_key')
    process_cfg = _require_dict(aiono_cfg, 'process')
    view_cfg = _require_dict(aiono_cfg, 'view')

    rhythm_classes = _require_str_list(process_cfg, 'rhythm_classes')
    shape_classes = _require_str_list(process_cfg, 'shape_classes')
    amplitude = _require_sampler_value(
      process_cfg,
      'amplitude',
      context='aiono.process',
      expected='float')
    missed_gap_factor = _require_sampler_value(
      process_cfg,
      'missed_gap_factor',
      context='aiono.process',
      expected='float')

    jitter_std = _require_sampler_value(
      view_cfg,
      'jitter_std',
      context='aiono.view',
      expected='float')
    max_delay = _require_sampler_value(
      view_cfg,
      'max_delay',
      context='aiono.view',
      expected='int')
    baseline_cfg = _require_dict(view_cfg, 'baseline_wander')
    baseline_amplitude_std = _require_sampler_value(
      baseline_cfg,
      'amplitude_std',
      context='aiono.view.baseline_wander',
      expected='float')
    baseline_freq_min = _require_sampler_value(
      baseline_cfg,
      'freq_min',
      context='aiono.view.baseline_wander',
      expected='float')
    baseline_freq_max = _require_sampler_value(
      baseline_cfg,
      'freq_max',
      context='aiono.view.baseline_wander',
      expected='float')
    noise_std = _require_sampler_value(
      view_cfg,
      'noise_std',
      context='aiono.view',
      expected='float')

    frequency_hz = _require_sampler_value(
      process_cfg,
      'frequency_hz',
      context='aiono.process',
      expected='float')
    kernel_frequency_hz = _require_optional_float(
      process_cfg,
      'kernel_frequency_hz',
      context='aiono.process')

    def _resolve_kernel_spacing_samples() -> float:
      """Resolve kernel spacing for the PQRST kernel bank."""
      # Step 1: prefer explicit kernel frequency overrides.
      if kernel_frequency_hz is not None:
        return _spacing_from_frequency(
          frequency_hz=kernel_frequency_hz,
          seq_len=seq_len,
          sample_rate_hz=sample_rate_hz,
          context='aiono.process.kernel_frequency_hz')

      # Step 2: fall back to fixed process frequency when available.
      if isinstance(frequency_hz, ConstantSampler):
        return _spacing_from_frequency(
          frequency_hz=float(frequency_hz.value),
          seq_len=seq_len,
          sample_rate_hz=sample_rate_hz,
          context='aiono.process.frequency_hz')
      if isinstance(frequency_hz, (int, float)):
        return _spacing_from_frequency(
          frequency_hz=float(frequency_hz),
          seq_len=seq_len,
          sample_rate_hz=sample_rate_hz,
          context='aiono.process.frequency_hz')

      # Step 3: require explicit kernel frequency when sampling.
      raise ValueError(
        'aiono.process.frequency_hz uses a sampler; '
        'set aiono.process.kernel_frequency_hz to a fixed value to build kernels.')

    kernel_spacing_samples = _resolve_kernel_spacing_samples()

    def _build_pipeline() -> torch.nn.Module:
      """Build a Aiono pipeline for ECG observations."""
      # Step 1: configure the pulse train process.
      process = PulseTrainProcess(
        seq_len=seq_len,
        frequency_hz=frequency_hz,
        sample_rate_hz=sample_rate_hz,
        rhythm_classes=rhythm_classes,
        shape_classes=shape_classes,
        latent_mode='pqrst3',
        amplitude=amplitude,
        missed_gap_factor=missed_gap_factor,
      )

      # Step 2: build PQRST kernels for event rendering.
      spacing = kernel_spacing_samples
      kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
      kernels = make_pqrst_kernel_bank(
        shape_names=process.shape_classes,
        spacing=spacing,
        kernel_size=kernel_size,
        device=device,
      )  # [K, T, W]
      padding = kernel_size // 2

      # Step 3: build the view chain for ECG observations.
      num_latent = int(kernels.shape[0])
      A0 = utils_make_canonical_A0(num_leads=config.num_channels, num_latent=num_latent)  # [C, K]
      views = {
        view_name: torch.nn.Sequential(
          EventImpulseView(seq_len=process.seq_len, amplitude_param='amplitude', rounding='nearest'),
          KernelConvView(kernels=kernels, padding=padding),
          ECGLeadsView(A0=A0, jitter_std=jitter_std, max_delay=max_delay),
          BaselineWanderView(
            amplitude_std=baseline_amplitude_std,
            freq_min=baseline_freq_min,
            freq_max=baseline_freq_max),
          GaussianNoiseView(noise_std=noise_std),
          NormalizeView(),
        )
      }
      return SynthPipeline(process, views).to(device)

    label_keys, num_classes_per_key, probe_class_names, probe_group_to_classes = (
      _build_aiono_label_spec(
        probe_label_key=probe_label_key,
        rhythm_classes=rhythm_classes,
        shape_classes=shape_classes,
      )
    )
    num_classes = sum(num_classes_per_key)

    if dense_probe_cfg is not None:
      if not isinstance(dense_probe_cfg, dict):
        raise ValueError(
          f'aiono.dense_probe must be a dict, got {type(dense_probe_cfg).__name__}')
      dense_probe_enabled = _require_bool(dense_probe_cfg, 'enabled')
      if dense_probe_enabled:
        if config.offline_probe_eval_interval is None:
          raise ValueError(
            'aiono.dense_probe.enabled requires offline_probe_eval_interval to be set.')
        dense_targets_cfg = dense_probe_cfg.get('targets')
        if not isinstance(dense_targets_cfg, list):
          raise ValueError(
            f'aiono.dense_probe.targets must be a list, got {type(dense_targets_cfg).__name__}')
        dense_target_names = aiono_dense_targets_validate_config(
          targets_cfg=dense_targets_cfg)
        dense_probe_log_per_target = _require_bool(dense_probe_cfg, 'log_per_target')
  else:
    try:
      from aiono import (
        AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
        AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY,
        AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION,
        AionoBasicComponentsPeriodicConfig,
        ConstantLatentNode,
        EnableComponentsNode,
        EventRenderView,
        EventSchema,
        GaussianNoiseView,
        GateEventsByEnabledNode,
        LinearTrendView,
        LogTrendView,
        PiecewiseLinearTrendView,
        ProcessGraph,
        QuadraticTrendView,
        RandomWalkNoiseView,
        SawtoothWaveView,
        SigmoidTrendView,
        SineWaveView,
        SingleEventNode,
        SquareWaveView,
        SynthPipeline,
        UnionEventsNode,
        UniformNoiseView,
        ViewChain,
        resolve_aiono_basic_components_periodic_contract,
      )
    except Exception as exc:
      raise ValueError(
        'Failed to import the public Aionoscope basic-components API. '
        'Ensure the environment contains the upstream `aiono` package. '
        f'Original error: {exc!r}') from exc

    forbidden_keys = ('probe_label_key', 'process', 'view')
    present_forbidden = [key for key in forbidden_keys if key in aiono_cfg]
    if present_forbidden:
      raise ValueError(
        f'aiono.kind="basic_components" does not use {present_forbidden}; remove them from the config.')

    if config.num_channels != 1:
      raise ValueError(
        'aiono.kind="basic_components" produces 1-channel signals. '
        f'Set channels to a single entry (e.g. channels=[I]). Got num_channels={config.num_channels}.')

    basic_cfg = _require_dict(aiono_cfg, 'basic_components')
    benchmark_family_raw = aiono_cfg.get('benchmark_family')
    benchmark_version_raw = aiono_cfg.get('benchmark_version')
    benchmark_configured = benchmark_family_raw is not None or benchmark_version_raw is not None
    periodic_contract = None
    if benchmark_configured:
      benchmark_family = _require_str(aiono_cfg, 'benchmark_family')
      benchmark_version = _require_str(aiono_cfg, 'benchmark_version')
      if benchmark_family != AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY:
        raise ValueError(
          'Aionoscope basic-components requires benchmark_family='
          f'{AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY!r}, got {benchmark_family!r}.')
      if benchmark_version != AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION:
        raise ValueError(
          'Aionoscope basic-components requires benchmark_version='
          f'{AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION!r}, got {benchmark_version!r}.')
      if int(config.sampling_frequency) != AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ:
        raise ValueError(
          'Aionoscope basic-components v1 requires sampling_frequency='
          f'{AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ} Hz, '
          f'got sampling_frequency={int(config.sampling_frequency)} Hz.')
    else:
      if basic_cfg.get('periodic') is not None:
        raise ValueError(
          'aiono.basic_components.periodic requires explicit benchmark_family and benchmark_version. '
          'Use dataset=aiono_basic_components_balanced or dataset=aiono_basic_components_imbalanced.')
      raise ValueError(
        'aiono.kind="basic_components" requires explicit benchmark_family and benchmark_version. '
        'Use dataset=aiono_basic_components_balanced or dataset=aiono_basic_components_imbalanced.')

    raw_component_keys = basic_cfg.get('component_keys')
    if not isinstance(raw_component_keys, list) or not raw_component_keys:
      raise ValueError(
        f'aiono.basic_components.component_keys must be a non-empty list[str], got {raw_component_keys!r}')
    if not all(isinstance(item, str) and item.strip() for item in raw_component_keys):
      raise ValueError(
        'aiono.basic_components.component_keys must contain only non-empty strings, '
        f'got {raw_component_keys!r}')
    component_keys = [str(item) for item in raw_component_keys]
    if len(set(component_keys)) != len(component_keys):
      raise ValueError(
        'aiono.basic_components.component_keys must be unique. '
        f'Got {component_keys}')
    if 'constant' not in component_keys:
      raise ValueError(
        'aiono.basic_components.component_keys must include "constant" '
        '(required to define the latent baseline).')
    if benchmark_configured and tuple(component_keys) != AIONO_BASIC_COMPONENTS_V1_COMPONENT_KEYS:
      raise ValueError(
        'Benchmark-labeled Aionoscope basic-components runs must use the canonical '
        f'component order {list(AIONO_BASIC_COMPONENTS_V1_COMPONENT_KEYS)}. '
        f'Got {component_keys}.')

    supported_component_keys = {
      'constant',
      'gaussian_noise',
      'uniform_noise',
      'random_walk_noise',
      'linear_trend',
      'quadratic_trend',
      'log_trend',
      'sigmoid_trend',
      'piecewise_linear_trend',
      'sine',
      'sawtooth',
      'square',
      'spike',
      'level_change',
      'gaussian',
    }
    unsupported = sorted(set(component_keys) - supported_component_keys)
    if unsupported:
      raise ValueError(
        'aiono.basic_components.component_keys contains unsupported keys. '
        f'Unsupported={unsupported}, supported={sorted(supported_component_keys)}.')
    if len(component_keys) < 2:
      raise ValueError(
        'aiono.basic_components.component_keys must contain at least 2 entries '
        '(e.g. constant + one enabled component).')

    num_enabled = basic_cfg.get('num_enabled')
    if not isinstance(num_enabled, int):
      raise ValueError(
        'aiono.basic_components.num_enabled must be an int. '
        f'Got {type(num_enabled).__name__}: {num_enabled!r}')
    if num_enabled < 1 or num_enabled > len(component_keys):
      raise ValueError(
        'aiono.basic_components.num_enabled must satisfy 1 <= num_enabled <= len(component_keys). '
        f'Got num_enabled={num_enabled}, len(component_keys)={len(component_keys)}.')

    if benchmark_configured:
      periodic_cfg = _require_dict(basic_cfg, 'periodic')
      periodic_contract = resolve_aiono_basic_components_periodic_contract(
        seq_len=seq_len,
        sampling_frequency_hz=int(config.sampling_frequency),
        config=AionoBasicComponentsPeriodicConfig.from_mapping(periodic_cfg),
        benchmark_family=benchmark_family,
        benchmark_version=benchmark_version,
      )
      benchmark_manifest = {
        **periodic_contract.manifest_fields(),
        'sampling_frequency': int(config.sampling_frequency),
        'view_name': view_name,
        'component_keys': list(component_keys),
        'num_enabled': int(num_enabled),
        'validation_seed_values': [int(seed_value) for seed_value in validation_seed_values],
        'validation_seed_offset': int(validation_seed_offset),
        'validation_generator_seeds': [int(seed_value) for seed_value in validation_generator_seeds],
        'validation_seed_to_generator_seed': {
          str(seed_value): int(generator_seed)
          for seed_value, generator_seed in validation_seed_to_generator_seed.items()
        },
        'validation_seed_count': int(len(validation_seed_values)),
      }
      benchmark_comparable = True

    component_order_sampler = None
    raw_component_weights_by_key = basic_cfg.get('component_weights_by_key')
    if raw_component_weights_by_key is not None:
      # Step 1: validate weight mapping keys and values.
      if not isinstance(raw_component_weights_by_key, dict):
        raise ValueError(
          'aiono.basic_components.component_weights_by_key must be a dict[str, float], '
          f'got {type(raw_component_weights_by_key).__name__}: {raw_component_weights_by_key!r}')
      component_weights_by_key: dict[str, float] = {}
      for raw_key, raw_value in raw_component_weights_by_key.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
          raise ValueError(
            'aiono.basic_components.component_weights_by_key keys must be non-empty strings. '
            f'Got {raw_key!r}')
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
          raise ValueError(
            'aiono.basic_components.component_weights_by_key values must be numeric. '
            f'Got {raw_key!r}: {raw_value!r}')
        component_weights_by_key[str(raw_key)] = float(raw_value)

      # Step 2: require exact key coverage, aligned to component_keys.
      if set(component_weights_by_key.keys()) != set(component_keys):
        missing = sorted(set(component_keys) - set(component_weights_by_key.keys()))
        extra = sorted(set(component_weights_by_key.keys()) - set(component_keys))
        raise ValueError(
          'aiono.basic_components.component_weights_by_key keys must match '
          'aiono.basic_components.component_keys exactly. '
          f'missing={missing}, extra={extra}.')

      probs = [float(component_weights_by_key[key]) for key in component_keys]
      if any((not math.isfinite(prob)) or (prob < 0.0) for prob in probs):
        raise ValueError(
          'aiono.basic_components.component_weights_by_key values must be finite and non-negative. '
          f'Got probs={probs}.')
      if sum(probs) <= 0.0:
        raise ValueError(
          'aiono.basic_components.component_weights_by_key must have a positive sum. '
          f'Got probs={probs}.')

      num_positive = sum(prob > 0.0 for prob in probs)
      if int(num_enabled) > num_positive:
        raise ValueError(
          'aiono.basic_components.num_enabled cannot exceed the number of positive-weight classes '
          'when aiono.basic_components.component_weights_by_key is set. '
          f'Got num_enabled={num_enabled}, num_positive_weights={num_positive}.')

      # Step 3: build a weighted per-sample component order sampler (Aiono example 08).
      component_order_sampler = WeightedPermutationSampler(probs=probs)

    def _build_pipeline() -> torch.nn.Module:
      """Build the benchmark-aligned Aionoscope basic-components pipeline."""
      # Step 1: define the event schema (matches the upstream Aionoscope benchmark builder).
      schema = EventSchema(
        type_names=['spike', 'level_change', 'gaussian'],
        param_names=['amplitude', 'sigma_sec'],
        time_unit='samples',
      )

      event_time_min = int(seq_len * 0.15)
      event_time_max = int(seq_len * 0.85)

      # Step 2: configure the process graph with enabled component masks.
      process_nodes = [
        EnableComponentsNode(
          component_keys=component_keys,
          num_enabled=int(num_enabled),
          component_order=component_order_sampler),
        ConstantLatentNode(
          seq_len=seq_len,
          channels=1,
          value=UniformSampler(-1.0, 1.0),
          enabled_key='constant',
          out_key='latent',
        ),
      ]

      event_in_keys: list[str] = []
      if 'spike' in component_keys:
        process_nodes.extend([
          SingleEventNode(
            seq_len=seq_len,
            schema=schema,
            type_name='spike',
            time_min=event_time_min,
            time_max=event_time_max,
            amplitude=UniformSampler(0.8, 1.2),
            amplitude_param='amplitude',
            out_key='spike',
          ),
          GateEventsByEnabledNode(in_key='spike', enabled_key='spike', out_key='spike.gated'),
        ])
        event_in_keys.append('spike.gated')

      if 'level_change' in component_keys:
        process_nodes.extend([
          SingleEventNode(
            seq_len=seq_len,
            schema=schema,
            type_name='level_change',
            time_min=event_time_min,
            time_max=event_time_max,
            amplitude=UniformSampler(-1.0, 1.0),
            amplitude_param='amplitude',
            out_key='level_change',
          ),
          GateEventsByEnabledNode(
            in_key='level_change',
            enabled_key='level_change',
            out_key='level_change.gated',
          ),
        ])
        event_in_keys.append('level_change.gated')

      if 'gaussian' in component_keys:
        process_nodes.extend([
          SingleEventNode(
            seq_len=seq_len,
            schema=schema,
            type_name='gaussian',
            time_min=event_time_min,
            time_max=event_time_max,
            amplitude=UniformSampler(-1.0, 1.0),
            amplitude_param='amplitude',
            extra_params={'sigma_sec': UniformSampler(0.01, 0.06)},
            out_key='gaussian',
          ),
          GateEventsByEnabledNode(
            in_key='gaussian',
            enabled_key='gaussian',
            out_key='gaussian.gated',
          ),
        ])
        event_in_keys.append('gaussian.gated')

      outputs = {'latent'}
      if event_in_keys:
        process_nodes.append(
          UnionEventsNode(
            in_keys=list(event_in_keys),
            out_key='events',
          )
        )
        outputs.add('events')

      base_meta = {
        'seq_len': seq_len,
        'sample_rate_hz': sample_rate_hz,
        'component_keys': list(component_keys),
        'num_enabled': int(num_enabled),
      }
      if component_order_sampler is not None:
        base_meta['component_order_spec'] = component_order_sampler.spec()

      process = ProcessGraph(
        name='BasicComponentsProcess',
        outputs=outputs,
        base_meta=base_meta,
        graph=process_nodes,
      )

      # Step 3: configure the view chain with the shared periodic contract.
      views: list[torch.nn.Module] = []
      if event_in_keys:
        views.append(
          EventRenderView(
            seq_len=seq_len,
            amplitude_param='amplitude',
            rounding='nearest',
            sigma_sec_param='sigma_sec',
          )
        )
      if 'gaussian_noise' in component_keys:
        views.append(GaussianNoiseView(noise_std=UniformSampler(0.02, 0.15), enabled_key='gaussian_noise'))
      if 'uniform_noise' in component_keys:
        views.append(UniformNoiseView(amplitude=UniformSampler(0.05, 0.25), enabled_key='uniform_noise'))
      if 'random_walk_noise' in component_keys:
        views.append(RandomWalkNoiseView(step_std=UniformSampler(0.01, 0.08), enabled_key='random_walk_noise'))
      if 'linear_trend' in component_keys:
        views.append(
          LinearTrendView(
            seq_len=seq_len,
            slope=UniformSampler(-2.0, 2.0),
            intercept=UniformSampler(-0.5, 0.5),
            enabled_key='linear_trend',
          )
        )
      if 'quadratic_trend' in component_keys:
        views.append(
          QuadraticTrendView(
            seq_len=seq_len,
            a=UniformSampler(-4.0, 4.0),
            b=UniformSampler(-2.0, 2.0),
            c=UniformSampler(-0.5, 0.5),
            enabled_key='quadratic_trend',
          )
        )
      if 'log_trend' in component_keys:
        views.append(
          LogTrendView(
            seq_len=seq_len,
            amplitude=UniformSampler(-2.0, 2.0),
            offset=UniformSampler(-0.5, 0.5),
            epsilon=1e-3,
            enabled_key='log_trend',
          )
        )
      if 'sigmoid_trend' in component_keys:
        views.append(
          SigmoidTrendView(
            seq_len=seq_len,
            amplitude=UniformSampler(-2.0, 2.0),
            center=UniformSampler(0.2, 0.8),
            sharpness=UniformSampler(5.0, 20.0),
            offset=UniformSampler(-0.5, 0.5),
            enabled_key='sigmoid_trend',
          )
        )
      if 'piecewise_linear_trend' in component_keys:
        views.append(
          PiecewiseLinearTrendView(
            seq_len=seq_len,
            slope1=UniformSampler(-2.0, 2.0),
            slope2=UniformSampler(-2.0, 2.0),
            change_t=UniformSampler(0.2, 0.8),
            intercept=UniformSampler(-0.5, 0.5),
            enabled_key='piecewise_linear_trend',
          )
        )
      # Step 3a: benchmark-labeled runs consume the shared upstream resolver; local
      # Aiono presets keep the previous periodic ranges and are marked
      # non-comparable in metadata.
      if 'sine' in component_keys:
        if periodic_contract is None:
          views.append(
            SineWaveView(
              seq_len=seq_len,
              amplitude=UniformSampler(0.2, 1.2),
              frequency_hz=UniformSampler(0.2, 6.0),
              phase=UniformSampler(0.0, 2.0 * math.pi),
              offset=UniformSampler(-0.2, 0.2),
              enabled_key='sine',
            )
          )
        else:
          views.append(
            SineWaveView(
              seq_len=seq_len,
              **periodic_contract.signal('sine').view_kwargs(),
              enabled_key='sine',
            )
          )
      if 'sawtooth' in component_keys:
        if periodic_contract is None:
          views.append(
            SawtoothWaveView(
              seq_len=seq_len,
              amplitude=UniformSampler(0.2, 1.2),
              frequency_hz=UniformSampler(0.2, 6.0),
              phase=UniformSampler(0.0, 2.0 * math.pi),
              offset=UniformSampler(-0.2, 0.2),
              enabled_key='sawtooth',
            )
          )
        else:
          views.append(
            SawtoothWaveView(
              seq_len=seq_len,
              **periodic_contract.signal('sawtooth').view_kwargs(),
              enabled_key='sawtooth',
            )
          )
      if 'square' in component_keys:
        if periodic_contract is None:
          views.append(
            SquareWaveView(
              seq_len=seq_len,
              amplitude=UniformSampler(0.2, 1.2),
              frequency_hz=UniformSampler(0.2, 6.0),
              phase=UniformSampler(0.0, 2.0 * math.pi),
              offset=UniformSampler(-0.2, 0.2),
              duty_cycle=UniformSampler(0.1, 0.9),
              enabled_key='square',
            )
          )
        else:
          views.append(
            SquareWaveView(
              seq_len=seq_len,
              **periodic_contract.signal('square').view_kwargs(),
              enabled_key='square',
            )
          )

      if not views:
        raise ValueError(
          'aiono.basic_components.component_keys must include at least one non-constant component '
          'to produce an observation view.')
      view = ViewChain(*views)

      return SynthPipeline(process=process, views={view_name: view}).to(device)

    enabled_component_keys = list(component_keys)
    num_classes = len(component_keys)
    probe_class_names = list(component_keys)
    group_specs = {
      'noise': {
        'gaussian_noise',
        'uniform_noise',
        'random_walk_noise',
      },
      'trend': {
        'linear_trend',
        'quadratic_trend',
        'log_trend',
        'sigmoid_trend',
        'piecewise_linear_trend',
      },
      'periodic': {
        'sine',
        'sawtooth',
        'square',
      },
      'events': {
        'spike',
        'level_change',
        'gaussian',
      },
    }
    probe_group_to_classes = {}
    for group_name, group_keys in group_specs.items():
      group_classes = [key for key in component_keys if key in group_keys]
      if group_classes:
        probe_group_to_classes[group_name] = group_classes
    probe_group_to_classes['all'] = list(component_keys)

    if dense_probe_cfg is not None:
      if not isinstance(dense_probe_cfg, dict):
        raise ValueError(
          f'aiono.dense_probe must be a dict, got {type(dense_probe_cfg).__name__}')
      dense_probe_enabled = _require_bool(dense_probe_cfg, 'enabled')
      if dense_probe_enabled:
        if config.offline_probe_eval_interval is None:
          raise ValueError(
            'aiono.dense_probe.enabled requires offline_probe_eval_interval to be set.')
        dense_targets_cfg = dense_probe_cfg.get('targets')
        if not isinstance(dense_targets_cfg, list):
          raise ValueError(
            f'aiono.dense_probe.targets must be a list, got {type(dense_targets_cfg).__name__}')
        dense_target_names = aiono_dense_targets_validate_config(
          targets_cfg=dense_targets_cfg)
        dense_probe_log_per_target = _require_bool(dense_probe_cfg, 'log_per_target')

  train_pipeline = _build_pipeline()
  emit_oracle_patch_mask = (
    kind == 'basic_components'
    and config.nepa_sigreg_time_saliency == 'aiono_oracle_mask'
  )

  # Step 5: build iterators/loaders for training, validation, and offline probe.
  train_dataset = AionoBatchIterableDataset(
    pipeline=train_pipeline,
    batch_size=config.batch_size,
    device=device,
    seed=train_seed,
    max_batches=None,
    view_name=view_name,
    num_channels=config.num_channels,
    channel_size=config.channel_size,
    end_truncate=config.end_truncate,
    labeled=labeled,
    label_keys=label_keys,
    num_classes_per_key=num_classes_per_key,
    enabled_component_keys=enabled_component_keys,
    dense_targets_cfg=None,
    dense_target_names=None,
    emit_oracle_patch_mask=emit_oracle_patch_mask,
    patch_size=int(config.patch_size),
  )
  train_iterator = iter(train_dataset)
  train_iterator = map_to_device(
    data_iterator=train_iterator,
    device=device,
    using_cuda=using_cuda,
    labeled=labeled,
  )
  train_iterator = prefetch_batch(train_iterator)

  probe_val_loader = None
  if labeled:
    val_dataset = AionoBatchIterableDataset(
      pipeline=train_pipeline,
      batch_size=config.batch_size,
      device=device,
      seed=val_seed,
      max_batches=val_batches,
      view_name=view_name,
      num_channels=config.num_channels,
      channel_size=config.channel_size,
      end_truncate=config.end_truncate,
      labeled=True,
      label_keys=label_keys,
      num_classes_per_key=num_classes_per_key,
      enabled_component_keys=enabled_component_keys,
    )
    probe_val_loader = DataLoader(
      dataset=val_dataset,
      batch_size=None,
      shuffle=False,
      pin_memory=False,
      num_workers=0)

  offline_probe_train_loader = None
  offline_probe_val_loader = None
  offline_probe_val_loaders_by_seed = None
  offline_probe_num_classes = None
  offline_probe_class_names = None
  offline_probe_group_to_classes = None
  offline_dense_probe_target_names = None
  offline_dense_probe_log_per_target = None

  if (config.offline_probe_eval_interval is not None
      and 'aiono' in config.offline_probe_sources_resolved):
    if not labeled:
      raise ValueError('Aionoscope offline probe requires probe_source="aiono".')
    offline_train_batches = _require_int(aiono_cfg, 'offline_probe_train_batches')
    offline_val_batches = _require_int(aiono_cfg, 'offline_probe_val_batches')

    # Use a separate pipeline so offline probing does not consume the training RNG stream.
    offline_pipeline = _build_pipeline()

    offline_train_dataset = AionoBatchIterableDataset(
      pipeline=offline_pipeline,
      batch_size=config.offline_probe_batch_size,
      device=device,
      seed=train_seed,
      max_batches=offline_train_batches,
      view_name=view_name,
      num_channels=config.num_channels,
      channel_size=config.channel_size,
      end_truncate=config.end_truncate,
      labeled=True,
      label_keys=label_keys,
      num_classes_per_key=num_classes_per_key,
      enabled_component_keys=enabled_component_keys,
      dense_targets_cfg=dense_targets_cfg if dense_probe_enabled else None,
      dense_target_names=dense_target_names if dense_probe_enabled else None,
    )
    offline_probe_train_loader = DataLoader(
      dataset=offline_train_dataset,
      batch_size=None,
      shuffle=False,
      pin_memory=False,
      num_workers=0)

    offline_probe_val_loaders_by_seed = {}
    for seed_value, generator_seed in validation_seed_to_generator_seed.items():
      offline_val_dataset = AionoBatchIterableDataset(
        pipeline=offline_pipeline,
        batch_size=config.offline_probe_batch_size,
        device=device,
        seed=int(generator_seed),
        max_batches=offline_val_batches,
        view_name=view_name,
        num_channels=config.num_channels,
        channel_size=config.channel_size,
        end_truncate=config.end_truncate,
        labeled=True,
        label_keys=label_keys,
        num_classes_per_key=num_classes_per_key,
        enabled_component_keys=enabled_component_keys,
        dense_targets_cfg=dense_targets_cfg if dense_probe_enabled else None,
        dense_target_names=dense_target_names if dense_probe_enabled else None,
      )
      offline_probe_val_loaders_by_seed[int(seed_value)] = DataLoader(
        dataset=offline_val_dataset,
        batch_size=None,
        shuffle=False,
        pin_memory=False,
        num_workers=0)
    offline_probe_val_loader = offline_probe_val_loaders_by_seed[int(validation_seed_values[0])]

    offline_probe_num_classes = num_classes
    offline_probe_class_names = probe_class_names
    offline_probe_group_to_classes = probe_group_to_classes
    offline_dense_probe_target_names = dense_target_names if dense_probe_enabled else None
    offline_dense_probe_log_per_target = (
      dense_probe_log_per_target if dense_probe_enabled else None)

  return PretrainDataBundle(
    train_iterator=train_iterator,
    labeled=labeled,
    probe_val_loader=probe_val_loader,
    num_classes=num_classes,
    probe_class_names=probe_class_names,
    probe_group_to_classes=probe_group_to_classes,
    offline_probe_train_loader=offline_probe_train_loader,
    offline_probe_val_loader=offline_probe_val_loader,
    offline_probe_num_classes=offline_probe_num_classes,
    offline_probe_class_names=offline_probe_class_names,
    offline_probe_group_to_classes=offline_probe_group_to_classes,
    offline_dense_probe_target_names=offline_dense_probe_target_names,
    offline_dense_probe_log_per_target=offline_dense_probe_log_per_target,
    offline_probe_val_loaders_by_seed=offline_probe_val_loaders_by_seed,
    offline_probe_source_name='aiono',
    offline_probe_benchmark_family=benchmark_family,
    offline_probe_benchmark_version=benchmark_version,
    offline_probe_benchmark_manifest=benchmark_manifest,
    offline_probe_benchmark_comparable=benchmark_comparable,
    offline_probe_validation_seed_values=list(validation_seed_values),
    offline_probe_validation_seed_offset=int(validation_seed_offset),
    offline_probe_validation_seed_to_generator_seed=dict(validation_seed_to_generator_seed),
  )


def build_aiono_backend(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
) -> PretrainDataBundle:
  """Build the canonical Aionoscope backend."""
  return _build_aiono_backend(
    cfg=cfg,
    config=config,
    device=device,
    using_cuda=using_cuda,
  )


def _require_int(mapping: dict[str, object], key: str) -> int:
  """Extract and validate an int value from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, int):
    raise ValueError(f'aiono.{key} must be an int, got {type(value).__name__}: {value!r}')
  return value


def _require_float(mapping: dict[str, object], key: str) -> float:
  """Extract and validate a float value from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, (int, float)):
    raise ValueError(f'aiono.{key} must be a float, got {type(value).__name__}: {value!r}')
  return float(value)


def _require_str(mapping: dict[str, object], key: str) -> str:
  """Extract and validate a non-empty string value from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'aiono.{key} must be a non-empty string, got {value!r}')
  return value


def _require_dict(mapping: dict[str, object], key: str) -> dict[str, object]:
  """Extract and validate a dict value from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, dict):
    raise ValueError(f'aiono.{key} must be a dict, got {type(value).__name__}: {value!r}')
  return value


def _require_str_list(mapping: dict[str, object], key: str) -> list[str]:
  """Extract and validate a non-empty list of strings from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, list) or not value:
    raise ValueError(f'aiono.{key} must be a non-empty list[str], got {value!r}')
  if not all(isinstance(item, str) and item.strip() for item in value):
    raise ValueError(f'aiono.{key} must contain only non-empty strings, got {value!r}')
  return [str(item) for item in value]


def _require_bool(mapping: dict[str, object], key: str) -> bool:
  """Extract and validate a bool value from aiono config."""
  value = mapping.get(key)
  if not isinstance(value, bool):
    raise ValueError(f'aiono.{key} must be a bool, got {type(value).__name__}: {value!r}')
  return value
