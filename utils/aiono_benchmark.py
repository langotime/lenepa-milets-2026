from __future__ import annotations

"""Shared helpers for Aionoscope benchmark metadata and aggregation."""

from collections.abc import Mapping, Sequence

import numpy as np

AIONO_BASIC_COMPONENTS_V1_COMPONENT_KEYS = (
  'constant',
  'gaussian_noise',
  'uniform_noise',
  'random_walk_noise',
  'linear_trend',
  'quadratic_trend',
  'log_trend',
  'sigmoid_trend',
  'sine',
  'sawtooth',
  'square',
  'spike',
  'level_change',
  'gaussian',
)
"""Canonical class order for `aiono_basic_components/v1`."""


def aiono_stat_summary(values: Sequence[float | int]) -> dict[str, object]:
  """Summarize numeric values with median/std while preserving non-finite entries."""
  if not values:
    raise ValueError('values must be non-empty')
  array = np.asarray([float(value) for value in values], dtype=float)
  finite_mask = np.isfinite(array)
  finite_values = array[finite_mask]
  serialized_values = [
    float(value) if is_finite else None
    for value, is_finite in zip(array.tolist(), finite_mask.tolist(), strict=True)
  ]
  if int(finite_values.size) < 1:
    return {
      'values': serialized_values,
      'median': None,
      'std': None,
      'n': int(array.size),
      'n_finite': 0,
    }
  std = float(np.std(finite_values, ddof=1)) if int(finite_values.size) >= 2 else 0.0
  return {
    'values': serialized_values,
    'median': float(np.median(finite_values)),
    'std': float(std),
    'n': int(array.size),
    'n_finite': int(finite_values.size),
  }


def aiono_flatten_mapping(
  *,
  prefix: str,
  mapping: Mapping[str, object],
) -> dict[str, object]:
  """Flatten a nested mapping into slash-delimited keys."""
  if not prefix:
    raise ValueError('prefix must be non-empty')
  flat: dict[str, object] = {}
  for key, value in mapping.items():
    if not isinstance(key, str) or not key:
      raise ValueError(f'mapping keys must be non-empty strings, got {key!r}')
    flat_key = f'{prefix}/{key}'
    if isinstance(value, Mapping):
      flat.update(aiono_flatten_mapping(prefix=flat_key, mapping=value))
      continue
    flat[flat_key] = value
  return flat


def aiono_contract_logs(
  *,
  source: str,
  manifest: Mapping[str, object],
  actual_sampling_frequency_hz: int | None,
  benchmark_comparable: bool,
) -> dict[str, object]:
  """Build flat W&B-friendly contract metadata logs."""
  if not source:
    raise ValueError('source must be non-empty')
  prefix = f'offline_probe_contract/{source}'
  logs = aiono_flatten_mapping(prefix=prefix, mapping=manifest)
  if actual_sampling_frequency_hz is not None:
    logs[f'{prefix}/actual_sampling_frequency_hz'] = int(actual_sampling_frequency_hz)
  logs[f'{prefix}/benchmark_comparable'] = bool(benchmark_comparable)
  return logs


def aiono_contract_lookup(
  *,
  source: str,
  summary: Mapping[str, object] | None = None,
  config: Mapping[str, object] | None = None,
  field: str,
) -> object:
  """Look up a flattened contract field from W&B summary/config payloads."""
  if not source:
    raise ValueError('source must be non-empty')
  if not field:
    raise ValueError('field must be non-empty')
  key = f'offline_probe_contract/{source}/{field}'
  for payload in (summary, config):
    if payload is None:
      continue
    if key in payload:
      return payload[key]
  return None


def aiono_is_benchmark_comparable(
  *,
  source: str,
  summary: Mapping[str, object] | None = None,
  config: Mapping[str, object] | None = None,
) -> bool:
  """Return whether W&B metadata marks a run as benchmark-comparable."""
  comparable = aiono_contract_lookup(
    source=source,
    summary=summary,
    config=config,
    field='benchmark_comparable',
  )
  if comparable is not True:
    return False
  family = aiono_contract_lookup(
    source=source,
    summary=summary,
    config=config,
    field='benchmark_family',
  )
  version = aiono_contract_lookup(
    source=source,
    summary=summary,
    config=config,
    field='benchmark_version',
  )
  baseline_sampling_frequency_hz = aiono_contract_lookup(
    source=source,
    summary=summary,
    config=config,
    field='baseline_sampling_frequency_hz',
  )
  validation_seed_count = aiono_contract_lookup(
    source=source,
    summary=summary,
    config=config,
    field='validation_seed_count',
  )
  return (
    family == 'aiono_basic_components'
    and version == 'v1'
    and int(baseline_sampling_frequency_hz) == 500
    and validation_seed_count is not None
    and int(validation_seed_count) > 0
  )
