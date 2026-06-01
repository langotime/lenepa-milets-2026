from __future__ import annotations

from typing import Any

import torch


def aiono_dense_targets_validate_config(
  *, targets_cfg: list[dict[str, object]]
) -> list[str]:
  """Validate dense target config entries and return ordered target names.

  Args:
    targets_cfg: List of dense target config dicts. Each entry must include
      a unique "name", a "source" of "process" or "view", and the locator
      fields required by that source. Optional fields:
        - "enabled_key": When set, the extracted value is considered defined
          only for samples where obs.meta["process"]["enabled"][enabled_key] is
          true; disabled samples are emitted as NaN.
        - "normalize": Optional postprocessing for numeric targets. The only
          supported value is "unit_interval", which divides by (seq_len - 1)
          where seq_len is derived from obs.x.shape[-1] at extraction time.

  Returns:
    Ordered list of target names matching the config order.
  """
  # Step 1: ensure the config is a non-empty list of dicts.
  if not targets_cfg:
    raise ValueError('aiono.dense_probe.targets must be a non-empty list.')
  for entry in targets_cfg:
    if not isinstance(entry, dict):
      raise ValueError(
        'aiono.dense_probe.targets entries must be dicts. '
        f'Got {type(entry).__name__}: {entry!r}')

  # Step 2: validate entries and collect target names.
  target_names: list[str] = []
  for index, entry in enumerate(targets_cfg):
    name = entry.get('name')
    if not isinstance(name, str) or not name.strip():
      raise ValueError(
        'aiono.dense_probe.targets entries must include a non-empty "name". '
        f'Entry[{index}]={entry!r}')
    source = entry.get('source')
    if source not in ('process', 'view'):
      raise ValueError(
        'aiono.dense_probe.targets entries must include "source" '
        'set to "process" or "view". '
        f'Entry[{index}]={entry!r}')
    if source == 'process':
      if not isinstance(entry.get('node'), str) or not str(entry.get('node')).strip():
        raise ValueError(
          'Process dense targets require "node" (non-empty string). '
          f'Entry[{index}]={entry!r}')
    if source == 'view':
      if not isinstance(entry.get('view'), str) or not str(entry.get('view')).strip():
        raise ValueError(
          'View dense targets require "view" (non-empty string). '
          f'Entry[{index}]={entry!r}')
    if not isinstance(entry.get('param'), str) or not str(entry.get('param')).strip():
      raise ValueError(
        'Dense targets require "param" (non-empty string). '
        f'Entry[{index}]={entry!r}')
    enabled_key = entry.get('enabled_key')
    if enabled_key is not None and (
      not isinstance(enabled_key, str) or not enabled_key.strip()
    ):
      raise ValueError(
        'Dense targets "enabled_key" must be a non-empty string when provided. '
        f'Entry[{index}]={entry!r}')
    normalize = entry.get('normalize')
    if normalize is not None:
      if normalize not in ('unit_interval',):
        raise ValueError(
          'Dense targets "normalize" must be one of ["unit_interval"] when provided. '
          f'Entry[{index}]={entry!r}')
    target_names.append(str(name))

  # Step 3: enforce name uniqueness.
  if len(set(target_names)) != len(target_names):
    raise ValueError(
      'aiono.dense_probe.targets names must be unique. '
      f'Got {target_names}')

  return target_names


def aiono_dense_targets_extract(
  *,
  obs: Any,
  targets_cfg: list[dict[str, object]],
  target_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
  """Extract dense probe targets from a Aiono observation.

  This function expects Aiono to store sampled process parameters under
  obs.meta["process"]["samples"][node][param] and view parameters under
  obs.view_meta(<ViewName>)["samples"][param]. Only per-sample scalars are
  supported (shapes [B] or [B, 1]).

  Args:
    obs: Aiono Observation object with .meta and .view_meta().
    targets_cfg: Dense target config list validated by
      aiono_dense_targets_validate_config.
    target_names: Optional prevalidated target name list matching targets_cfg.

  Returns:
    (y_dense, target_names) where y_dense is float32 [B, D_dense]. Targets
    configured with "enabled_key" are emitted as NaN for disabled samples.
  """
  # Step 1: validate config and collect names.
  if target_names is None:
    target_names = aiono_dense_targets_validate_config(targets_cfg=targets_cfg)
  else:
    if not target_names:
      raise ValueError('target_names must be non-empty when provided.')
    if len(target_names) != len(targets_cfg):
      raise ValueError(
        'target_names length must match targets_cfg length. '
        f'Got {len(target_names)} vs {len(targets_cfg)}.')
    if len(set(target_names)) != len(target_names):
      raise ValueError(f'target_names must be unique, got {target_names}')

  # Step 2: validate observation payload and collect batch size.
  x = getattr(obs, 'x', None)
  if not isinstance(x, torch.Tensor):
    raise ValueError(f'Aiono observation x must be a torch.Tensor, got {type(x).__name__}.')
  if x.ndim != 3:
    raise ValueError(f'Aiono observation x must be 3D [B, C, L], got {tuple(x.shape)}')
  batch_size = int(x.shape[0])
  seq_len = int(x.shape[-1])
  device = x.device

  # Step 3: locate process/view samples and extract per-target tensors.
  if not isinstance(getattr(obs, 'meta', None), dict):
    raise ValueError('Aiono observation meta must be a dict.')
  process_meta = obs.meta.get('process')
  if not isinstance(process_meta, dict):
    raise ValueError('Aiono observation meta["process"] must be a dict.')
  process_samples = process_meta.get('samples')
  if not isinstance(process_samples, dict):
    raise ValueError(
      'Aiono observation meta["process"]["samples"] must be a dict. '
      'Ensure Aiono exports process samples.')

  target_tensors: list[torch.Tensor] = []  # each [B]

  for entry in targets_cfg:
    source = entry['source']
    param = str(entry['param'])
    enabled_key = entry.get('enabled_key')
    enabled_mask = None
    if enabled_key is not None:
      enabled = process_meta.get('enabled')
      if not isinstance(enabled, dict):
        raise ValueError(
          'Dense targets with enabled_key require obs.meta["process"]["enabled"] to be a dict. '
          f'Got {type(enabled).__name__}.')
      mask = enabled.get(str(enabled_key))
      if not isinstance(mask, torch.Tensor):
        raise ValueError(
          f'Dense target enabled mask {enabled_key!r} must be a torch.Tensor, '
          f'got {type(mask).__name__}.')
      if mask.dtype != torch.bool:
        raise ValueError(
          f'Dense target enabled mask {enabled_key!r} must be bool, got {mask.dtype}.')
      if mask.ndim != 1:
        raise ValueError(
          f'Dense target enabled mask {enabled_key!r} must be 1D [B], got {tuple(mask.shape)}')
      if int(mask.shape[0]) != batch_size:
        raise ValueError(
          f'Dense target enabled mask {enabled_key!r} batch mismatch. '
          f'Got B={int(mask.shape[0])}, expected B={batch_size}.')
      enabled_mask = mask.to(device=device)
    allow_missing_value = enabled_mask is not None and not bool(torch.any(enabled_mask).item())

    if source == 'process':
      node = str(entry['node'])
      node_samples = process_samples.get(node)
      if not isinstance(node_samples, dict):
        if allow_missing_value:
          value = None
          target = torch.full((batch_size,), float('nan'), dtype=torch.float32, device=device)  # [B]
          target_tensors.append(target)
          continue
        available_nodes = sorted(process_samples)
        raise ValueError(
          f'Missing process samples for node {node!r}. '
          f'Available nodes: {available_nodes}')
      value = node_samples.get(param)
      if value is None:
        if allow_missing_value:
          target = torch.full((batch_size,), float('nan'), dtype=torch.float32, device=device)  # [B]
          target_tensors.append(target)
          continue
        available_params = sorted(node_samples)
        raise ValueError(
          f'Missing process sample param {param!r} in node {node!r}. '
          f'Available params: {available_params}')
    else:
      view_name = str(entry['view'])
      try:
        view_meta = obs.view_meta(view_name)
      except Exception as exc:
        view_names = _aiono_dense_targets_list_view_names(obs.meta)
        raise ValueError(
          f'Failed to resolve view {view_name!r} for dense target {param!r}. '
          f'Available views: {view_names}') from exc
      view_samples = view_meta.get('samples')
      if not isinstance(view_samples, dict):
        if allow_missing_value:
          value = None
        else:
          raise ValueError(
            f'View meta for {view_name!r} is missing "samples". '
            'Ensure Aiono exports view samples.')
      else:
        value = view_samples.get(param)
        if value is None:
          if allow_missing_value:
            value = None
          else:
            available_params = sorted(view_samples)
            raise ValueError(
              f'Missing view sample param {param!r} in view {view_name!r}. '
              f'Available params: {available_params}')

    if value is None:
      target = torch.full((batch_size,), float('nan'), dtype=torch.float32, device=device)  # [B]
      target_tensors.append(target)
      continue

    if not isinstance(value, torch.Tensor):
      raise ValueError(
        f'Dense target {param!r} must be a torch.Tensor, got {type(value).__name__}.')
    # value: [B] or [B, 1] expected
    numeric_int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if not (torch.is_floating_point(value) or value.dtype in numeric_int_dtypes):
      raise ValueError(
        f'Dense target {param!r} must be numeric, got dtype {value.dtype}.')

    # Step 4: validate shape and coerce to float32 [B].
    if value.ndim == 1:
      target = value  # [B]
    elif value.ndim == 2 and int(value.shape[1]) == 1:
      target = value.squeeze(1)  # [B]
    else:
      raise ValueError(
        f'Dense target {param!r} must be shape [B] or [B, 1], got {tuple(value.shape)}.')
    if int(target.shape[0]) != batch_size:
      raise ValueError(
        'Dense target batch size mismatch. '
        f'Expected {batch_size}, got {int(target.shape[0])} for {param!r}.')
    target = target.to(dtype=torch.float32, device=device)  # [B]

    normalize = entry.get('normalize')
    if normalize == 'unit_interval':
      if seq_len <= 1:
        raise ValueError(
          'Dense target extraction requires seq_len > 1 to normalize unit_interval targets. '
          f'Got seq_len={seq_len}.')
      target = target / float(seq_len - 1)  # [B]

    if enabled_mask is not None:
      target = target.masked_fill(~enabled_mask, float('nan'))  # [B]
    target_tensors.append(target)

  if not target_tensors:
    raise ValueError('Dense target extraction produced no targets.')

  # Step 5: stack into [B, D_dense] with consistent ordering.
  y_dense = torch.stack(target_tensors, dim=1)  # [B, D]
  return y_dense, target_names


def _aiono_dense_targets_list_view_names(meta: dict[str, Any]) -> list[str]:
  """List available view names from Aiono observation meta."""
  views = meta.get('views')
  if not isinstance(views, list):
    return []
  view_names = []
  for entry in views:
    if isinstance(entry, dict) and isinstance(entry.get('view'), str):
      view_names.append(entry['view'])
  return sorted(set(view_names))
