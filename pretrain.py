import json
import dataclasses
import logging.config
import math
import random
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from os import path, makedirs
from time import time, perf_counter

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import wandb

import configs
from configs.hydra_utils import (
  hydra_get_config_dict,
)
from data.datasets import PTB_XL
from data.pretrain_data_backend import pretrain_data_build
from data.offline_probe_data_backend import OfflineProbeDataBundle, offline_probe_data_build
from models import JEPA, LeJEPA, NEPA
from utils.monitoring import (
  get_cpu_count,
  get_memory_usage
)
from utils.schedules import (
  linear_schedule,
  cosine_schedule,
  update_weight_decay_,
  update_learning_rate_
)
from utils.probe_metrics import (
  probe_build_wandb_val_metric_logs,
  probe_compute_metrics,
)
from utils.offline_probe import (
  OfflineProbeConfig,
  offline_probe_run_linear_multihead,
  offline_probe_run_linear_multihead_by_layer,
)
from utils.offline_probe_logging import (
  offline_probe_build_classification_logs,
  offline_probe_build_confusion_logs,
  offline_probe_build_dense_logs,
  offline_probe_build_best_across_layers_categorical_logs,
  offline_probe_build_prefix,
)
from utils.aiono_benchmark import aiono_contract_logs
from utils.batch_label_stats import BatchLabelStatsTracker
from utils.label_exposure import LabelExposureTracker
from utils.seeding import derive_seed, seed_everything
from utils.ucr_benchmark import UCRBenchmarkConfig, ucr_benchmark_eval

# Runtime keys that are NOT part of configs.pretrain.Config
_RUNTIME_KEYS = frozenset({
  'run_name',
  'data',
  'offline_probe_data',
  'cauker',
  'ucr',
  'out',
  'chkpt',
  'amp',
  'compile',
  'ptb_xl_data_dir',
  'stop_after_steps',
  'offline_probe_eval_only',
})

# PTB-XL online probe task types
PROBE_TASKS = ('all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm')
_RETIRED_AIONO_ALIAS = 'to' 'yts'
_RETIRED_AIONO_CONFIG_KEY = 'to' 'yts'
_RETIRED_AIONO_OFFLINE_KEY = 'offline_probe_' 'to' 'yts'

def _compute_training_end_step(
  *,
  start_step: int,
  schedule_total_steps: int,
  stop_after_steps: int | None,
  resuming: bool,
) -> int:
  """Compute the exclusive end step for the training loop.

  This function exists to keep schedule-related settings (e.g., `config.steps`)
  stable while allowing short smoke runs and long resume runs via the
  `stop_after_steps` runtime knob.

  Notation: the training loop iterates `for step in range(start_step, end_step):`,
  and uses `log_step = step + 1` for checkpoint naming and W&B steps.

  Args:
    start_step: Starting step index (0-based). For resumed runs this is loaded
      from the checkpoint payload.
    schedule_total_steps: The schedule horizon (`config.steps`) baked into the
      checkpoint/config. Schedules clamp to their final values once
      `step > schedule_total_steps`.
    stop_after_steps: Optional runtime knob that limits how many steps to run
      in this invocation (relative to `start_step`).
    resuming: Whether we are resuming from a checkpoint.

  Returns:
    Exclusive end step for the Python range.

  Raises:
    ValueError: If the computed horizon is empty (would run zero steps).
  """
  if start_step < 0:
    raise ValueError(f'start_step must be >= 0, got {start_step}')
  if schedule_total_steps <= 0:
    raise ValueError(f'schedule_total_steps must be > 0, got {schedule_total_steps}')

  if stop_after_steps is not None:
    if stop_after_steps <= 0:
      raise ValueError(f'stop_after_steps must be > 0, got {stop_after_steps}')
    if resuming:
      end_step = start_step + stop_after_steps
    else:
      end_step = min(schedule_total_steps, start_step + stop_after_steps)
  else:
    end_step = schedule_total_steps

  if end_step <= start_step:
    raise ValueError(
      'Computed an empty training horizon (end_step <= start_step). '
      f'Got start_step={start_step} end_step={end_step}. '
      'Fix: set stop_after_steps to a positive integer (additional steps to run).'
    )
  return int(end_step)


def _grad_norm_params(params: Iterable[nn.Parameter]) -> torch.Tensor:
  """Compute L2 norm of gradients for a parameter collection."""
  total = None
  for param in params:
    if param.grad is None:
      continue
    grad = param.grad.detach()  # [*]
    grad_sq = grad.to(dtype=torch.float32).pow(2).sum()  # scalar
    total = grad_sq if total is None else total + grad_sq  # scalar
  if total is None:
    nan = torch.full((), float('nan'))  # scalar
    return nan
  return total.sqrt()


def _canonicalize_checkpoint_config_payload(
  payload: dict[str, object],
) -> dict[str, object]:
  """Normalize checkpoint config payloads onto canonical Aionoscope field names."""
  normalized = dict(payload)

  # Step 1: reject retired Aionoscope alias fields explicitly.
  if _RETIRED_AIONO_CONFIG_KEY in normalized or _RETIRED_AIONO_OFFLINE_KEY in normalized:
    raise ValueError(
      'Checkpoint config still uses a retired Aionoscope alias. '
      'Fix: migrate the checkpoint payload to use "aiono" and "offline_probe_aiono".')
  if normalized.get('data_backend') == _RETIRED_AIONO_ALIAS:
    raise ValueError(
      'Checkpoint config still uses a retired Aionoscope alias for data_backend. '
      'Fix: migrate it to data_backend="aiono".')
  if normalized.get('probe_source') == _RETIRED_AIONO_ALIAS:
    raise ValueError(
      'Checkpoint config still uses a retired Aionoscope alias for probe_source. '
      'Fix: migrate it to probe_source="aiono".')
  if normalized.get('offline_probe_source') == _RETIRED_AIONO_ALIAS:
    raise ValueError(
      'Checkpoint config still uses a retired Aionoscope alias for offline_probe_source. '
      'Fix: migrate it to offline_probe_source="aiono".')
  offline_probe_sources = normalized.get('offline_probe_sources')
  if isinstance(offline_probe_sources, (list, tuple)):
    if any(source == _RETIRED_AIONO_ALIAS for source in offline_probe_sources):
      raise ValueError(
        'Checkpoint config still uses offline_probe_sources containing a retired Aionoscope alias. '
        'Fix: migrate them to "aiono".')
    normalized['offline_probe_sources'] = list(offline_probe_sources)
  datasets = normalized.get('datasets')
  if isinstance(datasets, dict) and _RETIRED_AIONO_ALIAS in datasets:
    raise ValueError(
      'Checkpoint config still uses datasets with a retired Aionoscope alias. '
      'Fix: migrate it to datasets.aiono.')

  # Step 2: keep the offline-probe defaults aligned with the canonical blocks.
  normalized.setdefault('offline_probe_source', normalized.get('probe_source'))
  if normalized.get('offline_probe_aiono') is None and normalized.get('aiono') is not None:
    normalized['offline_probe_aiono'] = normalized.get('aiono')
  return normalized


def _config_to_checkpoint_payload(
  config: configs.pretrain.Config,
) -> dict[str, object]:
  """Serialize the runtime config using canonical Aionoscope field names."""
  return _canonicalize_checkpoint_config_payload(dataclasses.asdict(config))


def _is_aiono_basic_components_v1(mapping: object) -> bool:
  """Return whether a synthetic config requests benchmark-labeled Aionoscope v1."""
  if not isinstance(mapping, dict):
    return False
  return (
    mapping.get('kind') == 'basic_components'
    and mapping.get('benchmark_family') == 'aiono_basic_components'
    and mapping.get('benchmark_version') == 'v1'
  )


@hydra.main(version_base=None, config_path="configs/pretrain", config_name="ViTXS_ptbxl")
def main(cfg: DictConfig):
  # NOTE: we compute mean and standard deviation of ptb-xl over the train folds (1-8).
  #  We only use these folds during pre-training.
  PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
                0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
  PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
                0.235, 0.338, 0.335, 0.299, 0.294, 0.242]

  # Validate required runtime keys
  if not cfg.get('run_name'):
    raise ValueError("Missing required config key: 'run_name'. Provide via CLI: +run_name=myrun")
  stop_after_steps = cfg.get('stop_after_steps')
  if stop_after_steps is not None:
    if not isinstance(stop_after_steps, int):
      raise ValueError(f"stop_after_steps must be an int, got {type(stop_after_steps)}")
    if stop_after_steps <= 0:
      raise ValueError(f"stop_after_steps must be a positive integer, got {stop_after_steps}")

  offline_probe_eval_only = cfg.get('offline_probe_eval_only', False)
  if not isinstance(offline_probe_eval_only, bool):
    raise ValueError(
      'offline_probe_eval_only must be a bool, '
      f'got {type(offline_probe_eval_only).__name__}')

  run_name = cfg.run_name
  out_base = cfg.get('out', 'pretrain')
  out_dir = path.join(out_base, run_name)
  makedirs(out_dir, exist_ok=True)
  logging.config.fileConfig('logging.ini')
  logger = logging.getLogger('app')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  logger.debug(f'using {device} accelerator and {num_cpus} CPUs')

  # Disabled to enable cloudpickle which is required for parallel processing
  # if using_cuda:
  #   logger.debug('TF32 tensor cores are enabled')
  #   torch.backends.cuda.matmul.allow_tf32 = True
  #   torch.backends.cudnn.allow_tf32 = True
  #   torch.backends.cudnn.benchmark = True

  amp_mode = cfg.get('amp', 'float32')
  if amp_mode == 'float32' or not using_cuda:  # don't use AMP on a CPU
    logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif amp_mode == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.amp.autocast('cuda', dtype=torch.bfloat16)
  else:
    raise ValueError(f"Invalid amp mode: '{amp_mode}'. Must be 'float32' or 'bfloat16'.")

  chkpt_path = cfg.get('chkpt')
  if offline_probe_eval_only and not chkpt_path:
    raise ValueError(
      'offline_probe_eval_only=true requires chkpt=/path/to/checkpoint.pt. '
      'Fix: provide +chkpt=... on the command line.')
  if chkpt_path:
    logger.debug(f'resuming from checkpoint {chkpt_path}')
    # NOTE: PyTorch >= 2.6 defaults `weights_only=True`, which breaks loading
    # our local training checkpoints (they include non-tensor metadata).
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    raw_checkpoint_config = dict(chkpt['config'])
    checkpoint_sampling_frequency = raw_checkpoint_config.get('sampling_frequency')
    config_payload = _canonicalize_checkpoint_config_payload(raw_checkpoint_config)
    if offline_probe_eval_only:
      hydra_config_payload = hydra_get_config_dict(cfg, set(_RUNTIME_KEYS))
      for key in ('seed', 'seed_init', 'seed_data', 'seed_sigreg', 'seed_offline_probe'):
        override_value = cfg.get(key)
        if override_value is not None:
          config_payload[key] = override_value
      for key, value in hydra_config_payload.items():
        if key.startswith('offline_probe_'):
          config_payload[key] = value
      config_payload['sampling_frequency'] = hydra_config_payload.get(
        'sampling_frequency',
        config_payload.get('sampling_frequency'),
      )
    compat_downstream_rep_layer = config_payload.pop('downstream_rep_layer', None)
    if (
      config_payload.get('use_nepa') is True
      and config_payload.get('offline_probe_eval_interval') is not None
      and config_payload.get('offline_probe_layers_categorical') is None
      and config_payload.get('offline_probe_layers_dense') is None
      and config_payload.get('offline_probe_layers_confusion') is None
    ):
      depth = config_payload.get('depth')
      if not isinstance(depth, int):
        raise ValueError(
          'Checkpoint config is missing `depth`; cannot migrate retired downstream_rep_layer '
          f'to offline_probe_layers_*. chkpt={chkpt_path}')
      rep_layer = depth if compat_downstream_rep_layer is None else compat_downstream_rep_layer
      config_payload['offline_probe_layers_categorical'] = [rep_layer]
      config_payload['offline_probe_layers_dense'] = [rep_layer]
      logger.warning(
        'Resuming older NEPA checkpoint without offline_probe_layers_* configured; '
        f'migrating from downstream_rep_layer={compat_downstream_rep_layer} to '
        f'offline_probe_layers_categorical/dense=[{rep_layer}].')
    for compat_key in (
      'offline_probe_cache_on_device',
      'offline_probe_eval_logits_on_device',
      'offline_probe_pairwise_confusion_on_device',
      'offline_probe_regression_eval_on_device',
    ):
      config_payload.pop(compat_key, None)
    config = configs.pretrain.Config(**config_payload)
    if (
      offline_probe_eval_only
      and _is_aiono_basic_components_v1(config.offline_probe_synthetic_config)
    ):
      if not isinstance(checkpoint_sampling_frequency, int):
        raise ValueError(
          'Checkpoint config is missing sampling_frequency, but offline_probe_eval_only '
          'requested aiono_basic_components/v1. Fix: evaluate with a benchmark-compatible checkpoint.')
      if int(checkpoint_sampling_frequency) != 500:
        raise ValueError(
          'Checkpoint was trained with sampling_frequency='
          f'{int(checkpoint_sampling_frequency)} Hz, but offline_probe_eval_only requested '
          'aiono_basic_components/v1, which requires a 500 Hz checkpoint to stay benchmark-comparable. '
          'Fix: use a 500 Hz checkpoint or switch to a non-benchmark offline probe config.')
  else:
    # Build config from Hydra-composed config (excluding runtime keys)
    config_dict = hydra_get_config_dict(cfg, set(_RUNTIME_KEYS))
    config = configs.pretrain.Config(**config_dict)
    logger.debug(f'using Hydra config:\n{OmegaConf.to_yaml(cfg)}')
    chkpt = None

  strict = chkpt is None
  probe_source = config.probe_source
  if probe_source is None and (not strict) and set(config.datasets) == {'ptb-xl'}:
    probe_source = 'ptb-xl'

  labels_available = probe_source is not None

  ucr_benchmark_config = None
  ucr_cfg_raw = cfg.get('ucr')
  if ucr_cfg_raw is not None:
    if not isinstance(ucr_cfg_raw, (dict, DictConfig)):
      raise ValueError(
        'ucr runtime config must be a dict when provided. '
        f'Got type={type(ucr_cfg_raw).__name__}')
    ucr_payload = (
      OmegaConf.to_container(ucr_cfg_raw, resolve=True)
      if isinstance(ucr_cfg_raw, DictConfig) else dict(ucr_cfg_raw)
    )
    if not isinstance(ucr_payload, dict):
      raise ValueError(
        'ucr runtime config must convert to a dict. '
        f'Got type={type(ucr_payload).__name__}')
    required_base = (
      'archive_root',
      'resize_mode',
      'resize_target_length',
      'eval_batch_size',
      'classifier',
    )
    missing = [key for key in required_base if ucr_payload.get(key) in (None, '???')]
    if missing:
      raise ValueError(
        'ucr benchmark is enabled but missing required keys (no silent defaults): '
        f'{missing}')

    classifier = str(ucr_payload['classifier'])
    if classifier not in ('rf', 'logreg'):
      raise ValueError(
        'ucr.classifier must be "rf" or "logreg" (no silent defaults). '
        f'Got {classifier!r}')

    required_classifier = (
      ('rf_n_estimators', 'rf_n_jobs')
      if classifier == 'rf' else
      ('logreg_max_iter',)
    )
    missing_classifier = [
      key for key in required_classifier
      if ucr_payload.get(key) in (None, '???')
    ]
    if missing_classifier:
      raise ValueError(
        f'ucr benchmark classifier={classifier!r} is enabled but missing required keys (no silent defaults): '
        f'{missing_classifier}')
    dataset_names = ucr_payload.get('dataset_names')
    dataset_names_tuple = None
    if dataset_names is not None:
      if not isinstance(dataset_names, (list, tuple)):
        raise ValueError(
          'ucr.dataset_names must be a list/tuple of dataset names or null. '
          f'Got type={type(dataset_names).__name__}')
      dataset_names_tuple = tuple(str(name) for name in dataset_names)
    rf_n_estimators = ucr_payload.get('rf_n_estimators')
    rf_n_jobs = ucr_payload.get('rf_n_jobs')
    logreg_max_iter = ucr_payload.get('logreg_max_iter')

    ucr_benchmark_config = UCRBenchmarkConfig(
      archive_root=str(ucr_payload['archive_root']),
      resize_mode=str(ucr_payload['resize_mode']),
      resize_target_length=int(ucr_payload['resize_target_length']),
      eval_batch_size=int(ucr_payload['eval_batch_size']),
      classifier=str(classifier),
      rf_n_estimators=None if rf_n_estimators in (None, '???') else int(rf_n_estimators),
      rf_n_jobs=None if rf_n_jobs in (None, '???') else int(rf_n_jobs),
      logreg_max_iter=None if logreg_max_iter in (None, '???') else int(logreg_max_iter),
      dataset_names=dataset_names_tuple,
    )

  use_offline_probe = config.offline_probe_eval_interval is not None

  if labels_available and (not use_offline_probe):
    raise ValueError(
      'offline_probe_eval_interval is required when labels are available. '
      'Set offline_probe_eval_interval and related offline probe settings in your config.')

  if (not strict) and use_offline_probe and (
    config.offline_probe_source is None and config.offline_probe_sources is None
  ):
    config = dataclasses.replace(
      config,
      offline_probe_source=config.probe_source_resolved,
      offline_probe_aiono=config.synthetic_config,
    )

  offline_probe_sources = config.offline_probe_sources_resolved
  use_offline_probe_datasets = bool(offline_probe_sources)
  use_ucr_benchmark = ucr_benchmark_config is not None

  if use_ucr_benchmark and not use_offline_probe:
    raise ValueError(
      'ucr benchmark requires offline_probe_eval_interval to be set, because it must '
      'follow the same offline evaluation cadence (offline_probe_eval_interval).')
  if labels_available and not use_offline_probe_datasets:
    raise ValueError(
      'probe_source is set (labeled training), but offline_probe_source(s) are empty. '
      'Fix: set offline_probe_dataset=same_as_train/ptbxl/aiono_basic_components_balanced/'
      'aiono_basic_components_imbalanced/both_basic_components, '
      'or set offline_probe_source(s) directly.')

  if config.seed is not None:
    if not isinstance(config.seed, int):
      raise ValueError(f'seed must be an int or null, got {type(config.seed).__name__}')
    if config.seed < 0:
      raise ValueError(f'seed must be >= 0, got {config.seed}')

  if config.label_exposure_train_log_interval is not None:
    if not isinstance(config.label_exposure_train_log_interval, int):
      raise ValueError(
        'label_exposure_train_log_interval must be an int or null, '
        f'got {type(config.label_exposure_train_log_interval).__name__}')
    if config.label_exposure_train_log_interval <= 0:
      raise ValueError(
        'label_exposure_train_log_interval must be > 0, '
        f'got {config.label_exposure_train_log_interval}')
    if not labels_available:
      raise ValueError(
        'label_exposure_train_log_interval requires labeled training batches. '
        'Set probe_source (e.g., probe_source="ptb-xl" or probe_source="aiono").')

  if config.batch_label_stats_train_log_interval is not None:
    if not labels_available:
      raise ValueError(
        'batch_label_stats_train_log_interval requires labeled training batches. '
        'Set probe_source (e.g., probe_source="ptb-xl" or probe_source="aiono").')
    if config.gradient_accumulation_steps != 1:
      raise ValueError(
        'batch_label_stats_train_log_interval requires gradient_accumulation_steps=1 '
        f'(no silent defaults). Got gradient_accumulation_steps={config.gradient_accumulation_steps}')

  if not isinstance(config.label_exposure_offline_probe_log, bool):
    raise ValueError(
      'label_exposure_offline_probe_log must be a bool, '
      f'got {type(config.label_exposure_offline_probe_log).__name__}')
  if config.label_exposure_offline_probe_log:
    if not use_offline_probe:
      raise ValueError(
        'label_exposure_offline_probe_log requires offline probing to be enabled. '
        'Set offline_probe_eval_interval and offline_probe_dataset/source(s).')
    if 'ptb-xl' not in offline_probe_sources:
      raise ValueError(
        'label_exposure_offline_probe_log currently supports PTB-XL offline probing only. '
        'Fix: include "ptb-xl" in offline_probe_sources (e.g., offline_probe_dataset=ptbxl or both).')

  if use_offline_probe:
    if not isinstance(config.offline_probe_eval_interval, int):
      raise ValueError(
        f'offline_probe_eval_interval must be an int, got {type(config.offline_probe_eval_interval)}')
    if config.offline_probe_eval_interval <= 0:
      raise ValueError(
        f'offline_probe_eval_interval must be > 0, got {config.offline_probe_eval_interval}')
    if (not offline_probe_sources) and (not use_ucr_benchmark):
      raise ValueError(
        'offline_probe_eval_interval is set, but no offline evaluators are enabled. '
        'Fix: set offline_probe_dataset/source(s) and/or configure ucr:{...}.')
    if use_offline_probe_datasets:
      if config.offline_probe_task is not None and 'ptb-xl' not in offline_probe_sources:
        raise ValueError(
          'offline_probe_task must be null when offline_probe_sources does not include "ptb-xl".')
      if 'ptb-xl' in offline_probe_sources:
        if config.offline_probe_task is None:
          raise ValueError('offline_probe_task is required when offline probe is enabled')
        if config.offline_probe_task not in PROBE_TASKS:
          raise ValueError(
            f'offline_probe_task must be one of {PROBE_TASKS}, got {config.offline_probe_task}')
      if 'aiono' in offline_probe_sources and config.offline_probe_synthetic_config is None:
        raise ValueError(
          'offline_probe_aiono is required when offline_probe_sources includes "aiono". '
          'Set offline_probe_dataset=aiono_basic_components_balanced/'
          'aiono_basic_components_imbalanced, or provide '
          'offline_probe_aiono overrides.')
      offline_required = (
        'offline_probe_steps',
        'offline_probe_batch_size',
        'offline_probe_learning_rate',
        'offline_probe_final_learning_rate',
        'offline_probe_learning_rate_warmup_steps',
        'offline_probe_weight_decay',
        'offline_probe_opt_betas',
        'offline_probe_gradient_clip',
        'offline_probe_checkpoint_interval',
      )
      missing_required = [
        name for name in offline_required if getattr(config, name) is None
      ]
      if missing_required:
        raise ValueError(
          'Offline probe requires the following config keys to be set: '
          f'{missing_required}')
      if not isinstance(config.offline_probe_steps, int):
        raise ValueError(
          f'offline_probe_steps must be an int, got {type(config.offline_probe_steps)}')
      if config.offline_probe_steps <= 0:
        raise ValueError(
          f'offline_probe_steps must be > 0, got {config.offline_probe_steps}')
      if not isinstance(config.offline_probe_batch_size, int):
        raise ValueError(
          f'offline_probe_batch_size must be an int, got {type(config.offline_probe_batch_size)}')
      if config.offline_probe_batch_size <= 0:
        raise ValueError(
          f'offline_probe_batch_size must be > 0, got {config.offline_probe_batch_size}')
      if not isinstance(config.offline_probe_learning_rate, (int, float)):
        raise ValueError(
          'offline_probe_learning_rate must be a float, '
          f'got {type(config.offline_probe_learning_rate)}')
      if config.offline_probe_learning_rate <= 0:
        raise ValueError(
          f'offline_probe_learning_rate must be > 0, got {config.offline_probe_learning_rate}')
      if not isinstance(config.offline_probe_final_learning_rate, (int, float)):
        raise ValueError(
          'offline_probe_final_learning_rate must be a float, '
          f'got {type(config.offline_probe_final_learning_rate)}')
      if config.offline_probe_final_learning_rate < 0:
        raise ValueError(
          'offline_probe_final_learning_rate must be >= 0, '
          f'got {config.offline_probe_final_learning_rate}')
      if not isinstance(config.offline_probe_learning_rate_warmup_steps, int):
        raise ValueError(
          'offline_probe_learning_rate_warmup_steps must be an int, '
          f'got {type(config.offline_probe_learning_rate_warmup_steps)}')
      if config.offline_probe_learning_rate_warmup_steps < 0:
        raise ValueError(
          'offline_probe_learning_rate_warmup_steps must be >= 0, '
          f'got {config.offline_probe_learning_rate_warmup_steps}')
      if not isinstance(config.offline_probe_weight_decay, (int, float)):
        raise ValueError(
          'offline_probe_weight_decay must be a float, '
          f'got {type(config.offline_probe_weight_decay)}')
      if config.offline_probe_weight_decay < 0:
        raise ValueError(
          f'offline_probe_weight_decay must be >= 0, got {config.offline_probe_weight_decay}')
      if not isinstance(config.offline_probe_opt_betas, (tuple, list)):
        raise ValueError(
          'offline_probe_opt_betas must be a tuple/list of two floats, '
          f'got {type(config.offline_probe_opt_betas)}')
      if len(config.offline_probe_opt_betas) != 2:
        raise ValueError(
          'offline_probe_opt_betas must contain two values, '
          f'got {config.offline_probe_opt_betas}')
      for beta in config.offline_probe_opt_betas:
        if not isinstance(beta, (int, float)):
          raise ValueError(
            f'offline_probe_opt_betas values must be floats, got {type(beta)}')
        if beta <= 0 or beta >= 1:
          raise ValueError(
            f'offline_probe_opt_betas values must be in (0, 1), got {beta}')
      if not isinstance(config.offline_probe_gradient_clip, (int, float)):
        raise ValueError(
          'offline_probe_gradient_clip must be a float, '
          f'got {type(config.offline_probe_gradient_clip)}')
      if config.offline_probe_gradient_clip < 0:
        raise ValueError(
          f'offline_probe_gradient_clip must be >= 0, got {config.offline_probe_gradient_clip}')
      if not isinstance(config.offline_probe_checkpoint_interval, int):
        raise ValueError(
          'offline_probe_checkpoint_interval must be an int, '
          f'got {type(config.offline_probe_checkpoint_interval)}')
      if config.offline_probe_checkpoint_interval <= 0:
        raise ValueError(
          f'offline_probe_checkpoint_interval must be > 0, got {config.offline_probe_checkpoint_interval}')
      if config.offline_probe_checkpoint_interval > config.offline_probe_steps:
        raise ValueError(
          'offline_probe_checkpoint_interval must be <= offline_probe_steps, '
          f'got {config.offline_probe_checkpoint_interval} > {config.offline_probe_steps}')

  seed_mode = 'none'
  if config.seed is not None:
    seed_mode = 'single'
  elif config.seed_init is not None:
    seed_mode = 'split'

  if seed_mode == 'single':
    logger.info(f'Using fixed seed={config.seed}')
    seed_everything(int(config.seed))
  elif seed_mode == 'split':
    logger.info(
      'Using split seeds: seed_init=%d seed_data=%d seed_sigreg=%d seed_offline_probe=%d',
      int(config.seed_init),
      int(config.seed_data),
      int(config.seed_sigreg),
      int(config.seed_offline_probe),
    )
  wandb.init(project="ECG-LeJEPA", name=run_name, config=OmegaConf.to_container(cfg, resolve=True))

  data_bundle = pretrain_data_build(
    cfg=cfg,
    config=config,
    device=device,
    using_cuda=using_cuda,
    num_cpus=num_cpus,
    strict=strict,
  )
  use_online_probe = data_bundle.labeled

  if use_online_probe:
    if config.probe_lr is None:
      raise ValueError('probe_lr is required when probing is enabled')
    if config.probe_weight_decay is None:
      raise ValueError('probe_weight_decay is required when probing is enabled')
    if config.probe_eval_interval is None:
      raise ValueError('probe_eval_interval is required when probing is enabled')

  probe_val_loader = data_bundle.probe_val_loader
  num_classes = data_bundle.num_classes
  probe_class_names = data_bundle.probe_class_names
  probe_group_to_classes = data_bundle.probe_group_to_classes
  offline_probe_train_loader = data_bundle.offline_probe_train_loader
  offline_probe_val_loader = data_bundle.offline_probe_val_loader
  offline_probe_num_classes = data_bundle.offline_probe_num_classes
  offline_probe_class_names = data_bundle.offline_probe_class_names
  offline_probe_group_to_classes = data_bundle.offline_probe_group_to_classes
  offline_dense_probe_target_names = data_bundle.offline_dense_probe_target_names
  offline_dense_probe_log_per_target = data_bundle.offline_dense_probe_log_per_target

  if use_online_probe and (
    probe_val_loader is None
    or num_classes is None
    or probe_class_names is None
    or probe_group_to_classes is None
  ):
    raise ValueError('Probe requested but data backend did not initialize probe datasets.')

  label_exposure_dir = None
  if (config.label_exposure_train_log_interval is not None) or config.label_exposure_offline_probe_log:
    if wandb.run is None:
      raise ValueError('W&B run is not initialized; cannot write label exposure logs.')
    label_exposure_dir = path.join(wandb.run.dir, 'label_exposure')
    makedirs(label_exposure_dir, exist_ok=True)

  train_label_exposure_tracker = None
  train_label_exposure_snapshots: list[dict[str, object]] = []
  train_label_exposure_path = None
  if config.label_exposure_train_log_interval is not None:
    if not use_online_probe or probe_class_names is None:
      raise ValueError(
        'label_exposure_train_log_interval requires labeled training batches, '
        'but the data backend is unlabeled.')
    train_label_exposure_tracker = LabelExposureTracker(class_names=list(probe_class_names))
    train_label_exposure_path = path.join(label_exposure_dir, 'train.json')

  batch_label_stats_tracker = None
  batch_label_stats_snapshots: list[dict[str, object]] = []
  batch_label_stats_path = None
  if config.batch_label_stats_train_log_interval is not None:
    if not use_online_probe or probe_class_names is None:
      raise ValueError(
        'batch_label_stats_train_log_interval requires labeled training batches, '
        'but the data backend is unlabeled.')
    train_positive_counts = data_bundle.train_class_positive_counts
    if train_positive_counts is None:
      raise ValueError(
        'batch_label_stats_train_log_interval requires train_class_positive_counts from the data backend. '
        'Fix: use PTB-XL probe mode (probe_source="ptb-xl").')
    rare_bottom_k = int(config.batch_label_stats_rare_bottom_k)
    if rare_bottom_k > len(probe_class_names):
      raise ValueError(
        'batch_label_stats_rare_bottom_k must be <= num_classes. '
        f'Got rare_bottom_k={rare_bottom_k} > num_classes={len(probe_class_names)}')

    rare_class_indices = sorted(
      range(len(train_positive_counts)),
      key=lambda index: (int(train_positive_counts[index]), int(index)),
    )[:rare_bottom_k]

    if wandb.run is None:
      raise ValueError('W&B run is not initialized; cannot write batch label stats logs.')
    batch_label_stats_dir = path.join(wandb.run.dir, 'batch_label_stats')
    makedirs(batch_label_stats_dir, exist_ok=True)
    batch_label_stats_path = path.join(batch_label_stats_dir, 'train.json')
    batch_label_stats_tracker = BatchLabelStatsTracker(
      class_names=list(probe_class_names),
      train_positive_counts=list(train_positive_counts),
      rare_class_indices=rare_class_indices,
      rare_bottom_k=rare_bottom_k,
    )
    logger.info(
      'Batch label stats: rare_bottom_k=%d rare_classes=%s',
      rare_bottom_k,
      ','.join(batch_label_stats_tracker.rare_class_names),
    )

  offline_probe_eval_config = None
  offline_probe_bundles: dict[str, OfflineProbeDataBundle] = {}
  if use_offline_probe_datasets:
    training_offline_source = None
    if config.data_backend_resolved == 'aiono':
      training_offline_source = 'aiono'
    elif probe_source == 'ptb-xl':
      training_offline_source = 'ptb-xl'

    if (training_offline_source is not None
        and training_offline_source in offline_probe_sources
        and offline_probe_train_loader is not None
        and offline_probe_val_loader is not None
        and offline_probe_group_to_classes is not None
        and offline_probe_class_names is not None
        and offline_probe_num_classes is not None):
      offline_probe_bundles[training_offline_source] = OfflineProbeDataBundle(
        train_loader=offline_probe_train_loader,
        val_loader=offline_probe_val_loader,
        num_classes=int(offline_probe_num_classes),
        class_names=list(offline_probe_class_names),
        group_to_classes=dict(offline_probe_group_to_classes),
        dense_target_names=(
          list(offline_dense_probe_target_names)
          if offline_dense_probe_target_names is not None else None),
        dense_log_per_target=offline_dense_probe_log_per_target,
        val_loaders_by_seed=data_bundle.offline_probe_val_loaders_by_seed,
        source_name=data_bundle.offline_probe_source_name,
        benchmark_family=data_bundle.offline_probe_benchmark_family,
        benchmark_version=data_bundle.offline_probe_benchmark_version,
        benchmark_manifest=data_bundle.offline_probe_benchmark_manifest,
        benchmark_comparable=data_bundle.offline_probe_benchmark_comparable,
        validation_seed_values=data_bundle.offline_probe_validation_seed_values,
        validation_seed_offset=data_bundle.offline_probe_validation_seed_offset,
        validation_seed_to_generator_seed=data_bundle.offline_probe_validation_seed_to_generator_seed,
      )

    for source in offline_probe_sources:
      if source in offline_probe_bundles:
        continue
      offline_bundle = offline_probe_data_build(
        cfg=cfg,
        config=config,
        device=device,
        using_cuda=using_cuda,
        num_cpus=num_cpus,
        offline_probe_source=source,
      )
      offline_probe_bundles[source] = offline_bundle

    contract_summary_logs: dict[str, object] = {}
    for source, offline_bundle in offline_probe_bundles.items():
      if source != 'aiono':
        continue
      manifest = (
        dict(offline_bundle.benchmark_manifest)
        if offline_bundle.benchmark_manifest is not None else {}
      )
      contract_summary_logs.update(
        aiono_contract_logs(
          source='aiono',
          manifest=manifest,
          actual_sampling_frequency_hz=int(config.sampling_frequency),
          benchmark_comparable=bool(offline_bundle.benchmark_comparable),
        )
      )
    if contract_summary_logs:
      if wandb.run is None:
        raise ValueError('W&B run is not initialized; cannot write Aionoscope contract metadata.')
      wandb.run.summary.update(contract_summary_logs)

    offline_probe_eval_config = OfflineProbeConfig(
      steps=config.offline_probe_steps,
      batch_size=config.offline_probe_batch_size,
      learning_rate=float(config.offline_probe_learning_rate),
      final_learning_rate=float(config.offline_probe_final_learning_rate),
      learning_rate_warmup_steps=config.offline_probe_learning_rate_warmup_steps,
      weight_decay=float(config.offline_probe_weight_decay),
      opt_betas=(float(config.offline_probe_opt_betas[0]), float(config.offline_probe_opt_betas[1])),
      gradient_clip=float(config.offline_probe_gradient_clip),
      checkpoint_interval=config.offline_probe_checkpoint_interval,
    )

  logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')
  train_iterator = data_bundle.train_iterator

  # setup hyperparameter schedules
  start_step = chkpt['step'] if chkpt is not None else 0
  try:
    end_step = _compute_training_end_step(
      start_step=int(start_step),
      schedule_total_steps=int(config.steps),
      stop_after_steps=int(stop_after_steps) if stop_after_steps is not None else None,
      resuming=chkpt is not None,
    )
  except ValueError as e:
    if chkpt is not None and stop_after_steps is None and int(start_step) >= int(config.steps):
      raise ValueError(
        f'Checkpoint step={int(start_step)} is already at or beyond config.steps={int(config.steps)}. '
        'Fix: provide stop_after_steps=N to continue training for N additional steps.'
      ) from e
    raise
  if stop_after_steps is not None:
    logger.info(
      'Stopping after %d step(s) this run (stop_after_steps=%d, start_step=%d, end_step=%d, schedule_total_steps=%d)',
      end_step - start_step,
      stop_after_steps,
      start_step,
      end_step,
      config.steps,
    )

  momentum_schedule = linear_schedule(
    total_steps=config.steps,
    start_value=config.encoder_momentum,
    final_value=config.final_encoder_momentum,
    step=start_step)
  lr_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.learning_rate,
    final_value=config.final_learning_rate,
    warmup_steps=config.learning_rate_warmup_steps,
    warmup_start_value=1e-6,
    step=start_step)
  wd_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.weight_decay,
    final_value=config.final_weight_decay,
    step=start_step)

  if seed_mode == 'split':
    logger.info(f'Seeding model/training RNG with seed_init={int(config.seed_init)}')
    seed_everything(int(config.seed_init))

  # setup model
  if config.use_nepa:
    model = original_model = NEPA(
      config=config,
      use_sdp_kernel=using_cuda
    ).to(device)
  elif config.use_lejepa:
    model = original_model = LeJEPA(
      config=config,
      use_sdp_kernel=using_cuda
    ).to(device)
  else:
    model = original_model = JEPA(
      config=config,
      momentum_schedule=momentum_schedule,
      use_sdp_kernel=using_cuda
    ).to(device)
  optimizer = model.get_optimizer(fused=using_cuda)

  if chkpt is not None:  # resume training from checkpoint
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])

  # setup online probe (if PTB-XL mode)
  probe = None
  probe_optimizer = None
  if use_online_probe:
    probe = nn.Sequential(
      nn.LayerNorm(config.dim),
      nn.Linear(config.dim, num_classes)
    ).to(device)
    probe_optimizer = torch.optim.AdamW(
      probe.parameters(),
      lr=config.probe_lr,
      weight_decay=config.probe_weight_decay,
      fused=using_cuda)
    if chkpt is not None and 'probe' in chkpt:
      probe.load_state_dict(chkpt['probe'])
      probe_optimizer.load_state_dict(chkpt['probe_optimizer'])
    logger.debug(f'Online probe initialized: {num_classes} classes, lr={config.probe_lr}, wd={config.probe_weight_decay}')

  if cfg.get('compile', False):
    model = torch.compile(model)

  log_patch_loss = False
  log_patch_loss_interval = None
  if config.use_nepa and config.nepa_log_patch_loss:
    log_patch_loss = True
    log_patch_loss_interval = config.nepa_patch_loss_log_interval

  last_val_step = None
  last_val_auc = None
  last_val_auprc = None
  last_per_class_auc = None
  last_per_class_auprc = None

  offline_probe_base_seed = None
  if config.seed is not None:
    offline_probe_base_seed = int(config.seed)
  elif config.seed_offline_probe is not None:
    offline_probe_base_seed = int(config.seed_offline_probe)

  if ucr_benchmark_config is not None and offline_probe_base_seed is not None:
    ucr_seed = derive_seed(seed=offline_probe_base_seed, salt='ucr_classifier')
    ucr_benchmark_config = dataclasses.replace(ucr_benchmark_config, seed=ucr_seed)

  @contextmanager
  def _offline_probe_rng_context(*, log_step: int, salt: str):
    """Optionally isolate RNG state for one offline probe call (opt-in via seeding)."""
    # Step 1: keep default behavior when seeding is disabled.
    if offline_probe_base_seed is None:
      yield
      return
    if not isinstance(salt, str) or not salt:
      raise ValueError('salt must be a non-empty string')

    # Step 2: snapshot Python/NumPy RNG so offline probing cannot perturb training.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()

    # Step 3: fork torch RNG (CPU + current CUDA device) and seed deterministically.
    devices = [torch.cuda.current_device()] if using_cuda else []
    probe_seed = derive_seed(seed=offline_probe_base_seed, salt=f'offline_probe/{salt}/step={log_step}')
    with torch.random.fork_rng(devices=devices, enabled=True):
      seed_everything(probe_seed)
      try:
        yield
      finally:
        pass

    # Step 4: restore Python/NumPy RNG state.
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)

  def _offline_probe_eval_and_log(*, log_step: int) -> None:
    """Run offline probe evaluation and log metrics to W&B at the given step."""
    # Step 1: validate offline probe bundles and configs.
    total_start = perf_counter()
    if offline_probe_eval_config is None or not offline_probe_bundles:
      raise ValueError('Offline probe requested but not initialized.')
    if not isinstance(log_step, int) or log_step < 0:
      raise ValueError(f'log_step must be a non-negative int, got {log_step!r}')
    missing_sources = [
      source for source in offline_probe_sources
      if source not in offline_probe_bundles
    ]
    if missing_sources:
      raise ValueError(
        'Offline probe bundles are missing for sources '
        f'{missing_sources}.')

    offline_probe_layer_mode = any(
      value is not None
      for value in (
        config.offline_probe_layers_categorical,
        config.offline_probe_layers_dense,
        config.offline_probe_layers_confusion,
      )
    )
    multi_source = len(offline_probe_sources) > 1

    # Step 2: pick the encoder / representation function(s) used for offline probing.
    layers_categorical: tuple[int, ...] = ()
    layers_dense: tuple[int, ...] = ()
    layers_confusion: tuple[int, ...] = ()
    layers_union: tuple[int, ...] = ()
    layers_confusion_set: set[int] = set()

    offline_representation_fn = None
    if offline_probe_layer_mode:
      layers_categorical = tuple(config.offline_probe_layers_categorical or ())
      layers_dense = tuple(config.offline_probe_layers_dense or ())
      layers_confusion = tuple(config.offline_probe_layers_confusion or ())
      layers_union = tuple(sorted(set(layers_categorical) | set(layers_dense) | set(layers_confusion)))
      if not layers_union:
        raise ValueError('offline_probe_layers_* set but no layers were requested.')
      layers_confusion_set = set(layers_confusion)

      # Multi-layer offline probing: select an encoder that can produce `vit_layers`.
      if isinstance(original_model, NEPA):
        offline_encoder = original_model.encoder

        def _forward_vit_layers(x, *, _encoder=offline_encoder):
          vit_layers, _ = _encoder(x)
          return vit_layers
      else:
        offline_encoder = getattr(original_model, 'target_encoder', None) or original_model.encoder
        if not hasattr(offline_encoder, 'forward_vit_layers'):
          raise ValueError(
            'offline_probe_layers_* require the encoder to implement forward_vit_layers(x). '
            f'Got model={type(original_model).__name__}, encoder={type(offline_encoder).__name__}.')

        def _forward_vit_layers(x, *, _encoder=offline_encoder):
          vit_layers, _ = _encoder.forward_vit_layers(x)
          return vit_layers

      def offline_representation_fn(
        x,
        *,
        _forward=_forward_vit_layers,
        _layers=layers_union,
      ) -> dict[int, torch.Tensor]:
        """Extract representations `[B, D]` for requested ViT layers."""
        vit_layers = _forward(x)
        reps = {layer: vit_layers.representation(layer) for layer in _layers}  # each [B, D]
        return reps
    else:
      if isinstance(original_model, NEPA):
        offline_encoder = original_model.encoder

        def offline_representation_fn(
          x,
          *,
          _encoder=offline_encoder,
        ):
          """Extract NEPA layer representations for offline probing."""
          vit_layers, _ = _encoder(x)
          return vit_layers.representation(vit_layers.depth)  # [B, D]
      else:
        offline_encoder = getattr(original_model, 'target_encoder', None) or original_model.encoder

    if offline_encoder is None:
      raise ValueError('Offline probe requires an encoder or teacher_encoder.')

    # Step 3: evaluate each offline probe dataset independently.
    offline_probe_logs_combined: dict[str, object] = {}
    offline_probe_exposure: dict[str, object] | None = None
    if config.label_exposure_offline_probe_log:
      offline_probe_exposure = {'log_step': int(log_step), 'sources': {}}
    for source in offline_probe_sources:
      offline_bundle = offline_probe_bundles[source]
      if offline_probe_layer_mode:
        # Multi-layer NEPA probes: evaluate once per source and log per-layer metrics.
        log_label_exposure = config.label_exposure_offline_probe_log and source == 'ptb-xl'
        with _offline_probe_rng_context(log_step=log_step, salt=f'source={source}:mode=layered'):
          offline_layered = offline_probe_run_linear_multihead_by_layer(
            encoder=offline_encoder,
            representation_fn=offline_representation_fn,
            train_loader=offline_bundle.train_loader,
            val_loader=offline_bundle.val_loader,
            val_loaders_by_seed=offline_bundle.val_loaders_by_seed,
            num_classes=offline_bundle.num_classes,
            class_names=offline_bundle.class_names,
            eval_config=offline_probe_eval_config,
            device=device,
            auto_mixed_precision=auto_mixed_precision,
            layers_categorical=layers_categorical,
            layers_dense=layers_dense,
            layers_confusion=layers_confusion,
            dense_target_names=offline_bundle.dense_target_names,
            dense_log_per_target=offline_bundle.dense_log_per_target,
            label_exposure=log_label_exposure,
          )

        dense_targets_available = bool(offline_layered.get('dense_targets_available', False))
        if layers_dense and not dense_targets_available:
          logger.info(
            f'[step {log_step}] offline dense probe skipped (no dense targets) for source={source}')

        # Step 3a: log categorical (AUROC/AUPRC) metrics per requested layer.
        label_exposure_by_layer: dict[str, object] = {}
        for layer in layers_categorical:
          layer_metrics = offline_layered['categorical'].get(layer)
          if layer_metrics is None:
            raise ValueError(f'Missing categorical metrics for offline probe layer={layer}')
          if log_label_exposure:
            exposure_payload = layer_metrics.get('label_exposure')
            if exposure_payload is None:
              raise ValueError('Offline probe did not return label exposure payload.')
            label_exposure_by_layer[str(layer)] = exposure_payload
          offline_best_auc = layer_metrics['best_auc']
          offline_best_auprc = layer_metrics['best_auprc']

          probe_prefix = offline_probe_build_prefix(
            root='offline_probe',
            source=source,
            layer=layer,
            multi_source=multi_source)
          best_auprc_prefix = offline_probe_build_prefix(
            root='offline_probe_best_auprc',
            source=source,
            layer=layer,
            multi_source=multi_source)
          logger.info(
            f'[step {log_step}] {probe_prefix}/best_auc: {offline_best_auc["macro_auc"]:.4f} '
            f'{probe_prefix}/best_auprc: {offline_best_auprc["macro_auprc"]:.4f}')

          probe_logs, probe_best_auprc_logs = offline_probe_build_classification_logs(
            probe_prefix=probe_prefix,
            best_auprc_prefix=best_auprc_prefix,
            best_auc=offline_best_auc,
            best_auprc=offline_best_auprc,
            group_to_classes=offline_bundle.group_to_classes,
            feature_dim=int(layer_metrics['feature_dim']),
            train_size=int(layer_metrics['train_size']),
            val_size=int(layer_metrics['val_size']),
            val_num_crops=int(layer_metrics['val_num_crops']),
          )
          offline_probe_logs_combined.update(probe_logs)
          offline_probe_logs_combined.update(probe_best_auprc_logs)

          if layer in layers_confusion_set:
            title_suffix = f' layer={layer}' + (f' source={source}' if multi_source else '')
            confusion_logs = offline_probe_build_confusion_logs(
              probe_prefix=probe_prefix,
              best_auprc_prefix=best_auprc_prefix,
              best_auc=offline_best_auc,
              best_auprc=offline_best_auprc,
              title_suffix=title_suffix,
              error_context=f'layer={layer}',
              timing_logs=None,
            )
            offline_probe_logs_combined.update(confusion_logs)

        if log_label_exposure:
          offline_probe_exposure['sources'][source] = {
            'mode': 'layered',
            'categorical_by_layer': label_exposure_by_layer,
          }

        # Step 3b: log dense probe metrics per requested layer (if targets exist).
        if dense_targets_available:
          for layer in layers_dense:
            dense_best = offline_layered['dense'].get(layer)
            if dense_best is None:
              raise ValueError(f'Missing dense metrics for offline probe layer={layer}')
            dense_prefix = offline_probe_build_prefix(
              root='offline_dense_probe',
              source=source,
              layer=layer,
              multi_source=multi_source)
            dense_logs = offline_probe_build_dense_logs(
              dense_prefix=dense_prefix,
              dense_best=dense_best,
              feature_dim=int(offline_layered['feature_dim']),
              train_size=int(offline_layered['train_size']),
              val_size=int(offline_layered['val_size']),
              val_num_crops=int(offline_layered['val_num_crops']),
            )
            offline_probe_logs_combined.update(dense_logs)

        # Step 3c: log no-layer "best across layers" keys for compatibility.
        if layers_categorical:
          compat_probe_prefix = offline_probe_build_prefix(
            root='offline_probe',
            source=source,
            layer=None,
            multi_source=multi_source)
          compat_categorical_logs = offline_probe_build_best_across_layers_categorical_logs(
            probe_prefix=compat_probe_prefix,
            categorical_by_layer=dict(offline_layered['categorical']),
            group_to_classes=offline_bundle.group_to_classes,
            groups=('events', 'noise', 'periodic', 'trend'),
          )
          offline_probe_logs_combined.update(compat_categorical_logs)

        if dense_targets_available and layers_dense:
          compat_dense_prefix = offline_probe_build_prefix(
            root='offline_dense_probe',
            source=source,
            layer=None,
            multi_source=multi_source)
          mse_values = [float(offline_layered['dense'][layer]['macro_mse']) for layer in layers_dense]
          mae_values = [float(offline_layered['dense'][layer]['macro_mae']) for layer in layers_dense]
          r2_values = [float(offline_layered['dense'][layer]['macro_r2']) for layer in layers_dense]
          pearson_values = [float(offline_layered['dense'][layer]['macro_pearson']) for layer in layers_dense]

          mse_finite = [value for value in mse_values if math.isfinite(value)]
          mae_finite = [value for value in mae_values if math.isfinite(value)]
          r2_finite = [value for value in r2_values if math.isfinite(value)]
          pearson_finite = [value for value in pearson_values if math.isfinite(value)]

          best_mse = min(mse_finite) if mse_finite else float('nan')
          best_mae = min(mae_finite) if mae_finite else float('nan')
          best_r2 = max(r2_finite) if r2_finite else float('nan')
          best_pearson = max(pearson_finite) if pearson_finite else float('nan')
          offline_probe_logs_combined[f'{compat_dense_prefix}/mse'] = best_mse
          offline_probe_logs_combined[f'{compat_dense_prefix}/mae'] = best_mae
          offline_probe_logs_combined[f'{compat_dense_prefix}/r2'] = best_r2
          offline_probe_logs_combined[f'{compat_dense_prefix}/pearson'] = best_pearson
      else:
        # Single-layer probes: evaluate once per source and log metrics.
        source_start = perf_counter()
        log_label_exposure = config.label_exposure_offline_probe_log and source == 'ptb-xl'
        with _offline_probe_rng_context(log_step=log_step, salt=f'source={source}:mode=single'):
          offline_metrics = offline_probe_run_linear_multihead(
            encoder=offline_encoder,
            representation_fn=offline_representation_fn,
            train_loader=offline_bundle.train_loader,
            val_loader=offline_bundle.val_loader,
            val_loaders_by_seed=offline_bundle.val_loaders_by_seed,
            num_classes=offline_bundle.num_classes,
            class_names=offline_bundle.class_names,
            eval_config=offline_probe_eval_config,
            device=device,
            auto_mixed_precision=auto_mixed_precision,
            dense_target_names=offline_bundle.dense_target_names,
            dense_log_per_target=offline_bundle.dense_log_per_target,
            label_exposure=log_label_exposure,
          )
        source_total_s = perf_counter() - source_start
        if log_label_exposure:
          exposure_payload = offline_metrics.get('label_exposure')
          if exposure_payload is None:
            raise ValueError('Offline probe did not return label exposure payload.')
          offline_probe_exposure['sources'][source] = exposure_payload
        offline_best_auc = offline_metrics['best_auc']
        offline_best_auprc = offline_metrics['best_auprc']

        probe_prefix = offline_probe_build_prefix(
          root='offline_probe',
          source=source,
          layer=None,
          multi_source=multi_source)
        best_auprc_prefix = offline_probe_build_prefix(
          root='offline_probe_best_auprc',
          source=source,
          layer=None,
          multi_source=multi_source)
        dense_prefix = offline_probe_build_prefix(
          root='offline_dense_probe',
          source=source,
          layer=None,
          multi_source=multi_source)
        logger.info(
          f'[step {log_step}] {probe_prefix}/best_auc: {offline_best_auc["macro_auc"]:.4f} '
          f'{probe_prefix}/best_auprc: {offline_best_auprc["macro_auprc"]:.4f}')

        offline_timing_logs: dict[str, float | int] = {}
        timings = offline_metrics.get('timings')
        if isinstance(timings, dict):
          offline_timing_logs[f'{probe_prefix}/time/total_s'] = float(timings.get('total_s', 0.0))
          offline_timing_logs[f'{probe_prefix}/time/eval_call_total_s'] = float(source_total_s)
          collect_train = timings.get('collect_train')
          if isinstance(collect_train, dict):
            offline_timing_logs[f'{probe_prefix}/time/collect_train_total_s'] = float(
              collect_train.get('total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_to_device_s'] = float(
              collect_train.get('to_device_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_forward_s'] = float(
              collect_train.get('forward_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_cpu_copy_s'] = float(
              collect_train.get('cpu_copy_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_cat_s'] = float(
              collect_train.get('cat_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_batches'] = int(
              collect_train.get('batches', 0))
            offline_timing_logs[f'{probe_prefix}/time/collect_train_samples'] = int(
              collect_train.get('samples', 0))
          collect_val = timings.get('collect_val')
          if isinstance(collect_val, dict):
            offline_timing_logs[f'{probe_prefix}/time/collect_val_total_s'] = float(
              collect_val.get('total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_to_device_s'] = float(
              collect_val.get('to_device_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_forward_s'] = float(
              collect_val.get('forward_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_cpu_copy_s'] = float(
              collect_val.get('cpu_copy_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_cat_s'] = float(
              collect_val.get('cat_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_batches'] = int(
              collect_val.get('batches', 0))
            offline_timing_logs[f'{probe_prefix}/time/collect_val_samples'] = int(
              collect_val.get('samples', 0))
          classification = timings.get('classification')
          if isinstance(classification, dict):
            offline_timing_logs[f'{probe_prefix}/time/classification_total_s'] = float(
              classification.get('total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_train_steps_s'] = float(
              classification.get('train_steps_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_total_s'] = float(
              classification.get('eval_total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_calls'] = int(
              classification.get('eval_calls', 0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_forward_s'] = float(
              classification.get('eval_forward_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_numpy_s'] = float(
              classification.get('eval_numpy_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_metrics_s'] = float(
              classification.get('eval_metrics_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/classification_eval_pairwise_confusion_s'] = float(
              classification.get('eval_pairwise_confusion_s', 0.0))
          regression = timings.get('regression')
          if isinstance(regression, dict):
            offline_timing_logs[f'{probe_prefix}/time/regression_total_s'] = float(
              regression.get('total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/regression_train_steps_s'] = float(
              regression.get('train_steps_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/regression_eval_total_s'] = float(
              regression.get('eval_total_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/regression_eval_calls'] = int(
              regression.get('eval_calls', 0))
            offline_timing_logs[f'{probe_prefix}/time/regression_eval_forward_s'] = float(
              regression.get('eval_forward_s', 0.0))
            offline_timing_logs[f'{probe_prefix}/time/regression_eval_finalize_s'] = float(
              regression.get('eval_finalize_s', 0.0))

        probe_logs, probe_best_auprc_logs = offline_probe_build_classification_logs(
          probe_prefix=probe_prefix,
          best_auprc_prefix=best_auprc_prefix,
          best_auc=offline_best_auc,
          best_auprc=offline_best_auprc,
          group_to_classes=offline_bundle.group_to_classes,
          feature_dim=int(offline_metrics['feature_dim']),
          train_size=int(offline_metrics['train_size']),
          val_size=int(offline_metrics['val_size']),
          val_num_crops=int(offline_metrics['val_num_crops']),
        )
        offline_probe_logs_combined.update(probe_logs)
        offline_probe_logs_combined.update(probe_best_auprc_logs)

        title_suffix = f' source={source}' if multi_source else ''
        confusion_logs = offline_probe_build_confusion_logs(
          probe_prefix=probe_prefix,
          best_auprc_prefix=best_auprc_prefix,
          best_auc=offline_best_auc,
          best_auprc=offline_best_auprc,
          title_suffix=title_suffix,
          error_context=source,
          timing_logs=offline_timing_logs,
        )
        offline_probe_logs_combined.update(confusion_logs)

        dense_best = offline_metrics.get('dense_best')
        if dense_best is not None:
          dense_logs = offline_probe_build_dense_logs(
            dense_prefix=dense_prefix,
            dense_best=dense_best,
            feature_dim=int(offline_metrics['feature_dim']),
            train_size=int(offline_metrics['train_size']),
            val_size=int(offline_metrics['val_size']),
            val_num_crops=int(offline_metrics['val_num_crops']),
          )
          offline_probe_logs_combined.update(dense_logs)

        offline_probe_logs_combined.update(offline_timing_logs)
        if offline_timing_logs:
          logger.info(
            f'[step {log_step}] {probe_prefix} timing total={offline_timing_logs.get(f"{probe_prefix}/time/total_s", 0.0):.2f}s '
            f'collect_train={offline_timing_logs.get(f"{probe_prefix}/time/collect_train_total_s", 0.0):.2f}s '
            f'collect_val={offline_timing_logs.get(f"{probe_prefix}/time/collect_val_total_s", 0.0):.2f}s '
            f'classification={offline_timing_logs.get(f"{probe_prefix}/time/classification_total_s", 0.0):.2f}s')

    if offline_probe_exposure is not None:
      sources = offline_probe_exposure.get('sources')
      if isinstance(sources, dict) and sources:
        exposure_path = path.join(label_exposure_dir, f'offline_probe_step{log_step}.json')
        with open(exposure_path, 'w', encoding='utf-8') as handle:
          json.dump(offline_probe_exposure, handle, separators=(',', ':'))

    offline_probe_logs_combined['offline_probe/time/eval_and_log_total_s'] = float(
      perf_counter() - total_start)
    wandb.log(offline_probe_logs_combined, step=log_step)

  def _ucr_benchmark_eval_and_log(*, log_step: int) -> None:
    """Run UCR benchmark evaluation and log metrics to W&B at the given step."""
    # Step 1: validate required state.
    if not use_ucr_benchmark or ucr_benchmark_config is None:
      raise ValueError('UCR benchmark requested but not initialized.')
    if not isinstance(log_step, int) or log_step < 0:
      raise ValueError(f'log_step must be a non-negative int, got {log_step!r}')

    # Step 2: select the frozen encoder used for representation extraction.
    ucr_encoder = getattr(original_model, 'target_encoder', None) or original_model.encoder
    if ucr_encoder is None:
      raise ValueError('UCR benchmark requires a model encoder.')

    # Step 2b: if offline probe layer probing is configured, reuse that layer set for UCR.
    ucr_layers_categorical = tuple(config.offline_probe_layers_categorical or ())
    ucr_layers_dense = tuple(config.offline_probe_layers_dense or ())
    ucr_layers_confusion = tuple(config.offline_probe_layers_confusion or ())
    ucr_layers_union = tuple(sorted(set(ucr_layers_categorical) | set(ucr_layers_dense) | set(ucr_layers_confusion)))

    ucr_representation_fn = None
    if ucr_layers_union:
      # Multi-layer UCR evaluation: we must return pooled representations per requested layer.
      if isinstance(original_model, NEPA):
        def ucr_representation_fn(
          x: torch.Tensor,
          *,
          _encoder=ucr_encoder,
          _layers=ucr_layers_union,
        ) -> dict[int, torch.Tensor]:
          vit_layers, _ = _encoder(x)
          return {layer: vit_layers.representation(layer) for layer in _layers}  # each [B, D]
      else:
        if not hasattr(ucr_encoder, 'forward_vit_layers'):
          raise ValueError(
            'UCR multi-layer evaluation requires the encoder to implement forward_vit_layers(x). '
            f'Got model={type(original_model).__name__}, encoder={type(ucr_encoder).__name__}.')

        def ucr_representation_fn(
          x: torch.Tensor,
          *,
          _encoder=ucr_encoder,
          _layers=ucr_layers_union,
        ) -> dict[int, torch.Tensor]:
          vit_layers, _ = _encoder.forward_vit_layers(x)
          return {layer: vit_layers.representation(layer) for layer in _layers}  # each [B, D]

    # Step 3: run the benchmark under the offline-probe RNG isolation context.
    with _offline_probe_rng_context(
      log_step=log_step,
      salt=f'ucr:resize_mode={ucr_benchmark_config.resize_mode}:layers={ucr_layers_union}',
    ):
      ucr_result = ucr_benchmark_eval(
        encoder=ucr_encoder,
        representation_fn=ucr_representation_fn,
        cfg=ucr_benchmark_config,
        patch_size=config.patch_size,
        device=device,
        using_cuda=using_cuda,
        auto_mixed_precision=auto_mixed_precision,
        layers=ucr_layers_union if ucr_layers_union else None,
      )

    # Step 4: log scalars to W&B and persist per-dataset results to disk.
    timings = ucr_result.get('timings')
    if not isinstance(timings, dict):
      raise RuntimeError('UCR benchmark did not return timings; this is a bug.')
    ucr_logs: dict[str, object] = {
      'ucr/mean_test_acc': float(ucr_result['mean_acc']),
      'ucr/num_datasets': int(ucr_result['num_datasets']),
      'ucr/resize_mode': str(ucr_result['resize_mode']),
      'ucr/classifier': str(ucr_result.get('classifier', ucr_benchmark_config.classifier)),
      'ucr/time/total_s': float(timings.get('total_s', 0.0)),
      'ucr/time/load_s': float(timings.get('load_s', 0.0)),
      'ucr/time/embed_s': float(timings.get('embed_s', 0.0)),
      'ucr/time/clf_s': float(timings.get('clf_s', 0.0)),
      'ucr/time/rf_s': float(timings.get('rf_s', 0.0)),
    }
    mean_acc_by_layer = ucr_result.get('mean_acc_by_layer')
    if isinstance(mean_acc_by_layer, dict):
      for layer_str, value in mean_acc_by_layer.items():
        ucr_logs[f'ucr/layer{layer_str}/mean_test_acc'] = float(value)
      best_layer = ucr_result.get('best_layer')
      best_mean_acc = ucr_result.get('best_mean_acc')
      if best_layer is not None:
        ucr_logs['ucr/best_layer'] = int(best_layer)
      if best_mean_acc is not None:
        ucr_logs['ucr/best_mean_test_acc'] = float(best_mean_acc)
    wandb.log(ucr_logs, step=log_step)

    if wandb.run is None:
      raise ValueError('W&B run is not initialized; cannot write UCR benchmark logs.')
    ucr_dir = path.join(wandb.run.dir, 'ucr_benchmark')
    makedirs(ucr_dir, exist_ok=True)
    ucr_path = path.join(ucr_dir, f'ucr_step{log_step}.json')
    with open(ucr_path, 'w', encoding='utf-8') as handle:
      json.dump({'log_step': int(log_step), **ucr_result}, handle, separators=(',', ':'))

  def _offline_eval_and_log(*, log_step: int) -> None:
    """Run all enabled offline evaluators (offline probes and/or UCR benchmark)."""
    if use_offline_probe_datasets:
      _offline_probe_eval_and_log(log_step=log_step)
    if use_ucr_benchmark:
      _ucr_benchmark_eval_and_log(log_step=log_step)

  if offline_probe_eval_only:
    if not use_offline_probe:
      raise ValueError(
        'offline_probe_eval_only=true requires offline probing to be enabled. '
        'Fix: set offline_probe_eval_interval and offline_probe_dataset/source(s).')
    logger.info(
      'offline_probe_eval_only=true: running offline probe evaluation at step=%d and exiting.',
      start_step,
    )
    _offline_eval_and_log(log_step=start_step)
    return

  if use_offline_probe and config.offline_probe_eval_at_step0:
    if start_step == 0:
      _offline_eval_and_log(log_step=0)
    else:
      logger.info(
        f'offline_probe_eval_at_step0=True but start_step={start_step}; '
        'skipping step0 offline probe evaluation.')

  progress = trange(
    start_step,
    end_step,
    initial=start_step,
    total=end_step,
    desc='pretrain',
    dynamic_ncols=True,
  )
  for step in progress:
    log_step = step + 1
    step_start = time()
    log_patch_loss_this_step = log_patch_loss and (log_step % log_patch_loss_interval == 0)
    # update hyperparameters according to schedule
    learning_rate = next(lr_schedule)
    weight_decay = next(wd_schedule)
    update_learning_rate_(optimizer, learning_rate)
    update_weight_decay_(optimizer, weight_decay)
    # forward and backward pass
    batch_losses = {}
    batch_probe_loss = 0.
    patch_loss_accum = None
    batch_label_stats_snapshot = None
    for _ in range(config.gradient_accumulation_steps):
      batch = next(train_iterator)
      if use_online_probe:
        if not isinstance(batch, (tuple, list)):
          raise ValueError(
            'Online-probe batches must be tuples/lists. '
            f'Got {type(batch).__name__}.')
        if len(batch) == 2:
          x, y = batch
        elif len(batch) == 3:
          x, y, _ = batch
        else:
          raise ValueError(
            'Online-probe batches must have length 2 or 3. '
            f'Got len(batch)={len(batch)}.')
        if train_label_exposure_tracker is not None:
          train_label_exposure_tracker.update(y, step=log_step)
        if batch_label_stats_tracker is not None:
          batch_label_stats_snapshot = batch_label_stats_tracker.update(y, step=log_step)
        with auto_mixed_precision:
          if log_patch_loss_this_step:
            losses, rep, patch_loss = model(x, return_representation=True, return_patch_loss=True)
          else:
            losses, rep = model(x, return_representation=True)
          jepa_loss = losses['loss'] / config.gradient_accumulation_steps
          # probe on detached representation
          probe_logits = probe(rep.detach())
          probe_loss = F.binary_cross_entropy_with_logits(probe_logits, y)
          probe_loss = probe_loss / config.gradient_accumulation_steps
        jepa_loss.backward()
        probe_loss.backward()
        for k, v in losses.items():
          batch_losses[k] = batch_losses.get(k, 0.0) + v.item() / config.gradient_accumulation_steps
        if log_patch_loss_this_step:
          patch_loss = patch_loss.detach().cpu()  # [T-1]
          if patch_loss_accum is None:
            patch_loss_accum = patch_loss  # [T-1]
          else:
            patch_loss_accum = patch_loss_accum + patch_loss  # [T-1]
        batch_probe_loss += probe_loss.item()
      else:
        x = batch
        with auto_mixed_precision:
          if log_patch_loss_this_step:
            losses, patch_loss = model(x, return_patch_loss=True)
          else:
            losses = model(x)
          loss = losses['loss'] / config.gradient_accumulation_steps
        loss.backward()
        for k, v in losses.items():
          batch_losses[k] = batch_losses.get(k, 0.0) + v.item() / config.gradient_accumulation_steps
        if log_patch_loss_this_step:
          patch_loss = patch_loss.detach().cpu()  # [T-1]
          if patch_loss_accum is None:
            patch_loss_accum = patch_loss  # [T-1]
          else:
            patch_loss_accum = patch_loss_accum + patch_loss  # [T-1]
    # update weights (compute grad norm before step, clip if configured)
    max_norm = config.gradient_clip if config.gradient_clip > 0 else float('inf')
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    grad_stats = {}
    if config.use_nepa and config.nepa_use_dynamic_chunking:
      encoder = original_model.encoder
      transformer_params = list(encoder.blocks.parameters()) + list(encoder.norm.parameters())
      if encoder.pred_in_proj is not None:
        transformer_params += list(encoder.pred_in_proj.parameters())
        transformer_params += list(encoder.pred_norm.parameters())
        transformer_params += list(encoder.pred_out_proj.parameters())
      grad_stats = {
        'train/grad_norm_local_encoder': _grad_norm_params(encoder.local_encoder.parameters()).item(),
        'train/grad_norm_router': _grad_norm_params(encoder.router.parameters()).item(),
        'train/grad_norm_transformer': _grad_norm_params(transformer_params).item(),
        'train/grad_norm_input_proj': _grad_norm_params(encoder.input_proj.parameters()).item(),
      }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    probe_grad_norm = None
    if use_online_probe:
      probe_grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), float('inf'))
      probe_optimizer.step()
      probe_optimizer.zero_grad(set_to_none=True)
    # finalize train step
    step_end = time()
    step_time = step_end - step_start
    total_loss = batch_losses.get('loss', 0.0)
    postfix = {'loss': f'{total_loss:.4f}', 'lr': f'{learning_rate:.2e}', 'wd': f'{weight_decay:.2e}'}
    log_dict = {
      'train/step_time': step_time,
      'train/learning_rate': learning_rate,
      'train/weight_decay': weight_decay,
      'train/grad_norm': grad_norm.item(),
      **{f'train/{k}': v for k, v in batch_losses.items()},
    }
    log_dict.update(grad_stats)
    if log_patch_loss_this_step:
      patch_loss_mean = patch_loss_accum / config.gradient_accumulation_steps  # [T-1]
      patch_loss_rows = [
        [log_step, patch_index, float(loss_value)]
        for patch_index, loss_value in enumerate(patch_loss_mean.tolist())
      ]
      log_dict['train/patch_loss_table'] = wandb.Table(
        data=patch_loss_rows,
        columns=['step', 'patch_idx', 'loss'])
    if use_online_probe:
      postfix['probe'] = f'{batch_probe_loss:.4f}'
      log_dict['train/probe_loss'] = batch_probe_loss
      log_dict['train/probe_grad_norm'] = probe_grad_norm.item()
    progress.set_postfix(**postfix)
    wandb.log(log_dict, step=log_step)

    if batch_label_stats_tracker is not None and batch_label_stats_snapshot is not None:
      interval = int(config.batch_label_stats_train_log_interval)
      if (log_step % interval) == 0:
        batch_label_stats_snapshots.append(dataclasses.asdict(batch_label_stats_snapshot))
        if batch_label_stats_path is None:
          raise ValueError('batch_label_stats_path is not initialized.')
        with open(batch_label_stats_path, 'w', encoding='utf-8') as handle:
          json.dump(
            {
              'summary': batch_label_stats_tracker.gap_summary(),
              'snapshots': batch_label_stats_snapshots,
            },
            handle,
            separators=(',', ':'),
          )
        wandb.log(
          {
            'batch_label_stats/train/coverage_count': batch_label_stats_snapshot.coverage_count,
            'batch_label_stats/train/rare_label_coverage_count': batch_label_stats_snapshot.rare_label_coverage_count,
            'batch_label_stats/train/rare_positive_instances': batch_label_stats_snapshot.rare_positive_instances,
          },
          step=log_step,
        )

    if train_label_exposure_tracker is not None:
      interval = int(config.label_exposure_train_log_interval)
      if (log_step % interval) == 0:
        snapshot = train_label_exposure_tracker.snapshot(step=log_step)
        train_label_exposure_snapshots.append(dataclasses.asdict(snapshot))
        with open(train_label_exposure_path, 'w', encoding='utf-8') as handle:
          json.dump(
            {
              'class_names': list(train_label_exposure_tracker.class_names),
              'snapshots': train_label_exposure_snapshots,
            },
            handle,
            separators=(',', ':'),
          )
        wandb.log(
          {
            'label_exposure/train/coverage_count': snapshot.coverage_count,
            'label_exposure/train/coverage_fraction': snapshot.coverage_fraction,
          },
          step=log_step,
        )

    # periodic probe validation
    if use_online_probe and (step + 1) % config.probe_eval_interval == 0:
      val_auc, per_class_auc, val_auprc, per_class_auprc = evaluate_probe(
        model=original_model,
        probe=probe,
        val_loader=probe_val_loader,
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        class_names=probe_class_names)
      last_val_step = step + 1
      last_val_auc = val_auc
      last_val_auprc = val_auprc
      last_per_class_auc = per_class_auc
      last_per_class_auprc = per_class_auprc
      logger.info(
        f'[step {step + 1}] val/probe_auc: {val_auc:.4f} val/probe_auprc: {val_auprc:.4f}')
      wandb.log(
        probe_build_wandb_val_metric_logs(
          macro_auc=val_auc,
          per_class_auc=per_class_auc,
          macro_auprc=val_auprc,
          per_class_auprc=per_class_auprc,
          group_to_classes=probe_group_to_classes),
        step=step + 1)

    should_checkpoint = (
      ((log_step % config.checkpoint_interval) == 0) or (log_step == end_step)
    )
    if should_checkpoint:
      chkpt_dict = {
        'model': original_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': _config_to_checkpoint_payload(config),
        'step': step + 1,
      }
      if use_online_probe:
        chkpt_dict['probe'] = probe.state_dict()
        chkpt_dict['probe_optimizer'] = probe_optimizer.state_dict()
      torch.save(chkpt_dict, path.join(out_dir, f'chkpt_{step + 1}.pt'))

    if use_offline_probe and (step + 1) % config.offline_probe_eval_interval == 0:
      _offline_eval_and_log(log_step=log_step)

  final_val_auc = None
  final_val_auprc = None
  if use_online_probe:
    if last_val_step == end_step:
      final_val_auc = last_val_auc
      final_val_auprc = last_val_auprc
    else:
      final_val_auc, per_class_auc, final_val_auprc, per_class_auprc = evaluate_probe(
        model=original_model,
        probe=probe,
        val_loader=probe_val_loader,
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        class_names=probe_class_names)
      last_per_class_auc = per_class_auc
      last_per_class_auprc = per_class_auprc
      logger.info(
        f'[final] val/probe_auc: {final_val_auc:.4f} val/probe_auprc: {final_val_auprc:.4f}')
      wandb.log(
        probe_build_wandb_val_metric_logs(
          macro_auc=final_val_auc,
          per_class_auc=last_per_class_auc,
          macro_auprc=final_val_auprc,
          per_class_auprc=last_per_class_auprc,
          group_to_classes=probe_group_to_classes),
        step=end_step)

  wandb.finish()
  return final_val_auc, final_val_auprc


def evaluate_probe(model, probe, val_loader, device, auto_mixed_precision, class_names):
  """Evaluate probe on validation set and return macro + per-class AUC/AUPRC."""
  model.eval()
  probe.eval()
  all_logits = []
  all_targets = []
  with torch.inference_mode():
    for batch in val_loader:
      x, y = batch
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      with auto_mixed_precision:
        # Get representation from encoder (CLS token at index 0 for ViT-style encoders).
        if isinstance(model, NEPA):
          vit_layers, _ = model.encoder(x)
          rep = vit_layers.representation(vit_layers.depth)  # [B, D]
        else:
          h = model.encoder(x)
          if isinstance(h, torch.Tensor):
            rep = h[:, 0] if h.dim() == 3 else h
          elif isinstance(h, (tuple, list)):
            rep_candidate = None
            token_candidate = None
            for item in h:
              if not isinstance(item, torch.Tensor):
                continue
              if item.dim() == 2:
                rep_candidate = item
              elif item.dim() == 3 and token_candidate is None:
                token_candidate = item
            if rep_candidate is not None:
              rep = rep_candidate
            elif token_candidate is not None:
              rep = token_candidate[:, 0]
            else:
              raise ValueError(
                'Failed to extract a representation tensor from encoder outputs. '
                f'Got items={[type(item).__name__ for item in h]}')
          else:
            raise ValueError(
              'Failed to extract a representation tensor from encoder outputs. '
              f'Got type={type(h).__name__}.')
        logits = probe(rep)
      all_logits.append(logits.cpu())
      all_targets.append(y.cpu())
  model.train()
  probe.train()
  predictions = torch.cat(all_logits).float().sigmoid().numpy()
  targets = torch.cat(all_targets).float().numpy()
  return probe_compute_metrics(
    targets=targets,
    predictions=predictions,
    class_names=class_names)


if __name__ == '__main__':
  main()
