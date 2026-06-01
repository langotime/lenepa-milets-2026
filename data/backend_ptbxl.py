"""PTB-XL and dump-based data backends for pretraining.

This module provides data loading backends for:
- Generic unlabeled dump files (.npy/.npz) from any supported dataset
- PTB-XL dataset with labeled probe support for online/offline evaluation

Both backends load preprocessed ECG data from numpy dump files, apply
normalization and resampling, and return PretrainDataBundle instances.
"""
from __future__ import annotations

from os import path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

import configs
from configs.hydra_utils import hydra_get_data_mapping
from data import utils as datautils
from data.backend_base import (
  PROBE_TASKS,
  PretrainDataBundle,
  cycle,
  map_to_device,
  prefetch_batch,
)
from data.datasets import DATASETS, PTB_XL
from data.preprocess import PreprocessECG, TransformECG
from data.utils import DatasetRouter, TensorDataset, VariableTensorDataset
from utils.seeding import derive_seed, seed_worker_init_fn, torch_generator
from utils.probe_metrics import probe_build_auc_groups


def _resolve_dataloader_base_seed(config: configs.pretrain.Config) -> int | None:
  """Return the base seed used for DataLoader order (legacy `seed` or split `seed_data`)."""
  if config.seed is not None:
    return int(config.seed)
  if config.seed_data is not None:
    return int(config.seed_data)
  return None


def build_dump_unlabeled_backend(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
  num_cpus: int,
) -> PretrainDataBundle:
  """Build a data backend for unlabeled dump files.

  Loads ECG data from .npy or .npz dump files without labels. Supports multiple
  datasets with weighted sampling via DatasetRouter.

  Args:
    cfg: Hydra config containing 'data' mapping of dataset names to dump file paths.
    config: Pretrain config with dataset weights, sampling frequency, channels, etc.
    device: Target device for training tensors.
    using_cuda: Whether to use CUDA optimizations (non-blocking transfers).
    num_cpus: Number of CPU workers for parallel data loading.

  Returns:
    PretrainDataBundle with unlabeled training iterator. All probe fields are None.

  Raises:
    ValueError: If data config is missing, dataset is unknown, or dump file not found.
  """
  if config.ptbxl_train_sampling != 'uniform':
    raise ValueError(
      'ptbxl_train_sampling is only supported for PTB-XL probe mode (probe_source="ptb-xl"). '
      f'Got ptbxl_train_sampling={config.ptbxl_train_sampling!r}')

  if not cfg.get('data'):
    raise ValueError(
      "Missing required config key: 'data' for data_backend=\"dump\". "
      "Provide via CLI: '+data.DATASET=/path/to/dump.npy'")

  dump_files = hydra_get_data_mapping(cfg.data)

  datasets: dict[str, tuple[TensorDataset | VariableTensorDataset, float]] = {}
  for dataset_name, weight in config.datasets.items():
    if dataset_name not in DATASETS:
      raise ValueError(f'Unknown dataset {dataset_name}. Available datasets are {list(DATASETS)}')
    if dataset_name not in dump_files:
      raise ValueError(
        f"Missing {dataset_name} dataset in 'data' config. Provide via CLI: '+data.{dataset_name}=/path/to/dump.npy'")

    dump_file = dump_files[dataset_name]
    if not path.isfile(dump_file):
      raise ValueError(f'Dataset does not exist {dump_file}')
    _, ext = path.splitext(dump_file)
    if ext not in ('.npy', '.npz'):
      raise ValueError(f'Unsupported dataset format: {dump_file}')

    dataset_cls = DATASETS[dataset_name]
    resample_ratio = config.sampling_frequency / dataset_cls.sampling_frequency
    channel_order = datautils.get_channel_order(dataset_cls.channels, config.channels)
    mean = np.array([dataset_cls.mean], dtype=np.float16)  # [1, C]
    std = np.array([dataset_cls.std], dtype=np.float16)  # [1, C]

    if ext == '.npy':
      dataset = TensorDataset(
        data=datautils.load_data_dump(
          dump_file=dump_file,
          transform=PreprocessECG(
            mean_std=(mean, std),
            resample_ratio=resample_ratio,
            channel_order=channel_order,
            normalize=config.data_preprocess_normalize),
          processes=num_cpus),
        transform=TransformECG(
          crop_size=config.channel_size,
          end_truncate=config.end_truncate))
    elif ext == '.npz':
      dataset = VariableTensorDataset(
        *datautils.load_packed_variable_data_dump(
          dump_file=dump_file,
          min_channel_size=config.channel_size + config.end_truncate,
          transform=PreprocessECG(
            mean_std=(mean, std),
            resample_ratio=resample_ratio,
            channel_order=channel_order,
            normalize=config.data_preprocess_normalize),
          processes=num_cpus),
        transform=TransformECG(
          crop_size=config.channel_size,
          end_truncate=config.end_truncate))
    else:
      raise ValueError(f'Unsupported dataset format: {dump_file}')

    datasets[dataset_name] = (dataset, float(weight))

  dataloader_generator = None
  worker_init_fn = None
  base_seed = _resolve_dataloader_base_seed(config)
  if base_seed is not None:
    loader_seed = derive_seed(seed=base_seed, salt='dump_train_loader')
    dataloader_generator = torch_generator(loader_seed)
    worker_init_fn = seed_worker_init_fn

  train_loader = DataLoader(
    dataset=DatasetRouter(datasets.values()),
    batch_size=config.batch_size,
    pin_memory=using_cuda,
    num_workers=2,
    generator=dataloader_generator,
    worker_init_fn=worker_init_fn)
  train_iterator = iter(train_loader)
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


def build_ptbxl_probe_backend(
  *,
  cfg: DictConfig,
  config: configs.pretrain.Config,
  device: torch.device,
  using_cuda: bool,
  num_cpus: int,
) -> PretrainDataBundle:
  """Build a data backend for PTB-XL with labeled probe support.

  Loads PTB-XL ECG data with multi-label classification targets for online
  and offline probe evaluation. Uses stratified folds (1-8 train, 9 val).

  Supports different probe tasks:
  - 'all': All 71 diagnostic labels
  - 'diagnostic': Diagnostic class labels
  - 'subdiagnostic': Subdiagnostic class labels
  - 'superdiagnostic': Superdiagnostic class labels
  - 'form': Form-based labels
  - 'rhythm': Rhythm-based labels

  Args:
    cfg: Hydra config with 'ptb_xl_data_dir' and 'data.ptb-xl' dump path.
    config: Pretrain config with probe_task, offline_probe_task, etc.
    device: Target device for training tensors.
    using_cuda: Whether to use CUDA optimizations (non-blocking transfers).
    num_cpus: Number of CPU workers for parallel data loading.

  Returns:
    PretrainDataBundle with labeled training iterator and probe data loaders.

  Raises:
    ValueError: If PTB-XL is not the only dataset, probe_task is invalid,
      ptb_xl_data_dir is missing, or dump file not found.
  """
  if set(config.datasets) != {'ptb-xl'}:
    raise ValueError(
      'PTB-XL probe mode requires PTB-XL as the only dataset. '
      f'Got datasets={sorted(config.datasets)}')

  if config.probe_task not in PROBE_TASKS:
    raise ValueError(f'probe_task must be one of {PROBE_TASKS}, got {config.probe_task}')

  ptb_xl_data_dir = cfg.get('ptb_xl_data_dir')
  if not ptb_xl_data_dir:
    raise ValueError(
      "Missing required config key: 'ptb_xl_data_dir' when probe_source=\"ptb-xl\". "
      "Provide via CLI: +ptb_xl_data_dir=/path/to/ptbxl")
  if not path.isdir(ptb_xl_data_dir):
    raise ValueError(f'ptb_xl_data_dir does not exist: {ptb_xl_data_dir}')

  if not cfg.get('data'):
    raise ValueError(
      "Missing required config key: 'data' for data_backend=\"dump\". "
      "Provide via CLI: '+data.ptb-xl=/path/to/ptbxl_dump.npy'")
  dump_files = hydra_get_data_mapping(cfg.data)
  if 'ptb-xl' not in dump_files:
    raise ValueError(
      "Missing ptb-xl dataset in 'data' config. Provide via CLI: '+data.ptb-xl=/path/to/dump.npy'")
  dump_file = dump_files['ptb-xl']
  if not path.isfile(dump_file):
    raise ValueError(f'Dataset does not exist {dump_file}')

  resample_ratio = config.sampling_frequency / PTB_XL.sampling_frequency
  channel_order = datautils.get_channel_order(PTB_XL.channels, config.channels)
  mean = np.array([PTB_XL.mean], dtype=np.float16)  # [1, C]
  std = np.array([PTB_XL.std], dtype=np.float16)  # [1, C]

  labels_df_raw = PTB_XL.load_raw_labels(ptb_xl_data_dir)

  x_full_raw = datautils.load_data_dump(
    dump_file=dump_file,
    transform=PreprocessECG(
      mean_std=(mean, std),
      resample_ratio=resample_ratio,
      channel_order=channel_order,
      normalize=config.data_preprocess_normalize),
    processes=num_cpus)

  if len(x_full_raw) != len(labels_df_raw):
    raise ValueError(
      f'Dump size ({len(x_full_raw)}) does not match labels size ({len(labels_df_raw)}). '
      f'Regenerate the full PTB-XL dump using: python -m scripts.dump_data --data-dir "{ptb_xl_data_dir}"')

  labels_df_online = PTB_XL.compute_label_aggregations(
    labels_df_raw.copy(), ptb_xl_data_dir, config.probe_task)
  x_online, labels_df_online, y_online_np, mlb_online = PTB_XL.select_data(
    x_full_raw, labels_df_online, config.probe_task, min_samples=0)
  y_online = torch.from_numpy(y_online_np).float()  # [N, C]
  num_classes = int(y_online.shape[1])
  probe_class_names = [str(name) for name in mlb_online.classes_]
  if len(probe_class_names) != num_classes:
    raise ValueError(
      f'PTB-XL probe class count mismatch: {len(probe_class_names)} class names vs '
      f'{num_classes} label columns. Check label binarizer alignment.')
  if config.probe_task == 'all' and len(probe_class_names) != 71:
    raise ValueError(
      f'Expected 71 classes for probe_task="all", got {len(probe_class_names)}. '
      'Ensure the full PTB-XL label set is used.')

  _, probe_group_to_classes = probe_build_auc_groups(
    ptb_xl_data_dir=ptb_xl_data_dir,
    class_names=probe_class_names)

  train_mask = (labels_df_online.strat_fold >= 1) & (labels_df_online.strat_fold <= 8)
  val_mask = labels_df_online.strat_fold == 9
  # NOTE: pandas may return a non-writeable NumPy view; PyTorch warns when it
  # converts such arrays (e.g. for boolean indexing). We only use these as masks,
  # so a cheap copy avoids the warning and keeps behavior unchanged.
  train_mask = train_mask.to_numpy(copy=True)
  val_mask = val_mask.to_numpy(copy=True)

  train_dataset = TensorDataset(
    data=x_online[train_mask],
    labels=y_online[train_mask],
    transform=TransformECG(crop_size=config.channel_size, end_truncate=config.end_truncate))
  val_dataset = TensorDataset(
    data=x_online[val_mask],
    labels=y_online[val_mask],
    transform=TransformECG(crop_size=config.channel_size, end_truncate=config.end_truncate))

  train_class_positive_counts = (y_online[train_mask] > 0).to(torch.int64).sum(dim=0)  # [C]
  train_class_positive_counts_list = [
    int(value) for value in train_class_positive_counts.cpu().tolist()
  ]

  train_loader_generator = None
  val_loader_generator = None
  worker_init_fn = None
  base_seed = _resolve_dataloader_base_seed(config)
  if base_seed is not None:
    train_seed = derive_seed(seed=base_seed, salt='ptbxl_train_loader')
    val_seed = derive_seed(seed=base_seed, salt='ptbxl_val_loader')
    train_loader_generator = torch_generator(train_seed)
    val_loader_generator = torch_generator(val_seed)
    worker_init_fn = seed_worker_init_fn

  sampler = None
  shuffle = True
  if config.ptbxl_train_sampling == 'weighted_rare':
    alpha = config.ptbxl_train_sampling_alpha
    if alpha is None:
      raise ValueError(
        'ptbxl_train_sampling_alpha is required when ptbxl_train_sampling="weighted_rare" '
        '(no silent defaults).')
    y_train = y_online[train_mask]  # [N_train, C]
    class_pos_counts = (y_train > 0).to(torch.int64).sum(dim=0)  # [C]
    class_weights = (class_pos_counts + 1).to(torch.float32).pow(-float(alpha))  # [C]
    sample_weights = (y_train > 0).to(torch.float32) @ class_weights  # [N_train]
    if bool((sample_weights <= 0).any().item()):
      raise ValueError(
        'ptbxl_train_sampling="weighted_rare" produced non-positive sample weights. '
        'Fix: ensure PTB-XL labels are multi-hot with at least one positive per sample.')
    sampler = WeightedRandomSampler(
      weights=sample_weights.tolist(),
      num_samples=len(train_dataset),
      replacement=True,
      generator=train_loader_generator,
    )
    shuffle = False

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=shuffle,
    sampler=sampler,
    drop_last=True,
    pin_memory=using_cuda,
    num_workers=2,
    generator=train_loader_generator,
    worker_init_fn=worker_init_fn)
  probe_val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=using_cuda,
    num_workers=2,
    generator=val_loader_generator,
    worker_init_fn=worker_init_fn)

  train_iterator = cycle(train_loader)
  train_iterator = map_to_device(
    data_iterator=train_iterator,
    device=device,
    using_cuda=using_cuda,
    labeled=True,
  )
  train_iterator = prefetch_batch(train_iterator)

  offline_probe_train_loader = None
  offline_probe_val_loader = None
  offline_probe_num_classes = None
  offline_probe_class_names = None
  offline_probe_group_to_classes = None
  if (config.offline_probe_eval_interval is not None
      and 'ptb-xl' in config.offline_probe_sources_resolved):
    offline_task = config.offline_probe_task
    if offline_task is None:
      raise ValueError('offline_probe_task is required when offline probe is enabled')

    if offline_task == config.probe_task:
      offline_x = x_online
      offline_labels_df = labels_df_online
      offline_y = y_online
      offline_mlb = mlb_online
    else:
      offline_labels_df = PTB_XL.compute_label_aggregations(
        labels_df_raw.copy(), ptb_xl_data_dir, offline_task)
      offline_x, offline_labels_df, offline_y_np, offline_mlb = PTB_XL.select_data(
        x_full_raw, offline_labels_df, offline_task, min_samples=0)
      offline_y = torch.from_numpy(offline_y_np).float()  # [N, C]

    offline_probe_num_classes = int(offline_y.shape[1])
    offline_probe_class_names = [str(name) for name in offline_mlb.classes_]
    if len(offline_probe_class_names) != offline_probe_num_classes:
      raise ValueError(
        f'PTB-XL offline probe class count mismatch: {len(offline_probe_class_names)} class names vs '
        f'{offline_probe_num_classes} label columns. Check label binarizer alignment.')
    if offline_task == 'all' and len(offline_probe_class_names) != 71:
      raise ValueError(
        f'Expected 71 classes for offline_probe_task="all", got {len(offline_probe_class_names)}. '
        'Ensure the full PTB-XL label set is used.')
    _, offline_probe_group_to_classes = probe_build_auc_groups(
      ptb_xl_data_dir=ptb_xl_data_dir,
      class_names=offline_probe_class_names)

    offline_train_mask = (offline_labels_df.strat_fold >= 1) & (offline_labels_df.strat_fold <= 8)
    offline_val_mask = offline_labels_df.strat_fold == 9
    offline_train_mask = offline_train_mask.to_numpy()
    offline_val_mask = offline_val_mask.to_numpy()

    offline_train_dataset = TensorDataset(
      data=offline_x[offline_train_mask],
      labels=offline_y[offline_train_mask],
      transform=TransformECG(crop_size=None, end_truncate=config.end_truncate))
    offline_val_dataset = TensorDataset(
      data=offline_x[offline_val_mask],
      labels=offline_y[offline_val_mask],
      transform=TransformECG(crop_size=None, end_truncate=config.end_truncate))

    offline_train_loader_generator = None
    offline_val_loader_generator = None
    offline_worker_init_fn = None
    base_seed = _resolve_dataloader_base_seed(config)
    if base_seed is not None:
      offline_train_seed = derive_seed(seed=base_seed, salt='ptbxl_offline_probe_train_loader')
      offline_val_seed = derive_seed(seed=base_seed, salt='ptbxl_offline_probe_val_loader')
      offline_train_loader_generator = torch_generator(offline_train_seed)
      offline_val_loader_generator = torch_generator(offline_val_seed)
      offline_worker_init_fn = seed_worker_init_fn

    offline_probe_train_loader = DataLoader(
      dataset=offline_train_dataset,
      batch_size=config.offline_probe_batch_size,
      shuffle=False,
      drop_last=False,
      pin_memory=using_cuda,
      num_workers=2,
      generator=offline_train_loader_generator,
      worker_init_fn=offline_worker_init_fn)
    offline_probe_val_loader = DataLoader(
      dataset=offline_val_dataset,
      batch_size=config.offline_probe_batch_size,
      shuffle=False,
      drop_last=False,
      pin_memory=using_cuda,
      num_workers=2,
      generator=offline_val_loader_generator,
      worker_init_fn=offline_worker_init_fn)

  return PretrainDataBundle(
    train_iterator=train_iterator,
    labeled=True,
    probe_val_loader=probe_val_loader,
    num_classes=num_classes,
    probe_class_names=probe_class_names,
    probe_group_to_classes=probe_group_to_classes,
    offline_probe_train_loader=offline_probe_train_loader,
    offline_probe_val_loader=offline_probe_val_loader,
    offline_probe_num_classes=offline_probe_num_classes,
    offline_probe_class_names=offline_probe_class_names,
    offline_probe_group_to_classes=offline_probe_group_to_classes,
    offline_dense_probe_target_names=None,
    offline_dense_probe_log_per_target=None,
    train_class_positive_counts=train_class_positive_counts_list,
  )
