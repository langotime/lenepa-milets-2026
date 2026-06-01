from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def hydra_get_config_dict(cfg: DictConfig, runtime_keys: set[str]) -> dict:
  if not isinstance(cfg, DictConfig):
    raise ValueError(f'cfg must be a DictConfig, got {type(cfg)}')
  if not runtime_keys:
    raise ValueError('runtime_keys must be a non-empty set of config keys to exclude')
  cfg_dict = OmegaConf.to_container(cfg, resolve=True)
  if not isinstance(cfg_dict, dict):
    raise ValueError(f'cfg must convert to dict, got {type(cfg_dict)}')
  return {key: value for key, value in cfg_dict.items() if key not in runtime_keys}


def hydra_get_data_mapping(data_cfg: DictConfig | dict) -> dict[str, str]:
  if isinstance(data_cfg, DictConfig):
    data_dict = OmegaConf.to_container(data_cfg, resolve=True)
  elif isinstance(data_cfg, dict):
    data_dict = dict(data_cfg)
  else:
    raise ValueError(
      f"data must be a dict mapping dataset names to file paths, got: {type(data_cfg)}")

  if not isinstance(data_dict, dict):
    raise ValueError(f"data must be a dict mapping dataset names to file paths, got: {type(data_dict)}")

  for key, value in data_dict.items():
    if not isinstance(key, str) or not isinstance(value, str):
      raise ValueError(
        f"data must map dataset names to file path strings, got {key!r}: {value!r}")

  return data_dict
