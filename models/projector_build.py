"""Projector builder (MLP vs InvertibleFlow).

This module centralizes projector construction so JEPA/NEPA/LeJEPA stay in sync.
"""

from __future__ import annotations

from torch import nn

import configs
from models.invertible_flow import C_InvertibleFlow
from models.modules import ProjectorMLP

__all__ = ["projector_build"]


def projector_build(config: configs.pretrain.Config) -> nn.Module | None:
  """Build the embedding projector defined by `config`.

  Semantics:
    - `use_projector=false` -> `None`
    - `proj_backend="mlp"` -> current `ProjectorMLP(..., norm_layer=BatchNorm1d)`
    - `proj_backend="invertible_flow"` -> `Sequential(Linear/Id, C_InvertibleFlow, Linear/Id)`

  Shapes:
    - Input:  `x[..., dim]`
    - Output:
      - `x[..., proj_dim]` when `proj_out_dim=null`
      - `x[..., proj_out_dim]` when `proj_out_dim` is set
  """
  if not config.use_projector:
    return None

  if config.proj_backend == "mlp":
    hidden_channels = [config.proj_hidden_dim] * config.proj_num_layers + [config.proj_dim]
    if config.proj_out_dim is not None:
      hidden_channels = [*hidden_channels, config.proj_out_dim]
    return ProjectorMLP(config.dim, hidden_channels, norm_layer=nn.BatchNorm1d)

  if config.proj_backend == "invertible_flow":
    in_proj: nn.Module
    if config.proj_dim != config.dim:
      in_proj = nn.Linear(config.dim, config.proj_dim)
    else:
      in_proj = nn.Identity()

    out_proj: nn.Module
    if config.proj_out_dim is not None:
      out_proj = nn.Linear(config.proj_dim, config.proj_out_dim)
    else:
      out_proj = nn.Identity()

    flow = C_InvertibleFlow(
      dim=config.proj_dim,
      n_layers=config.proj_flow_n_layers,
      hidden_dim=config.proj_flow_hidden_dim,
      n_blocks=config.proj_flow_n_blocks,
      s_max=config.proj_flow_s_max,
      mask_mode=config.proj_flow_mask_mode,
      permute_mode=config.proj_flow_permute_mode,
      permute_seed=config.proj_flow_permute_seed,
    )
    return nn.Sequential(in_proj, flow, out_proj)

  raise ValueError(f"Unknown proj_backend: {config.proj_backend!r}")
