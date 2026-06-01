import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

import configs
from losses.rectified_lpjepa import (
  diag_update_for_features as rdmreg_diag_update_for_features,
  determine_sigma_for_lp_dist,
  sample_unit_sphere_projections,
  swd_mean_over_views as rdmreg_swd_mean_over_views,
)
from losses.wristband_gaussian import C_WristbandGaussianLoss
from models.modules import PatchEmbedding, Block
from models.mantis_tokgen import MantisTokGen1D, MultiScaledScalarEncoder
from losses.sigreg import SIGReg
from models.projector_build import projector_build
from models.utils import get_1d_pos_embed, get_learned_pos_embed
from models.vit_layer_outputs import ViTLayerOutputs, _masked_batch_norm_features


def _masked_mean(
  loss_terms: torch.Tensor,
  token_mask: torch.Tensor | None,
) -> torch.Tensor:
  """Compute mean loss over batch/time with optional token mask."""
  if token_mask is None:
    return loss_terms.mean(dtype=torch.float32)  # scalar
  if token_mask.dtype is not torch.bool:
    raise ValueError(f'token_mask must be bool, got {token_mask.dtype}')
  if token_mask.dim() != 2:
    raise ValueError(f'token_mask must be [B, T], got {tuple(token_mask.shape)}')
  if loss_terms.dim() not in (2, 3):
    raise ValueError(f'loss_terms must be [B, T] or [B, T, D], got {tuple(loss_terms.shape)}')
  if loss_terms.size(0) != token_mask.size(0) or loss_terms.size(1) != token_mask.size(1):
    raise ValueError(
      'token_mask must match loss_terms first two dims, '
      f'loss_terms.shape={tuple(loss_terms.shape)}, token_mask.shape={tuple(token_mask.shape)}')
  if loss_terms.dim() == 3:
    masked = loss_terms * token_mask.unsqueeze(-1)  # [B, T, D]
    total = masked.sum(dtype=torch.float32)  # scalar
    count = token_mask.sum(dtype=torch.float32) * loss_terms.size(-1)  # scalar
  else:
    masked = loss_terms * token_mask  # [B, T]
    total = masked.sum(dtype=torch.float32)  # scalar
    count = token_mask.sum(dtype=torch.float32)  # scalar
  if count <= 0:
    raise ValueError('token_mask must include at least one valid element')
  return total / count  # scalar


def _masked_patch_loss(
  loss_terms: torch.Tensor,
  token_mask: torch.Tensor | None,
) -> torch.Tensor:
  """Compute per-position loss with optional token mask."""
  if token_mask is None:
    if loss_terms.dim() == 3:
      return loss_terms.detach().mean(dim=(0, 2), dtype=torch.float32)  # [T]
    return loss_terms.detach().mean(dim=0, dtype=torch.float32)  # [T]
  if token_mask.dtype is not torch.bool:
    raise ValueError(f'token_mask must be bool, got {token_mask.dtype}')
  if token_mask.dim() != 2:
    raise ValueError(f'token_mask must be [B, T], got {tuple(token_mask.shape)}')
  if loss_terms.dim() not in (2, 3):
    raise ValueError(f'loss_terms must be [B, T] or [B, T, D], got {tuple(loss_terms.shape)}')
  if loss_terms.size(0) != token_mask.size(0) or loss_terms.size(1) != token_mask.size(1):
    raise ValueError(
      'token_mask must match loss_terms first two dims, '
      f'loss_terms.shape={tuple(loss_terms.shape)}, token_mask.shape={tuple(token_mask.shape)}')
  if loss_terms.dim() == 3:
    masked = loss_terms * token_mask.unsqueeze(-1)  # [B, T, D]
    sum_per_pos = masked.sum(dim=(0, 2), dtype=torch.float32)  # [T]
    count_per_pos = token_mask.sum(dim=0, dtype=torch.float32) * loss_terms.size(-1)  # [T]
  else:
    masked = loss_terms * token_mask  # [B, T]
    sum_per_pos = masked.sum(dim=0, dtype=torch.float32)  # [T]
    count_per_pos = token_mask.sum(dim=0, dtype=torch.float32)  # [T]
  denom = count_per_pos.clamp_min(1.0)  # [T]
  patch_loss = sum_per_pos / denom  # [T]
  nan_fill = torch.full_like(patch_loss, float('nan'))  # [T]
  return torch.where(count_per_pos > 0, patch_loss, nan_fill)  # [T]


def _masked_mean_or_nan(
  values: torch.Tensor,
  mask: torch.Tensor,
) -> torch.Tensor:
  """Compute masked mean or return NaN if the mask is empty."""
  if mask.dtype is not torch.bool:
    raise ValueError(f'mask must be bool, got {mask.dtype}')
  if values.shape != mask.shape:
    raise ValueError(
      f'mask must match values shape, values.shape={tuple(values.shape)}, mask.shape={tuple(mask.shape)}')
  if not mask.any():
    return torch.full((), float('nan'), device=values.device, dtype=torch.float32)
  return values[mask].to(dtype=torch.float32).mean()


def _masked_embedding_distribution_stats(
  values: torch.Tensor,
  mask: torch.Tensor | None,
  *,
  prefix: str,
) -> dict[str, torch.Tensor]:
  """Compute mask-aware distribution diagnostics for embedding tensors.

  Args:
    values: [A, B, D] embedding tensor.
    mask: Optional [A, B] boolean mask selecting valid tokens.
    prefix: Prefix for stats keys.

  Returns:
    Dict with scalar float32 stats: token_mean_abs_mean, token_std_mean, token_rms_mean,
    global_mean, and global_std (all keyed by the provided prefix).
  """
  # Step 1: validate shapes.
  if values.dim() != 3:
    raise ValueError(f'values must be [A, B, D], got {tuple(values.shape)}')
  if mask is not None:
    if mask.dtype is not torch.bool:
      raise ValueError(f'mask must be bool, got {mask.dtype}')
    if mask.shape != values.shape[:2]:
      raise ValueError(
        'mask must match values first two dims, '
        f'values.shape={tuple(values.shape)}, mask.shape={tuple(mask.shape)}')

  # Step 2: compute per-token stats over the feature dimension.
  values_f32 = values.to(dtype=torch.float32)  # [A, B, D]
  token_mean = values_f32.mean(dim=-1)  # [A, B]
  token_var = values_f32.var(dim=-1, unbiased=False)  # [A, B]
  token_std = token_var.sqrt()  # [A, B]
  token_rms = values_f32.square().mean(dim=-1).sqrt()  # [A, B]

  # Step 3: reduce per-token stats with an optional mask.
  if mask is None:
    token_mean_abs_mean = token_mean.abs().mean()  # []
    token_std_mean = token_std.mean()  # []
    token_rms_mean = token_rms.mean()  # []
    flat = values_f32.reshape(-1)  # [A*B*D]
    global_mean = flat.mean()  # []
    global_std = flat.std(unbiased=False)  # []
  else:
    token_mean_abs_mean = _masked_mean_or_nan(token_mean.abs(), mask)  # []
    token_std_mean = _masked_mean_or_nan(token_std, mask)  # []
    token_rms_mean = _masked_mean_or_nan(token_rms, mask)  # []
    if mask.any():
      values_valid = values_f32[mask]  # [N, D]
      flat = values_valid.reshape(-1)  # [N*D]
      global_mean = flat.mean()  # []
      global_std = flat.std(unbiased=False)  # []
    else:
      nan = torch.full((), float('nan'), device=values.device, dtype=torch.float32)  # []
      global_mean = nan  # []
      global_std = nan  # []

  # Step 4: package scalars for logging.
  return {
    f'{prefix}_token_mean_abs_mean': token_mean_abs_mean,
    f'{prefix}_token_std_mean': token_std_mean,
    f'{prefix}_token_rms_mean': token_rms_mean,
    f'{prefix}_global_mean': global_mean,
    f'{prefix}_global_std': global_std,
  }


def _flat_stats(
  values: torch.Tensor,
  *,
  prefix: str,
  quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
  include_mean_std: bool = True,
  include_min_max: bool = False,
) -> dict[str, torch.Tensor]:
  """Compute summary stats (mean/std/min/max/quantiles) for a flat tensor."""
  flat = values.reshape(-1).to(dtype=torch.float32)  # [N]
  stats: dict[str, torch.Tensor] = {}
  if flat.numel() == 0:
    nan = torch.full((), float('nan'), device=values.device, dtype=torch.float32)
    if include_mean_std:
      stats[f'{prefix}_mean'] = nan
      stats[f'{prefix}_std'] = nan
    if include_min_max:
      stats[f'{prefix}_min'] = nan
      stats[f'{prefix}_max'] = nan
    for q in quantiles:
      stats[f'{prefix}_p{int(q * 100)}'] = nan
    return stats
  if include_mean_std:
    stats[f'{prefix}_mean'] = flat.mean()
    stats[f'{prefix}_std'] = flat.std(unbiased=False)
  if include_min_max:
    stats[f'{prefix}_min'] = flat.min()
    stats[f'{prefix}_max'] = flat.max()
  if quantiles:
    q_tensor = torch.tensor(quantiles, device=flat.device, dtype=flat.dtype)  # [Q]
    q_values = torch.quantile(flat, q_tensor)  # [Q]
    for idx, q in enumerate(quantiles):
      stats[f'{prefix}_p{int(q * 100)}'] = q_values[idx]
  return stats


def _select_topk_token_indices(
  tokens: torch.Tensor,
  *,
  k: int,
) -> torch.Tensor:
  """Select top-k token indices per sample using cosine dissimilarity boundary score.

  This implements the plan in `plans/53_nepa_static_topk_tokens.md`:

  - compute per-position score $s_t = 0.5 (1 - cos(x_t, x_{t-1}))$ for t>=1
  - always include endpoints {0, L-1}
  - select the remaining K-2 indices by top-k score over interior positions {1..L-2}
  - return indices sorted in ascending time order

  Args:
    tokens: [B, L, D] full-resolution token features.
    k: Number of selected tokens per sample (must satisfy 2 <= k <= L).

  Returns:
    [B, k] int64 tensor of sorted indices in [0, L-1].
  """
  if tokens.dim() != 3:
    raise ValueError(f'tokens must be [B, L, D], got {tuple(tokens.shape)}')
  if not isinstance(k, int):
    raise ValueError(f'k must be int, got {type(k).__name__}')
  B, L, _ = tokens.shape
  if k < 2:
    raise ValueError(f'k must be >= 2, got {k}')
  if k > L:
    raise ValueError(f'k must be <= L={L}, got {k}')
  if L < 2:
    raise ValueError(f'Expected L >= 2 for top-k selection, got L={L}')

  device = tokens.device
  if k == 2:
    start = torch.zeros((B, 1), device=device, dtype=torch.long)  # [B, 1]
    end = torch.full((B, 1), L - 1, device=device, dtype=torch.long)  # [B, 1]
    return torch.cat([start, end], dim=1)  # [B, 2]

  # Step 1: compute interior boundary scores (t=1..L-2).
  cos_sim = F.cosine_similarity(tokens[:, 1:, :], tokens[:, :-1, :], dim=-1)  # [B, L-1]
  score = 0.5 * (1.0 - cos_sim)  # [B, L-1]
  interior_score = score[:, :L - 2]  # [B, L-2] positions 1..L-2

  # Step 2: select top-k (excluding endpoints).
  idx_mid = interior_score.topk(k=k - 2, dim=1).indices.to(dtype=torch.long) + 1  # [B, k-2]

  # Step 3: add endpoints and sort in time.
  start = torch.zeros((B, 1), device=device, dtype=torch.long)  # [B, 1]
  end = torch.full((B, 1), L - 1, device=device, dtype=torch.long)  # [B, 1]
  idx = torch.cat([start, idx_mid, end], dim=1)  # [B, k]
  return idx.sort(dim=1).values  # [B, k]


def _pool_tokens_by_end_indices(
  tokens: torch.Tensor,
  *,
  end_indices: torch.Tensor,
  mode: str,
) -> torch.Tensor:
  """Pool full-resolution tokens into segments defined by per-sample end indices.

  Given tokens $x \\in \\mathbb{R}^{B \\times L \\times D}$ and strictly-increasing end indices
  $e \\in \\{0, \\dots, L-1\\}^{B \\times K}$, define segment boundaries:

  - $s_0 = 0$
  - $s_i = e_{i-1} + 1$ for $i \\ge 1$

  and segments $[s_i, e_i]$ (inclusive). We return:

  - `mode="last"`: $z_i = x_{e_i}$
  - `mode="mean"`: $z_i = \\frac{1}{e_i - s_i + 1} \\sum_{t=s_i}^{e_i} x_t$

  Args:
    tokens: [B, L, D] full-resolution token features.
    end_indices: [B, K] end indices per segment (strictly increasing along K).
    mode: "last" or "mean".

  Returns:
    [B, K, D] pooled segment representations.
  """
  if tokens.dim() != 3:
    raise ValueError(f'tokens must be [B, L, D], got {tuple(tokens.shape)}')
  if end_indices.dim() != 2:
    raise ValueError(f'end_indices must be [B, K], got {tuple(end_indices.shape)}')
  if mode not in ('last', 'mean'):
    raise ValueError(f'mode must be "last" or "mean", got {mode!r}')
  B, _, D = tokens.shape
  if end_indices.size(0) != B:
    raise ValueError(
      'end_indices batch dim must match tokens, '
      f'got end_indices.shape={tuple(end_indices.shape)}, tokens.shape={tuple(tokens.shape)}')
  if end_indices.size(1) < 1:
    raise ValueError(f'end_indices must have K >= 1, got K={end_indices.size(1)}')
  if (end_indices[:, 1:] <= end_indices[:, :-1]).any():
    raise ValueError('end_indices must be strictly increasing along dim=1')

  if mode == 'last':
    return tokens.gather(1, end_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, K, D]

  # Step 1: prefix sums to compute segment sums with 2 gathers.
  tokens_f32 = tokens.to(dtype=torch.float32)
  cumsum = tokens_f32.cumsum(dim=1)  # [B, L, D]

  # Step 2: segment ends.
  end_sum = cumsum.gather(1, end_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, K, D]

  # Step 3: segment starts (inclusive), derived from previous end + 1.
  start = torch.cat(
    [end_indices.new_zeros((B, 1)), end_indices[:, :-1] + 1],
    dim=1)  # [B, K]
  start_prev = (start - 1).clamp_min(0)  # [B, K]
  start_sum = cumsum.gather(1, start_prev.unsqueeze(-1).expand(-1, -1, D))  # [B, K, D]
  start_sum = start_sum.masked_fill((start == 0).unsqueeze(-1), 0.0)  # [B, K, D]

  # Step 4: mean pooling.
  segment_sum = end_sum - start_sum  # [B, K, D]
  segment_len = (end_indices - start + 1).to(dtype=torch.float32)  # [B, K]
  pooled = segment_sum / segment_len.unsqueeze(-1)  # [B, K, D]
  return pooled.to(dtype=tokens.dtype)  # [B, K, D]


def nepa_loss(
  z: torch.Tensor,
  z_hat: torch.Tensor,
  *,
  token_mask: torch.Tensor | None = None,
  loss_type: str = 'mse',
  return_patch_loss: bool = False,
  batch_norm: bool = False,
  batch_norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  """Compute NEPA next-token prediction loss.

  Args:
    z: [B, T, D] target embeddings (stop-grad applied by caller).
    z_hat: [B, T, D] predicted embeddings (already shifted if needed).
    token_mask: [B, T] boolean mask for valid tokens (True = valid).
    loss_type: One of 'mse' or 'cos'.
    return_patch_loss: Whether to return per-patch loss for logging.
    batch_norm: Whether to batch-normalize `pred` and `target` before computing MSE.
    batch_norm_eps: Epsilon for batch normalization.

  Returns:
    loss: Scalar loss value.
    patch_loss: [T-1] per-patch loss (if return_patch_loss=True, else None).
  """
  # NEPA Step 3 (shift + stop-grad + cosine): z/z_hat are [B, T, D].
  target = z  # [B, T, D] target embeddings.
  pred = z_hat[:, :-1, :]  # [B, T-1, D] predict t+1 from <=t.
  target = target[:, 1:, :]  # [B, T-1, D] shift target by one step.
  loss_mask = None
  if token_mask is not None:
    if token_mask.dtype is not torch.bool:
      raise ValueError(f'token_mask must be bool, got {token_mask.dtype}')
    if token_mask.dim() != 2:
      raise ValueError(f'token_mask must be [B, T], got {tuple(token_mask.shape)}')
    if token_mask.size(0) != target.size(0) or token_mask.size(1) != target.size(1) + 1:
      raise ValueError(
        'token_mask must match z shape [B, T], '
        f'z.shape={tuple(z.shape)}, token_mask.shape={tuple(token_mask.shape)}')
    loss_mask = token_mask[:, 1:]  # [B, T-1]

  patch_loss = None

  if loss_type == 'mse':
    if batch_norm:
      if pred.dim() != 3 or target.dim() != 3:
        raise ValueError(
          'Expected pred/target to be [B, T-1, D] for batch_norm, '
          f'got pred.shape={tuple(pred.shape)}, target.shape={tuple(target.shape)}')
      B, Tm1, D = pred.shape
      pred_flat = pred.reshape(-1, D)  # [B*(T-1), D]
      target_flat = target.reshape(-1, D)  # [B*(T-1), D]
      mask_flat = loss_mask.reshape(-1) if loss_mask is not None else None  # [B*(T-1)] or None
      pred_bn = _masked_batch_norm_features(pred_flat, mask_flat, eps=batch_norm_eps)  # [B*(T-1), D]
      target_bn = _masked_batch_norm_features(target_flat, mask_flat, eps=batch_norm_eps)  # [B*(T-1), D]
      pred = pred_bn.reshape(B, Tm1, D)  # [B, T-1, D]
      target = target_bn.reshape(B, Tm1, D)  # [B, T-1, D]
    diff = pred - target  # [B, T-1, D]
    diff_sq = diff.square()  # [B, T-1, D]
    loss = _masked_mean(diff_sq, loss_mask)  # scalar
    if return_patch_loss:
      patch_loss = _masked_patch_loss(diff_sq, loss_mask)  # [T-1]

  elif loss_type == 'cos':
    if batch_norm:
      raise ValueError('batch_norm is only supported for loss_type="mse"')
    pred_norm = F.normalize(pred, dim=-1)  # [B, T-1, D] L2-normalized.
    target_norm = F.normalize(target, dim=-1)  # [B, T-1, D] L2-normalized.
    loss_terms = 1.0 - (pred_norm * target_norm).sum(dim=-1)  # [B, T-1]
    loss = _masked_mean(loss_terms, loss_mask)  # scalar
    if return_patch_loss:
      patch_loss = _masked_patch_loss(loss_terms, loss_mask)  # [T-1]

  else:
    raise ValueError(f'Unknown loss type: {loss_type}')

  return loss, patch_loss


class LocalMambaEncoder(nn.Module):
  """Local full-resolution encoder using stacked Mamba blocks.

  Each layer applies LayerNorm + Mamba with a residual connection:
    h_{l+1} = h_l + Mamba(LN(h_l))
  """

  def __init__(
      self,
      *,
      dim: int,
      num_layers: int,
      d_state: int,
      d_conv: int,
      expand: int,
      norm_eps: float,
      bias: bool,
  ):
    super().__init__()
    if num_layers < 1:
      raise ValueError(f'num_layers must be >= 1, got {num_layers}')
    try:
      from mamba_ssm import Mamba
    except ImportError as exc:
      raise ImportError(
        'mamba-ssm is required for Mamba-based NEPA encoders; '
        'install it with `uv add mamba-ssm`') from exc
    self.layers = nn.ModuleList([
      nn.ModuleDict({
        'norm': nn.LayerNorm(dim, eps=norm_eps, bias=bias),
        'mamba': Mamba(
          d_model=dim,
          d_state=d_state,
          d_conv=d_conv,
          expand=expand),
      })
      for _ in range(num_layers)
    ])

  def forward(
      self,
      x: torch.Tensor,
      *,
      return_layer_stats: bool = False,
      stats_prefix: str | None = None,
  ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply stacked Mamba layers.

    Args:
      x: [B, L, D] full-resolution embeddings.
      return_layer_stats: When True, compute and return per-layer distribution diagnostics
        for the residual-stream outputs `h` after each layer.
      stats_prefix: Prefix used for per-layer stats keys (required when return_layer_stats=True).

    Returns:
      Either:
        - h: [B, L, D] contextualized embeddings, or
        - (h, stats): where `stats` maps `f"{stats_prefix}_layer{idx}_<stat>"` to scalar tensors.
    """
    if return_layer_stats and stats_prefix is None:
      raise ValueError('stats_prefix must be set when return_layer_stats=True')

    h = x  # [B, L, D]
    layer_stats: dict[str, torch.Tensor] | None = {} if return_layer_stats else None

    # Apply LayerNorm -> Mamba -> residual per layer.
    for layer_idx, layer in enumerate(self.layers):
      h_norm = layer['norm'](h)  # [B, L, D]
      h_mamba = layer['mamba'](h_norm)  # [B, L, D]
      h = h + h_mamba  # [B, L, D] residual update.
      if layer_stats is not None:
        with torch.no_grad():
          layer_stats.update(_masked_embedding_distribution_stats(
            h.detach(),
            None,
            prefix=f'{stats_prefix}_layer{layer_idx}'))

    if layer_stats is None:
      return h
    return h, layer_stats


class BoundaryRouterHNet(nn.Module):
  """Boundary router for H-Net dynamic chunking.

  Causal mode:
    p_0 = 1
    p_t = 0.5 * (1 - cos(Wq x_t, Wk x_{t-1}))
  Lookahead mode:
    p_{L-1} = 1
    p_t = 0.5 * (1 - cos(Wq x_{t+1}, Wk x_t))
  Boundaries: b_t = 1[p_t >= tau] with b_0 = b_{L-1} = 1.
  """

  def __init__(
      self,
      *,
      dim: int,
      boundary_threshold: float,
      mode: str,
      bias: bool = True,
  ):
    super().__init__()
    if mode not in ('causal', 'lookahead'):
      raise ValueError(f'mode must be "causal" or "lookahead", got {mode!r}')
    if not 0.0 < boundary_threshold < 1.0:
      raise ValueError(
        f'boundary_threshold must be in (0, 1), got {boundary_threshold}')
    self.mode = mode
    self.boundary_threshold = boundary_threshold
    self.query_proj = nn.Linear(dim, dim, bias=bias)
    self.key_proj = nn.Linear(dim, dim, bias=bias)

  def forward(
      self,
      x_hat: torch.Tensor,
      *,
      return_stats: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute boundary probabilities and hard boundaries.

    Args:
      x_hat: [B, L, D] full-resolution representations.
      return_stats: Whether to return router diagnostics for logging.

    Returns:
      p: [B, L] boundary probabilities.
      b: [B, L] boundary mask (bool, endpoints enforced).
      stats: Optional dict of summary statistics.
    """
    # Step 1: project to query/key spaces.
    B, L, _ = x_hat.shape
    if L < 2:
      raise ValueError(f'BoundaryRouterHNet requires L >= 2, got L={L}')
    q = self.query_proj(x_hat)  # [B, L, D]
    k = self.key_proj(x_hat)  # [B, L, D]
    # Step 2: cosine similarity between consecutive pairs.
    cos_sim = F.cosine_similarity(q[:, 1:, :], k[:, :-1, :], dim=-1)  # [B, L-1]
    p = torch.empty((B, L), device=x_hat.device, dtype=x_hat.dtype)  # [B, L]
    # Step 3: boundary probabilities (causal or lookahead).
    if self.mode == 'causal':
      p[:, 0] = 1.0  # [B]
      p[:, 1:] = 0.5 * (1.0 - cos_sim)  # [B, L-1]
    else:
      p[:, :-1] = 0.5 * (1.0 - cos_sim)  # [B, L-1]
      p[:, -1] = 1.0  # [B]
    # Step 4: threshold into hard boundaries (endpoints enforced).
    b_raw = p >= self.boundary_threshold  # [B, L]
    b = b_raw.clone()  # [B, L]
    b[:, 0] = True  # [B]
    b[:, -1] = True  # [B]
    if not return_stats:
      return p, b
    # Step 5: router diagnostics.
    p_margin = p - self.boundary_threshold  # [B, L]
    stats: dict[str, torch.Tensor] = {}
    stats.update(_flat_stats(p, prefix='nepa_chunk_p'))
    stats.update(_flat_stats(cos_sim, prefix='nepa_chunk_cos_sim'))
    stats.update(_flat_stats(
      p_margin,
      prefix='nepa_chunk_p_margin',
      include_mean_std=False))
    stats['nepa_chunk_p_on_boundaries_mean'] = _masked_mean_or_nan(p, b_raw)
    stats['nepa_chunk_p_off_boundaries_mean'] = _masked_mean_or_nan(p, ~b_raw)
    stats['nepa_chunk_p_ge_tau_mean'] = b_raw.to(dtype=torch.float32).mean()
    return p, b, stats


class Downsampler(nn.Module):
  """Pack variable-length boundary tokens into padded chunks."""

  def __init__(self, *, max_tokens: int):
    super().__init__()
    if max_tokens < 2:
      raise ValueError(f'max_tokens must be >= 2, got {max_tokens}')
    self.max_tokens = max_tokens

  def forward(
      self,
      x_hat: torch.Tensor,
      b: torch.Tensor,
      p: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select boundary tokens and pad to fixed length.

    Args:
      x_hat: [B, L, D] full-resolution representations.
      b: [B, L] boundary mask (bool).
      p: [B, L] boundary probabilities.

    Returns:
      z_padded: [B, M, D] selected tokens padded to M=max_tokens.
      z_mask: [B, M] validity mask for padded tokens.
      idx_padded: [B, M] original indices (pad = -1).
      p_padded: [B, M] boundary probabilities (pad = 0).
    """
    # Validate shapes/dtypes.
    if b.dtype is not torch.bool:
      raise ValueError(f'b must be bool, got {b.dtype}')
    if x_hat.dim() != 3:
      raise ValueError(f'x_hat must be [B, L, D], got {tuple(x_hat.shape)}')
    B, L, D = x_hat.shape
    if b.shape != (B, L):
      raise ValueError(f'b must be [B, L], got {tuple(b.shape)}')
    if p.shape != (B, L):
      raise ValueError(f'p must be [B, L], got {tuple(p.shape)}')
    if not (b[:, 0].all() and b[:, -1].all()):
      raise ValueError('b must include first and last tokens for each sample')

    z_padded = x_hat.new_zeros((B, self.max_tokens, D))  # [B, M, D]
    z_mask = torch.zeros((B, self.max_tokens), device=x_hat.device, dtype=torch.bool)  # [B, M]
    idx_padded = torch.full(
      (B, self.max_tokens), -1, device=x_hat.device, dtype=torch.long)  # [B, M]
    p_padded = p.new_zeros((B, self.max_tokens))  # [B, M]

    # Select indices per sample, then pad to max_tokens.
    for batch_idx in range(B):
      idx = torch.nonzero(b[batch_idx], as_tuple=False).squeeze(1)  # [K]
      if idx.numel() > self.max_tokens:
        idx_start = idx[:1]  # [1]
        idx_end = idx[-1:]  # [1]
        idx_mid = idx[1:-1]  # [K-2] or [0]
        if self.max_tokens > 2 and idx_mid.numel() > 0:
          keep_mid = self.max_tokens - 2
          if idx_mid.numel() > keep_mid:
            p_mid = p[batch_idx, idx_mid]  # [K-2]
            top_idx = torch.topk(p_mid, k=keep_mid, largest=True).indices  # [keep_mid]
            idx_mid = idx_mid[top_idx]  # [keep_mid]
        else:
          idx_mid = idx_mid[:0]  # [0] ensure empty tensor
        idx = torch.cat([idx_start, idx_mid, idx_end], dim=0)  # [M]
        idx, _ = idx.sort()  # [M]

      num_tokens = idx.numel()
      z_padded[batch_idx, :num_tokens] = x_hat[batch_idx, idx]  # [M, D]
      z_mask[batch_idx, :num_tokens] = True  # [M]
      idx_padded[batch_idx, :num_tokens] = idx  # [M]
      p_padded[batch_idx, :num_tokens] = p[batch_idx, idx]  # [M]

    return z_padded, z_mask, idx_padded, p_padded


class NEPAEncoder(nn.Module):
  """NEPA encoder with optional narrow predictor-style tail.

  By default, the encoder is a standard ViT stack of `config.depth` transformer `Block`s operating
  in the embedding space of width `config.dim` with `config.num_heads` attention heads.

  When `config.pred_depth > 0`, the last `pred_depth` blocks are run in a narrower space
  (`pred_dim`, `pred_num_heads`) similarly to a classical JEPA predictor. The narrow tail is
  projected back to `dim` for all returned layer outputs so downstream code keeps a single
  canonical embedding space.
  """
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.use_dynamic_chunking = config.nepa_use_dynamic_chunking
    self.static_tokenizer = config.nepa_static_tokenizer

    if self.use_dynamic_chunking:
      self.patch_embed = None
      self.input_proj = nn.Linear(config.num_channels, config.dim, bias=config.bias)
      self.local_encoder = LocalMambaEncoder(
        dim=config.dim,
        num_layers=config.nepa_local_mamba_layers,
        d_state=config.nepa_local_mamba_d_state,
        d_conv=config.nepa_local_mamba_d_conv,
        expand=config.nepa_local_mamba_expand,
        norm_eps=config.norm_eps,
        bias=config.bias)
      self.router = BoundaryRouterHNet(
        dim=config.dim,
        boundary_threshold=config.nepa_chunk_boundary_threshold,
        mode=config.nepa_chunk_router_mode,
        bias=config.bias)
      self.downsampler = Downsampler(max_tokens=config.nepa_chunk_max_tokens)
      self.patch_positions = None
      num_pos_tokens = config.channel_size
    else:
      if self.static_tokenizer == 'conv_patch_embed':
        if config.channel_size % config.patch_size != 0:
          raise ValueError(
            f'channel_size must be divisible by patch_size, '
            f'got channel_size={config.channel_size}, patch_size={config.patch_size}')
        num_patches = config.channel_size // config.patch_size
        self.nepa_patch_embed_scalar_stats_mode = config.nepa_patch_embed_scalar_stats_mode
        if self.nepa_patch_embed_scalar_stats_mode is None:
          cnn_dim = config.dim
          self.nepa_patch_embed_mean_encoder = None
          self.nepa_patch_embed_std_encoder = None
        else:
          cnn_dim = int(config.nepa_patch_embed_cnn_dim)
          scalar_hidden_dim = int(config.nepa_patch_embed_scalar_hidden_dim)
          scalar_scales = tuple(config.nepa_patch_embed_scalar_scales)
          scalar_epsilon = float(config.nepa_patch_embed_scalar_epsilon)
          self.nepa_patch_embed_mean_encoder = MultiScaledScalarEncoder(
            scales=scalar_scales,
            hidden_dim=scalar_hidden_dim,
            epsilon=scalar_epsilon,
            eps=float(config.norm_eps),
            bias=config.bias)
          self.nepa_patch_embed_std_encoder = MultiScaledScalarEncoder(
            scales=scalar_scales,
            hidden_dim=scalar_hidden_dim,
            epsilon=scalar_epsilon,
            eps=float(config.norm_eps),
            bias=config.bias)
        self.patch_embed = PatchEmbedding(
          dim=cnn_dim,
          in_channels=config.num_channels,
          patch_size=config.patch_size,
          bias=config.bias)
        self.input_proj = None
        self.local_encoder = None
        self.patch_positions = None
        num_pos_tokens = num_patches
      elif self.static_tokenizer == 'mamba_static_patches':
        if config.channel_size % config.patch_size != 0:
          raise ValueError(
            f'channel_size must be divisible by patch_size, '
            f'got channel_size={config.channel_size}, patch_size={config.patch_size}')
        num_patches = config.channel_size // config.patch_size
        self.patch_embed = None
        self.input_proj = nn.Linear(config.num_channels, config.dim, bias=config.bias)
        self.local_encoder = LocalMambaEncoder(
          dim=config.dim,
          num_layers=config.nepa_local_mamba_layers,
          d_state=config.nepa_local_mamba_d_state,
          d_conv=config.nepa_local_mamba_d_conv,
          expand=config.nepa_local_mamba_expand,
          norm_eps=config.norm_eps,
          bias=config.bias)
        self.mamba_patch_readout = config.nepa_local_mamba_readout
        if self.mamba_patch_readout == 'conv':
          self.mamba_patch_readout_conv = nn.Conv1d(
            in_channels=config.dim,
            out_channels=config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.bias)
        else:
          self.mamba_patch_readout_conv = None
        patch_positions = torch.arange(
          config.patch_size - 1,
          config.channel_size,
          config.patch_size,
          dtype=torch.long)  # [T] patch-end indices per fixed patch.
        self.register_buffer('patch_positions', patch_positions, persistent=False)
        num_pos_tokens = num_patches
      elif self.static_tokenizer == 'mamba_topk_tokens':
        self.patch_embed = None
        self.input_proj = nn.Linear(config.num_channels, config.dim, bias=config.bias)
        self.local_encoder = LocalMambaEncoder(
          dim=config.dim,
          num_layers=config.nepa_local_mamba_layers,
          d_state=config.nepa_local_mamba_d_state,
          d_conv=config.nepa_local_mamba_d_conv,
          expand=config.nepa_local_mamba_expand,
          norm_eps=config.norm_eps,
          bias=config.bias)
        self.mamba_patch_readout = config.nepa_local_mamba_readout
        self.mamba_patch_readout_conv = None
        self.patch_positions = None
        self.topk_tokens = config.nepa_topk_tokens
        num_pos_tokens = config.channel_size
      elif self.static_tokenizer == 'mantis_tokgen':
        if config.channel_size % config.patch_size != 0:
          raise ValueError(
            f'channel_size must be divisible by patch_size, '
            f'got channel_size={config.channel_size}, patch_size={config.patch_size}')
        num_patches = config.channel_size // config.patch_size
        self.patch_embed = None
        self.input_proj = None
        self.local_encoder = None
        self.patch_positions = None
        self.mantis_tokgen = MantisTokGen1D(
          dim=config.dim,
          in_channels=config.num_channels,
          patch_size=config.patch_size,
          kernel_size=int(config.nepa_mantis_kernel_size),
          conv_mode=str(config.nepa_mantis_conv_mode),
          instance_norm_mode=str(config.nepa_mantis_instance_norm_mode),
          instance_norm_eps=float(config.nepa_mantis_instance_norm_eps),
          stats_mode=str(config.nepa_mantis_stats_mode),
          stats_dim=int(config.nepa_mantis_stats_dim),
          scalar_scales=tuple(config.nepa_mantis_scalar_scales),
          scalar_hidden_dim=int(config.nepa_mantis_scalar_hidden_dim),
          scalar_epsilon=float(config.nepa_mantis_scalar_epsilon),
          bias=config.bias,
          norm_eps=float(config.norm_eps))
        num_pos_tokens = num_patches
      else:
        raise ValueError(f'Unknown static tokenizer: {self.static_tokenizer}')

    # Positional embeddings: sinusoidal (buffer), learned (parameter), or none.
    if config.pos_embed_type == 'learned':
      self.pos_embed = get_learned_pos_embed(dim=config.dim, num_patches=num_pos_tokens)
    elif config.pos_embed_type == 'sin':
      self.register_buffer(
        'pos_embed',
        get_1d_pos_embed(dim=config.dim, num_patches=num_pos_tokens),
        persistent=False)
    else:
      self.pos_embed = None
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    if config.pred_depth < 0:
      raise ValueError(f'pred_depth must be >= 0, got pred_depth={config.pred_depth}')
    if config.pred_depth > config.depth:
      raise ValueError(
        'pred_depth must be <= depth for NEPAEncoder, '
        f'got pred_depth={config.pred_depth}, depth={config.depth}')
    self.trunk_depth = config.depth - config.pred_depth

    # Transformer blocks: wide trunk (dim/num_heads) + optional narrow predictor tail.
    trunk_blocks = [
      Block(
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        layer_scale_eps=config.layer_scale_eps,
        use_rope=config.use_rope,
        rope_base=config.rope_base or 10000,
        use_qk_norm=config.use_qk_norm,
        qk_norm_eps=config.qk_norm_eps or 1e-6,
        use_swiglu=config.use_swiglu,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(self.trunk_depth)
    ]

    pred_blocks: list[Block] = []
    if config.pred_depth > 0:
      if config.pred_dim % config.pred_num_heads != 0:
        raise ValueError(
          'pred_dim must be divisible by pred_num_heads, '
          f'got pred_dim={config.pred_dim}, pred_num_heads={config.pred_num_heads}')
      if config.use_rope:
        pred_head_dim = config.pred_dim // config.pred_num_heads
        if pred_head_dim % 2 != 0:
          raise ValueError(
            'RoPE requires even pred head_dim (=pred_dim/pred_num_heads), '
            f'got pred_dim={config.pred_dim}, pred_num_heads={config.pred_num_heads}, head_dim={pred_head_dim}')
      pred_blocks = [
        Block(
          dim=config.pred_dim,
          num_heads=config.pred_num_heads,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          bias=config.bias,
          dropout=config.dropout,
          attn_dropout=config.attn_dropout,
          eps=config.norm_eps,
          layer_scale_eps=config.layer_scale_eps,
          use_rope=config.use_rope,
          rope_base=config.rope_base or 10000,
          use_qk_norm=config.use_qk_norm,
          qk_norm_eps=config.qk_norm_eps or 1e-6,
          use_swiglu=config.use_swiglu,
          use_sdp_kernel=use_sdp_kernel)
        for _ in range(config.pred_depth)
      ]
      self.pred_in_proj = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
      self.pred_norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
      self.pred_out_proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)
    else:
      self.pred_in_proj = None
      self.pred_norm = None
      self.pred_out_proj = None

    self.blocks = nn.ModuleList([*trunk_blocks, *pred_blocks])
    if config.nepa_final_norm == 'ln':
      self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)
    elif config.nepa_final_norm == 'rms':
      self.norm = nn.RMSNorm((config.dim,), eps=config.norm_eps)
    elif config.nepa_final_norm == 'none':
      self.norm = nn.Identity()
    else:
      raise ValueError(f'Unknown nepa_final_norm: {config.nepa_final_norm!r}')
    for name, module in self.named_modules():
      if name.startswith('local_encoder'):
        continue
      if isinstance(module, (nn.Linear, nn.Conv1d)):
        if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
          # residual projections are initialized with scaled std
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * config.depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if module.weight is not None:
          nn.init.ones_(module.weight)
        bias = getattr(module, 'bias', None)
        if bias is not None:
          nn.init.zeros_(bias)

  @property
  def trunk_blocks(self) -> list[Block]:
    """Return the wide ViT trunk blocks (operating in `dim`)."""
    return list(self.blocks[:self.trunk_depth])

  @property
  def pred_blocks(self) -> list[Block]:
    """Return the narrow predictor-tail blocks (operating in `pred_dim`)."""
    return list(self.blocks[self.trunk_depth:])

  def _run_vit_and_collect_layers(
    self,
    h0_full: torch.Tensor,
    *,
    is_causal: bool,
    rope_prefix_len: int,
    rope_positions: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    token_mask: torch.Tensor | None,
  ) -> tuple[ViTLayerOutputs, dict[str, torch.Tensor] | None]:
    """Run transformer blocks and return per-layer patch tokens (registers stripped).

    Args:
      h0_full: [B, R+T, D] transformer input tokens (optional register prefix).
      is_causal: Whether to use causal attention (only allowed when attn_mask is None).
      rope_prefix_len: Number of prefix tokens excluded from RoPE rotation.
      rope_positions: Optional [B, T] integer positions for RoPE on the non-prefix tokens (irregular indices).
      attn_mask: Optional [B, 1, N, N] boolean mask (True = attend).
      token_mask: Optional [B, T] boolean mask for valid patch tokens (dynamic chunking).

    Returns:
      (vit_layers, vit_drift_stats)
        - vit_layers: ViTLayerOutputs with `patch_tokens_pre_norm[layer]` for layer=0..depth.
        - vit_drift_stats: Optional dict with `nepa_diag_vit_*` drift stats.
    """
    # Step 1: strip registers for layer 0 outputs.
    num_registers = self.config.num_registers
    if num_registers > 0:
      patch_tokens0 = h0_full[:, num_registers:]  # [B, T, D]
    else:
      patch_tokens0 = h0_full  # [B, T, D]
    if patch_tokens0.size(1) < 1:
      raise ValueError(
        'Expected at least one patch token after stripping registers, '
        f'got patch_tokens0.shape={tuple(patch_tokens0.shape)}')

    patch_tokens_pre_norm: list[torch.Tensor] = [patch_tokens0]

    # Step 2: run blocks and collect per-layer patch tokens (pre-norm).
    h = h0_full  # [B, R+T, D]
    vit_drift_stats: dict[str, torch.Tensor] | None = None
    drift_layers_set: set[int] = set()
    if self.config.nepa_vit_drift_diagnostics:
      vit_drift_layers = self.config.nepa_vit_drift_layers
      if vit_drift_layers is None:
        raise RuntimeError(
          'nepa_vit_drift_diagnostics=True but nepa_vit_drift_layers is None; '
          'this should have been validated in the config.')
      vit_drift_stats = {}
      drift_layers_set = set(vit_drift_layers)

    for layer_idx, block in enumerate(self.trunk_blocks):
      collect = layer_idx in drift_layers_set
      h = block(
        h,
        is_causal=is_causal,
        rope_prefix_len=rope_prefix_len,
        rope_positions=rope_positions,
        attn_mask=attn_mask,
        drift_stats=vit_drift_stats if collect else None,
        drift_layer_idx=layer_idx if collect else None)  # [B, R+T, D]
      if num_registers > 0:
        patch_tokens = h[:, num_registers:]  # [B, T, D]
      else:
        patch_tokens = h  # [B, T, D]
      patch_tokens_pre_norm.append(patch_tokens)

    # Step 2b: optional narrow predictor tail (compute in pred_dim, emit in dim).
    if self.pred_in_proj is not None:
      if self.pred_out_proj is None or self.pred_norm is None:
        raise RuntimeError('NEPAEncoder predictor tail is partially initialized; this is a bug.')
      p = self.pred_in_proj(h)  # [B, R+T, pred_dim]
      pred_blocks = self.pred_blocks
      for pred_idx, block in enumerate(pred_blocks):
        layer_idx = self.trunk_depth + pred_idx
        collect = layer_idx in drift_layers_set
        p = block(
          p,
          is_causal=is_causal,
          rope_prefix_len=rope_prefix_len,
          rope_positions=rope_positions,
          attn_mask=attn_mask,
          drift_stats=vit_drift_stats if collect else None,
          drift_layer_idx=layer_idx if collect else None)  # [B, R+T, pred_dim]
        if pred_idx == len(pred_blocks) - 1:
          p = self.pred_norm(p)  # [B, R+T, pred_dim]
        h_pred = self.pred_out_proj(p)  # [B, R+T, D]
        if num_registers > 0:
          patch_tokens = h_pred[:, num_registers:]  # [B, T, D]
        else:
          patch_tokens = h_pred  # [B, T, D]
        patch_tokens_pre_norm.append(patch_tokens)

    # Step 3: build the output container.
    final_norm_weight = getattr(self.norm, 'weight', None)
    final_norm_bias = getattr(self.norm, 'bias', None)
    final_norm_eps = getattr(self.norm, 'eps', None)
    vit_layers = ViTLayerOutputs(
      patch_tokens_pre_norm=tuple(patch_tokens_pre_norm),
      token_mask=token_mask,
      final_norm=self.config.nepa_final_norm,
      final_norm_weight=final_norm_weight,
      final_norm_bias=final_norm_bias,
      final_norm_eps=final_norm_eps,
      output_norm=self.config.nepa_layer_output_norm,
      output_bn_eps=self.config.nepa_layer_output_bn_eps,
      rep_pooling=self.config.nepa_rep_pooling)
    return vit_layers, vit_drift_stats

  def forward(self, x):
    """Run the NEPA encoder and return all ViT layer outputs.

    Returns:
      (vit_layers, encoder_stats):
        - vit_layers: ViTLayerOutputs container with per-layer patch tokens.
        - encoder_stats: Optional diagnostics dict (dynamic chunking stats and/or tokenizer distribution stats).
    """
    # x: [B, C, L]
    token_mask = None
    encoder_stats = None

    if self.use_dynamic_chunking:
      # Step 1: full-resolution input projection.
      x_t = x.transpose(1, 2)  # [B, L, C]
      h0 = self.input_proj(x_t)  # [B, L, D]
      mamba_tokenizer_stats: dict[str, torch.Tensor] | None = None
      if self.config.nepa_distribution_diagnostics:
        with torch.no_grad():
          mamba_tokenizer_stats = {}
          mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
            h0.detach(), None, prefix='nepa_diag_tok_mamba_h0'))

      # Step 2: local contextualizer (Mamba).
      if self.config.nepa_distribution_diagnostics:
        x_hat, layer_stats = self.local_encoder(
          h0,
          return_layer_stats=True,
          stats_prefix='nepa_diag_tok_mamba')  # [B, L, D], dict
        if mamba_tokenizer_stats is None:
          raise RuntimeError('Expected mamba_tokenizer_stats to be initialized; this is a bug.')
        mamba_tokenizer_stats.update(layer_stats)
        with torch.no_grad():
          mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
            x_hat.detach(), None, prefix='nepa_diag_tok_mamba_x_hat'))
      else:
        x_hat = self.local_encoder(h0)  # [B, L, D]

      # Step 3: boundary router (H-Net).
      p, b, router_stats = self.router(x_hat, return_stats=True)  # [B, L], [B, L]

      # Step 4: downsampling (pack + pad).
      z, z_mask, idx_padded, _ = self.downsampler(x_hat, b, p)  # [B, M, D], [B, M]
      z_in = z  # [B, M, D] patch embeddings fed into the transformer.
      B, T, D = z_in.size()
      token_mask = z_mask  # [B, M]
      tokenizer_stats = None
      if self.config.nepa_distribution_diagnostics:
        with torch.no_grad():
          tokenizer_stats = {}
          if mamba_tokenizer_stats is not None:
            tokenizer_stats.update(mamba_tokenizer_stats)
          tokenizer_stats.update(_masked_embedding_distribution_stats(
            z_in, token_mask, prefix='nepa_diag_tok_in'))

      # Step 5: positional embeddings and RoPE positions for selected indices.
      idx_safe: torch.Tensor | None = None
      if self.config.use_rope or self.config.nepa_target_uses_pos_embed:
        idx_safe = idx_padded.clamp_min(0)  # [B, M]
      if self.config.nepa_target_uses_pos_embed:
        if self.pos_embed is None:
          raise ValueError('pos_embed is required when nepa_target_uses_pos_embed=True')
        if idx_safe is None:
          raise RuntimeError('Expected idx_safe to be set when nepa_target_uses_pos_embed=True; this is a bug.')
        pos_full = self.pos_embed.expand(B, -1, -1)  # [B, L, D]
        pos_selected = pos_full.gather(
          1, idx_safe.unsqueeze(-1).expand(-1, -1, pos_full.size(-1)))  # [B, M, D]
        z_in = z_in + pos_selected * z_mask.unsqueeze(-1)  # [B, M, D]
      rope_positions = idx_safe if self.config.use_rope else None  # [B, M] or None

      # Step 6: build transformer input with registers.
      if self.config.num_registers > 0:
        registers = self.registers.expand(B, -1, -1)  # [B, R, D]
        h0_full = torch.cat([registers, z_in], dim=1)  # [B, R+M, D]
      else:
        h0_full = z_in  # [B, M, D]

      # RoPE prefix length: don't rotate register tokens.
      rope_prefix_len = self.config.num_registers if self.config.use_rope else 0

      # Step 7: attention mask (causal + padding).
      if self.config.num_registers > 0:
        reg_mask = torch.ones(
          (B, self.config.num_registers), device=x.device, dtype=torch.bool)  # [B, R]
        key_is_valid = torch.cat([reg_mask, z_mask], dim=1)  # [B, R+M]
      else:
        key_is_valid = z_mask  # [B, M]
      N = key_is_valid.size(1)
      causal = torch.ones((N, N), device=x.device, dtype=torch.bool).tril()  # [N, N]
      attn_mask = causal.unsqueeze(0).unsqueeze(0) & key_is_valid.unsqueeze(1).unsqueeze(2)  # [B, 1, N, N]

      # Step 8: run ViT once and capture all patch-token layers.
      vit_layers, vit_drift_stats = self._run_vit_and_collect_layers(
        h0_full,
        is_causal=False,
        rope_prefix_len=rope_prefix_len,
        rope_positions=rope_positions,
        attn_mask=attn_mask,
        token_mask=token_mask)
      if vit_drift_stats is not None:
        if encoder_stats is None:
          encoder_stats = {}
        encoder_stats.update(vit_drift_stats)

      # Step 9: ratio loss + summary stats.
      p_mean = p.to(dtype=torch.float32).mean(dim=1)  # [B]
      b_mean = b.to(dtype=torch.float32).mean(dim=1)  # [B]
      target = float(self.config.nepa_chunk_ratio_target)
      ratio_terms = target / (target - 1.0) * (
        (target - 1.0) * b_mean * p_mean + (1.0 - b_mean) * (1.0 - p_mean))  # [B]
      ratio_loss = ratio_terms.mean(dtype=torch.float32)  # scalar
      num_tokens_raw = b.to(dtype=torch.float32).sum(dim=1)  # [B]
      num_tokens = z_mask.to(dtype=torch.float32).sum(dim=1)  # [B]
      chunk_stats = {
        'loss_nepa_ratio': ratio_loss,
        'nepa_chunk_p_mean': p_mean.mean(dtype=torch.float32),
        'nepa_chunk_b_mean': b_mean.mean(dtype=torch.float32),
        'nepa_chunk_tokens_mean': num_tokens.mean(dtype=torch.float32),
        'nepa_chunk_tokens_min': num_tokens.min(),
        'nepa_chunk_tokens_max': num_tokens.max(),
        'nepa_chunk_tokens_raw_mean': num_tokens_raw.mean(dtype=torch.float32),
        'nepa_chunk_tokens_raw_max': num_tokens_raw.max(),
      }
      chunk_stats.update(router_stats)

      # Step 10: chunk length diagnostics (delta between boundary indices).
      delta_list = []
      for batch_idx in range(B):
        idx_valid = idx_padded[batch_idx, z_mask[batch_idx]]  # [K]
        if idx_valid.numel() > 1:
          delta = idx_valid[1:] - idx_valid[:-1]  # [K-1]
          delta_list.append(delta)
      if delta_list:
        delta_all = torch.cat(delta_list, dim=0)  # [N]
      else:
        delta_all = idx_padded.new_empty((0,))  # [0]
      chunk_stats.update(_flat_stats(
        delta_all,
        prefix='nepa_chunk_delta_idx',
        quantiles=(0.5, 0.9),
        include_min_max=True))

      # Step 11: chunk cap/utilization diagnostics.
      cap_rate = (num_tokens_raw > self.config.nepa_chunk_max_tokens).to(dtype=torch.float32).mean()  # scalar
      tokens_dropped = (num_tokens_raw - num_tokens).clamp_min(0.0)  # [B]
      utilization = (num_tokens / float(self.config.nepa_chunk_max_tokens)).mean(dtype=torch.float32)  # scalar
      chunk_stats['nepa_chunk_cap_rate'] = cap_rate
      chunk_stats['nepa_chunk_tokens_dropped_mean'] = tokens_dropped.mean(dtype=torch.float32)
      chunk_stats['nepa_chunk_utilization'] = utilization
      if tokenizer_stats is not None:
        chunk_stats.update(tokenizer_stats)
      encoder_stats = chunk_stats
    else:
      # NEPA Step 1: static tokenizer (conv patch embed, static Mamba patches, or top-k Mamba tokens).
      mamba_tokenizer_stats: dict[str, torch.Tensor] | None = None
      idx_selected: torch.Tensor | None = None
      rope_positions: torch.Tensor | None = None
      if self.static_tokenizer == 'conv_patch_embed':
        if self.nepa_patch_embed_scalar_stats_mode is None:
          z_in = self.patch_embed(x)  # [B, T, D]
        else:
          if self.nepa_patch_embed_mean_encoder is None or self.nepa_patch_embed_std_encoder is None:
            raise RuntimeError('Scalar encoders are not initialized; this is a bug.')
          B, C, L = x.shape
          if L % self.config.patch_size != 0:
            raise RuntimeError(
              'Expected L divisible by patch_size for conv_patch_embed, '
              f'got L={L}, patch_size={self.config.patch_size}')
          T = L // self.config.patch_size
          x_f32 = x.to(dtype=torch.float32)  # [B, C, L]
          mode = self.nepa_patch_embed_scalar_stats_mode
          if mode == 'sequence_norm':
            mean_seq = x_f32.mean(dim=(1, 2))  # [B]
            std_seq = x_f32.std(dim=(1, 2), unbiased=False)  # [B]
            mean_broadcast = mean_seq[:, None, None]  # [B, 1, 1]
            std_broadcast = std_seq[:, None, None]  # [B, 1, 1]
            x_norm_f32 = (x_f32 - mean_broadcast) / (std_broadcast + self.config.norm_eps)  # [B, C, L]
            mean_vals = mean_seq[:, None].expand(B, T)  # [B, T]
            std_vals = std_seq[:, None].expand(B, T)  # [B, T]
          elif mode == 'patch_norm':
            P = self.config.patch_size
            x_patches = x_f32.reshape(B, C, T, P)  # [B, C, T, P]
            mean_patch = x_patches.mean(dim=(1, 3))  # [B, T]
            std_patch = x_patches.std(dim=(1, 3), unbiased=False)  # [B, T]
            mean_broadcast = mean_patch[:, None, :, None]  # [B, 1, T, 1]
            std_broadcast = std_patch[:, None, :, None]  # [B, 1, T, 1]
            x_norm_patches = (x_patches - mean_broadcast) / (std_broadcast + self.config.norm_eps)  # [B, C, T, P]
            x_norm_f32 = x_norm_patches.reshape(B, C, L)  # [B, C, L]
            mean_vals = mean_patch  # [B, T]
            std_vals = std_patch  # [B, T]
          else:
            raise RuntimeError(
              'Unknown nepa_patch_embed_scalar_stats_mode; this should have been validated in the config. '
              f'Got {mode!r}')

          x_norm = x_norm_f32.to(dtype=x.dtype)  # [B, C, L]
          z_cnn = self.patch_embed(x_norm)  # [B, T, D_cnn]
          e_mean = self.nepa_patch_embed_mean_encoder(mean_vals.unsqueeze(-1))  # [B, T, H]
          e_std = self.nepa_patch_embed_std_encoder(std_vals.unsqueeze(-1))  # [B, T, H]
          z_in = torch.cat(
            [z_cnn, e_mean.to(dtype=z_cnn.dtype), e_std.to(dtype=z_cnn.dtype)],
            dim=-1,
          )  # [B, T, D]
          if z_in.size(-1) != self.config.dim:
            raise RuntimeError(
              'conv_patch_embed scalar concat produced unexpected token dim; '
              f'got z_in.shape={tuple(z_in.shape)}, expected last dim={self.config.dim}')
      elif self.static_tokenizer == 'mantis_tokgen':
        z_in = self.mantis_tokgen(x)  # [B, T, D]
      elif self.static_tokenizer == 'mamba_static_patches':
        x_t = x.transpose(1, 2)  # [B, L, C]
        h0 = self.input_proj(x_t)  # [B, L, D]
        if self.config.nepa_distribution_diagnostics:
          with torch.no_grad():
            mamba_tokenizer_stats = {}
            mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
              h0.detach(), None, prefix='nepa_diag_tok_mamba_h0'))
          x_hat, layer_stats = self.local_encoder(
            h0,
            return_layer_stats=True,
            stats_prefix='nepa_diag_tok_mamba')  # [B, L, D], dict
          mamba_tokenizer_stats.update(layer_stats)
          with torch.no_grad():
            mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
              x_hat.detach(), None, prefix='nepa_diag_tok_mamba_x_hat'))
        else:
          x_hat = self.local_encoder(h0)  # [B, L, D]
        if self.mamba_patch_readout == 'last':
          patch_positions = self.patch_positions  # [T]
          z_in = x_hat[:, patch_positions, :]  # [B, T, D]
        elif self.mamba_patch_readout == 'mean':
          B, L, D = x_hat.shape
          if L % self.config.patch_size != 0:
            raise RuntimeError(
              'Expected full-resolution length divisible by patch_size for mean patch readout, '
              f'got L={L}, patch_size={self.config.patch_size}')
          T = L // self.config.patch_size
          z_in = x_hat.view(B, T, self.config.patch_size, D).mean(dim=2)  # [B, T, D]
        elif self.mamba_patch_readout == 'conv':
          if self.mamba_patch_readout_conv is None:
            raise RuntimeError('mamba_patch_readout_conv is not initialized; this is a bug.')
          x_hat_t = x_hat.transpose(1, 2)  # [B, D, L]
          z_t = self.mamba_patch_readout_conv(x_hat_t)  # [B, D, T]
          z_in = z_t.transpose(1, 2)  # [B, T, D]
        else:
          raise ValueError(f'Unknown nepa_local_mamba_readout: {self.mamba_patch_readout!r}')
      elif self.static_tokenizer == 'mamba_topk_tokens':
        x_t = x.transpose(1, 2)  # [B, L, C]
        h0 = self.input_proj(x_t)  # [B, L, D]
        if self.config.nepa_distribution_diagnostics:
          with torch.no_grad():
            mamba_tokenizer_stats = {}
            mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
              h0.detach(), None, prefix='nepa_diag_tok_mamba_h0'))
          x_hat, layer_stats = self.local_encoder(
            h0,
            return_layer_stats=True,
            stats_prefix='nepa_diag_tok_mamba')  # [B, L, D], dict
          mamba_tokenizer_stats.update(layer_stats)
          with torch.no_grad():
            mamba_tokenizer_stats.update(_masked_embedding_distribution_stats(
              x_hat.detach(), None, prefix='nepa_diag_tok_mamba_x_hat'))
        else:
          x_hat = self.local_encoder(h0)  # [B, L, D]

        if self.topk_tokens is None:
          raise RuntimeError('topk_tokens is not initialized; this should have been validated in the config.')
        idx_selected = _select_topk_token_indices(x_hat, k=self.topk_tokens)  # [B, K]
        z_in = _pool_tokens_by_end_indices(
          x_hat,
          end_indices=idx_selected,
          mode=self.mamba_patch_readout)  # [B, K, D]
        rope_positions = idx_selected if self.config.use_rope else None
      else:
        raise ValueError(f'Unknown static tokenizer: {self.static_tokenizer}')

      if self.config.nepa_distribution_diagnostics:
        with torch.no_grad():
          encoder_stats = {}
          if mamba_tokenizer_stats is not None:
            encoder_stats.update(mamba_tokenizer_stats)
          encoder_stats.update(_masked_embedding_distribution_stats(z_in, None, prefix='nepa_diag_tok_in'))
          if self.static_tokenizer == 'mamba_topk_tokens':
            if idx_selected is None:
              raise RuntimeError('Expected idx_selected to be set for mamba_topk_tokens; this is a bug.')
            deltas = idx_selected[:, 1:] - idx_selected[:, :-1]  # [B, K-1]
            encoder_stats.update(_flat_stats(
              deltas,
              prefix='nepa_topk_delta_idx',
              quantiles=(0.5, 0.9),
              include_min_max=True))
      B, T, D = z_in.size()
      if self.config.nepa_target_uses_pos_embed:
        if self.pos_embed is None:
          raise ValueError('pos_embed is required when nepa_target_uses_pos_embed=True')
        if self.static_tokenizer == 'mamba_topk_tokens':
          if idx_selected is None:
            raise RuntimeError('Expected idx_selected to be set for mamba_topk_tokens; this is a bug.')
          pos_full = self.pos_embed.expand(B, -1, -1)  # [B, L, D]
          pos_selected = pos_full.gather(
            1, idx_selected.unsqueeze(-1).expand(-1, -1, pos_full.size(-1)))  # [B, T, D]
          z_in = z_in + pos_selected  # [B, T, D]
        else:
          z_in = z_in + self.pos_embed[:, :T]  # [B, T, D] regular grid.
      if self.config.num_registers > 0:
        registers = self.registers.expand(B, -1, -1)  # [B, R, D]
        h0_full = torch.cat([registers, z_in], dim=1)  # [B, R+T, D] prefix registers.
      else:
        h0_full = z_in  # [B, T, D]

      # RoPE prefix length: don't rotate register tokens
      rope_prefix_len = self.config.num_registers if self.config.use_rope else 0

      # NEPA Step 2: causal transformer (all layers).
      vit_layers, vit_drift_stats = self._run_vit_and_collect_layers(
        h0_full,
        is_causal=True,
        rope_prefix_len=rope_prefix_len,
        rope_positions=rope_positions,
        attn_mask=None,
        token_mask=None)
      if vit_drift_stats is not None:
        if encoder_stats is None:
          encoder_stats = {}
        encoder_stats.update(vit_drift_stats)
    return vit_layers, encoder_stats


class NEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.encoder = NEPAEncoder(config, use_sdp_kernel=use_sdp_kernel)

    # Projector (MLP vs Flow; unified builder).
    self.proj = projector_build(config)

    sigreg_seed = None
    if config.seed is None and config.seed_sigreg is not None:
      sigreg_seed = int(config.seed_sigreg)
    self.sigreg = SIGReg(normalize=config.sigreg_normalize, seed=sigreg_seed)

    # Wristband Gaussian stability (replacement for SIGReg).
    self.wristband: C_WristbandGaussianLoss | None = None
    if config.nepa_stability == "wristband":
      self.wristband = C_WristbandGaussianLoss(
        beta=float(config.nepa_wristband_beta),
        alpha=config.nepa_wristband_alpha,
        angular=str(config.nepa_wristband_angular),
        reduction=str(config.nepa_wristband_reduction),
        lambda_rad=float(config.nepa_wristband_lambda_rad),
        lambda_ang=float(config.nepa_wristband_lambda_ang),
        moment=str(config.nepa_wristband_moment),
        lambda_mom=float(config.nepa_wristband_lambda_mom),
      )

  def forward(self, x, return_representation=False, return_patch_loss=False):
    # Step 1: compute all per-layer ViT patch tokens.
    vit_layers, encoder_stats = self.encoder(x)

    # Step 2: select layers for targets/context/representation.
    depth = vit_layers.depth
    target_layer = 0 if self.config.nepa_target_layer is None else self.config.nepa_target_layer

    z_hat = vit_layers.patch_tokens(depth)  # [B, T, D]
    z = vit_layers.patch_tokens(target_layer, apply_norm=(target_layer != 0))  # [B, T, D]
    rep = vit_layers.representation(depth)  # [B, D]
    token_mask = vit_layers.token_mask  # [B, T] or None

    # z: [B, T, D], z_hat: [B, T, D], rep: [B, D]
    B, T, D = z.size()
    if z_hat.size(1) != T:
      raise ValueError(
        'NEPAEncoder must return patch tokens with consistent sequence length: '
        f'expected z_hat.shape[1]==T={T}, got z_hat.shape={tuple(z_hat.shape)}')

    z_target = z.detach() if self.config.nepa_stability == 'stopgrad' else z  # [B, T, D]
    z_hat_preproj = z_hat  # [B, T, D]
    z_target_preproj = z_target  # [B, T, D]
    diag_stats: dict[str, torch.Tensor] | None = None
    if self.config.nepa_distribution_diagnostics or self.config.nepa_vit_drift_diagnostics:
      diag_stats = {}
      with torch.no_grad():
        if self.config.nepa_distribution_diagnostics:
          preproj_z_hat_stats = _masked_embedding_distribution_stats(
            z_hat_preproj.detach(), token_mask, prefix='nepa_diag_preproj_z_hat')
          preproj_z_target_stats = _masked_embedding_distribution_stats(
            z_target_preproj.detach(), token_mask, prefix='nepa_diag_preproj_z_target')
          diag_stats.update(preproj_z_hat_stats)
          diag_stats.update(preproj_z_target_stats)
          preproj_rms_hat = preproj_z_hat_stats['nepa_diag_preproj_z_hat_token_rms_mean']  # []
          preproj_rms_target = preproj_z_target_stats['nepa_diag_preproj_z_target_token_rms_mean']  # []
          diag_stats['nepa_diag_preproj_scale_ratio_rms'] = preproj_rms_hat / preproj_rms_target  # []
          preproj_mean_hat = preproj_z_hat_stats['nepa_diag_preproj_z_hat_global_mean']  # []
          preproj_mean_target = preproj_z_target_stats['nepa_diag_preproj_z_target_global_mean']  # []
          diag_stats['nepa_diag_preproj_center_shift'] = preproj_mean_target - preproj_mean_hat  # []

        if self.config.nepa_vit_drift_diagnostics:
          drift_layers = self.config.nepa_vit_drift_layers
          if drift_layers is None:
            raise RuntimeError(
              'nepa_vit_drift_diagnostics=True but nepa_vit_drift_layers is None; '
              'this should have been validated in the config.')
          for layer in drift_layers:
            values = vit_layers.patch_tokens_pre_norm[layer]  # [B, T, D]
            diag_stats.update(_masked_embedding_distribution_stats(
              values.detach(),
              token_mask,
              prefix=f'nepa_diag_vit_pre_norm_layer{layer}'))

    # Projector (for z_hat and z_target)
    if self.proj is not None:
      mode = self.config.nepa_projector_mode
      if mode == 'both':
        z_hat = self.proj(z_hat.reshape(-1, D)).reshape(B, T, -1) # [B, T, proj_dim/proj_out_dim]
        if self.config.nepa_stability == 'stopgrad':
          with torch.no_grad():
            z_target = self.proj(z_target.reshape(-1, D)).reshape(B, T, -1) # [B, T, proj_dim/proj_out_dim]
        else:
          z_target = self.proj(z_target.reshape(-1, D)).reshape(B, T, -1) # [B, T, proj_dim/proj_out_dim]
      elif mode == 'pred_only':
        z_hat = self.proj(z_hat.reshape(-1, D)).reshape(B, T, -1) # [B, T, dim]
      elif mode == 'target_only':
        if self.config.nepa_stability == 'stopgrad':
          with torch.no_grad():
            z_target = self.proj(z_target.reshape(-1, D)).reshape(B, T, -1) # [B, T, dim]
        else:
          z_target = self.proj(z_target.reshape(-1, D)).reshape(B, T, -1) # [B, T, dim]
      else:
        raise ValueError(f'Unknown nepa_projector_mode: {mode!r}')
    if self.config.nepa_distribution_diagnostics and diag_stats is not None and self.proj is not None:
      with torch.no_grad():
        postproj_z_hat_stats = _masked_embedding_distribution_stats(
          z_hat.detach(), token_mask, prefix='nepa_diag_postproj_z_hat')
        postproj_z_target_stats = _masked_embedding_distribution_stats(
          z_target.detach(), token_mask, prefix='nepa_diag_postproj_z_target')
      diag_stats.update(postproj_z_hat_stats)
      diag_stats.update(postproj_z_target_stats)
      postproj_rms_hat = postproj_z_hat_stats['nepa_diag_postproj_z_hat_token_rms_mean']  # []
      postproj_rms_target = postproj_z_target_stats['nepa_diag_postproj_z_target_token_rms_mean']  # []
      diag_stats['nepa_diag_postproj_scale_ratio_rms'] = postproj_rms_hat / postproj_rms_target  # []
      postproj_mean_hat = postproj_z_hat_stats['nepa_diag_postproj_z_hat_global_mean']  # []
      postproj_mean_target = postproj_z_target_stats['nepa_diag_postproj_z_target_global_mean']  # []
      diag_stats['nepa_diag_postproj_center_shift'] = postproj_mean_target - postproj_mean_hat  # []

    # Loss
    z_hat_patches = z_hat  # [B, T, D/proj_dim] patch tokens only.
    pred = z_hat_patches[:, :-1, :]  # [B, T-1, D/proj_dim] predict t+1 from <=t.
    target_next = z_target[:, 1:, :]  # [B, T-1, D/proj_dim]
    loss_mask = token_mask[:, 1:] if token_mask is not None else None  # [B, T-1] or None
    if self.config.nepa_distribution_diagnostics and diag_stats is not None:
      with torch.no_grad():
        pred_stats = _masked_embedding_distribution_stats(
          pred.detach(), loss_mask, prefix='nepa_diag_pred')
        target_next_stats = _masked_embedding_distribution_stats(
          target_next.detach(), loss_mask, prefix='nepa_diag_target_next')
      diag_stats.update(pred_stats)
      diag_stats.update(target_next_stats)
    pred_target_cos = F.cosine_similarity(pred, target_next, dim=-1)  # [B, T-1]
    pred_target_mse = (pred - target_next).square()  # [B, T-1, D/proj_dim]
    pred_target_stats = {
      'nepa_pred_target_cos_mean': _masked_mean(pred_target_cos, loss_mask),
      'nepa_pred_target_mse_mean': _masked_mean(pred_target_mse, loss_mask),
    }
    loss_nepa, patch_loss = nepa_loss(
      z_target,
      z_hat_patches,
      token_mask=token_mask,
      loss_type=self.config.nepa_loss_type,
      return_patch_loss=return_patch_loss,
      batch_norm=self.config.nepa_loss_batch_norm,
      batch_norm_eps=self.config.nepa_loss_batch_norm_eps,
    )  # scalar, [T-1] or None
    losses = {'loss_nepa': loss_nepa}
    losses.update(pred_target_stats)

    loss = self.config.nepa_loss_scale*loss_nepa
    if encoder_stats is not None:
      if 'loss_nepa_ratio' in encoder_stats:
        ratio_loss = encoder_stats['loss_nepa_ratio']
        losses['loss_nepa_ratio'] = ratio_loss
        loss += self.config.nepa_chunk_ratio_loss_scale * ratio_loss
      for key, value in encoder_stats.items():
        if key != 'loss_nepa_ratio':
          losses[key] = value

    if self.config.nepa_stability in ('sigreg', 'rdmreg', 'wristband'):
      layer_tokens_cache: dict[int, torch.Tensor] = {
        depth: z_hat_patches,
        target_layer: z_target,
      }

      def _stability_tokens(layer: int) -> torch.Tensor:
        """Return patch tokens for stability regularization at the given ViT layer.

        This matches the existing SIGReg semantics exactly and is reused for RDMReg/Wristband.
        """
        cached = layer_tokens_cache.get(layer)
        if cached is not None:
          return cached  # [B, T, D/proj_dim]
        tokens = vit_layers.patch_tokens(layer)  # [B, T, D]
        if self.proj is not None:
          tokens = self.proj(tokens.reshape(-1, tokens.size(-1))).reshape(B, T, -1)  # [B, T, proj_dim]
        layer_tokens_cache[layer] = tokens
        return tokens  # [B, T, D/proj_dim]

    if self.config.nepa_stability == 'sigreg':

      # SIGReg_batch
      if self.config.nepa_sigreg_scale is not None and self.config.nepa_sigreg_scale > 0:
        sigreg_layers = self.config.nepa_sigreg_layers
        if sigreg_layers is None:
          raise ValueError('nepa_sigreg_layers must be set when nepa_sigreg_scale > 0')
        sigreg_tokens = torch.cat(
          [_stability_tokens(layer) for layer in sigreg_layers], dim=0)  # [L*B, T, D/proj_dim]
        sigreg_in = sigreg_tokens.transpose(0, 1)  # [T, L*B, D/proj_dim]
        if token_mask is None:
          sigreg_mask = None
        else:
          token_mask_time = token_mask.transpose(0, 1)  # [T, B]
          sigreg_mask = token_mask_time.repeat(1, len(sigreg_layers))  # [T, L*B]

        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          with torch.no_grad():
            sigreg_batch_stats = _masked_embedding_distribution_stats(
              sigreg_in.detach(), sigreg_mask, prefix='nepa_diag_sigreg_batch_in')
          diag_stats.update(sigreg_batch_stats)
        if sigreg_mask is None:
          loss_sigreg = self.sigreg(sigreg_in)  # scalar
        else:
          loss_sigreg = self.sigreg(sigreg_in, mask=sigreg_mask)  # scalar
        losses['loss_sigreg'] = loss_sigreg
        loss += self.config.nepa_sigreg_scale * loss_sigreg

      # SIGReg_time
      if self.config.nepa_sigreg_scale_time is not None and self.config.nepa_sigreg_scale_time > 0:
        sigreg_time_layers = self.config.nepa_sigreg_time_layers
        if sigreg_time_layers is None:
          raise ValueError('nepa_sigreg_time_layers must be set when nepa_sigreg_scale_time is set')
        sigreg_in = torch.cat(
          [_stability_tokens(layer) for layer in sigreg_time_layers], dim=0)  # [L*B, T, D/proj_dim]
        if token_mask is None:
          sigreg_mask = None
        else:
          sigreg_mask = token_mask.repeat(len(sigreg_time_layers), 1)  # [L*B, T]
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          with torch.no_grad():
            sigreg_time_stats = _masked_embedding_distribution_stats(
              sigreg_in.detach(), sigreg_mask, prefix='nepa_diag_sigreg_time_in')
          diag_stats.update(sigreg_time_stats)
        loss_sigreg_time = self.sigreg(sigreg_in, mask=sigreg_mask)  # scalar
        losses['loss_sigreg_time'] = loss_sigreg_time
        loss += self.config.nepa_sigreg_scale_time * loss_sigreg_time

      # SIGReg_batch_time
      if self.config.nepa_sigreg_scale_batch_time is not None and self.config.nepa_sigreg_scale_batch_time > 0:
        sigreg_batch_time_layers = self.config.nepa_sigreg_batch_time_layers
        if sigreg_batch_time_layers is None:
          raise ValueError(
            'nepa_sigreg_batch_time_layers must be set when nepa_sigreg_scale_batch_time is set')
        sigreg_batch_time_tokens = torch.stack(
          [_stability_tokens(layer) for layer in sigreg_batch_time_layers], dim=0)  # [L, B, T, D/proj_dim]
        L = sigreg_batch_time_tokens.size(0)
        sigreg_batch_time_in = sigreg_batch_time_tokens.reshape(L, B * T, -1)  # [L, B*T, D/proj_dim]
        if token_mask is None:
          sigreg_batch_time_mask = None
        else:
          sigreg_batch_time_mask = token_mask.reshape(1, B * T).repeat(L, 1)  # [L, B*T]
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          with torch.no_grad():
            sigreg_batch_time_stats = _masked_embedding_distribution_stats(
              sigreg_batch_time_in.detach(), sigreg_batch_time_mask, prefix='nepa_diag_sigreg_batch_time_in')
          diag_stats.update(sigreg_batch_time_stats)
        loss_sigreg_batch_time = self.sigreg(sigreg_batch_time_in, mask=sigreg_batch_time_mask)  # scalar
        losses['loss_sigreg_batch_time'] = loss_sigreg_batch_time
        loss += self.config.nepa_sigreg_scale_batch_time * loss_sigreg_batch_time

      # SIGReg_innovation
      if self.config.nepa_sigreg_innov_scale is not None and self.config.nepa_sigreg_innov_scale > 0:
        if T < 2:
          raise ValueError('nepa_sigreg_innov_scale requires at least 2 tokens to form innovations')
        sigreg_innov_layers = self.config.nepa_sigreg_innov_layers
        if sigreg_innov_layers is None:
          raise ValueError('nepa_sigreg_innov_layers must be set when nepa_sigreg_innov_scale is set')
        sigreg_innov_tokens = torch.cat(
          [_stability_tokens(layer) for layer in sigreg_innov_layers], dim=0)  # [L*B, T, D/proj_dim]
        innov = sigreg_innov_tokens[:, 1:, :] - sigreg_innov_tokens[:, :-1, :]  # [L*B, T-1, D/proj_dim]
        sigreg_innov_in = innov.transpose(0, 1)  # [T-1, L*B, D/proj_dim]
        if token_mask is None:
          sigreg_innov_mask = None
        else:
          innov_mask = token_mask[:, 1:] & token_mask[:, :-1]  # [B, T-1]
          if not innov_mask.any():
            raise ValueError('token_mask leaves no valid innovation pairs for SIGReg')
          innov_mask_time = innov_mask.transpose(0, 1)  # [T-1, B]
          sigreg_innov_mask = innov_mask_time.repeat(1, len(sigreg_innov_layers))  # [T-1, L*B]
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          with torch.no_grad():
            sigreg_innov_stats = _masked_embedding_distribution_stats(
              sigreg_innov_in.detach(), sigreg_innov_mask, prefix='nepa_diag_sigreg_innov_in')
          diag_stats.update(sigreg_innov_stats)
        if sigreg_innov_mask is None:
          loss_sigreg_innov = self.sigreg(sigreg_innov_in)  # scalar
        else:
          loss_sigreg_innov = self.sigreg(sigreg_innov_in, mask=sigreg_innov_mask)  # scalar
        losses['loss_sigreg_innov'] = loss_sigreg_innov
        loss += self.config.nepa_sigreg_innov_scale * loss_sigreg_innov

      # SIGReg_rep
      if self.config.nepa_sigreg_rep_scale is not None and self.config.nepa_sigreg_rep_scale > 0:
        rep_sigreg_layers = self.config.nepa_sigreg_rep_layers
        if rep_sigreg_layers is None:
          raise ValueError('nepa_sigreg_rep_layers must be set when nepa_sigreg_rep_scale is set')
        reps = torch.stack(
          [vit_layers.representation(layer) for layer in rep_sigreg_layers], dim=0)  # [L, B, D]
        if self.config.nepa_sigreg_rep_use_projector:
          L = reps.size(0)
          rep_sigreg_in = self.proj(reps.reshape(-1, reps.size(-1))).reshape(L, B, -1)  # [L, B, proj_dim]
        else:
          rep_sigreg_in = reps  # [L, B, D]
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          with torch.no_grad():
            sigreg_rep_stats = _masked_embedding_distribution_stats(
              rep_sigreg_in.detach(), None, prefix='nepa_diag_sigreg_rep_in')
          diag_stats.update(sigreg_rep_stats)
        loss_sigreg_rep = self.sigreg(rep_sigreg_in)  # scalar
        losses['loss_sigreg_rep'] = loss_sigreg_rep
        loss += self.config.nepa_sigreg_rep_scale * loss_sigreg_rep

    elif self.config.nepa_stability == 'rdmreg':
      # Step 1: configure SWD target parameters.
      num_projections = self.config.nepa_rdmreg_num_projections
      if num_projections is None:
        raise RuntimeError('nepa_rdmreg_num_projections must be set when nepa_stability="rdmreg"')
      lp_norm_parameter = self.config.nepa_rdmreg_p
      if lp_norm_parameter is None:
        raise RuntimeError('nepa_rdmreg_p must be set when nepa_stability="rdmreg"')
      mean_shift_value = self.config.nepa_rdmreg_mean_shift
      if mean_shift_value is None:
        raise RuntimeError('nepa_rdmreg_mean_shift must be set when nepa_stability="rdmreg"')
      sigma_mode = self.config.nepa_rdmreg_sigma_mode
      if sigma_mode is None:
        raise RuntimeError('nepa_rdmreg_sigma_mode must be set when nepa_stability="rdmreg"')
      if sigma_mode == 'gn_unit_var':
        chosen_sigma = float(determine_sigma_for_lp_dist(float(lp_norm_parameter)))
      elif sigma_mode == 'manual':
        chosen_sigma = self.config.nepa_rdmreg_sigma
        if chosen_sigma is None:
          raise RuntimeError('nepa_rdmreg_sigma must be set when nepa_rdmreg_sigma_mode="manual"')
      else:
        raise RuntimeError(f'Invalid nepa_rdmreg_sigma_mode: {sigma_mode!r}; this should be validated in the config.')
      activation = self.config.nepa_rdmreg_feature_activation
      if activation is None:
        raise RuntimeError('nepa_rdmreg_feature_activation must be set when nepa_stability="rdmreg"')
      target_dist = self.config.nepa_rdmreg_target_dist
      if target_dist is None:
        raise RuntimeError('nepa_rdmreg_target_dist must be set when nepa_stability="rdmreg"')

      # RDMReg_batch (matches SIGReg_batch sites)
      if self.config.nepa_sigreg_scale is not None and self.config.nepa_sigreg_scale > 0:
        rdmreg_layers = self.config.nepa_sigreg_layers
        if rdmreg_layers is None:
          raise ValueError('nepa_sigreg_layers must be set when nepa_sigreg_scale > 0')
        tokens = torch.cat([_stability_tokens(layer) for layer in rdmreg_layers], dim=0)  # [L*B, T, D']
        features_by_view = tokens.transpose(0, 1)  # [T, L*B, D']
        if token_mask is None:
          mask_by_view = None
        else:
          token_mask_time = token_mask.transpose(0, 1)  # [T, B]
          mask_by_view = token_mask_time.repeat(1, len(rdmreg_layers))  # [T, L*B]
        projections = sample_unit_sphere_projections(
          num_projections=num_projections,
          dim=features_by_view.size(-1),
          device=features_by_view.device,
          dtype=features_by_view.dtype)  # [N, D']
        loss_rdmreg = rdmreg_swd_mean_over_views(
          features_by_view=features_by_view,
          mask_by_view=mask_by_view,
          projection_vectors=projections,
          target_dist=str(target_dist),
          mean_shift_value=float(mean_shift_value),
          lp_norm_parameter=float(lp_norm_parameter),
          chosen_sigma=float(chosen_sigma),
          feature_activation=str(activation),
        )
        losses['loss_rdmreg'] = loss_rdmreg
        loss += self.config.nepa_sigreg_scale * loss_rdmreg
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          flat = features_by_view.reshape(-1, features_by_view.size(-1))  # [T*L*B, D']
          with torch.no_grad():
            diag_stats.update(rdmreg_diag_update_for_features(
              diag_key_prefix='nepa_diag_rdmreg_batch_features',
              features_2d=flat.detach(),
              feature_activation=str(activation)))

      # RDMReg_time (matches SIGReg_time sites)
      if self.config.nepa_sigreg_scale_time is not None and self.config.nepa_sigreg_scale_time > 0:
        rdmreg_time_layers = self.config.nepa_sigreg_time_layers
        if rdmreg_time_layers is None:
          raise ValueError('nepa_sigreg_time_layers must be set when nepa_sigreg_scale_time is set')
        tokens = torch.cat([_stability_tokens(layer) for layer in rdmreg_time_layers], dim=0)  # [L*B, T, D']
        features_by_view = tokens  # [L*B, T, D'] (views = samples)
        if token_mask is None:
          mask_by_view = None
        else:
          mask_by_view = token_mask.repeat(len(rdmreg_time_layers), 1)  # [L*B, T]
        projections = sample_unit_sphere_projections(
          num_projections=num_projections,
          dim=features_by_view.size(-1),
          device=features_by_view.device,
          dtype=features_by_view.dtype)  # [N, D']
        loss_rdmreg_time = rdmreg_swd_mean_over_views(
          features_by_view=features_by_view,
          mask_by_view=mask_by_view,
          projection_vectors=projections,
          target_dist=str(target_dist),
          mean_shift_value=float(mean_shift_value),
          lp_norm_parameter=float(lp_norm_parameter),
          chosen_sigma=float(chosen_sigma),
          feature_activation=str(activation),
        )
        losses['loss_rdmreg_time'] = loss_rdmreg_time
        loss += self.config.nepa_sigreg_scale_time * loss_rdmreg_time
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          flat = features_by_view.reshape(-1, features_by_view.size(-1))  # [L*B*T, D']
          with torch.no_grad():
            diag_stats.update(rdmreg_diag_update_for_features(
              diag_key_prefix='nepa_diag_rdmreg_time_features',
              features_2d=flat.detach(),
              feature_activation=str(activation)))

      # RDMReg_batch_time (matches SIGReg_batch_time sites)
      if self.config.nepa_sigreg_scale_batch_time is not None and self.config.nepa_sigreg_scale_batch_time > 0:
        rdmreg_batch_time_layers = self.config.nepa_sigreg_batch_time_layers
        if rdmreg_batch_time_layers is None:
          raise ValueError(
            'nepa_sigreg_batch_time_layers must be set when nepa_sigreg_scale_batch_time is set')
        tokens = torch.stack([_stability_tokens(layer) for layer in rdmreg_batch_time_layers], dim=0)  # [L, B, T, D']
        L_layers = tokens.size(0)
        features_by_view = tokens.reshape(L_layers, B * T, -1)  # [L, B*T, D']
        if token_mask is None:
          mask_by_view = None
        else:
          mask_by_view = token_mask.reshape(1, B * T).repeat(L_layers, 1)  # [L, B*T]
        projections = sample_unit_sphere_projections(
          num_projections=num_projections,
          dim=features_by_view.size(-1),
          device=features_by_view.device,
          dtype=features_by_view.dtype)  # [N, D']
        loss_rdmreg_batch_time = rdmreg_swd_mean_over_views(
          features_by_view=features_by_view,
          mask_by_view=mask_by_view,
          projection_vectors=projections,
          target_dist=str(target_dist),
          mean_shift_value=float(mean_shift_value),
          lp_norm_parameter=float(lp_norm_parameter),
          chosen_sigma=float(chosen_sigma),
          feature_activation=str(activation),
        )
        losses['loss_rdmreg_batch_time'] = loss_rdmreg_batch_time
        loss += self.config.nepa_sigreg_scale_batch_time * loss_rdmreg_batch_time
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          flat = features_by_view.reshape(-1, features_by_view.size(-1))  # [L*B*T, D']
          with torch.no_grad():
            diag_stats.update(rdmreg_diag_update_for_features(
              diag_key_prefix='nepa_diag_rdmreg_batch_time_features',
              features_2d=flat.detach(),
              feature_activation=str(activation)))

      # RDMReg_innovation (matches SIGReg_innovation sites)
      if self.config.nepa_sigreg_innov_scale is not None and self.config.nepa_sigreg_innov_scale > 0:
        if T < 2:
          raise ValueError('nepa_sigreg_innov_scale requires at least 2 tokens to form innovations')
        rdmreg_innov_layers = self.config.nepa_sigreg_innov_layers
        if rdmreg_innov_layers is None:
          raise ValueError('nepa_sigreg_innov_layers must be set when nepa_sigreg_innov_scale is set')
        tokens = torch.cat([_stability_tokens(layer) for layer in rdmreg_innov_layers], dim=0)  # [L*B, T, D']
        innov = tokens[:, 1:, :] - tokens[:, :-1, :]  # [L*B, T-1, D']
        features_by_view = innov.transpose(0, 1)  # [T-1, L*B, D']
        if token_mask is None:
          mask_by_view = None
        else:
          innov_mask = token_mask[:, 1:] & token_mask[:, :-1]  # [B, T-1]
          if not innov_mask.any():
            raise ValueError('token_mask leaves no valid innovation pairs for RDMReg')
          innov_mask_time = innov_mask.transpose(0, 1)  # [T-1, B]
          mask_by_view = innov_mask_time.repeat(1, len(rdmreg_innov_layers))  # [T-1, L*B]
        projections = sample_unit_sphere_projections(
          num_projections=num_projections,
          dim=features_by_view.size(-1),
          device=features_by_view.device,
          dtype=features_by_view.dtype)  # [N, D']
        loss_rdmreg_innov = rdmreg_swd_mean_over_views(
          features_by_view=features_by_view,
          mask_by_view=mask_by_view,
          projection_vectors=projections,
          target_dist=str(target_dist),
          mean_shift_value=float(mean_shift_value),
          lp_norm_parameter=float(lp_norm_parameter),
          chosen_sigma=float(chosen_sigma),
          feature_activation=str(activation),
        )
        losses['loss_rdmreg_innov'] = loss_rdmreg_innov
        loss += self.config.nepa_sigreg_innov_scale * loss_rdmreg_innov
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          flat = features_by_view.reshape(-1, features_by_view.size(-1))  # [(T-1)*L*B, D']
          with torch.no_grad():
            diag_stats.update(rdmreg_diag_update_for_features(
              diag_key_prefix='nepa_diag_rdmreg_innov_features',
              features_2d=flat.detach(),
              feature_activation=str(activation)))

      # RDMReg_rep (matches SIGReg_rep sites)
      if self.config.nepa_sigreg_rep_scale is not None and self.config.nepa_sigreg_rep_scale > 0:
        rep_layers = self.config.nepa_sigreg_rep_layers
        if rep_layers is None:
          raise ValueError('nepa_sigreg_rep_layers must be set when nepa_sigreg_rep_scale is set')
        reps = torch.stack([vit_layers.representation(layer) for layer in rep_layers], dim=0)  # [L, B, D]
        if self.config.nepa_sigreg_rep_use_projector:
          L_layers = reps.size(0)
          reps = self.proj(reps.reshape(-1, reps.size(-1))).reshape(L_layers, B, -1)  # [L, B, D']
        features_by_view = reps  # [L, B, D']
        projections = sample_unit_sphere_projections(
          num_projections=num_projections,
          dim=features_by_view.size(-1),
          device=features_by_view.device,
          dtype=features_by_view.dtype)  # [N, D']
        loss_rdmreg_rep = rdmreg_swd_mean_over_views(
          features_by_view=features_by_view,
          mask_by_view=None,
          projection_vectors=projections,
          target_dist=str(target_dist),
          mean_shift_value=float(mean_shift_value),
          lp_norm_parameter=float(lp_norm_parameter),
          chosen_sigma=float(chosen_sigma),
          feature_activation=str(activation),
        )
        losses['loss_rdmreg_rep'] = loss_rdmreg_rep
        loss += self.config.nepa_sigreg_rep_scale * loss_rdmreg_rep
        if self.config.nepa_distribution_diagnostics and diag_stats is not None:
          flat = features_by_view.reshape(-1, features_by_view.size(-1))  # [L*B, D']
          with torch.no_grad():
            diag_stats.update(rdmreg_diag_update_for_features(
              diag_key_prefix='nepa_diag_rdmreg_rep_features',
              features_2d=flat.detach(),
              feature_activation=str(activation)))

    elif self.config.nepa_stability == "wristband":
      wristband = self.wristband
      if wristband is None:
        raise RuntimeError(
          'wristband loss is not initialized; expected when nepa_stability="wristband". '
          'This is a bug.')
      if token_mask is not None:
        raise ValueError(
          'nepa_stability="wristband" currently requires token_mask=null '
          '(fixed-length tokens; wristband masking is not implemented).')

      max_samples = self.config.nepa_wristband_max_samples
      if max_samples is None:
        raise RuntimeError(
          'nepa_wristband_max_samples must be set when nepa_stability="wristband"; '
          'this should have been validated in the config.')
      max_samples = int(max_samples)

      def _subsample(features_by_view: torch.Tensor) -> torch.Tensor:
        """Randomly subsample the sample axis to cap wristband O(N^2) cost."""
        # features_by_view: [V, N, D']
        if features_by_view.dim() != 3:
          raise ValueError(f'features_by_view must be [V, N, D], got {tuple(features_by_view.shape)}')
        V, N, _Dproj = features_by_view.shape
        if N < 2:
          return features_by_view
        if N <= max_samples:
          return features_by_view
        idx = torch.randperm(N, device=features_by_view.device)[:max_samples]  # [max_samples]
        return features_by_view.index_select(1, idx)  # [V, max_samples, D']

      def _log_and_accumulate(*, prefix: str, scale: float, features_by_view: torch.Tensor) -> None:
        """Compute wristband components, log them, and add to total loss."""
        nonlocal loss
        features_by_view = _subsample(features_by_view)  # [V, N<=max, D']
        lc = wristband(features_by_view)
        losses[f"loss_wristband_{prefix}"] = lc.total
        losses[f"loss_wristband_{prefix}_rep"] = lc.rep
        losses[f"loss_wristband_{prefix}_rad"] = lc.rad
        losses[f"loss_wristband_{prefix}_ang"] = lc.ang
        losses[f"loss_wristband_{prefix}_mom"] = lc.mom
        loss += float(scale) * lc.total

      # Wristband_time (priority: mirrors Plan 110 baseline 1:1).
      if self.config.nepa_wristband_scale_time is not None and self.config.nepa_wristband_scale_time > 0:
        time_layers = self.config.nepa_wristband_time_layers
        if time_layers is None:
          raise RuntimeError(
            'nepa_wristband_time_layers must be set when nepa_wristband_scale_time is set; '
            'this should have been validated in the config.')
        tokens = torch.cat([_stability_tokens(layer) for layer in time_layers], dim=0)  # [L*B, T, D']
        _log_and_accumulate(
          prefix="time",
          scale=float(self.config.nepa_wristband_scale_time),
          features_by_view=tokens,
        )

      # Wristband_batch (mirrors SIGReg_batch sites).
      if self.config.nepa_wristband_scale is not None and self.config.nepa_wristband_scale > 0:
        batch_layers = self.config.nepa_wristband_layers
        if batch_layers is None:
          raise RuntimeError(
            'nepa_wristband_layers must be set when nepa_wristband_scale is set; '
            'this should have been validated in the config.')
        tokens = torch.cat([_stability_tokens(layer) for layer in batch_layers], dim=0)  # [L*B, T, D']
        features_by_view = tokens.transpose(0, 1)  # [T, L*B, D']
        _log_and_accumulate(
          prefix="batch",
          scale=float(self.config.nepa_wristband_scale),
          features_by_view=features_by_view,
        )

      # Wristband_batch_time (mirrors SIGReg_batch_time sites).
      if self.config.nepa_wristband_scale_batch_time is not None and self.config.nepa_wristband_scale_batch_time > 0:
        batch_time_layers = self.config.nepa_wristband_batch_time_layers
        if batch_time_layers is None:
          raise RuntimeError(
            'nepa_wristband_batch_time_layers must be set when nepa_wristband_scale_batch_time is set; '
            'this should have been validated in the config.')
        tokens = torch.stack([_stability_tokens(layer) for layer in batch_time_layers], dim=0)  # [L, B, T, D']
        L_layers = tokens.size(0)
        features_by_view = tokens.reshape(L_layers, B * T, -1)  # [L, B*T, D']
        _log_and_accumulate(
          prefix="batch_time",
          scale=float(self.config.nepa_wristband_scale_batch_time),
          features_by_view=features_by_view,
        )

      # Wristband_innovation (mirrors SIGReg_innovation sites).
      if self.config.nepa_wristband_innov_scale is not None and self.config.nepa_wristband_innov_scale > 0:
        if T < 2:
          raise ValueError('nepa_wristband_innov_scale requires at least 2 tokens to form innovations')
        innov_layers = self.config.nepa_wristband_innov_layers
        if innov_layers is None:
          raise RuntimeError(
            'nepa_wristband_innov_layers must be set when nepa_wristband_innov_scale is set; '
            'this should have been validated in the config.')
        tokens = torch.cat([_stability_tokens(layer) for layer in innov_layers], dim=0)  # [L*B, T, D']
        innov = tokens[:, 1:, :] - tokens[:, :-1, :]  # [L*B, T-1, D']
        _log_and_accumulate(
          prefix="innov",
          scale=float(self.config.nepa_wristband_innov_scale),
          features_by_view=innov,
        )

      # Wristband_rep (mirrors SIGReg_rep sites).
      if self.config.nepa_wristband_rep_scale is not None and self.config.nepa_wristband_rep_scale > 0:
        rep_layers = self.config.nepa_wristband_rep_layers
        if rep_layers is None:
          raise RuntimeError(
            'nepa_wristband_rep_layers must be set when nepa_wristband_rep_scale is set; '
            'this should have been validated in the config.')
        reps = torch.stack([vit_layers.representation(layer) for layer in rep_layers], dim=0)  # [L, B, D]
        if self.proj is not None:
          L_layers = reps.size(0)
          reps = self.proj(reps.reshape(-1, reps.size(-1))).reshape(L_layers, B, -1)  # [L, B, D']
        _log_and_accumulate(
          prefix="rep",
          scale=float(self.config.nepa_wristband_rep_scale),
          features_by_view=reps,
        )

    if diag_stats is not None:
      for key, value in diag_stats.items():
        losses[key] = value

    # Store the final loss.
    losses['loss'] = loss

    if return_representation:
      if return_patch_loss:
        return losses, rep, patch_loss
      return losses, rep
    if return_patch_loss:
      return losses, patch_loss
    return losses

  def get_optimizer(self, fused=False):
    decay_modules = (nn.Linear, nn.Conv1d)
    decay = set()
    for module_name, module in self.named_modules():
      for param_name, param in module.named_parameters():
        if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
          param_name = f'{module_name}.{param_name}' if module_name else param_name
          decay.add(param_name)

    decay_params, non_decay_params = OrderedDict(), OrderedDict()
    for name, param in self.named_parameters():
      if param.requires_grad:
        if name in decay:
          decay_params[name] = param
        else:
          non_decay_params[name] = param

    param_groups = [
      {'params': list(decay_params.values()),
       'weight_decay': self.config.weight_decay,
       'use_weight_decay': True},
      {'params': list(non_decay_params.values()),
       'weight_decay': 0.,
       'use_weight_decay': False}
    ]

    optimizer = torch.optim.AdamW(
      param_groups,
      lr=self.config.learning_rate,
      betas=self.config.opt_betas,
      eps=self.config.opt_eps,
      weight_decay=0.,
      fused=fused)

    return optimizer
