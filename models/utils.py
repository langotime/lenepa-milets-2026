import math

import torch


def apply_mask(x, mask):  # x: (B, N, D); mask: (B, K) with values in range [0, N)
  mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
  x = torch.gather(x, dim=1, index=mask)
  return x  # x: (B, K, D)


def zero_masked_inputs(x, mask, patch_size):  # x: (B, C, L); mask: (B, K) keep indices
  if x.dim() != 3:
    raise ValueError(f'Expected x with shape [B, C, L], got {tuple(x.shape)}')
  if mask.dim() != 2:
    raise ValueError(f'Expected mask with shape [B, K], got {tuple(mask.shape)}')
  if x.size(0) != mask.size(0):
    raise ValueError(f'Expected mask batch {mask.size(0)} to match x batch {x.size(0)}')
  if x.size(-1) % patch_size != 0:
    raise ValueError(
      f'channel_size {x.size(-1)} must be divisible by patch_size {patch_size}')
  if mask.device != x.device:
    raise ValueError('mask must be on the same device as x')
  if mask.dtype not in (torch.int32, torch.int64):
    raise ValueError(f'mask must be int tensor, got {mask.dtype}')
  num_patches = x.size(-1) // patch_size
  if mask.numel() > 0:
    min_idx = int(mask.min().item())
    max_idx = int(mask.max().item())
    if min_idx < 0 or max_idx >= num_patches:
      raise ValueError(
        f'mask indices must be within [0, {num_patches - 1}], got min {min_idx}, max {max_idx}')
  keep = torch.zeros((x.size(0), num_patches), device=x.device, dtype=torch.bool)
  keep.scatter_(1, mask, True)
  keep = keep.repeat_interleave(patch_size, dim=1)
  return x * keep.unsqueeze(1).to(dtype=x.dtype)


def get_1d_pos_embed(dim, num_patches):
  assert dim % 2 == 0
  position = torch.arange(num_patches).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
  pos_embed = torch.zeros(1, num_patches, dim)
  pos_embed[0, :, 0::2] = torch.sin(position * div_term)
  pos_embed[0, :, 1::2] = torch.cos(position * div_term)
  return pos_embed


def get_learned_pos_embed(dim, num_patches):
  """Create learnable positional embeddings initialized with trunc_normal."""
  from torch import nn
  pos_embed = nn.Parameter(torch.empty(1, num_patches, dim))
  nn.init.trunc_normal_(pos_embed, mean=0., std=0.02)
  return pos_embed


def loss_patch_inv_masked_local_global(
  *,
  local_patches: torch.Tensor,
  global_patches: torch.Tensor,
  mask_encoder: torch.Tensor,
  mask_encoder_global: torch.Tensor | None,
  num_local_views: int,
  num_global_views: int,
) -> torch.Tensor:
  # Algorithm:
  # (1) mark masked local patches,
  # (2) build global target from unmasked globals,
  # (3) compute per-patch MSE for masked locals, normalized by masked-view count,
  # (4) average only over patches with valid global targets.
  if local_patches.dim() != 3:
    raise ValueError(f'local_patches must be [B*V, N, D], got {tuple(local_patches.shape)}')
  if global_patches.dim() != 3:
    raise ValueError(f'global_patches must be [B*Vg, N, D], got {tuple(global_patches.shape)}')
  if local_patches.size(0) % num_local_views != 0:
    raise ValueError(
      f'local_patches batch {local_patches.size(0)} must be divisible by num_local_views {num_local_views}')
  if global_patches.size(0) % num_global_views != 0:
    raise ValueError(
      f'global_patches batch {global_patches.size(0)} must be divisible by num_global_views {num_global_views}')
  batch_size = local_patches.size(0) // num_local_views
  if global_patches.size(0) // num_global_views != batch_size:
    raise ValueError(
      'local_patches and global_patches must have the same batch size after view grouping')
  if local_patches.size(1) != global_patches.size(1):
    raise ValueError(
      f'local_patches N={local_patches.size(1)} must match global_patches N={global_patches.size(1)}')
  if mask_encoder.size(0) != batch_size * num_local_views:
    raise ValueError(
      f'mask_encoder batch {mask_encoder.size(0)} must match local_patches {batch_size * num_local_views}')
  if mask_encoder_global is not None and mask_encoder_global.size(0) != batch_size * num_global_views:
    raise ValueError(
      f'mask_encoder_global batch {mask_encoder_global.size(0)} must match global_patches {batch_size * num_global_views}')

  num_patches = local_patches.size(1)
  device = local_patches.device

  # Group views per sample for patch-wise aggregation.
  local = local_patches.reshape(batch_size, num_local_views, num_patches, -1)     # [B, V, N, D]
  global_ = global_patches.reshape(batch_size, num_global_views, num_patches, -1) # [B, Vg, N, D]

  # Local: masked positions are those not kept by mask_encoder.
  keep_local = torch.zeros(
    (batch_size * num_local_views, num_patches), device=device, dtype=torch.bool)
  keep_local.scatter_(1, mask_encoder, True)
  keep_local = keep_local.reshape(batch_size, num_local_views, num_patches) # [B, V, N]
  masked_local = ~keep_local # [B, V, N]

  # Global: only unmasked patches contribute to target computation.
  if mask_encoder_global is None:
    keep_global = torch.ones((batch_size, num_global_views, num_patches), device=device, dtype=torch.bool) # [B, Vg, N]
  else:
    keep_global_flat = torch.zeros((batch_size * num_global_views, num_patches), device=device, dtype=torch.bool)
    keep_global_flat.scatter_(1, mask_encoder_global, True)
    keep_global = keep_global_flat.reshape(batch_size, num_global_views, num_patches) # [B, Vg, N]

  # Target per patch index: mean over unmasked global views only.
  global_counts = keep_global.sum(dim=1)
  target_sum = (global_ * keep_global.unsqueeze(-1)).sum(dim=1)
  targets = torch.zeros_like(target_sum)
  valid_global = global_counts > 0
  if valid_global.any():
    targets[valid_global] = (target_sum[valid_global] / global_counts[valid_global].to(target_sum.dtype).unsqueeze(-1))

  # MSE per patch, normalized by how many local views masked that patch.
  diff = (local - targets.unsqueeze(1)).square().mean(dim=-1) # [B, V, N]
  local_counts = masked_local.sum(dim=1)
  patch_sum = (diff * masked_local.to(diff.dtype)).sum(dim=1)
  per_patch = torch.zeros_like(patch_sum)
  valid_local = local_counts > 0
  if valid_local.any():
    per_patch[valid_local] = (patch_sum[valid_local] / local_counts[valid_local].to(patch_sum.dtype))

  valid = valid_local & valid_global
  if not valid.any():
    raise ValueError(
      'loss_patch_inv_masked_local_global found no patches masked locally and unmasked globally')
  return per_patch[valid].mean()
