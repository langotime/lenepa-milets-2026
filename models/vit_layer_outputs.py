from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _masked_batch_norm_features(
  values: torch.Tensor,
  mask: torch.Tensor | None,
  *,
  eps: float,
) -> torch.Tensor:
  """Batch-normalize features over the first dimension with an optional mask.

  This is a stateless alternative to `nn.BatchNorm1d(affine=False)` for 2D tensors.

  Args:
    values: [N, D] feature tensor.
    mask: Optional [N] boolean mask selecting valid rows (True = valid).
    eps: Small constant added to the variance for numerical stability.

  Returns:
    [N, D] float32 batch-normalized tensor (mean≈0, var≈1 over valid rows).
  """
  # Step 1: validate shapes/dtypes.
  if values.dim() != 2:
    raise ValueError(f'values must be [N, D], got {tuple(values.shape)}')
  if not isinstance(eps, (int, float)):
    raise ValueError(f'eps must be a float, got {type(eps)}')
  if eps <= 0:
    raise ValueError(f'eps must be > 0, got {eps}')
  values_f32 = values.to(dtype=torch.float32)  # [N, D]
  num_rows = values_f32.size(0)
  if num_rows < 1:
    raise ValueError(f'values must have N >= 1, got {tuple(values.shape)}')

  # Step 2: compute moments over valid rows.
  if mask is None:
    mean = values_f32.mean(dim=0, keepdim=True)  # [1, D]
    var = values_f32.var(dim=0, unbiased=False, keepdim=True)  # [1, D]
  else:
    if mask.dtype is not torch.bool:
      raise ValueError(f'mask must be bool, got {mask.dtype}')
    if mask.dim() != 1:
      raise ValueError(f'mask must be [N], got {tuple(mask.shape)}')
    if mask.size(0) != num_rows:
      raise ValueError(
        'mask must match values first dim, '
        f'values.shape={tuple(values.shape)}, mask.shape={tuple(mask.shape)}')
    if not mask.any():
      raise ValueError('mask must include at least one valid row')
    valid = values_f32[mask]  # [K, D]
    mean = valid.mean(dim=0, keepdim=True)  # [1, D]
    var = valid.var(dim=0, unbiased=False, keepdim=True)  # [1, D]

  # Step 3: apply normalization.
  return (values_f32 - mean) / torch.sqrt(var + eps)  # [N, D]


@dataclass(frozen=True)
class ViTLayerOutputs:
  """All-layer ViT outputs container (shared by NEPA and ViT-based JEPA encoders).

  Stores patch tokens for every transformer layer (with special tokens already stripped),
  plus optional per-layer CLS tokens for encoders that expose them (JEPA/LeJEPA).

  Layer indexing (depth = len(patch_tokens_pre_norm) - 1):
    - 0: token state before any transformer blocks (after patch embed + pos embed + special token assembly).
    - k in [1, depth]: token state after block (k-1), before output normalization.

  Notes:
    - `patch_tokens_pre_norm[layer]` are `[B, T, D]` for all layers.
    - `cls_tokens_pre_norm[layer]` (when present) are `[B, D]` for all layers.
    - `token_mask` (if present) is `[B, T]` and is used for masked pooling and mask-aware BatchNorm.
  """

  patch_tokens_pre_norm: tuple[torch.Tensor, ...]
  token_mask: torch.Tensor | None
  final_norm: str
  final_norm_weight: torch.Tensor | None
  final_norm_bias: torch.Tensor | None
  final_norm_eps: float | None
  output_norm: str
  output_bn_eps: float
  rep_pooling: str
  cls_tokens_pre_norm: tuple[torch.Tensor, ...] | None = None

  @property
  def depth(self) -> int:
    """Return the number of transformer blocks (depth)."""
    return len(self.patch_tokens_pre_norm) - 1

  def _apply_final_norm(self, tokens: torch.Tensor) -> torch.Tensor:
    """Apply the encoder's final normalization to a token tensor."""
    if self.final_norm == 'none':
      return tokens
    D = tokens.size(-1)
    eps = 1e-6 if self.final_norm_eps is None else self.final_norm_eps
    if self.final_norm == 'ln':
      if self.final_norm_weight is None:
        raise RuntimeError('final_norm_weight is required when final_norm="ln"')
      return F.layer_norm(
        tokens,
        normalized_shape=(D,),
        weight=self.final_norm_weight,
        bias=self.final_norm_bias,
        eps=eps)
    if self.final_norm == 'rms':
      if self.final_norm_weight is None:
        raise RuntimeError('final_norm_weight is required when final_norm="rms"')
      tokens_norm = tokens.to(dtype=self.final_norm_weight.dtype)
      normed = F.rms_norm(
        tokens_norm,
        normalized_shape=(D,),
        weight=self.final_norm_weight,
        eps=eps)
      return normed.to(dtype=tokens.dtype)
    raise ValueError(f'Unknown final_norm: {self.final_norm!r}')

  def patch_tokens(self, layer: int, apply_norm: bool | None = None) -> torch.Tensor:
    """Return patch tokens at the given layer.

    Semantics:
      - layer == depth: returns the canonical encoder output space (post-final normalization),
        regardless of `output_norm`. This keeps the model's main output stable even when
        layer-output normalization is disabled for intermediate layers.
      - layer in [0, depth-1]: applies `output_norm` to the requested layer tokens.

    Args:
      layer: ViT layer index in [0, depth].
      apply_norm: Optional override for the layer output normalization:
        - None: use the default semantics above.
        - False: return raw `patch_tokens_pre_norm[layer]` (also disables the final norm for layer=depth).
        - True: apply the default semantics above (kept for experimental toggles).
    """
    # Step 1: validate the layer index.
    if not isinstance(layer, int):
      raise ValueError(f'layer must be an int, got {layer!r}')
    if layer < 0 or layer > self.depth:
      raise ValueError(f'layer must be in [0, {self.depth}], got {layer}')

    # Step 2: select pre-normalized tokens.
    tokens = self.patch_tokens_pre_norm[layer]  # [B, T, D]
    if apply_norm is False:
      return tokens  # [B, T, D]

    # Step 3: apply output normalization.
    if layer == self.depth:
      return self._apply_final_norm(tokens)  # [B, T, D]

    if self.output_norm == 'none':
      return tokens  # [B, T, D]

    if self.output_norm == 'ln':
      if self.final_norm != 'ln':
        raise ValueError(
          'output_norm="ln" requires final_norm="ln", '
          f'got final_norm={self.final_norm!r}')
      D = tokens.size(-1)
      return F.layer_norm(
        tokens,
        normalized_shape=(D,),
        weight=self.final_norm_weight,
        bias=self.final_norm_bias,
        eps=self.final_norm_eps)  # [B, T, D]

    if self.output_norm == 'rms':
      if self.final_norm != 'rms':
        raise ValueError(
          'output_norm="rms" requires final_norm="rms", '
          f'got final_norm={self.final_norm!r}')
      D = tokens.size(-1)
      tokens_norm = tokens.to(dtype=self.final_norm_weight.dtype)
      normed = F.rms_norm(
        tokens_norm,
        normalized_shape=(D,),
        weight=self.final_norm_weight,
        eps=self.final_norm_eps)
      return normed.to(dtype=tokens.dtype)

    if self.output_norm == 'bn':
      if tokens.dim() != 3:
        raise ValueError(f'Expected tokens to be [B, T, D], got {tuple(tokens.shape)}')
      B, T, D = tokens.shape
      flat = tokens.reshape(-1, D)  # [B*T, D]
      mask_flat = None
      if self.token_mask is not None:
        if self.token_mask.shape != (B, T):
          raise ValueError(
            'token_mask must match tokens first two dims, '
            f'tokens.shape={tuple(tokens.shape)}, token_mask.shape={tuple(self.token_mask.shape)}')
        mask_flat = self.token_mask.reshape(-1)  # [B*T]
      flat_bn = _masked_batch_norm_features(flat, mask_flat, eps=self.output_bn_eps)  # [B*T, D] fp32
      return flat_bn.to(dtype=tokens.dtype).reshape(B, T, D)  # [B, T, D]

    raise ValueError(f'Unknown output_norm: {self.output_norm!r}')

  def representation(self, layer: int, apply_norm: bool | None = None) -> torch.Tensor:
    """Return pooled representation `[B, D]` at the given layer."""
    # Step 1: handle CLS pooling (JEPA-style representations).
    if self.rep_pooling == 'cls':
      if not isinstance(layer, int):
        raise ValueError(f'layer must be an int, got {layer!r}')
      if layer < 0 or layer > self.depth:
        raise ValueError(f'layer must be in [0, {self.depth}], got {layer}')
      if self.cls_tokens_pre_norm is None:
        raise ValueError('rep_pooling="cls" requires cls_tokens_pre_norm to be provided.')
      if len(self.cls_tokens_pre_norm) != self.depth + 1:
        raise ValueError(
          'cls_tokens_pre_norm must have one entry per layer in [0, depth]. '
          f'Got len(cls_tokens_pre_norm)={len(self.cls_tokens_pre_norm)}, depth={self.depth}')
      cls_tokens = self.cls_tokens_pre_norm[layer]  # [B, D]
      if cls_tokens.dim() != 2:
        raise ValueError(f'cls_tokens must be [B, D], got {tuple(cls_tokens.shape)}')
      if apply_norm is False:
        return cls_tokens  # [B, D]
      if layer == self.depth:
        return self._apply_final_norm(cls_tokens)  # [B, D]
      return cls_tokens  # [B, D]

    # Step 2: fetch normalized patch tokens.
    patch_tokens = self.patch_tokens(layer, apply_norm=apply_norm)  # [B, T, D]

    # Step 3: apply pooling.
    if self.rep_pooling == 'mean':
      if self.token_mask is None:
        return patch_tokens.mean(dim=1)  # [B, D]
      mask_f = self.token_mask.to(dtype=patch_tokens.dtype).unsqueeze(-1)  # [B, T, 1]
      denom = mask_f.sum(dim=1)  # [B, 1]
      if (denom <= 0).any():
        raise ValueError('token_mask must include at least one valid token per sample')
      return (patch_tokens * mask_f).sum(dim=1) / denom  # [B, D]

    if self.rep_pooling == 'last':
      if self.token_mask is None:
        return patch_tokens[:, -1, :]  # [B, D]
      valid_counts = self.token_mask.to(dtype=torch.long).sum(dim=1)  # [B]
      if (valid_counts <= 0).any():
        raise ValueError('token_mask must include at least one valid token per sample')
      last_idx = (valid_counts - 1).clamp_min(0)  # [B]
      rep = patch_tokens.gather(
        1, last_idx.view(-1, 1, 1).expand(-1, 1, patch_tokens.size(-1))).squeeze(1)  # [B, D]
      return rep  # [B, D]

    raise ValueError(f'Unknown rep_pooling: {self.rep_pooling!r}')

