"""Causal Mantis-like tokenizer for NEPA/LeNEPA.

This module implements a Mantis-style token generator adapted for NEPA:

- Input: multichannel 1D signal `x` with shape [B, C, L]
- Output: patch tokens with shape [B, T, D], where T = L / patch_size

Key constraints for NEPA:
- No future leakage: token t must depend only on samples within patches <= t.
- Patch-local instance normalization is used to avoid non-causal global statistics.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ScalarEncoder(nn.Module):
  """Affine + LayerNorm scalar encoder used inside MultiScaledScalarEncoder.

  For scalar input $x \\in \\mathbb{R}$, the encoder computes:
  $$z = x \\cdot w + k \\cdot b,\\quad y = \\mathrm{LN}(z)$$

  where `w, b` are learnable vectors in R^{H} and `k` is the fixed scale.
  """

  def __init__(self, *, k: float, hidden_dim: int, eps: float, bias: bool):
    super().__init__()
    if hidden_dim < 1:
      raise ValueError(f'hidden_dim must be >= 1, got {hidden_dim}')
    if not isinstance(k, (int, float)):
      raise ValueError(f'k must be a number, got {type(k).__name__}')
    if eps <= 0:
      raise ValueError(f'eps must be > 0, got {eps}')

    self.k = float(k)
    self.w = nn.Parameter(torch.rand((1, hidden_dim), dtype=torch.float32, requires_grad=True))
    self.b = nn.Parameter(torch.rand((1, hidden_dim), dtype=torch.float32, requires_grad=True))
    self.norm = nn.LayerNorm(hidden_dim, eps=eps, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode scalar tensor.

    Args:
      x: [..., 1] scalar values.

    Returns:
      [..., H] encoded features.
    """
    if x.size(-1) != 1:
      raise ValueError(f'Expected x[..., 1], got x.shape={tuple(x.shape)}')
    z = x * self.w + self.k * self.b  # [..., H]
    return self.norm(z)  # [..., H]


class MultiScaledScalarEncoder(nn.Module):
  """Multi-scaled encoding of a scalar variable (NuTime-style).

  Given scalar x and a set of scales {s_i}, we compute per-scale encoder outputs y_i(x)
  and blend them with weights alpha_i(x) depending on the relative magnitude of x:

  $$\\alpha_i(x) \\propto \\left|\\frac{1}{\\log(|x|/s_i + \\epsilon)}\\right|,$$
  normalized to sum to 1 over i.
  """

  def __init__(
    self,
    *,
    scales: tuple[float, ...],
    hidden_dim: int,
    epsilon: float,
    eps: float,
    bias: bool,
  ):
    super().__init__()
    if not scales:
      raise ValueError('scales must be non-empty')
    if any(s <= 0 for s in scales):
      raise ValueError(f'All scales must be > 0, got scales={scales}')
    if hidden_dim < 1:
      raise ValueError(f'hidden_dim must be >= 1, got {hidden_dim}')
    if epsilon <= 0:
      raise ValueError(f'epsilon must be > 0, got {epsilon}')
    if eps <= 0:
      raise ValueError(f'eps must be > 0, got {eps}')

    scales_t = torch.tensor(scales, dtype=torch.float32)  # [S]
    self.register_buffer('scales', scales_t, persistent=False)
    self.epsilon = float(epsilon)
    self.encoders = nn.ModuleList([
      ScalarEncoder(k=float(scale), hidden_dim=hidden_dim, eps=eps, bias=bias)
      for scale in scales
    ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode scalar tensor with multi-scale mixing.

    Args:
      x: [..., 1] scalar values.

    Returns:
      [..., H] encoded features.
    """
    if x.size(-1) != 1:
      raise ValueError(f'Expected x[..., 1], got x.shape={tuple(x.shape)}')

    # Step 1: compute scale weights alpha(x) in float32 for stability.
    x_abs = x.abs().to(dtype=torch.float32)  # [..., 1]
    scales = self.scales.to(device=x.device)  # [S]
    inv_scales = (1.0 / scales).reshape(1, -1)  # [1, S]
    ratio = torch.matmul(x_abs, inv_scales)  # [..., S]
    alpha = (1.0 / torch.log(ratio + self.epsilon)).abs()  # [..., S]
    alpha = alpha / alpha.sum(dim=-1, keepdim=True)  # [..., S]
    alpha = alpha.unsqueeze(-1)  # [..., S, 1]

    # Step 2: encode at each scale and blend.
    encoded_list = [encoder(x) for encoder in self.encoders]  # S * [..., H]
    encoded = torch.stack(encoded_list, dim=-2)  # [..., S, H]
    mixed = (encoded.to(dtype=torch.float32) * alpha).sum(dim=-2)  # [..., H]
    return mixed.to(dtype=x.dtype)  # [..., H]


class CausalConv1d(nn.Module):
  """Causal Conv1d implemented as left-pad + Conv1d(padding=0).

  PyTorch Conv1d padding is symmetric, so to guarantee causality we explicitly left-pad
  by (kernel_size - 1) samples.
  """

  def __init__(
    self,
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool,
  ):
    super().__init__()
    if kernel_size < 1:
      raise ValueError(f'kernel_size must be >= 1, got {kernel_size}')
    self.kernel_size = int(kernel_size)
    self.conv = nn.Conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=1,
      padding=0,
      bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply causal convolution.

    Args:
      x: [B, C, L]

    Returns:
      [B, D, L]
    """
    if x.dim() != 3:
      raise ValueError(f'Expected x with shape [B, C, L], got {tuple(x.shape)}')
    pad_len = self.kernel_size - 1
    x_pad = F.pad(x, (pad_len, 0))  # [B, C, L+pad_len]
    return self.conv(x_pad)  # [B, D, L]


class MantisTokGen1D(nn.Module):
  """Causal Mantis-like token generator producing patch tokens [B, T, D].

  Branches (per patch t):
  - signal conv branch on patch-normalized x
  - diff conv branch on patch-normalized first differences
  - raw patch statistics (per-channel mean/std) encoded with MSSE

  Notes on causality:
  - Convolutions are causal in `stride1_pool` mode via left padding.
  - Patch-local instance normalization is used (per patch) to avoid non-causal global stats.
  """

  def __init__(
    self,
    *,
    dim: int,
    in_channels: int,
    patch_size: int,
    kernel_size: int,
    conv_mode: str,
    instance_norm_mode: str,
    instance_norm_eps: float,
    stats_mode: str,
    stats_dim: int,
    scalar_scales: tuple[float, ...],
    scalar_hidden_dim: int,
    scalar_epsilon: float,
    bias: bool,
    norm_eps: float,
  ):
    super().__init__()
    if dim < 1:
      raise ValueError(f'dim must be >= 1, got {dim}')
    if in_channels < 1:
      raise ValueError(f'in_channels must be >= 1, got {in_channels}')
    if patch_size < 1:
      raise ValueError(f'patch_size must be >= 1, got {patch_size}')
    if kernel_size < 1:
      raise ValueError(f'kernel_size must be >= 1, got {kernel_size}')
    if stats_dim < 1:
      raise ValueError(f'stats_dim must be >= 1, got {stats_dim}')
    if instance_norm_eps <= 0:
      raise ValueError(f'instance_norm_eps must be > 0, got {instance_norm_eps}')
    if norm_eps <= 0:
      raise ValueError(f'norm_eps must be > 0, got {norm_eps}')

    allowed_conv_modes = ('patch_embed', 'stride1_pool')
    if conv_mode not in allowed_conv_modes:
      raise ValueError(f'conv_mode must be one of {sorted(allowed_conv_modes)}, got {conv_mode!r}')
    allowed_norm_modes = ('per_channel', 'global_over_channels')
    if instance_norm_mode not in allowed_norm_modes:
      raise ValueError(
        'instance_norm_mode must be one of '
        f'{sorted(allowed_norm_modes)}, got {instance_norm_mode!r}')
    allowed_stats_modes = ('flatten_project', 'per_channel_slots')
    if stats_mode not in allowed_stats_modes:
      raise ValueError(f'stats_mode must be one of {sorted(allowed_stats_modes)}, got {stats_mode!r}')
    if conv_mode == 'patch_embed' and kernel_size != patch_size:
      raise ValueError(
        'conv_mode="patch_embed" requires kernel_size==patch_size for unambiguous patch tokens, '
        f'got kernel_size={kernel_size}, patch_size={patch_size}')

    self.dim = int(dim)
    self.in_channels = int(in_channels)
    self.patch_size = int(patch_size)
    self.kernel_size = int(kernel_size)
    self.conv_mode = str(conv_mode)
    self.instance_norm_mode = str(instance_norm_mode)
    self.instance_norm_eps = float(instance_norm_eps)
    self.stats_mode = str(stats_mode)
    self.stats_dim = int(stats_dim)

    # Scalar encoders: separate MSSE modules for mean and std (as in Mantis).
    self.mean_encoder = MultiScaledScalarEncoder(
      scales=scalar_scales,
      hidden_dim=scalar_hidden_dim,
      epsilon=scalar_epsilon,
      eps=norm_eps,
      bias=bias)
    self.std_encoder = MultiScaledScalarEncoder(
      scales=scalar_scales,
      hidden_dim=scalar_hidden_dim,
      epsilon=scalar_epsilon,
      eps=norm_eps,
      bias=bias)
    self.scalar_hidden_dim = int(scalar_hidden_dim)

    # Conv branches.
    if self.conv_mode == 'patch_embed':
      self.signal_conv = nn.Conv1d(
        in_channels=self.in_channels,
        out_channels=self.dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        padding=0,
        bias=bias)
      self.diff_conv = nn.Conv1d(
        in_channels=self.in_channels,
        out_channels=self.dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        padding=0,
        bias=bias)
    else:
      self.signal_conv = CausalConv1d(
        in_channels=self.in_channels,
        out_channels=self.dim,
        kernel_size=self.kernel_size,
        bias=bias)
      self.diff_conv = CausalConv1d(
        in_channels=self.in_channels,
        out_channels=self.dim,
        kernel_size=self.kernel_size,
        bias=bias)

    self.signal_ln = nn.LayerNorm(self.dim, eps=norm_eps, bias=bias)
    self.diff_ln = nn.LayerNorm(self.dim, eps=norm_eps, bias=bias)

    # Stats branch projection.
    stats_in_dim = self.in_channels * self.scalar_hidden_dim  # [C*H] after per-channel slots.
    if self.stats_mode == 'flatten_project':
      flat_in_dim = self.in_channels * 2 * self.scalar_hidden_dim  # [C*(2H)]
      self.stats_proj = nn.Linear(flat_in_dim, self.stats_dim, bias=bias)
      self.stats_norm = nn.LayerNorm(self.stats_dim, eps=norm_eps, bias=bias)
      self.stats_channel_proj = None
      self.stats_channel_norms = None
      self.stats_slots_proj = None
    else:
      # Per-channel slots: encode each channel independently to H, then project to stats_dim.
      self.stats_channel_proj = nn.ModuleList([
        nn.Linear(2 * self.scalar_hidden_dim, self.scalar_hidden_dim, bias=bias)
        for _ in range(self.in_channels)
      ])
      self.stats_channel_norms = nn.ModuleList([
        nn.LayerNorm(self.scalar_hidden_dim, eps=norm_eps, bias=bias)
        for _ in range(self.in_channels)
      ])
      self.stats_slots_proj = nn.Linear(stats_in_dim, self.stats_dim, bias=bias)
      self.stats_norm = nn.LayerNorm(self.stats_dim, eps=norm_eps, bias=bias)
      self.stats_proj = None

    # Final token projection (concat signal/diff/stats).
    final_in_dim = 2 * self.dim + self.stats_dim
    self.final_proj = nn.Linear(final_in_dim, self.dim, bias=bias)
    self.final_norm = nn.LayerNorm(self.dim, eps=norm_eps, bias=bias)

  def _patch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std per patch.

    Args:
      x: [B, C, L]

    Returns:
      (mean_patch, std_patch): both [B, C, T, 1] in float32.
    """
    B, C, L = x.shape
    if L % self.patch_size != 0:
      raise ValueError(
        'Expected L divisible by patch_size, '
        f'got L={L}, patch_size={self.patch_size}')
    T = L // self.patch_size
    x_patches = x.reshape(B, C, T, self.patch_size).to(dtype=torch.float32)  # [B, C, T, P]
    mean_patch = x_patches.mean(dim=-1, keepdim=True)  # [B, C, T, 1]
    var_patch = x_patches.var(dim=-1, unbiased=False, keepdim=True)  # [B, C, T, 1]
    std_patch = var_patch.sqrt()  # [B, C, T, 1]
    return mean_patch, std_patch

  def _patch_instance_norm(
    self,
    x: torch.Tensor,
    *,
    mean_patch: torch.Tensor,
    std_patch: torch.Tensor,
  ) -> torch.Tensor:
    """Patch-local instance normalization.

    Args:
      x: [B, C, L]
      mean_patch: [B, C, T, 1] per-channel patch means (float32).
      std_patch: [B, C, T, 1] per-channel patch stds (float32).

    Returns:
      x_norm: [B, C, L] normalized in x.dtype.
    """
    B, C, L = x.shape
    T = L // self.patch_size
    x_patches = x.reshape(B, C, T, self.patch_size)  # [B, C, T, P]

    if self.instance_norm_mode == 'per_channel':
      mean = mean_patch.to(dtype=x.dtype)  # [B, C, T, 1]
      std = std_patch.to(dtype=x.dtype)  # [B, C, T, 1]
    else:
      # One mean/std per patch shared across channels (preserves relative channel scales).
      mean = mean_patch.mean(dim=1, keepdim=True).to(dtype=x.dtype)  # [B, 1, T, 1]
      x_f32 = x_patches.to(dtype=torch.float32)  # [B, C, T, P]
      mean_f32 = mean.to(dtype=torch.float32)  # [B, 1, T, 1]
      var = (x_f32 - mean_f32).square().mean(dim=(1, 3), keepdim=True)  # [B, 1, T, 1]
      std = var.sqrt().to(dtype=x.dtype)  # [B, 1, T, 1]

    x_norm = (x_patches - mean) / (std + self.instance_norm_eps)  # [B, C, T, P]
    return x_norm.reshape(B, C, L)  # [B, C, L]

  def _encode_stats(
    self,
    *,
    mean_patch: torch.Tensor,
    std_patch: torch.Tensor,
  ) -> torch.Tensor:
    """Encode per-patch raw statistics into stats tokens.

    Args:
      mean_patch: [B, C, T, 1] float32.
      std_patch: [B, C, T, 1] float32.

    Returns:
      stats: [B, T, D_stats] (same dtype as encoder parameters, usually float32).
    """
    # Step 1: MSSE encode mean/std (per channel).
    e_mean = self.mean_encoder(mean_patch)  # [B, C, T, H]
    e_std = self.std_encoder(std_patch)  # [B, C, T, H]
    e = torch.cat([e_mean, e_std], dim=-1)  # [B, C, T, 2H]

    # Step 2: combine channels according to stats_mode.
    if self.stats_mode == 'flatten_project':
      B, C, T, D2 = e.shape
      e_flat = e.permute(0, 2, 1, 3).reshape(B, T, C * D2)  # [B, T, C*(2H)]
      if self.stats_proj is None:
        raise RuntimeError('stats_proj is not initialized for flatten_project; this is a bug.')
      stats = self.stats_proj(e_flat)  # [B, T, D_stats]
      return self.stats_norm(stats)  # [B, T, D_stats]

    if self.stats_channel_proj is None or self.stats_channel_norms is None or self.stats_slots_proj is None:
      raise RuntimeError('Per-channel stats modules are not initialized; this is a bug.')

    channel_slots = []
    for ch in range(self.in_channels):
      e_ch = e[:, ch, :, :]  # [B, T, 2H]
      slot = self.stats_channel_proj[ch](e_ch)  # [B, T, H]
      slot = self.stats_channel_norms[ch](slot)  # [B, T, H]
      channel_slots.append(slot)
    slots = torch.cat(channel_slots, dim=-1)  # [B, T, C*H]
    stats = self.stats_slots_proj(slots)  # [B, T, D_stats]
    return self.stats_norm(stats)  # [B, T, D_stats]

  def _diff_causal(self, x: torch.Tensor) -> torch.Tensor:
    """Compute first differences with a causal 0 at t=0.

    Args:
      x: [B, C, L]

    Returns:
      d: [B, C, L] where d[:,:,0]=0 and d[:,:,t]=x[:,:,t]-x[:,:,t-1].
    """
    B, C, L = x.shape
    if L < 1:
      raise ValueError(f'Expected L >= 1, got L={L}')
    d0 = x.new_zeros((B, C, 1))  # [B, C, 1]
    d_rest = x[:, :, 1:] - x[:, :, :-1]  # [B, C, L-1]
    return torch.cat([d0, d_rest], dim=-1)  # [B, C, L]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Generate patch tokens.

    Args:
      x: [B, C, L]

    Returns:
      tokens: [B, T, D]
    """
    # Step 0: validate shapes.
    if x.dim() != 3:
      raise ValueError(f'Expected x with shape [B, C, L], got {tuple(x.shape)}')
    B, C, L = x.shape
    if C != self.in_channels:
      raise ValueError(
        'Expected x channels to match in_channels, '
        f'got x.shape={tuple(x.shape)}, in_channels={self.in_channels}')
    if L % self.patch_size != 0:
      raise ValueError(
        'Expected L divisible by patch_size, '
        f'got L={L}, patch_size={self.patch_size}')
    T = L // self.patch_size

    # Step 1: per-patch stats on raw signal (float32 for stability).
    mean_patch, std_patch = self._patch_stats(x)  # [B, C, T, 1], [B, C, T, 1]

    # Step 2: patch-local instance normalization (causal at patch resolution).
    x_norm = self._patch_instance_norm(x, mean_patch=mean_patch, std_patch=std_patch)  # [B, C, L]

    # Step 3: diff branch input.
    d = self._diff_causal(x_norm)  # [B, C, L]

    # Step 4: conv branches (signal + diff).
    if self.conv_mode == 'patch_embed':
      if not isinstance(self.signal_conv, nn.Conv1d) or not isinstance(self.diff_conv, nn.Conv1d):
        raise RuntimeError('Expected Conv1d modules for patch_embed; this is a bug.')
      ux_t = self.signal_conv(x_norm)  # [B, D, T]
      ud_t = self.diff_conv(d)  # [B, D, T]
      ux = self.signal_ln(ux_t.transpose(1, 2))  # [B, T, D]
      ud = self.diff_ln(ud_t.transpose(1, 2))  # [B, T, D]
    else:
      fx_t = self.signal_conv(x_norm)  # [B, D, L]
      fd_t = self.diff_conv(d)  # [B, D, L]
      fx = self.signal_ln(fx_t.transpose(1, 2))  # [B, L, D]
      fd = self.diff_ln(fd_t.transpose(1, 2))  # [B, L, D]
      ux = fx.reshape(B, T, self.patch_size, self.dim).mean(dim=2)  # [B, T, D]
      ud = fd.reshape(B, T, self.patch_size, self.dim).mean(dim=2)  # [B, T, D]

    # Step 5: stats branch.
    stats = self._encode_stats(mean_patch=mean_patch, std_patch=std_patch)  # [B, T, D_stats]

    # Step 6: fuse branches into final tokens.
    fused = torch.cat([ux, ud, stats.to(dtype=ux.dtype)], dim=-1)  # [B, T, 2D+D_stats]
    tokens = self.final_proj(fused)  # [B, T, D]
    return self.final_norm(tokens)  # [B, T, D]

