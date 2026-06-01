from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
  def __init__(self, dim: int, base: int = 10000):
    super().__init__()
    if dim % 2 != 0:
      raise ValueError(f'RoPE requires even head_dim, got {dim}')
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer('inv_freq', inv_freq, persistent=False)
    self._seq_len_cached = None
    self._device_cached = None
    self._dtype_cached = None
    self._cos_cached = None
    self._sin_cached = None

  def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
    positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
    freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    self._seq_len_cached = seq_len
    self._device_cached = device
    self._dtype_cached = dtype
    self._cos_cached = cos
    self._sin_cached = sin

  def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
    if (self._cos_cached is None
        or self._seq_len_cached != seq_len
        or self._device_cached != device
        or self._dtype_cached != dtype):
      self._build_cache(seq_len, device, dtype)
    return self._cos_cached, self._sin_cached

  def _apply_rotary(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
  ) -> torch.Tensor:
    B, H, N, D = x.shape
    x = x.view(B, H, N, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    if cos.dim() == 2:
      cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D/2]
      sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D/2]
    elif cos.dim() == 3:
      if cos.size(0) != B or cos.size(1) != N:
        raise ValueError(
          'cos/sin must match x batch/seq dims when providing per-sample RoPE positions: '
          f'x.shape={tuple(x.shape)}, cos.shape={tuple(cos.shape)}')
      cos = cos.unsqueeze(1)  # [B, 1, N, D/2]
      sin = sin.unsqueeze(1)  # [B, 1, N, D/2]
    else:
      raise ValueError(f'cos must be [N, D/2] or [B, N, D/2], got {tuple(cos.shape)}')
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)

  def apply(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      *,
      prefix_len: int = 0,
      positions: torch.Tensor | None = None,
  ):
    seq_len = q.size(-2)
    if prefix_len < 0 or prefix_len > seq_len:
      raise ValueError(f'prefix_len must be in [0, {seq_len}], got {prefix_len}')
    if prefix_len == seq_len:
      if positions is not None and positions.numel() != 0:
        raise ValueError('positions must be empty when prefix_len==seq_len (no tokens to rotate)')
      return q, k
    q_prefix = q[..., :prefix_len, :]
    k_prefix = k[..., :prefix_len, :]
    q_rot = q[..., prefix_len:, :]
    k_rot = k[..., prefix_len:, :]
    if positions is None:
      cos, sin = self._get_cos_sin(q_rot.size(-2), q.device, q.dtype)  # [N, D/2]
    else:
      if positions.device != q.device:
        raise ValueError('positions must be on the same device as q/k')
      if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'positions must be int tensor, got {positions.dtype}')
      if positions.dim() == 1:
        if positions.size(0) != q_rot.size(-2):
          raise ValueError(
            'positions must match rotated sequence length: '
            f'positions.shape={tuple(positions.shape)}, expected ({q_rot.size(-2)},)')
        pos = positions.to(dtype=torch.long)  # [N]
        if (pos < 0).any():
          raise ValueError('positions must be non-negative')
        max_pos = int(pos.max().item())
        cos_full, sin_full = self._get_cos_sin(max_pos + 1, q.device, q.dtype)  # [P, D/2]
        cos = cos_full.index_select(0, pos)  # [N, D/2]
        sin = sin_full.index_select(0, pos)  # [N, D/2]
      elif positions.dim() == 2:
        if positions.size(0) != q_rot.size(0) or positions.size(1) != q_rot.size(-2):
          raise ValueError(
            'positions must be [B, N] matching rotated q/k: '
            f'positions.shape={tuple(positions.shape)}, q_rot.shape={tuple(q_rot.shape)}')
        pos = positions.to(dtype=torch.long)  # [B, N]
        if (pos < 0).any():
          raise ValueError('positions must be non-negative')
        max_pos = int(pos.max().item())
        cos_full, sin_full = self._get_cos_sin(max_pos + 1, q.device, q.dtype)  # [P, D/2]
        pos_flat = pos.reshape(-1)  # [B*N]
        cos = cos_full.index_select(0, pos_flat).view(pos.size(0), pos.size(1), -1)  # [B, N, D/2]
        sin = sin_full.index_select(0, pos_flat).view(pos.size(0), pos.size(1), -1)  # [B, N, D/2]
      else:
        raise ValueError(f'positions must be [N] or [B, N], got {tuple(positions.shape)}')
    q_rot = self._apply_rotary(q_rot, cos, sin)
    k_rot = self._apply_rotary(k_rot, cos, sin)
    q = torch.cat([q_prefix, q_rot], dim=-2)
    k = torch.cat([k_prefix, k_rot], dim=-2)
    return q, k


class QKNorm(nn.Module):
  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.norm(x)


class Block(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      mlp_ratio=4.,
      qkv_bias=False,
      bias=True,
      dropout=0.,
      attn_dropout=0.,
      eps=1e-6,
      layer_scale_eps=0.,
      use_rope=False,
      rope_base=10000,
      use_qk_norm=False,
      qk_norm_eps=1e-6,
      use_swiglu=False,
      use_sdp_kernel=True,
  ):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=eps, bias=bias)
    self.attn = Attention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      bias=bias,
      attn_dropout=attn_dropout,
      proj_dropout=dropout,
      use_rope=use_rope,
      rope_base=rope_base,
      use_qk_norm=use_qk_norm,
      qk_norm_eps=qk_norm_eps,
      use_sdp_kernel=use_sdp_kernel)
    self.attn_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
    self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
    if use_swiglu:
      mlp_hidden_dim = int((2 / 3) * mlp_ratio * dim)
      if mlp_hidden_dim <= 0:
        raise ValueError(f'swiglu hidden_dim must be > 0, got {mlp_hidden_dim}')
      self.mlp = GatedMLP(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        dropout=dropout,
        bias=bias)
    else:
      mlp_hidden_dim = int(dim * mlp_ratio)
      self.mlp = MLP(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        dropout=dropout,
        bias=bias)
    self.mlp_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()

  def forward(
      self,
      x,
      *,
      is_causal: bool = False,
      rope_prefix_len: int = 0,
      rope_positions: torch.Tensor | None = None,
      attn_mask: torch.Tensor | None = None,
      drift_stats: dict[str, torch.Tensor] | None = None,
      drift_layer_idx: int | None = None,
  ):
    """Apply attention + MLP block with optional attention mask."""
    def _rms_all(values: torch.Tensor) -> torch.Tensor:
      """Return scalar RMS over all elements (computed in float32)."""
      return values.to(dtype=torch.float32).square().mean().sqrt()  # []

    x_in = x  # [B, N, D]
    h = self.norm1(x_in)  # [B, N, D] normalized tokens.
    attn_out = self.attn(
      h,
      is_causal=is_causal,
      rope_prefix_len=rope_prefix_len,
      rope_positions=rope_positions,
      attn_mask=attn_mask)  # [B, N, D] attention output.
    attn_delta = self.attn_ls(attn_out)  # [B, N, D]
    x_after_attn = x_in + attn_delta  # [B, N, D] residual attention.
    mlp_out = self.mlp(self.norm2(x_after_attn))  # [B, N, D] MLP output.
    mlp_delta = self.mlp_ls(mlp_out)  # [B, N, D]
    x_out = x_after_attn + mlp_delta  # [B, N, D] residual MLP.

    if drift_stats is not None:
      if drift_layer_idx is None:
        raise ValueError('drift_layer_idx must be set when drift_stats is provided')
      x_in_rms = _rms_all(x_in)  # []
      x_after_attn_rms = _rms_all(x_after_attn)  # []
      if x_in_rms <= 0 or x_after_attn_rms <= 0:
        raise ValueError(
          'Expected positive RMS for residual stream when computing drift diagnostics, '
          f'got x_in_rms={x_in_rms.item()}, x_after_attn_rms={x_after_attn_rms.item()}')
      drift_stats[f'nepa_diag_vit_attn_delta_rms_ratio_layer{drift_layer_idx}'] = (
        _rms_all(attn_delta) / x_in_rms)  # []
      drift_stats[f'nepa_diag_vit_mlp_delta_rms_ratio_layer{drift_layer_idx}'] = (
        _rms_all(mlp_delta) / x_after_attn_rms)  # []

    return x_out


class CrossAttentionBlock(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      mlp_ratio=4.,
      qkv_bias=False,
      bias=True,
      eps=1e-6,
      use_qk_norm=False,
      qk_norm_eps=1e-6,
      use_swiglu=False,
      use_sdp_kernel=True,
  ):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=eps, bias=bias)
    self.xattn = CrossAttention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      bias=bias,
      use_qk_norm=use_qk_norm,
      qk_norm_eps=qk_norm_eps,
      use_sdp_kernel=use_sdp_kernel)
    self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
    if use_swiglu:
      mlp_hidden_dim = int((2 / 3) * mlp_ratio * dim)
      if mlp_hidden_dim <= 0:
        raise ValueError(f'swiglu hidden_dim must be > 0, got {mlp_hidden_dim}')
      self.mlp = GatedMLP(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        bias=bias)
    else:
      mlp_hidden_dim = int(dim * mlp_ratio)
      self.mlp = MLP(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        bias=bias)

  def forward(self, q, x):
    q = q + self.xattn(q, self.norm1(x))
    q = q + self.mlp(self.norm2(q))
    return q


class AttentivePooler(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      num_queries=1,
      depth=1,
      mlp_ratio=4.,
      qkv_bias=False,
      bias=True,
      complete_block=False,
      proj_dim=None,
      eps=1e-6,
      use_sdp_kernel=True
  ):
    super().__init__()
    self.query_token = nn.Parameter(torch.empty(1, num_queries, dim))
    nn.init.trunc_normal_(self.query_token, mean=0., std=0.02)
    self.blocks = None
    if complete_block:
      self.cross_attention_block = CrossAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        bias=bias,
        eps=eps,
        use_sdp_kernel=use_sdp_kernel)
      if depth > 1:
        self.blocks = nn.ModuleList([
          Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            bias=bias,
            eps=1e-6,
            use_sdp_kernel=use_sdp_kernel)
          for _ in range(depth - 1)])
    else:
      self.cross_attention_block = CrossAttention(
        dim=dim,
        num_heads=num_heads,
        proj_dim=proj_dim,
        qkv_bias=qkv_bias,
        bias=bias,
        use_sdp_kernel=use_sdp_kernel)

  def forward(self, x):
    q = self.query_token.repeat(len(x), 1, 1)
    q = self.cross_attention_block(q, x)
    if self.blocks is not None:
      for block in self.blocks:
        q = block(q)
    return q


class Attention(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      qkv_bias=False,
      bias=True,
      attn_dropout=0.,
      proj_dropout=0.,
      use_rope=False,
      rope_base=10000,
      use_qk_norm=False,
      qk_norm_eps=1e-6,
      use_sdp_kernel=True
  ):
    super().__init__()
    self.num_heads = num_heads
    self.use_sdp_kernel = use_sdp_kernel
    assert dim % num_heads == 0
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5
    self.rope = RotaryEmbedding(head_dim, base=rope_base) if use_rope else None
    self.qk_norm = QKNorm(head_dim, eps=qk_norm_eps) if use_qk_norm else None
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    if self.use_sdp_kernel:
      self.attn_dropout = attn_dropout
    else:
      self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout else nn.Identity()
    self.proj = nn.Linear(dim, dim, bias=bias)
    self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout else nn.Identity()

  def forward(
      self,
      x,
      *,
      is_causal: bool = False,
      rope_prefix_len: int = 0,
      rope_positions: torch.Tensor | None = None,
      attn_mask: torch.Tensor | None = None,
  ):
    """Compute self-attention with optional mask (True = attend)."""
    if attn_mask is not None and is_causal:
      raise ValueError('attn_mask cannot be combined with is_causal=True; bake causal into attn_mask.')
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
    if self.rope is not None:
      q, k = self.rope.apply(q, k, prefix_len=rope_prefix_len, positions=rope_positions)
    elif rope_positions is not None:
      raise ValueError('rope_positions provided but use_rope=False (rope is not initialized)')
    if self.qk_norm is not None:
      q = self.qk_norm(q)
      k = self.qk_norm(k)
    if self.use_sdp_kernel:
      if attn_mask is None:
        x = F.scaled_dot_product_attention(
          q, k, v, dropout_p=self.attn_dropout, is_causal=is_causal)
      else:
        x = F.scaled_dot_product_attention(
          q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=False)
    else:
      attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
      if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
          attn = attn.masked_fill(~attn_mask, torch.finfo(attn.dtype).min)
        else:
          attn = attn + attn_mask
      elif is_causal:
        causal_mask = torch.ones((N, N), device=attn.device, dtype=torch.bool).triu(diagonal=1)
        attn = attn.masked_fill(causal_mask, torch.finfo(attn.dtype).min)
      attn = attn.softmax(dim=-1)
      attn = self.attn_dropout(attn)
      x = (attn @ v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_dropout(x)
    return x


class CrossAttention(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      proj_dim=None,
      qkv_bias=False,
      bias=True,
      use_qk_norm=False,
      qk_norm_eps=1e-6,
      use_sdp_kernel=True
  ):
    super().__init__()
    self.num_heads = num_heads
    assert dim % num_heads == 0
    head_dim = dim // num_heads
    proj_dim = proj_dim or dim
    self.scale = head_dim ** -0.5
    self.qk_norm = QKNorm(head_dim, eps=qk_norm_eps) if use_qk_norm else None
    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
    self.proj = nn.Linear(dim, proj_dim, bias=bias)
    self.use_sdp_kernel = use_sdp_kernel

  def forward(self, q, x):
    B, n, C = q.shape
    q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    B, N, C = x.shape
    kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)
    if self.qk_norm is not None:
      q = self.qk_norm(q)
      k = self.qk_norm(k)
    if self.use_sdp_kernel:
      q = F.scaled_dot_product_attention(q, k, v)
    else:
      xattn = (q @ k.transpose(-2, -1)) * self.scale
      xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
      self.xattn = xattn.detach()  # store attention weights for later inspection
      q = (xattn @ v)
    q = q.transpose(1, 2).reshape(B, n, C)
    q = self.proj(q)
    return q


class ProjectorMLP(nn.Sequential):
  """This block implements the multi-layer perceptron (MLP) module.

  Args:
      in_channels (int): Number of channels of the input
      hidden_channels (List[int]): List of the hidden channel dimensions
      norm_layer (Callable[..., nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
      activation_layer (Callable[..., nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``nn.ReLU``
      inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
          Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
      bias (bool): Whether to use bias in the linear layer. Default ``True``
      dropout (float): The probability for the dropout layer. Default: 0.0
  """

  def __init__(
      self,
      in_channels: int,
      hidden_channels: list[int],
      norm_layer: Callable[..., nn.Module] | None = None,
      activation_layer: Callable[..., nn.Module] = nn.ReLU,
      inplace: bool | None = None,
      bias: bool = True,
      dropout: float = 0.0,
  ):
    # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
    # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
    params = {} if inplace is None else {"inplace": inplace}

    layers = []
    in_dim = in_channels
    for hidden_dim in hidden_channels[:-1]:
      layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
      if norm_layer is not None:
        layers.append(norm_layer(hidden_dim))
      layers.append(activation_layer(**params))
      layers.append(nn.Dropout(dropout, **params))
      in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
    layers.append(nn.Dropout(dropout, **params))

    super().__init__(*layers)


class MLP(nn.Module):
  def __init__(
      self,
      in_features,
      hidden_features=None,
      out_features=None,
      dropout=0.,
      bias=True
  ):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.act = nn.GELU()
    self.drop1 = nn.Dropout(dropout) if dropout else nn.Identity()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop2 = nn.Dropout(dropout) if dropout else nn.Identity()

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class GatedMLP(nn.Module):
  def __init__(
      self,
      in_features,
      hidden_features=None,
      out_features=None,
      dropout=0.,
      bias=True
  ):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
    self.act = nn.SiLU()
    self.drop1 = nn.Dropout(dropout) if dropout else nn.Identity()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop2 = nn.Dropout(dropout) if dropout else nn.Identity()

  def forward(self, x):
    x = self.fc1(x)
    gate, value = x.chunk(2, dim=-1)
    x = self.act(gate) * value
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class PatchEmbedding(nn.Module):
  def __init__(
      self,
      dim,
      in_channels,
      patch_size,
      bias=True
  ):
    super().__init__()
    self.proj = nn.Conv1d(
      in_channels,
      dim,
      kernel_size=patch_size,
      stride=patch_size,
      bias=bias)

  def forward(self, x):
    x = self.proj(x).transpose(1, 2)
    return x


class LayerScale(nn.Module):
  """https://arxiv.org/abs/2103.17239"""
  def __init__(self, dim: int, eps: float = 0.1):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(eps * torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.weight
