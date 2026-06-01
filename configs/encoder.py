from dataclasses import dataclass


@dataclass
class EncoderConfig:
  # data
  sampling_frequency: int
  channels: tuple[str, ...]
  channel_size: int
  patch_size: int
  masking_strategy: str
  # model architecture
  dim: int
  depth: int
  num_heads: int
  mlp_ratio: float
  qkv_bias: bool
  dropout: float
  attn_dropout: float
  num_registers: int
  bias: bool
  norm_eps: float
  layer_scale_eps: float
  # architecture optimizations
  use_rope: bool | None
  rope_base: int | None
  use_swiglu: bool | None
  use_qk_norm: bool | None
  qk_norm_eps: float | None
  pos_embed_type: str | None

  @property
  def num_channels(self) -> int:
    return len(self.channels)

  def __post_init__(self):
    self.channels = tuple(self.channels)

    if self.use_rope is None:
      raise ValueError('use_rope must be explicitly set (True or False)')
    if self.use_swiglu is None:
      raise ValueError('use_swiglu must be explicitly set (True or False)')
    if self.use_qk_norm is None:
      raise ValueError('use_qk_norm must be explicitly set (True or False)')
    if self.pos_embed_type is None:
      raise ValueError('pos_embed_type must be explicitly set ("sin", "learned", or "none")')
    allowed_pos_embed = {'sin', 'learned', 'none'}
    if self.pos_embed_type not in allowed_pos_embed:
      raise ValueError(
        f'pos_embed_type must be one of {sorted(allowed_pos_embed)}, got {self.pos_embed_type}')
    allowed_masking = {'remove', 'zero', 'mask'}
    if self.masking_strategy not in allowed_masking:
      raise ValueError(
        f'masking_strategy must be one of {sorted(allowed_masking)}, got {self.masking_strategy}')
    if self.use_rope:
      if self.rope_base is None:
        raise ValueError('rope_base must be set when use_rope=True')
      head_dim = self.dim // self.num_heads
      if head_dim % 2 != 0:
        raise ValueError(f'RoPE requires even head_dim, got {head_dim} (dim={self.dim}, num_heads={self.num_heads})')
    if self.use_qk_norm and self.qk_norm_eps is None:
      raise ValueError('qk_norm_eps must be set when use_qk_norm=True')
