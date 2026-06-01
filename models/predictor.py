import math

import torch
from torch import nn

import configs
from models.modules import Block
from models.utils import get_1d_pos_embed, get_learned_pos_embed, apply_mask


class Predictor(nn.Module):
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    self.num_patches = num_patches
    self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, config.pred_dim))
    # Positional embeddings: sinusoidal (buffer), learned (parameter), or none.
    if config.pos_embed_type == 'learned':
      self.pos_embed = get_learned_pos_embed(dim=config.pred_dim, num_patches=num_patches)
    elif config.pos_embed_type == 'sin':
      self.register_buffer(
        'pos_embed',
        get_1d_pos_embed(dim=config.pred_dim, num_patches=num_patches),
        persistent=False)
    else:
      self.pos_embed = None
    # Note: Predictor blocks use pred_num_heads, so we calculate head_dim for RoPE validation
    # If use_rope=True but pred_dim // pred_num_heads is odd, RoPE will raise an error
    self.blocks = nn.ModuleList([
      Block(
        dim=config.pred_dim,
        num_heads=config.pred_num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        use_rope=config.use_rope,
        rope_base=config.rope_base or 10000,
        use_qk_norm=config.use_qk_norm,
        qk_norm_eps=config.qk_norm_eps or 1e-6,
        use_swiglu=config.use_swiglu,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(config.pred_depth)
    ])
    self.norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
    self.proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
          # residual projections are initialized with scaled std
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * config.depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
          nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
    nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)

  def forward(self, x, mask_encoder, mask_predictor):
    # x: [B, 1+M, D] for remove, [B, 1+N, D] for zero/mask
    B, K = mask_predictor.size()
    # Split CLS from patches
    cls, patches = x[:, :1], x[:, 1:]
    # Prepare positional embeddings for patches only
    pos_embed = self.pos_embed.repeat(B, 1, 1) if self.pos_embed is not None else None  # [B, N, pred_dim] or None
    num_patches = self.num_patches
    if self.config.masking_strategy == 'remove':
      if patches.size(1) != mask_encoder.size(1):
        raise ValueError(
          f'Expected {mask_encoder.size(1)} kept patches, got {patches.size(1)}')
      # Embed and add positional info to patches (no position for CLS)
      cls = self.embed(cls)
      patches = self.embed(patches)
      # Build mask tokens with positional embeddings
      mask_token = self.mask_token.repeat(B, K, 1)
      if pos_embed is not None:
        pos_encoder = apply_mask(pos_embed, mask_encoder)
        pos_predictor = apply_mask(pos_embed, mask_predictor)
        patches = patches + pos_encoder
        mask_token = mask_token + pos_predictor
      # Construct sequence: [CLS, kept_patches, mask_tokens]
      x = torch.cat([cls, patches, mask_token], dim=1)
    elif self.config.masking_strategy in ('zero', 'mask'):
      if patches.size(1) != num_patches:
        raise ValueError(
          f'Expected {num_patches} patch tokens for zero/mask masking, got {patches.size(1)}')
      cls = self.embed(cls)
      patches = self.embed(patches)
      if pos_embed is not None:
        patches = patches + pos_embed
      if K > 0:
        mask_token = self.mask_token.repeat(B, K, 1)
        if pos_embed is not None:
          pos_predictor = apply_mask(pos_embed, mask_predictor)
          mask_token = mask_token + pos_predictor
        index = mask_predictor.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        patches = patches.scatter(1, index, mask_token)
      x = torch.cat([cls, patches], dim=1)
    else:
      raise ValueError(f'Unknown masking_strategy: {self.config.masking_strategy}')
    # RoPE prefix length: don't rotate CLS token
    rope_prefix_len = 1 if self.config.use_rope else 0
    for block in self.blocks:
      x = block(x, rope_prefix_len=rope_prefix_len)
    x = self.norm(x)
    x = self.proj(x)
    # Drop CLS from output, return only patch tokens [B, M+K, D] = [B, N, D]
    return x[:, 1:]


class AutoregressivePredictor(nn.Module):
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    assert config.channel_size % config.patch_size == 0
    self.num_patches = config.channel_size // config.patch_size
    # Step 1: project encoder tokens into predictor space.
    self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
    # Positional embeddings: sinusoidal (buffer), learned (parameter), or none.
    if config.pos_embed_type == 'learned':
      self.pos_embed = get_learned_pos_embed(dim=config.pred_dim, num_patches=self.num_patches)
    elif config.pos_embed_type == 'sin':
      self.register_buffer(
        'pos_embed',
        get_1d_pos_embed(dim=config.pred_dim, num_patches=self.num_patches),
        persistent=False)
    else:
      self.pos_embed = None
    # Note: Predictor blocks use pred_num_heads, so we calculate head_dim for RoPE validation
    # If use_rope=True but pred_dim // pred_num_heads is odd, RoPE will raise an error
    self.blocks = nn.ModuleList([
      Block(
        dim=config.pred_dim,
        num_heads=config.pred_num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        use_rope=config.use_rope,
        rope_base=config.rope_base or 10000,
        use_qk_norm=config.use_qk_norm,
        qk_norm_eps=config.qk_norm_eps or 1e-6,
        use_swiglu=config.use_swiglu,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(config.pred_depth)
    ])
    self.norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
    self.proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
          # residual projections are initialized with scaled std
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * config.depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
          nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, z_patches: torch.Tensor) -> torch.Tensor:
    # z_patches: [B, N, D]
    if z_patches.dim() != 3:
      raise ValueError(f'Expected z_patches with shape [B, N, D], got {tuple(z_patches.shape)}')
    if z_patches.size(1) != self.num_patches:
      raise ValueError(
        f'Expected {self.num_patches} patch tokens, got {z_patches.size(1)}')
    if z_patches.size(-1) != self.config.dim:
      raise ValueError(
        f'Expected embedding dim {self.config.dim}, got {z_patches.size(-1)}')
    # Step 2: add predictor positional embeddings for causal modeling.
    x = self.embed(z_patches)  # [B, N, pred_dim]
    if self.pos_embed is not None:
      x = x + self.pos_embed[:, :self.num_patches]  # [B, N, pred_dim]
    rope_prefix_len = 0
    # Step 3: causal transformer blocks (token t attends to <= t).
    for block in self.blocks:
      x = block(x, is_causal=True, rope_prefix_len=rope_prefix_len)  # [B, N, pred_dim]
    x = self.norm(x)  # [B, N, pred_dim]
    x = self.proj(x)  # [B, N, D]
    return x  # [B, N, D]
