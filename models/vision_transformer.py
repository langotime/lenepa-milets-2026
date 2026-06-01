import math

import torch
from torch import nn

import configs
from models.modules import PatchEmbedding, Block
from models.utils import get_1d_pos_embed, get_learned_pos_embed, apply_mask, zero_masked_inputs
from models.vit_layer_outputs import ViTLayerOutputs


class VisionTransformer(nn.Module):
  def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.keep_registers = keep_registers
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    self.patch_embed = PatchEmbedding(
      dim=config.dim,
      in_channels=config.num_channels,
      patch_size=config.patch_size,
      bias=config.bias)
    # Positional embeddings: sinusoidal (buffer), learned (parameter), or none.
    if config.pos_embed_type == 'learned':
      self.pos_embed = get_learned_pos_embed(dim=config.dim, num_patches=num_patches)
    elif config.pos_embed_type == 'sin':
      self.register_buffer(
        'pos_embed',
        get_1d_pos_embed(dim=config.dim, num_patches=num_patches),
        persistent=False)
    else:
      self.pos_embed = None
    # CLS token (no positional embedding, prepended to all sequences)
    self.cls_token = nn.Parameter(torch.empty(1, 1, config.dim))
    nn.init.trunc_normal_(self.cls_token, mean=0., std=0.02)
    self.mask_token = nn.Parameter(torch.empty(1, 1, config.dim))
    nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    self.blocks = nn.ModuleList([
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
      for _ in range(config.depth)
    ])
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, (nn.Linear, nn.Conv1d)):
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

  def forward(self, x, mask=None, *, is_causal: bool = False):
    if mask is not None and self.config.masking_strategy == 'zero':
      x = zero_masked_inputs(x, mask, self.config.patch_size)
    x = self.patch_embed(x)
    B, N, D = x.size()
    if mask is not None and self.config.masking_strategy == 'mask':
      keep = torch.zeros((B, N), device=x.device, dtype=torch.bool)
      keep.scatter_(1, mask, True)
      x = torch.where(keep.unsqueeze(-1), x, self.mask_token.to(dtype=x.dtype))
    if self.pos_embed is not None:
      x = x + self.pos_embed[:, :N]
    # Apply mask to patches only (masks are in patch index space)
    if mask is not None and self.config.masking_strategy == 'remove':
      x = apply_mask(x, mask)
    # Prepend special tokens: [CLS] + [REG_1..REG_R] + [PATCH_1..PATCH_N]
    cls = self.cls_token.expand(B, -1, -1)
    if self.config.num_registers > 0:
      registers = self.registers.expand(B, -1, -1)
      x = torch.cat([cls, registers, x], dim=1)
    else:
      x = torch.cat([cls, x], dim=1)
    # RoPE prefix length: don't rotate CLS and register tokens
    rope_prefix_len = (1 + self.config.num_registers) if self.config.use_rope else 0
    for block in self.blocks:
      x = block(x, is_causal=is_causal, rope_prefix_len=rope_prefix_len)
    x = self.norm(x)
    # Drop registers but keep CLS at index 0
    if not self.keep_registers and self.config.num_registers > 0:
      x = torch.cat([x[:, :1], x[:, 1 + self.config.num_registers:]], dim=1)
    return x

  def forward_vit_layers(self, x: torch.Tensor, *, is_causal: bool = False) -> tuple[ViTLayerOutputs, None]:
    """Run the ViT encoder while capturing per-layer token states for offline probing.

    Returns:
      (vit_layers, None):
        - vit_layers: `ViTLayerOutputs` with patch and CLS tokens for layers k in [0, depth].

    Layer indexing follows NEPA's `ViTLayerOutputs` convention:
      - k=0: token state after patch embed + pos embed + special token assembly (before any blocks).
      - k in [1, depth]: token state after block (k-1), pre-final-norm.
    """
    # Step 1: patch embedding + positional embedding.
    # x: [B, C, L]
    x = self.patch_embed(x)  # [B, N, D]
    B, N, _ = x.size()
    if self.pos_embed is not None:
      x = x + self.pos_embed[:, :N]  # [B, N, D]

    # Step 2: assemble special tokens: [CLS] + [REG_1..REG_R] + [PATCH_1..PATCH_N].
    cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
    if self.config.num_registers > 0:
      registers = self.registers.expand(B, -1, -1)  # [B, R, D]
      x = torch.cat([cls, registers, x], dim=1)  # [B, 1+R+N, D]
    else:
      x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]

    patch_start = 1 + self.config.num_registers
    patch_tokens_pre_norm: list[torch.Tensor] = [x[:, patch_start:]]  # each [B, T, D]
    cls_tokens_pre_norm: list[torch.Tensor] = [x[:, 0]]  # each [B, D]

    # Step 3: transformer blocks (capture tokens after each block).
    rope_prefix_len = patch_start if self.config.use_rope else 0
    for block in self.blocks:
      x = block(x, is_causal=is_causal, rope_prefix_len=rope_prefix_len)  # [B, 1+R+T, D]
      patch_tokens_pre_norm.append(x[:, patch_start:])  # [B, T, D]
      cls_tokens_pre_norm.append(x[:, 0])  # [B, D]

    # Step 4: build a shared ViTLayerOutputs container (final LN is applied on demand).
    final_norm_weight = getattr(self.norm, 'weight', None)
    final_norm_bias = getattr(self.norm, 'bias', None)
    final_norm_eps = getattr(self.norm, 'eps', None)
    rep_pooling = 'mean' if self.config.use_mean_pooling else 'cls'
    vit_layers = ViTLayerOutputs(
      patch_tokens_pre_norm=tuple(patch_tokens_pre_norm),
      cls_tokens_pre_norm=tuple(cls_tokens_pre_norm),
      token_mask=None,
      final_norm='ln',
      final_norm_weight=final_norm_weight,
      final_norm_bias=final_norm_bias,
      final_norm_eps=final_norm_eps,
      output_norm='none',
      output_bn_eps=1e-5,
      rep_pooling=rep_pooling,
    )
    return vit_layers, None
