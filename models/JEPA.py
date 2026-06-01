import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import configs
from models.projector_build import projector_build
from models.predictor import Predictor
from models.utils import apply_mask, loss_patch_inv_masked_local_global
from models.vision_transformer import VisionTransformer
from losses.sigreg import SIGReg


class MaskGenerator:
    """Generates block-based masks for JEPA pretraining."""

    def __init__(self, num_patches, min_block_size, min_keep_ratio, max_keep_ratio):
        self.num_patches = num_patches
        self.min_block_size = min_block_size
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

    def __call__(self, batch_size, num_views, device=None):
        """Generate masks for batch_size * num_views samples."""
        total = batch_size * num_views
        keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
        num_keep = math.ceil(keep_ratio * self.num_patches)
        mask_enc, mask_pred = [], []
        for _ in range(total):
            mask = self._sample_mask(num_keep)
            mask_enc.append(mask.nonzero().squeeze())
            mask_pred.append((1 - mask).nonzero().squeeze())
        mask_enc = torch.stack(mask_enc).to(device)
        mask_pred = torch.stack(mask_pred).to(device)
        return mask_enc, mask_pred

    def _sample_mask(self, num_keep):
        """Generate a single block-based mask."""
        num_patches = self.num_patches
        patch_intervals = [(0, num_patches)]
        num_mask = num_patches - num_keep
        total_mask_size = 0
        while total_mask_size < num_mask:
            interval_sizes = np.diff(patch_intervals).flatten()
            index = np.random.choice(len(patch_intervals), p=interval_sizes / interval_sizes.sum())
            start, end = patch_intervals.pop(index)
            interval_size = end - start
            max_block_size = num_mask - total_mask_size
            if max_block_size >= self.min_block_size:
                block_size = np.random.randint(self.min_block_size, max_block_size + 1)
            else:
                block_size = max_block_size
            if interval_size <= block_size:
                total_mask_size += interval_size
            else:
                if max_block_size >= self.min_block_size:
                    split = np.random.randint(start, end - block_size + 1)
                else:
                    attach_choices = []
                    if start > 0:
                        attach_choices.append(start)
                    if end < num_patches:
                        attach_choices.append(end - block_size)
                    if not attach_choices:
                        attach_choices.append(start)
                        attach_choices.append(end - block_size)
                    split = np.random.choice(attach_choices)
                patch_intervals.append((start, split))
                patch_intervals.append((split + block_size, end))
                total_mask_size += block_size
        mask = torch.zeros(num_patches)
        for start, end in patch_intervals:
            mask[start:end] = 1.0
        return mask


class RandomResizedCrop1D(nn.Module):
    """Randomly crop and resize a 1D signal to a fixed length."""

    def __init__(self, *, output_size: int, scale: tuple[float, float]):
        super().__init__()
        if output_size <= 0:
            raise ValueError(f'output_size must be > 0, got {output_size}')
        if not isinstance(scale, (tuple, list)) or len(scale) != 2:
            raise ValueError(f'scale must be a pair (min, max), got {scale}')
        scale_min, scale_max = float(scale[0]), float(scale[1])
        if scale_min <= 0 or scale_max > 1 or scale_min > scale_max:
            raise ValueError(f'scale must satisfy 0 < min <= max <= 1, got {scale}')
        self.output_size = int(output_size)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected x with shape (B, C, L), got {tuple(x.shape)}')
        batch_size, num_channels, length = x.shape
        if length != self.output_size:
            raise ValueError(f'Expected input length {self.output_size}, got {length}')
        if self.scale_min == 1.0 and self.scale_max == 1.0:
            return x
        scale = torch.empty((), device=x.device).uniform_(self.scale_min, self.scale_max)
        crop_len = int(round(float(scale.item()) * length))
        crop_len = max(1, min(crop_len, length))
        if crop_len == length:
            return x
        max_start = length - crop_len
        starts = torch.randint(0, max_start + 1, (batch_size,), device=x.device)
        offsets = torch.arange(crop_len, device=x.device)
        indices = starts[:, None] + offsets[None, :]
        indices = indices[:, None, :].expand(batch_size, num_channels, crop_len)
        cropped = torch.gather(x, dim=2, index=indices)
        resized = F.interpolate(cropped, size=length, mode='linear', align_corners=False)
        return resized


class JEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, momentum_schedule, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.momentum_schedule = momentum_schedule
    self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)
    self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)

    # Projector (MLP vs Flow; unified builder).
    self.proj = projector_build(config)

    # Mask generation
    num_patches = config.channel_size // config.patch_size
    self.mask_generator = MaskGenerator(
      num_patches=num_patches,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio,
      max_keep_ratio=config.max_keep_ratio)

    self.sigreg = SIGReg(normalize=config.sigreg_normalize)
    if config.ema_encoder:
      self.target_encoder = copy.deepcopy(self.encoder)

      for param in self.target_encoder.parameters():
        param.requires_grad = False
    else:
      self.target_encoder = None

  @torch.compiler.disable()
  def update_momentum_encoder(self):
    if self.target_encoder is None:
      return
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, return_representation=False):
    V = self.config.num_views
    device = x.device
    B = x.size(0)

    # expand x for V views: (B, C, L) -> (B*V, C, L)
    vs = x.unsqueeze(1).expand(-1, V, -1, -1).reshape(B * V, *x.shape[1:])

    ### Target encoder path (view-specific)
    if self.target_encoder is None:
      # No EMA, but with Stop-Grad
      with torch.no_grad():
        h_full_views = self.encoder(vs)  # [B*V, 1+N, D] (CLS + patches)
    else:
      with torch.no_grad():
        self.update_momentum_encoder()
        # compute prediction targets (full sequence for representation)
        h_full_views = self.target_encoder(vs)  # [B*V, 1+N, D] (CLS + patches)
    # Split CLS and patches
    h_patches = h_full_views[:, 1:]  # [B*V, N, D]
    _, N, D = h_patches.size()

    ### Context encoder -> predictor path
    # generate V masks per sample
    mask_encoder, mask_predictor = self.mask_generator(B, V, device) # [B*V, ??]
    # encode unmasked patches (M patches) + CLS
    z_enc = self.encoder(vs, mask_encoder)  # [B*V, 1+M, D] or [B*V, 1+N, D]
    # Split CLS and patches for later use
    z_cls = z_enc[:, 0]  # [B*V, D]
    if self.config.masking_strategy in ('zero', 'mask'):
      z_patches = apply_mask(z_enc[:, 1:], mask_encoder)  # [B*V, M, D]
    else:
      z_patches = z_enc[:, 1:]  # [B*V, M, D]
    # predict masked patches, predictor receives full z_enc (with CLS) and returns patch-only
    z_full = self.predictor(z_enc, mask_encoder, mask_predictor)  # [B*V, N, D]

    ### Loss
    _, K = mask_predictor.size()
    _, M = mask_encoder.size()
    assert K+M == N
    if self.config.masking_strategy in ('zero', 'mask'):
      z_masked = apply_mask(z_full, mask_predictor)  # [B*V, K, D]
    else:
      z_masked = z_full[:, -K:]  # [B*V, K, D]
    h_masked = apply_mask(h_patches, mask_predictor)  # [B*V, K, D]
    # Reshape to [V, B, K, D]
    z = z_masked.reshape(B, V, K, D).transpose(0, 1)  # [V, B, K, D]
    h = h_masked.reshape(B, V, K, D).transpose(0, 1)  # [V, B, K, D]
    if self.config.nepa_loss_type == 'mse':
      pred_loss = (z - h).square().mean()
    elif self.config.nepa_loss_type == 'cos':
      pred_loss = (1.0 - F.cosine_similarity(z, h, dim=-1)).mean()
    else:
      raise ValueError(
        'nepa_loss_type must be "mse" or "cos" for JEPA prediction loss, '
        f'got {self.config.nepa_loss_type!r}')
    losses = {'loss_pred': pred_loss}

    # SIGReg over CLS representation vectors (replaces mean pooling)
    rep_z = z_cls  # [B*V, D] CLS token
    if self.proj is not None:
      rep_z = self.proj(rep_z)  # [B*V, proj_dim]
    rep_z = rep_z.reshape(B, V, -1).transpose(0, 1)  # [V, B, D or proj_dim]
    sigreg_loss_rep = self.sigreg(rep_z)
    losses['loss_sigreg_rep'] = sigreg_loss_rep

    # SIGReg over time (patch tokens only, excluding CLS)
    z_patches_proj = z_patches
    if self.proj is not None:
      # Project each patch embedding: [B*V, M, D] -> [B*V*M, D] -> [B*V*M, proj_dim] -> [B*V, M, proj_dim]
      z_patches_proj = self.proj(z_patches.reshape(-1, D)).reshape(B*V, M, -1)
    # [B*V, M, D/proj_dim] -> [B, V, M, D/proj_dim] -> [M, V, B, D/proj_dim] -> [M*V, B, D/proj_dim]
    z_time = z_patches_proj.reshape(B, V, M, -1).transpose(0, 2).reshape(M*V, B, -1)  # [M*V, B, D/proj_dim]
    sigreg_loss_time_mv_b = self.sigreg(z_time)
    losses['loss_sigreg_time_mv_b'] = sigreg_loss_time_mv_b

    sigreg_loss = 0.0001*sigreg_loss_rep
    losses['loss_sigreg'] = sigreg_loss
    if self.config.add_sigreg_loss:
      losses['loss'] = pred_loss + sigreg_loss
    else:
      losses['loss'] = pred_loss

    # Log consts
    losses["B"] = torch.tensor(B)
    losses["V"] = torch.tensor(V)
    losses["K"] = torch.tensor(K)
    losses["M"] = torch.tensor(M)
    losses["N"] = torch.tensor(N)

    if return_representation:
      with torch.no_grad():
        if self.target_encoder is None:
          h_full = self.encoder(x)  # [B, 1+N, D]
        else:
          h_full = self.target_encoder(x)  # [B, 1+N, D]
      rep = h_full[:, 0]  # [B, D] CLS token as representation
      return losses, rep
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


class LeJEPA(nn.Module):
  """LeJEPA: No predictor, no EMA, no stop-gradient, uses SIGReg loss for regularization.
  """

  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)

    # Projector (MLP vs Flow; unified builder).
    self.proj = projector_build(config)

    # Mask generation
    num_patches = config.channel_size // config.patch_size
    self.mask_generator = MaskGenerator(
      num_patches=num_patches,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio,
      max_keep_ratio=config.max_keep_ratio)
    self.mask_generator_global = MaskGenerator(
      num_patches=num_patches,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio_global,
      max_keep_ratio=config.max_keep_ratio_global)

    self.sigreg = SIGReg(normalize=config.sigreg_normalize)

  def _get_view_embeddings(self, x, mask_encoder, V, return_patches):
    B = x.size(0)
    D = self.config.dim

    # Expand x for V views: (B, C, L) → (B*V, C, L)
    vs = x.unsqueeze(1).expand(-1, V, -1, -1).reshape(B * V, *x.shape[1:])

    # Encode masked views (CLS + M kept patches)
    z_enc = self.encoder(vs, mask_encoder)  # [B*V, 1+M, D]
    z_cls = z_enc[:, 0]  # [B*V, D]
    M = z_enc.size(1) - 1

    if self.config.use_mean_pooling:
      z_proj = z_enc[:, 1:].mean(dim=1)  # [B*V, D]
    else:
      z_proj = z_cls  # [B*V, D]
    
    z_proj_patches = None
    if return_patches:
      z_proj_patches = z_enc[:, 1:]  # [B*V, M, D]

    # Project embeddings
    if self.proj is not None:
      z_proj = self.proj(z_proj)  # [B*V, proj_dim/proj_out_dim]
      z_proj_patches = (
        self.proj(z_proj_patches.reshape(B * V * M, D)).reshape(B * V, M, -1)
        if z_proj_patches is not None
        else None
      )  # [B*V, M, proj_dim/proj_out_dim] or None
    
    return z_proj, z_proj_patches
    
  def forward(self, x, return_representation=False):
    V = self.config.num_views
    Vg = self.config.num_views_global
    device = x.device
    B = x.size(0)
    N = self.config.channel_size // self.config.patch_size

    # Generate V masks per sample
    mask_encoder, _ = self.mask_generator(B, V, device)  # [B*V, M]
    M = mask_encoder.size(1)
    if Vg == 1:
      mask_encoder_global = None
      Mg = N
    else:
      mask_encoder_global, _ = self.mask_generator_global(B, Vg, device)  # [B*Vg, M]
      Mg = mask_encoder_global.size(1)

    # Encode views
    need_local_patches = (
      self.config.loss_scale_sigreg_time is not None or
      self.config.loss_scale_inv_patch is not None)
    need_global_patches = (
      self.config.loss_scale_sigreg_time_global is not None or
      self.config.loss_scale_inv_patch is not None)
    z_proj, z_proj_patches = self._get_view_embeddings(x, mask_encoder, V, return_patches=need_local_patches)  # [B*V, D/proj_dim], [B*V, M, D/proj_dim]
    z_proj_global, z_proj_patches_global = self._get_view_embeddings(x, mask_encoder_global, Vg, return_patches=need_global_patches)  # [B*Vg, D/proj_dim], [B*Vg, M, D/proj_dim]

    # Reshape to [V, B, D/proj_dim] for loss computation
    z = z_proj.reshape(B, V, -1).transpose(0, 1)  # [V, B, D/proj_dim]
    z_global = z_proj_global.reshape(B, Vg, -1).transpose(0, 1)  # [Vg, B, D/proj_dim]
    z_all = torch.cat([z, z_global], dim=0)       # [V+Vg, B, D/proj_dim]

    # Invariance loss
    loss_inv = (z_all - z_global.mean(dim=0, keepdim=True)).square().mean()

    # SIGReg loss
    loss_sigreg = self.sigreg(z_all)
    loss_sigreg_time = self.sigreg(z_proj_patches) if z_proj_patches is not None else None
    loss_sigreg_time_global = self.sigreg(z_proj_patches_global) if z_proj_patches_global is not None else None

    # Patch-level invariance loss (masked local patches vs unmasked global patches)
    loss_inv_patch = None
    # if z_proj_patches is not None and z_proj_patches_global is not None:
    #   loss_inv_patch = (z_proj_patches.reshape(B, V, N, -1) - z_proj_patches_global.reshape(B, Vg, N, -1).mean(dim=1, keepdim=True)).square().mean()
    if self.config.loss_scale_inv_patch is not None:
      if self.config.masking_strategy != 'mask':
        raise ValueError('loss_scale_inv_patch requires masking_strategy="mask"')
      if z_proj_patches is None or z_proj_patches_global is None:
        raise ValueError('loss_scale_inv_patch requires patch embeddings for local and global views')
      loss_inv_patch = loss_patch_inv_masked_local_global(
        local_patches=z_proj_patches,
        global_patches=z_proj_patches_global,
        mask_encoder=mask_encoder,
        mask_encoder_global=mask_encoder_global,
        num_local_views=V,
        num_global_views=Vg)

    # LeJEPA loss
    loss = self.config.loss_scale_sigreg * loss_sigreg + self.config.loss_scale_inv * loss_inv
    if loss_inv_patch is not None and self.config.loss_scale_inv_patch is not None:
      loss += self.config.loss_scale_inv_patch * loss_inv_patch
    if loss_sigreg_time is not None and self.config.loss_scale_sigreg_time is not None:
      loss += self.config.loss_scale_sigreg_time * loss_sigreg_time
    if loss_sigreg_time_global is not None and self.config.loss_scale_sigreg_time_global is not None:
      loss += self.config.loss_scale_sigreg_time_global * loss_sigreg_time_global

    losses = {
      'loss_inv': loss_inv,
      'loss_sigreg': loss_sigreg,
      'loss': loss,
      # Log constants for debugging
      'B': torch.tensor(B),
      'V': torch.tensor(V),
      'Vg': torch.tensor(Vg),
      'M': torch.tensor(M),
      'Mg': torch.tensor(Mg),
      'N': torch.tensor(N),
    }
    if z_proj_patches is not None:
      losses['loss_sigreg_time'] = loss_sigreg_time
    if z_proj_patches_global is not None:
      losses['loss_sigreg_time_global'] = loss_sigreg_time_global
    if loss_inv_patch is not None:
      losses['loss_inv_patch'] = loss_inv_patch

    if return_representation:
      # Compute full, unmasked CLS for the online probe
      with torch.no_grad():
        h_full = self.encoder(x)  # [B, 1+N, D]
        rep = h_full[:, 0]  # [B, D]
      return losses, rep

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
