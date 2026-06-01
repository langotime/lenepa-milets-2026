from dataclasses import dataclass


@dataclass
class Config:
  # data
  sampling_frequency: int
  channels: tuple[str, ...]
  channel_size: int
  patch_size: int
  min_block_size: int
  min_keep_ratio: float
  max_keep_ratio: float
  min_keep_ratio_global: float
  max_keep_ratio_global: float
  datasets: dict[str, float]
  # model architecture
  dim: int
  depth: int
  num_heads: int
  pred_dim: int
  pred_depth: int
  pred_num_heads: int
  mlp_ratio: float
  qkv_bias: bool
  dropout: float
  attn_dropout: float
  num_registers: int
  bias: bool
  norm_eps: float
  layer_scale_eps: float
  # training
  steps: int
  batch_size: int
  encoder_momentum: float
  final_encoder_momentum: float
  learning_rate: float
  final_learning_rate: float
  learning_rate_warmup_steps: int
  weight_decay: float
  final_weight_decay: float
  opt_betas: tuple[float, float]
  opt_eps: float
  gradient_clip: float
  gradient_accumulation_steps: int
  checkpoint_interval: int
  seed: int | None = None  # Optional: fixed seed for reproducible runs (opt-in).
  seed_init: int | None = None  # Optional: split seed for model init / training RNG (opt-in).
  seed_data: int | None = None  # Optional: split seed for DataLoader order RNG (opt-in).
  seed_sigreg: int | None = None  # Optional: split seed for SIGReg projection RNG (opt-in).
  seed_offline_probe: int | None = None  # Optional: split seed for offline probe RNG (opt-in).
  label_exposure_train_log_interval: int | None = None  # Optional: log label exposure every N steps.
  label_exposure_offline_probe_log: bool = False  # Optional: log unique label exposure during offline probe training.
  batch_label_stats_train_log_interval: int | None = None  # Optional: log batch label-gap stats every N steps.
  batch_label_stats_rare_bottom_k: int | None = None  # Required when batch_label_stats_* is enabled.
  ptbxl_train_sampling: str = "uniform"  # "uniform" or "weighted_rare"
  ptbxl_train_sampling_alpha: float | None = None  # Required when ptbxl_train_sampling="weighted_rare".
  masking_strategy: str = 'remove'
  end_truncate: int = 0  # truncate this many samples from the end of the signal
  data_preprocess_normalize: bool = True  # Apply dataset mean/std normalization + clipping in PreprocessECG.
  data_backend: str | None = None  # "dump", "aiono", or "cauker"
  probe_source: str | None = None  # "ptb-xl" or "aiono" enables labeled probing
  aiono: dict[str, object] | None = None  # canonical synthetic config
  # online probe (required when 'ptb-xl' in datasets)
  probe_lr: float | None = None
  probe_weight_decay: float | None = None
  probe_eval_interval: int | None = None
  probe_task: str = 'all'
  # offline probe dataset (decoupled from training)
  offline_probe_source: str | None = None  # "ptb-xl" or "aiono"
  offline_probe_sources: tuple[str, ...] | None = None  # "ptb-xl", "aiono"
  offline_probe_aiono: dict[str, object] | None = None  # canonical offline probe synthetic config
  # offline probe (optional)
  offline_probe_eval_interval: int | None = None
  offline_probe_eval_at_step0: bool = True
  offline_probe_task: str | None = None
  offline_probe_steps: int | None = None
  offline_probe_batch_size: int | None = None
  offline_probe_learning_rate: float | None = None
  offline_probe_final_learning_rate: float | None = None
  offline_probe_learning_rate_warmup_steps: int | None = None
  offline_probe_weight_decay: float | None = None
  offline_probe_opt_betas: tuple[float, float] | None = None
  offline_probe_gradient_clip: float | None = None
  offline_probe_checkpoint_interval: int | None = None
  # offline probe layer selection (required when offline probing is enabled for NEPA; optional otherwise)
  offline_probe_layers_categorical: tuple[int, ...] | None = None
  offline_probe_layers_dense: tuple[int, ...] | None = None
  offline_probe_layers_confusion: tuple[int, ...] | None = None
  # LeJEPA
  ema_encoder: bool = True
  add_sigreg_loss: bool = False
  num_views: int = 1  # number of mask views per sample
  num_views_global: int = 2  # number of global mask views per sample
  # Projector
  use_mean_pooling: bool = False # Use mean pooling if True, CLS token if false
  use_projector: bool = False  # Whether to project embeddings before SIGReg
  proj_dim: int = 16  # Projection output dimension
  proj_out_dim: int | None = None  # Optional projector output dimension (when projecting back to encoder dim).
  proj_hidden_dim: int = 768  # Hidden dimension in projector MLP
  proj_num_layers: int = 1  # Number of hidden layers in projector MLP
  proj_backend: str = "mlp"  # "mlp" or "invertible_flow"
  # Flow projector hyperparameters (used only when proj_backend="invertible_flow").
  proj_flow_n_layers: int = 6
  proj_flow_hidden_dim: int = 128
  proj_flow_n_blocks: int = 2
  proj_flow_s_max: float = 2.0
  proj_flow_mask_mode: str = "alternating"  # "alternating" | "half"
  proj_flow_permute_mode: str = "per_pair"  # "none" | "per_layer" | "per_pair"
  proj_flow_permute_seed: int = 1337
  # NEPA
  use_nepa: bool = False  # Enable NEPA (next-embedding prediction)
  nepa_stability: str | None = None  # "stopgrad", "sigreg", "wristband", or "rdmreg"
  nepa_loss_type: str = 'cos'  # Prediction loss type ("mse" or "cos") for NEPA and JEPA.
  nepa_loss_scale: float = 1.0 # Scale of the NEPA loss
  nepa_sigreg_scale: float = 10.0 # Scale of the SIGReg loss
  nepa_sigreg_scale_time: float | None = None # Scale of the SIGReg over Time loss
  nepa_sigreg_time_saliency: str | None = None  # Optional time-saliency mode for SIGReg diagnostics.
  nepa_sigreg_scale_batch_time: float | None = None # Scale of SIGReg over batch+time per layer.
  nepa_sigreg_innov_scale: float | None = None  # Scale of SIGReg on temporal innovations.
  nepa_sigreg_rep_scale: float | None = None  # Scale of SIGReg applied to the probe representation.
  nepa_sigreg_rep_use_projector: bool | None = True  # Use projector before SIGReg on rep (ignored when nepa_sigreg_rep_scale is null).
  nepa_sigreg_layers: tuple[int, ...] | None = None  # ViT layers for loss_sigreg.
  nepa_sigreg_time_layers: tuple[int, ...] | None = None  # ViT layers for loss_sigreg_time.
  nepa_sigreg_batch_time_layers: tuple[int, ...] | None = None  # ViT layers for loss_sigreg_batch_time.
  nepa_sigreg_innov_layers: tuple[int, ...] | None = None  # ViT layers for loss_sigreg_innov.
  nepa_sigreg_rep_layers: tuple[int, ...] | None = None  # ViT layers for loss_sigreg_rep.
  # NEPA Wristband Gaussian stability (replacement for SIGReg when nepa_stability="wristband").
  nepa_wristband_scale: float | None = None
  nepa_wristband_layers: tuple[int, ...] | None = None
  nepa_wristband_scale_time: float | None = None
  nepa_wristband_time_layers: tuple[int, ...] | None = None
  nepa_wristband_scale_batch_time: float | None = None
  nepa_wristband_batch_time_layers: tuple[int, ...] | None = None
  nepa_wristband_innov_scale: float | None = None
  nepa_wristband_innov_layers: tuple[int, ...] | None = None
  nepa_wristband_rep_scale: float | None = None
  nepa_wristband_rep_layers: tuple[int, ...] | None = None
  # Wristband loss hyperparameters (passed to C_WristbandGaussianLoss).
  nepa_wristband_beta: float = 8.0
  nepa_wristband_alpha: float | None = None
  nepa_wristband_angular: str = "chordal"  # "chordal" | "geodesic"
  nepa_wristband_reduction: str = "per_point"  # "per_point" | "global"
  nepa_wristband_lambda_rad: float = 0.1
  nepa_wristband_lambda_ang: float = 0.0
  nepa_wristband_moment: str = "w2"  # "mu_only"|"kl_diag"|"kl_full"|"jeff_diag"|"jeff_full"|"w2"
  nepa_wristband_lambda_mom: float = 1.0
  # O(N^2) kernel complexity limit (required when nepa_stability="wristband").
  nepa_wristband_max_samples: int | None = None
  # NEPA RDMReg (Rectified Distribution Matching Regularization)
  nepa_rdmreg_num_projections: int | None = None  # Number of random projections for SWD.
  nepa_rdmreg_p: float | None = None  # Generalized Gaussian p parameter.
  nepa_rdmreg_mean_shift: float | None = None  # Mean shift μ for the target distribution.
  nepa_rdmreg_target_dist: str | None = None  # "rectified_lp_distribution" or "lp_distribution".
  nepa_rdmreg_sigma_mode: str | None = None  # "gn_unit_var" or "manual" (no silent defaults).
  nepa_rdmreg_sigma: float | None = None  # Required when nepa_rdmreg_sigma_mode="manual".
  nepa_rdmreg_feature_activation: str | None = None  # "none", "relu", or "reprelu" (paper uses "relu").
  nepa_target_uses_pos_embed: bool | None = None  # Required when use_nepa=True
  nepa_target_layer: int | None = None  # 0=input, depth=encoder/out (post-final-norm); requires use_nepa=True
  nepa_final_norm: str = 'ln'  # Final encoder output norm ("ln", "rms", or "none").
  nepa_layer_output_norm: str = 'ln'  # Per-layer output norm for layers < depth ("ln", "rms", "bn", "none"); layer=depth uses nepa_final_norm.
  nepa_layer_output_bn_eps: float = 1e-5  # Epsilon for nepa_layer_output_norm="bn".
  nepa_rep_pooling: str = 'mean'  # "mean" or "last" pooling for ViT layer representations.
  nepa_log_patch_loss: bool = False  # Log per-patch loss table in W&B
  nepa_patch_loss_log_interval: int | None = None  # Required when nepa_log_patch_loss=True
  nepa_distribution_diagnostics: bool = False  # Log nepa_diag_* embedding distribution stats.
  nepa_vit_drift_diagnostics: bool = False  # Log nepa_diag_vit_* diagnostics (explicit opt-in).
  nepa_vit_drift_layers: tuple[int, ...] | None = None  # Required when nepa_vit_drift_diagnostics=True.
  # NEPA/JEPA prediction loss parameters
  nepa_loss_batch_norm: bool = False  # Batch-normalize pred/target for MSE loss.
  nepa_loss_batch_norm_eps: float = 1e-5  # Epsilon for nepa_loss_batch_norm.
  nepa_projector_mode: str = 'both'  # "both", "target_only", or "pred_only" (NEPA-only).
  # NEPA architecture optimizations (RoPE, SwiGLU, QK-Norm, pos embed type)
  use_rope: bool | None = None
  rope_base: int | None = None
  use_swiglu: bool | None = None
  use_qk_norm: bool | None = None
  qk_norm_eps: float | None = None
  pos_embed_type: str | None = None  # "sin", "learned", or "none"
  # NEPA static tokenizer (used when nepa_use_dynamic_chunking=False)
  nepa_static_tokenizer: str = 'conv_patch_embed'  # "conv_patch_embed", "mamba_static_patches", "mamba_topk_tokens", or "mantis_tokgen"
  nepa_topk_tokens: int | None = None  # Required when nepa_static_tokenizer="mamba_topk_tokens" (excluding registers).
  # NEPA conv_patch_embed scalar stats (mean/std) encoded with MSSE and concatenated to CNN tokens.
  nepa_patch_embed_scalar_stats_mode: str | None = None  # null | "sequence_norm" | "patch_norm"
  nepa_patch_embed_cnn_dim: int | None = None  # Required when nepa_patch_embed_scalar_stats_mode is not null.
  nepa_patch_embed_scalar_hidden_dim: int = 32  # Hidden dim for each scalar embedding (mean and std).
  nepa_patch_embed_scalar_scales: tuple[float, ...] = (
    1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4,
  )
  nepa_patch_embed_scalar_epsilon: float = 1.1
  # NEPA Mantis-like tokenizer (used when nepa_static_tokenizer="mantis_tokgen"; required keys, no silent defaults).
  nepa_mantis_kernel_size: int | None = None  # Conv kernel size (patch_embed requires ==patch_size).
  nepa_mantis_conv_mode: str | None = None  # "patch_embed" or "stride1_pool"
  nepa_mantis_use_diff_branch: bool = True  # Enable the first-difference branch in MantisTokGen1D.
  nepa_mantis_instance_norm_mode: str | None = None  # "per_channel" or "global_over_channels"
  nepa_mantis_instance_norm_eps: float | None = None  # Epsilon for patch-local instance norm.
  nepa_mantis_stats_mode: str | None = None  # "flatten_project" or "per_channel_slots"
  nepa_mantis_stats_dim: int | None = None  # Output dimension for the stats branch.
  nepa_mantis_scalar_scales: tuple[float, ...] | None = None  # MultiScaledScalarEncoder scales (must be explicit).
  nepa_mantis_scalar_hidden_dim: int | None = None  # Scalar encoder hidden dimension.
  nepa_mantis_scalar_epsilon: float | None = None  # Scalar encoder epsilon for alpha weights.
  # NEPA full-resolution dynamic chunking (Mamba + H-Net)
  nepa_use_dynamic_chunking: bool = False
  nepa_chunk_router_mode: str | None = None  # "causal" or "lookahead"
  nepa_chunk_boundary_threshold: float | None = None
  nepa_chunk_ratio_target: int | None = None
  nepa_chunk_ratio_loss_scale: float | None = None
  nepa_chunk_max_tokens: int | None = None
  nepa_local_mamba_layers: int | None = None
  nepa_local_mamba_d_state: int | None = None
  nepa_local_mamba_d_conv: int | None = None
  nepa_local_mamba_expand: int | None = None
  nepa_local_mamba_readout: str = 'last'  # "last", "mean", or "conv" (static tokenizer only)
  # LeJEPA mode
  use_lejepa: bool = False  # Enable LeJEPA (no predictor, no EMA, no stop-grad)
  sigreg_normalize: bool = False # Enable or disable sigreg pre-normalization
  loss_scale_sigreg: float | None = None  # Scale for the SIGReg loss component
  loss_scale_sigreg_time: float | None = None  # Scale for the SIGReg Over Time (local views) loss component
  loss_scale_sigreg_time_global: float | None = None  # Scale for the SIGReg Over Time (global views) loss component
  loss_scale_inv: float | None = None  # Scale for the Invariance loss component
  loss_scale_inv_patch: float | None = None  # Scale for the Patch-level Invariance loss component

  @property
  def num_channels(self):
    return len(self.channels)

  @property
  def data_backend_resolved(self) -> str | None:
    """Return the configured data backend name."""
    return self.data_backend

  @property
  def probe_source_resolved(self) -> str | None:
    """Return the configured probe source name."""
    return self.probe_source

  @property
  def synthetic_config(self) -> dict[str, object] | None:
    """Return the synthetic dataset config."""
    return self.aiono

  @property
  def offline_probe_source_resolved(self) -> str | None:
    """Return the configured offline probe source name."""
    return self.offline_probe_source

  @property
  def offline_probe_synthetic_config(self) -> dict[str, object] | None:
    """Return the offline-probe synthetic dataset config."""
    if self.offline_probe_aiono is not None:
      return self.offline_probe_aiono
    if self.offline_probe_source_resolved == 'aiono':
      return self.synthetic_config
    return None

  @property
  def offline_probe_sources_resolved(self) -> tuple[str, ...]:
    """Return the configured offline probe sources."""
    if self.offline_probe_sources is not None:
      normalized = [str(source) for source in self.offline_probe_sources]
      return tuple(dict.fromkeys(normalized))
    if self.offline_probe_source_resolved is None:
      return ()
    return (self.offline_probe_source_resolved,)

  def __post_init__(self):
    def _validate_optional_seed(name: str, value: int | None) -> None:
      """Validate that an optional seed is a non-negative int when provided."""
      if value is None:
        return
      if not isinstance(value, int):
        raise ValueError(f'{name} must be an int or null, got {type(value).__name__}')
      if value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}')

    def _validate_mamba_settings(context: str) -> None:
      """Validate Mamba hyperparameters for the given config context."""
      # Step 1: ensure required Mamba hyperparameters are present and valid.
      if self.nepa_local_mamba_layers is None:
        raise ValueError(f'nepa_local_mamba_layers must be set when {context}')
      if self.nepa_local_mamba_layers < 1:
        raise ValueError(
          'nepa_local_mamba_layers must be >= 1, '
          f'got {self.nepa_local_mamba_layers}')
      if self.nepa_local_mamba_d_state is None:
        raise ValueError(f'nepa_local_mamba_d_state must be set when {context}')
      if self.nepa_local_mamba_d_state < 1:
        raise ValueError(
          'nepa_local_mamba_d_state must be >= 1, '
          f'got {self.nepa_local_mamba_d_state}')
      if self.nepa_local_mamba_d_conv is None:
        raise ValueError(f'nepa_local_mamba_d_conv must be set when {context}')
      if self.nepa_local_mamba_d_conv < 1:
        raise ValueError(
          'nepa_local_mamba_d_conv must be >= 1, '
          f'got {self.nepa_local_mamba_d_conv}')
      if self.nepa_local_mamba_expand is None:
        raise ValueError(f'nepa_local_mamba_expand must be set when {context}')
      if self.nepa_local_mamba_expand < 1:
        raise ValueError(
          'nepa_local_mamba_expand must be >= 1, '
          f'got {self.nepa_local_mamba_expand}')

    def _validate_layer_list(name: str, layers: object | None) -> tuple[int, ...] | None:
      """Validate a (possibly-null) list/tuple of ViT layer indices (empty == null)."""
      if layers is None:
        return None
      if not isinstance(layers, (list, tuple)):
        raise ValueError(
          f'{name} must be a list/tuple of ints, got {type(layers).__name__}')
      if not layers:
        return None
      normalized: list[int] = []
      for idx, layer in enumerate(layers):
        if not isinstance(layer, int):
          raise ValueError(f'{name}[{idx}] must be an int, got {layer!r}')
        if layer < 0 or layer > self.depth:
          raise ValueError(
            f'{name}[{idx}] must be in [0, depth], got {layer} (depth={self.depth})')
        normalized.append(layer)
      if len(set(normalized)) != len(normalized):
        raise ValueError(f'{name} must be unique, got {normalized}')
      return tuple(normalized)

    # Seeding configuration (single-seed vs split-seed mode).
    seed_fields = {
      'seed_init': self.seed_init,
      'seed_data': self.seed_data,
      'seed_sigreg': self.seed_sigreg,
      'seed_offline_probe': self.seed_offline_probe,
    }
    split_mode_requested = any(value is not None for value in seed_fields.values())

    _validate_optional_seed('seed', self.seed)
    for name, value in seed_fields.items():
      _validate_optional_seed(name, value)

    if self.seed is not None and split_mode_requested:
      raise ValueError('seed and seed_* cannot both be set. Use either a single seed or split seeds.')
    if self.seed is None and split_mode_requested:
      missing = [name for name, value in seed_fields.items() if value is None]
      if missing:
        raise ValueError(
          'Split-seed mode requested but some seed_* values are missing. '
          f'Missing: {missing}. Fix: set all of seed_init/seed_data/seed_sigreg/seed_offline_probe.')

    allowed_ptbxl_sampling = ('uniform', 'weighted_rare')
    if not isinstance(self.ptbxl_train_sampling, str):
      raise ValueError(
        'ptbxl_train_sampling must be a str, '
        f'got {type(self.ptbxl_train_sampling).__name__}')
    if self.ptbxl_train_sampling not in allowed_ptbxl_sampling:
      raise ValueError(
        f'ptbxl_train_sampling must be one of {sorted(allowed_ptbxl_sampling)}, '
        f'got {self.ptbxl_train_sampling!r}')
    if self.ptbxl_train_sampling == 'weighted_rare':
      if self.ptbxl_train_sampling_alpha is None:
        raise ValueError(
          'ptbxl_train_sampling_alpha is required when ptbxl_train_sampling="weighted_rare" '
          '(no silent defaults).')
      if not isinstance(self.ptbxl_train_sampling_alpha, (int, float)):
        raise ValueError(
          'ptbxl_train_sampling_alpha must be a float, '
          f'got {type(self.ptbxl_train_sampling_alpha).__name__}')
      if self.ptbxl_train_sampling_alpha <= 0:
        raise ValueError(
          'ptbxl_train_sampling_alpha must be > 0 when ptbxl_train_sampling="weighted_rare", '
          f'got {self.ptbxl_train_sampling_alpha}')
    elif self.ptbxl_train_sampling_alpha is not None:
      raise ValueError(
        'ptbxl_train_sampling_alpha is set but ptbxl_train_sampling!="weighted_rare". '
        'Fix: set ptbxl_train_sampling="weighted_rare" or unset ptbxl_train_sampling_alpha.')

    if not isinstance(self.data_preprocess_normalize, bool):
      raise ValueError(
        'data_preprocess_normalize must be a bool, '
        f'got {type(self.data_preprocess_normalize).__name__}')

    if self.batch_label_stats_train_log_interval is not None:
      if not isinstance(self.batch_label_stats_train_log_interval, int):
        raise ValueError(
          'batch_label_stats_train_log_interval must be an int or null, '
          f'got {type(self.batch_label_stats_train_log_interval).__name__}')
      if self.batch_label_stats_train_log_interval <= 0:
        raise ValueError(
          'batch_label_stats_train_log_interval must be > 0, '
          f'got {self.batch_label_stats_train_log_interval}')
      if self.batch_label_stats_rare_bottom_k is None:
        raise ValueError(
          'batch_label_stats_rare_bottom_k is required when batch_label_stats_train_log_interval is set '
          '(no silent defaults).')
    if self.batch_label_stats_rare_bottom_k is not None:
      if not isinstance(self.batch_label_stats_rare_bottom_k, int):
        raise ValueError(
          'batch_label_stats_rare_bottom_k must be an int or null, '
          f'got {type(self.batch_label_stats_rare_bottom_k).__name__}')
      if self.batch_label_stats_rare_bottom_k <= 0:
        raise ValueError(
          'batch_label_stats_rare_bottom_k must be > 0, '
          f'got {self.batch_label_stats_rare_bottom_k}')

    if self.data_backend is not None and self.data_backend not in ('dump', 'aiono', 'cauker'):
      raise ValueError(
        'data_backend must be "dump", "aiono", or "cauker", '
        f'got {self.data_backend!r}')
    if self.probe_source is not None and self.probe_source not in ('ptb-xl', 'aiono'):
      raise ValueError(
        'probe_source must be "ptb-xl", "aiono", or None, '
        f'got {self.probe_source!r}')
    if self.offline_probe_source is not None and self.offline_probe_source not in ('ptb-xl', 'aiono'):
      raise ValueError(
        'offline_probe_source must be "ptb-xl", "aiono", or None, '
        f'got {self.offline_probe_source!r}')
    if self.offline_probe_sources is not None:
      if self.offline_probe_source is not None:
        raise ValueError('offline_probe_source and offline_probe_sources cannot both be set.')
      if not isinstance(self.offline_probe_sources, (list, tuple)):
        raise ValueError(
          'offline_probe_sources must be a list/tuple of strings, '
          f'got {type(self.offline_probe_sources).__name__}')
      if not self.offline_probe_sources:
        raise ValueError('offline_probe_sources must be non-empty when provided.')
      if len(set(self.offline_probe_sources)) != len(self.offline_probe_sources):
        raise ValueError(f'offline_probe_sources must be unique, got {self.offline_probe_sources}')
      for source in self.offline_probe_sources:
        if source not in ('ptb-xl', 'aiono'):
          raise ValueError(
            'offline_probe_sources must contain only "ptb-xl" and "aiono". '
            f'Got {self.offline_probe_sources}')
    if self.data_backend == 'dump' and self.synthetic_config is not None:
      raise ValueError('aiono config must be null when data_backend="dump"')
    if self.data_backend_resolved == 'aiono' and self.synthetic_config is None:
      raise ValueError('aiono config is required when data_backend="aiono"')
    if self.data_backend == 'cauker' and self.synthetic_config is not None:
      raise ValueError('aiono config must be null when data_backend="cauker"')

    # Offline probe layer selection (used by NEPA and ViT-based JEPA encoders).
    offline_probe_layer_mode = any(
      value is not None
      for value in (
        self.offline_probe_layers_categorical,
        self.offline_probe_layers_dense,
        self.offline_probe_layers_confusion,
      )
    )
    if offline_probe_layer_mode:
      self.offline_probe_layers_categorical = _validate_layer_list(
        'offline_probe_layers_categorical', self.offline_probe_layers_categorical)
      self.offline_probe_layers_dense = _validate_layer_list(
        'offline_probe_layers_dense', self.offline_probe_layers_dense)
      self.offline_probe_layers_confusion = _validate_layer_list(
        'offline_probe_layers_confusion', self.offline_probe_layers_confusion)

      if (self.offline_probe_layers_categorical is None
          and self.offline_probe_layers_dense is None
          and self.offline_probe_layers_confusion is None):
        raise ValueError(
          'At least one of offline_probe_layers_categorical/offline_probe_layers_dense/'
          'offline_probe_layers_confusion must be non-empty when provided.')

      categorical_layers = set(self.offline_probe_layers_categorical or ())
      confusion_layers = set(self.offline_probe_layers_confusion or ())
      if not confusion_layers.issubset(categorical_layers):
        missing = sorted(confusion_layers - categorical_layers)
        raise ValueError(
          'offline_probe_layers_confusion must be a subset of offline_probe_layers_categorical. '
          f'Missing categorical layers: {missing}')
    elif (self.offline_probe_eval_interval is not None
          and self.use_nepa
          and self.offline_probe_sources_resolved):
      raise ValueError(
        'offline_probe_eval_interval is set and use_nepa=True, but offline_probe_layers_* are all null. '
        'Set at least one of offline_probe_layers_categorical/offline_probe_layers_dense/'
        'offline_probe_layers_confusion to a non-empty list, e.g. '
        'offline_probe_layers_categorical=[depth] offline_probe_layers_dense=[depth].')

    # Validate NEPA architecture optimizations (no silent defaults)
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
    if (self.use_nepa and self.pos_embed_type == 'none' and self.nepa_target_uses_pos_embed):
      raise ValueError('nepa_target_uses_pos_embed=True requires pos_embed_type != "none"')
    if self.use_rope:
      if self.rope_base is None:
        raise ValueError('rope_base must be set when use_rope=True')
      head_dim = self.dim // self.num_heads
      if head_dim % 2 != 0:
        raise ValueError(f'RoPE requires even head_dim, got {head_dim} (dim={self.dim}, num_heads={self.num_heads})')
    if self.use_qk_norm:
      if self.qk_norm_eps is None:
        raise ValueError('qk_norm_eps must be set when use_qk_norm=True')

    if self.nepa_sigreg_time_saliency is not None:
      allowed_time_saliency = {'robust_zscore', 'aiono_oracle_mask'}
      if self.nepa_sigreg_time_saliency not in allowed_time_saliency:
        raise ValueError(
          'nepa_sigreg_time_saliency must be one of '
          f'{sorted(allowed_time_saliency)}, got {self.nepa_sigreg_time_saliency!r}')

    if self.nepa_log_patch_loss and not self.use_nepa:
      raise ValueError('nepa_log_patch_loss requires use_nepa=True')
    if self.nepa_log_patch_loss:
      if self.nepa_patch_loss_log_interval is None:
        raise ValueError(
          'nepa_patch_loss_log_interval must be set when nepa_log_patch_loss=True')
      if not isinstance(self.nepa_patch_loss_log_interval, int):
        raise ValueError(
          'nepa_patch_loss_log_interval must be an int when nepa_log_patch_loss=True')
      if self.nepa_patch_loss_log_interval <= 0:
        raise ValueError(
          'nepa_patch_loss_log_interval must be > 0 when nepa_log_patch_loss=True')

    enabled_objectives = [
      name for name, enabled in (
        ('use_nepa', self.use_nepa),
        ('use_lejepa', self.use_lejepa),
      )
      if enabled
    ]
    if len(enabled_objectives) > 1:
      raise ValueError(f'Only one objective can be enabled, got {enabled_objectives}')

    allowed_loss_types = ('mse', 'cos')
    if not isinstance(self.nepa_loss_type, str):
      raise ValueError(
        'nepa_loss_type must be a str, '
        f'got {type(self.nepa_loss_type).__name__}')
    if self.nepa_loss_type not in allowed_loss_types:
      raise ValueError(
        f'nepa_loss_type must be one of {allowed_loss_types}, got {self.nepa_loss_type!r}')

    if self.nepa_target_layer is not None:
      if not self.use_nepa:
        raise ValueError('nepa_target_layer requires use_nepa=True')
      for name, layer in (
        ('nepa_target_layer', self.nepa_target_layer),
      ):
        if layer is None:
          continue
        if not isinstance(layer, int):
          raise ValueError(f'{name} must be an int, got {layer!r}')
        if layer < 0 or layer > self.depth:
          raise ValueError(
            f'{name} must be in [0, depth], got {layer} (depth={self.depth})')

    if self.nepa_sigreg_rep_use_projector is None:
      self.nepa_sigreg_rep_use_projector = True
    if not isinstance(self.nepa_sigreg_rep_use_projector, bool):
      raise ValueError(
        'nepa_sigreg_rep_use_projector must be a bool, '
        f'got {type(self.nepa_sigreg_rep_use_projector).__name__}')

    if self.nepa_sigreg_rep_scale is not None:
      if not self.use_nepa:
        raise ValueError('nepa_sigreg_rep_scale requires use_nepa=True')
      if self.nepa_sigreg_rep_scale <= 0:
        raise ValueError('nepa_sigreg_rep_scale must be > 0 when set')
      if self.nepa_sigreg_rep_use_projector and not self.use_projector:
        raise ValueError('nepa_sigreg_rep_use_projector=True requires use_projector=True')

    if self.nepa_sigreg_innov_scale is not None:
      if not self.use_nepa:
        raise ValueError('nepa_sigreg_innov_scale requires use_nepa=True')
      if self.nepa_sigreg_innov_scale <= 0:
        raise ValueError('nepa_sigreg_innov_scale must be > 0 when set')

    if self.use_nepa:
      if self.nepa_target_uses_pos_embed is None:
        raise ValueError('nepa_target_uses_pos_embed must be set when use_nepa=True')
      allowed_final_norm = ('ln', 'rms', 'none')
      if not isinstance(self.nepa_final_norm, str):
        raise ValueError(
          'nepa_final_norm must be a str, '
          f'got {type(self.nepa_final_norm).__name__}')
      if self.nepa_final_norm not in allowed_final_norm:
        raise ValueError(
          'nepa_final_norm must be one of '
          f'{sorted(allowed_final_norm)}, got {self.nepa_final_norm!r}')
      allowed_layer_output_norm = ('ln', 'rms', 'bn', 'none')
      if not isinstance(self.nepa_layer_output_norm, str):
        raise ValueError(
          'nepa_layer_output_norm must be a str, '
          f'got {type(self.nepa_layer_output_norm).__name__}')
      if self.nepa_layer_output_norm not in allowed_layer_output_norm:
        raise ValueError(
          'nepa_layer_output_norm must be one of '
          f'{sorted(allowed_layer_output_norm)}, got {self.nepa_layer_output_norm!r}')
      if self.nepa_layer_output_norm in ('ln', 'rms') and self.nepa_layer_output_norm != self.nepa_final_norm:
        raise ValueError(
          'nepa_layer_output_norm must match nepa_final_norm when using "ln" or "rms": '
          f'nepa_layer_output_norm={self.nepa_layer_output_norm!r}, '
          f'nepa_final_norm={self.nepa_final_norm!r}')
      if self.nepa_final_norm == 'none' and self.nepa_layer_output_norm in ('ln', 'rms'):
        raise ValueError(
          'nepa_layer_output_norm cannot be "ln" or "rms" when nepa_final_norm="none". '
          f'Got nepa_layer_output_norm={self.nepa_layer_output_norm!r}')
      if self.nepa_layer_output_norm == 'bn':
        if not isinstance(self.nepa_layer_output_bn_eps, (int, float)):
          raise ValueError(
            'nepa_layer_output_bn_eps must be a float, '
            f'got {type(self.nepa_layer_output_bn_eps).__name__}')
        if self.nepa_layer_output_bn_eps <= 0:
          raise ValueError(
            'nepa_layer_output_bn_eps must be > 0, '
            f'got {self.nepa_layer_output_bn_eps}')

      if self.nepa_vit_drift_diagnostics:
        if self.nepa_vit_drift_layers is None:
          raise ValueError(
            'nepa_vit_drift_layers must be set when nepa_vit_drift_diagnostics=True '
            '(no silent defaults).')
      self.nepa_vit_drift_layers = _validate_layer_list('nepa_vit_drift_layers', self.nepa_vit_drift_layers)

      allowed_rep_pooling = ('mean', 'last')
      if not isinstance(self.nepa_rep_pooling, str):
        raise ValueError(
          'nepa_rep_pooling must be a str, '
          f'got {type(self.nepa_rep_pooling).__name__}')
      if self.nepa_rep_pooling not in allowed_rep_pooling:
        raise ValueError(
          'nepa_rep_pooling must be one of '
          f'{sorted(allowed_rep_pooling)}, got {self.nepa_rep_pooling!r}')

      if self.nepa_stability is None:
        raise ValueError('nepa_stability must be set when use_nepa=True')
      if self.nepa_stability not in ('stopgrad', 'sigreg', 'wristband', 'rdmreg'):
        raise ValueError(
          'nepa_stability must be "stopgrad", "sigreg", "wristband", or "rdmreg", '
          f'got {self.nepa_stability}')

      self.nepa_sigreg_layers = _validate_layer_list('nepa_sigreg_layers', self.nepa_sigreg_layers)
      self.nepa_sigreg_time_layers = _validate_layer_list(
        'nepa_sigreg_time_layers', self.nepa_sigreg_time_layers)
      self.nepa_sigreg_batch_time_layers = _validate_layer_list(
        'nepa_sigreg_batch_time_layers', self.nepa_sigreg_batch_time_layers)
      self.nepa_sigreg_innov_layers = _validate_layer_list(
        'nepa_sigreg_innov_layers', self.nepa_sigreg_innov_layers)
      self.nepa_sigreg_rep_layers = _validate_layer_list(
        'nepa_sigreg_rep_layers', self.nepa_sigreg_rep_layers)

      self.nepa_wristband_layers = _validate_layer_list('nepa_wristband_layers', self.nepa_wristband_layers)
      self.nepa_wristband_time_layers = _validate_layer_list(
        'nepa_wristband_time_layers', self.nepa_wristband_time_layers)
      self.nepa_wristband_batch_time_layers = _validate_layer_list(
        'nepa_wristband_batch_time_layers', self.nepa_wristband_batch_time_layers)
      self.nepa_wristband_innov_layers = _validate_layer_list(
        'nepa_wristband_innov_layers', self.nepa_wristband_innov_layers)
      self.nepa_wristband_rep_layers = _validate_layer_list(
        'nepa_wristband_rep_layers', self.nepa_wristband_rep_layers)

      if self.nepa_sigreg_rep_scale is not None and self.nepa_sigreg_rep_layers is None:
        raise ValueError(
          'nepa_sigreg_rep_layers must be set when nepa_sigreg_rep_scale is set '
          'and use_nepa=True')

      if self.nepa_stability == 'stopgrad':
        if self.nepa_sigreg_scale != 0.0:
          raise ValueError('nepa_sigreg_scale must be 0.0 when nepa_stability="stopgrad"')
        if self.nepa_sigreg_scale_time not in (None, 0.0):
          raise ValueError(
            'nepa_sigreg_scale_time must be null or 0 when nepa_stability="stopgrad"')
        if self.nepa_sigreg_scale_batch_time not in (None, 0.0):
          raise ValueError(
            'nepa_sigreg_scale_batch_time must be null or 0 when nepa_stability="stopgrad"')
        if self.nepa_sigreg_innov_scale is not None:
          raise ValueError(
            'nepa_sigreg_innov_scale requires nepa_stability="sigreg" when use_nepa=True')
      elif self.nepa_stability in ('sigreg', 'rdmreg'):
        if self.nepa_sigreg_scale is not None and self.nepa_sigreg_scale < 0:
          raise ValueError('nepa_sigreg_scale must be >= 0 when nepa_stability in {"sigreg","rdmreg"}')
        if self.nepa_sigreg_scale_time is not None:
          if self.nepa_sigreg_scale_time <= 0:
            raise ValueError('nepa_sigreg_scale_time must be > 0 when set')
        if self.nepa_sigreg_scale_batch_time is not None:
          if self.nepa_sigreg_scale_batch_time <= 0:
            raise ValueError('nepa_sigreg_scale_batch_time must be > 0 when set')
        if self.nepa_sigreg_scale is not None and self.nepa_sigreg_scale > 0:
          if self.nepa_sigreg_layers is None:
            raise ValueError(
              'nepa_sigreg_layers must be set when nepa_sigreg_scale > 0 '
              'and nepa_stability in {"sigreg","rdmreg"}')
        if self.nepa_sigreg_scale_time is not None:
          if self.nepa_sigreg_time_layers is None:
            raise ValueError(
              'nepa_sigreg_time_layers must be set when nepa_sigreg_scale_time is set '
              'and nepa_stability in {"sigreg","rdmreg"}')
        if self.nepa_sigreg_scale_batch_time is not None:
          if self.nepa_sigreg_batch_time_layers is None:
            raise ValueError(
              'nepa_sigreg_batch_time_layers must be set when nepa_sigreg_scale_batch_time is set '
              'and nepa_stability in {"sigreg","rdmreg"}')
        if self.nepa_sigreg_innov_scale is not None:
          if self.nepa_sigreg_innov_layers is None:
            raise ValueError(
              'nepa_sigreg_innov_layers must be set when nepa_sigreg_innov_scale is set '
              'and nepa_stability in {"sigreg","rdmreg"}')
      else:  # nepa_stability == "wristband"
        if self.nepa_wristband_max_samples is None:
          raise ValueError(
            'nepa_wristband_max_samples must be set when nepa_stability="wristband" '
            '(wristband has an O(N^2) kernel; no silent defaults).')
        if not isinstance(self.nepa_wristband_max_samples, int):
          raise ValueError(
            'nepa_wristband_max_samples must be an int, '
            f'got {type(self.nepa_wristband_max_samples).__name__}')
        if self.nepa_wristband_max_samples < 2:
          raise ValueError(
            'nepa_wristband_max_samples must be >= 2 when nepa_stability="wristband", '
            f'got {self.nepa_wristband_max_samples}')

        if not isinstance(self.nepa_wristband_beta, (int, float)):
          raise ValueError(
            'nepa_wristband_beta must be a float, '
            f'got {type(self.nepa_wristband_beta).__name__}')
        if self.nepa_wristband_beta <= 0:
          raise ValueError(f'nepa_wristband_beta must be > 0, got {self.nepa_wristband_beta}')
        if self.nepa_wristband_angular not in ("chordal", "geodesic"):
          raise ValueError(
            'nepa_wristband_angular must be "chordal" or "geodesic", '
            f'got {self.nepa_wristband_angular!r}')
        if self.nepa_wristband_reduction not in ("per_point", "global"):
          raise ValueError(
            'nepa_wristband_reduction must be "per_point" or "global", '
            f'got {self.nepa_wristband_reduction!r}')
        if self.nepa_wristband_moment not in ("mu_only", "kl_diag", "kl_full", "jeff_diag", "jeff_full", "w2"):
          raise ValueError(
            'nepa_wristband_moment must be one of '
            '{"mu_only","kl_diag","kl_full","jeff_diag","jeff_full","w2"}, '
            f'got {self.nepa_wristband_moment!r}')
        for name in ("nepa_wristband_lambda_rad", "nepa_wristband_lambda_ang", "nepa_wristband_lambda_mom"):
          value = getattr(self, name)
          if not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a float, got {type(value).__name__}')
          if value < 0:
            raise ValueError(f'{name} must be >= 0, got {value}')

        active_sites: list[str] = []
        for scale_name, layers_name in (
          ("nepa_wristband_scale", "nepa_wristband_layers"),
          ("nepa_wristband_scale_time", "nepa_wristband_time_layers"),
          ("nepa_wristband_scale_batch_time", "nepa_wristband_batch_time_layers"),
          ("nepa_wristband_innov_scale", "nepa_wristband_innov_layers"),
          ("nepa_wristband_rep_scale", "nepa_wristband_rep_layers"),
        ):
          scale = getattr(self, scale_name)
          if scale is None:
            continue
          if not isinstance(scale, (int, float)):
            raise ValueError(f'{scale_name} must be a float or null, got {type(scale).__name__}')
          if scale <= 0:
            raise ValueError(f'{scale_name} must be > 0 when set (nepa_stability="wristband"), got {scale}')
          layers = getattr(self, layers_name)
          if layers is None:
            raise ValueError(f'{layers_name} must be set when {scale_name} is set and nepa_stability="wristband"')
          active_sites.append(scale_name)
        if not active_sites:
          raise ValueError(
            'nepa_stability="wristband" requires at least one active wristband site: '
            'set one of nepa_wristband_*scale* > 0 and its corresponding *_layers.')

        # Wristband is a replacement for SIGReg: forbid enabling SIGReg scales.
        for name in (
          "nepa_sigreg_scale",
          "nepa_sigreg_scale_time",
          "nepa_sigreg_scale_batch_time",
          "nepa_sigreg_innov_scale",
          "nepa_sigreg_rep_scale",
        ):
          value = getattr(self, name)
          if value not in (None, 0.0):
            raise ValueError(
              f'{name} must be null/0 when nepa_stability="wristband" '
              '(no SIGReg + wristband mixing).')

      # If stability is not wristband, wristband scales must be disabled.
      if self.nepa_stability != "wristband":
        for name in (
          "nepa_wristband_scale",
          "nepa_wristband_scale_time",
          "nepa_wristband_scale_batch_time",
          "nepa_wristband_innov_scale",
          "nepa_wristband_rep_scale",
        ):
          value = getattr(self, name)
          if value not in (None, 0.0):
            raise ValueError(
              f'{name} must be null/0 unless nepa_stability="wristband" (no silent mixing).')

      # NEPA RDMReg hyperparameters (only when enabled).
      if self.nepa_stability == 'rdmreg':
        if self.nepa_rdmreg_num_projections is None:
          raise ValueError('nepa_rdmreg_num_projections must be set when nepa_stability="rdmreg"')
        if self.nepa_rdmreg_num_projections < 1:
          raise ValueError(
            'nepa_rdmreg_num_projections must be >= 1, '
            f'got {self.nepa_rdmreg_num_projections}')
        if self.nepa_rdmreg_p is None:
          raise ValueError('nepa_rdmreg_p must be set when nepa_stability="rdmreg"')
        if not isinstance(self.nepa_rdmreg_p, (int, float)):
          raise ValueError(
            'nepa_rdmreg_p must be a float, '
            f'got {type(self.nepa_rdmreg_p).__name__}')
        if self.nepa_rdmreg_p <= 0:
          raise ValueError(f'nepa_rdmreg_p must be > 0, got {self.nepa_rdmreg_p}')
        if self.nepa_rdmreg_mean_shift is None:
          raise ValueError('nepa_rdmreg_mean_shift must be set when nepa_stability="rdmreg"')
        if not isinstance(self.nepa_rdmreg_mean_shift, (int, float)):
          raise ValueError(
            'nepa_rdmreg_mean_shift must be a float, '
            f'got {type(self.nepa_rdmreg_mean_shift).__name__}')
        if self.nepa_rdmreg_target_dist is None:
          raise ValueError('nepa_rdmreg_target_dist must be set when nepa_stability="rdmreg"')
        if self.nepa_rdmreg_target_dist not in ('rectified_lp_distribution', 'lp_distribution'):
          raise ValueError(
            'nepa_rdmreg_target_dist must be "rectified_lp_distribution" or "lp_distribution", '
            f'got {self.nepa_rdmreg_target_dist!r}')
        if self.nepa_rdmreg_sigma_mode is None:
          raise ValueError('nepa_rdmreg_sigma_mode must be set when nepa_stability="rdmreg"')
        if self.nepa_rdmreg_sigma_mode not in ('gn_unit_var', 'manual'):
          raise ValueError(
            'nepa_rdmreg_sigma_mode must be "gn_unit_var" or "manual", '
            f'got {self.nepa_rdmreg_sigma_mode!r}')
        if self.nepa_rdmreg_sigma_mode == 'manual':
          if self.nepa_rdmreg_sigma is None:
            raise ValueError('nepa_rdmreg_sigma must be set when nepa_rdmreg_sigma_mode="manual"')
          if not isinstance(self.nepa_rdmreg_sigma, (int, float)):
            raise ValueError(
              'nepa_rdmreg_sigma must be a float, '
              f'got {type(self.nepa_rdmreg_sigma).__name__}')
          if self.nepa_rdmreg_sigma <= 0:
            raise ValueError(f'nepa_rdmreg_sigma must be > 0, got {self.nepa_rdmreg_sigma}')
        else:
          if self.nepa_rdmreg_sigma is not None:
            raise ValueError('nepa_rdmreg_sigma must be null when nepa_rdmreg_sigma_mode="gn_unit_var"')
        if self.nepa_rdmreg_feature_activation is None:
          raise ValueError('nepa_rdmreg_feature_activation must be set when nepa_stability="rdmreg"')
        if self.nepa_rdmreg_feature_activation not in ('none', 'relu', 'reprelu'):
          raise ValueError(
            'nepa_rdmreg_feature_activation must be "none", "relu", or "reprelu", '
            f'got {self.nepa_rdmreg_feature_activation!r}')
      else:
        for name in (
          'nepa_rdmreg_num_projections',
          'nepa_rdmreg_p',
          'nepa_rdmreg_mean_shift',
          'nepa_rdmreg_target_dist',
          'nepa_rdmreg_sigma_mode',
          'nepa_rdmreg_sigma',
          'nepa_rdmreg_feature_activation',
        ):
          if getattr(self, name) is not None:
            raise ValueError(f'{name} must be null unless nepa_stability="rdmreg"')

      # NEPA loss parameters
      if self.nepa_loss_batch_norm:
        if self.nepa_loss_type != 'mse':
          raise ValueError('nepa_loss_batch_norm requires nepa_loss_type="mse"')
        if not isinstance(self.nepa_loss_batch_norm_eps, (int, float)):
          raise ValueError(
            'nepa_loss_batch_norm_eps must be a float, '
            f'got {type(self.nepa_loss_batch_norm_eps)}')
        if self.nepa_loss_batch_norm_eps <= 0:
          raise ValueError(
            'nepa_loss_batch_norm_eps must be > 0, '
            f'got {self.nepa_loss_batch_norm_eps}')

      # NEPA projector semantics.
      allowed_projector_modes = ('both', 'target_only', 'pred_only')
      if not isinstance(self.nepa_projector_mode, str):
        raise ValueError(
          'nepa_projector_mode must be a str, '
          f'got {type(self.nepa_projector_mode).__name__}')
      if self.nepa_projector_mode not in allowed_projector_modes:
        raise ValueError(
          'nepa_projector_mode must be one of '
          f'{sorted(allowed_projector_modes)}, got {self.nepa_projector_mode!r}')
      if self.nepa_projector_mode != 'both':
        if not self.use_projector:
          raise ValueError('nepa_projector_mode!=both requires use_projector=True')
        if self.proj_out_dim is None:
          raise ValueError(
            'proj_out_dim must be set when nepa_projector_mode!=both '
            '(we need to project back to the encoder dim without silent defaults).')
        if not isinstance(self.proj_out_dim, int) or self.proj_out_dim <= 0:
          raise ValueError(f'proj_out_dim must be a positive int, got {self.proj_out_dim!r}')
        if self.proj_out_dim != self.dim:
          raise ValueError(
            'proj_out_dim must match dim when nepa_projector_mode!=both '
            f'(proj_out_dim={self.proj_out_dim}, dim={self.dim})')
      allowed_static_tokenizers = ('conv_patch_embed', 'mamba_static_patches', 'mamba_topk_tokens', 'mantis_tokgen')
      if self.nepa_static_tokenizer not in allowed_static_tokenizers:
        raise ValueError(
          'nepa_static_tokenizer must be one of '
          f'{sorted(allowed_static_tokenizers)}, got {self.nepa_static_tokenizer!r}')

      # conv_patch_embed scalar stats (mean/std) settings.
      if self.nepa_patch_embed_scalar_stats_mode is not None:
        if not isinstance(self.nepa_patch_embed_scalar_stats_mode, str):
          raise ValueError(
            'nepa_patch_embed_scalar_stats_mode must be a str or null, '
            f'got {type(self.nepa_patch_embed_scalar_stats_mode).__name__}')
      allowed_patch_embed_stats_modes = ('sequence_norm', 'patch_norm')
      if (self.nepa_patch_embed_scalar_stats_mode is not None
          and self.nepa_patch_embed_scalar_stats_mode not in allowed_patch_embed_stats_modes):
        raise ValueError(
          'nepa_patch_embed_scalar_stats_mode must be one of '
          f'{sorted(allowed_patch_embed_stats_modes)} or null, '
          f'got {self.nepa_patch_embed_scalar_stats_mode!r}')
      if self.nepa_static_tokenizer != 'conv_patch_embed':
        if self.nepa_patch_embed_scalar_stats_mode is not None:
          raise ValueError(
            'nepa_patch_embed_scalar_stats_mode is only supported when nepa_static_tokenizer="conv_patch_embed". '
            f'Got nepa_static_tokenizer={self.nepa_static_tokenizer!r}')
        if self.nepa_patch_embed_cnn_dim is not None:
          raise ValueError(
            'nepa_patch_embed_cnn_dim must be null unless nepa_static_tokenizer="conv_patch_embed" '
            'and nepa_patch_embed_scalar_stats_mode is set.')
      else:
        if self.nepa_patch_embed_scalar_stats_mode is None:
          if self.nepa_patch_embed_cnn_dim is not None:
            raise ValueError(
              'nepa_patch_embed_cnn_dim must be null when nepa_patch_embed_scalar_stats_mode is null.')
        else:
          if self.data_preprocess_normalize:
            raise ValueError(
              'nepa_patch_embed_scalar_stats_mode requires data_preprocess_normalize=false '
              '(mean/std must be computed on raw, unclipped samples).')
          if self.nepa_use_dynamic_chunking:
            raise ValueError(
              'nepa_patch_embed_scalar_stats_mode requires nepa_use_dynamic_chunking=false '
              '(static conv_patch_embed tokenizer only).')

          if self.nepa_patch_embed_cnn_dim is None:
            raise ValueError(
              'nepa_patch_embed_cnn_dim must be set when nepa_patch_embed_scalar_stats_mode is set')
          if not isinstance(self.nepa_patch_embed_cnn_dim, int):
            raise ValueError(
              'nepa_patch_embed_cnn_dim must be an int, '
              f'got {type(self.nepa_patch_embed_cnn_dim).__name__}')
          if self.nepa_patch_embed_cnn_dim < 1:
            raise ValueError(
              'nepa_patch_embed_cnn_dim must be >= 1, '
              f'got {self.nepa_patch_embed_cnn_dim}')

          if not isinstance(self.nepa_patch_embed_scalar_hidden_dim, int):
            raise ValueError(
              'nepa_patch_embed_scalar_hidden_dim must be an int, '
              f'got {type(self.nepa_patch_embed_scalar_hidden_dim).__name__}')
          if self.nepa_patch_embed_scalar_hidden_dim < 1:
            raise ValueError(
              'nepa_patch_embed_scalar_hidden_dim must be >= 1, '
              f'got {self.nepa_patch_embed_scalar_hidden_dim}')

          if self.nepa_patch_embed_scalar_scales is None:
            raise ValueError(
              'nepa_patch_embed_scalar_scales must be set when nepa_patch_embed_scalar_stats_mode is set')
          if not isinstance(self.nepa_patch_embed_scalar_scales, (list, tuple)):
            raise ValueError(
              'nepa_patch_embed_scalar_scales must be a list/tuple of floats, '
              f'got {type(self.nepa_patch_embed_scalar_scales).__name__}')
          if not self.nepa_patch_embed_scalar_scales:
            raise ValueError('nepa_patch_embed_scalar_scales must be non-empty')
          normalized_scales: list[float] = []
          for idx, scale in enumerate(self.nepa_patch_embed_scalar_scales):
            if not isinstance(scale, (int, float)):
              raise ValueError(f'nepa_patch_embed_scalar_scales[{idx}] must be a float, got {scale!r}')
            if scale <= 0:
              raise ValueError(f'nepa_patch_embed_scalar_scales[{idx}] must be > 0, got {scale}')
            normalized_scales.append(float(scale))
          self.nepa_patch_embed_scalar_scales = tuple(normalized_scales)

          if not isinstance(self.nepa_patch_embed_scalar_epsilon, (int, float)):
            raise ValueError(
              'nepa_patch_embed_scalar_epsilon must be a float, '
              f'got {type(self.nepa_patch_embed_scalar_epsilon).__name__}')
          if self.nepa_patch_embed_scalar_epsilon <= 0:
            raise ValueError(
              'nepa_patch_embed_scalar_epsilon must be > 0, '
              f'got {self.nepa_patch_embed_scalar_epsilon}')

          expected_dim = self.nepa_patch_embed_cnn_dim + 2 * self.nepa_patch_embed_scalar_hidden_dim
          if expected_dim != self.dim:
            raise ValueError(
              'dim must equal nepa_patch_embed_cnn_dim + 2*nepa_patch_embed_scalar_hidden_dim when '
              'nepa_patch_embed_scalar_stats_mode is set. '
              f'Got dim={self.dim}, nepa_patch_embed_cnn_dim={self.nepa_patch_embed_cnn_dim}, '
              f'nepa_patch_embed_scalar_hidden_dim={self.nepa_patch_embed_scalar_hidden_dim}')
      if self.nepa_static_tokenizer == 'mamba_static_patches':
        allowed_readouts = ('last', 'mean', 'conv')
        if self.nepa_local_mamba_readout not in allowed_readouts:
          raise ValueError(
            'nepa_local_mamba_readout must be one of '
            f'{sorted(allowed_readouts)}, got {self.nepa_local_mamba_readout!r}')
      if self.nepa_static_tokenizer == 'mamba_topk_tokens':
        allowed_readouts = ('last', 'mean')
        if self.nepa_local_mamba_readout not in allowed_readouts:
          raise ValueError(
            'nepa_local_mamba_readout must be one of '
            f'{sorted(allowed_readouts)} when nepa_static_tokenizer="mamba_topk_tokens", '
            f'got {self.nepa_local_mamba_readout!r}')
        if self.nepa_topk_tokens is None:
          raise ValueError('nepa_topk_tokens must be set when nepa_static_tokenizer="mamba_topk_tokens"')
        if not isinstance(self.nepa_topk_tokens, int):
          raise ValueError(f'nepa_topk_tokens must be int, got {type(self.nepa_topk_tokens).__name__}')
        if self.nepa_topk_tokens < 2:
          raise ValueError(f'nepa_topk_tokens must be >= 2, got {self.nepa_topk_tokens}')
        if self.nepa_topk_tokens > self.channel_size:
          raise ValueError(
            'nepa_topk_tokens must be <= channel_size, '
            f'got nepa_topk_tokens={self.nepa_topk_tokens}, channel_size={self.channel_size}')
      if self.nepa_static_tokenizer == 'mantis_tokgen':
        if self.channel_size % self.patch_size != 0:
          raise ValueError(
            'nepa_static_tokenizer="mantis_tokgen" requires channel_size divisible by patch_size, '
            f'got channel_size={self.channel_size}, patch_size={self.patch_size}')

        if self.nepa_mantis_kernel_size is None:
          raise ValueError('nepa_mantis_kernel_size must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_kernel_size, int):
          raise ValueError(f'nepa_mantis_kernel_size must be int, got {type(self.nepa_mantis_kernel_size).__name__}')
        if self.nepa_mantis_kernel_size < 1:
          raise ValueError(f'nepa_mantis_kernel_size must be >= 1, got {self.nepa_mantis_kernel_size}')

        if self.nepa_mantis_conv_mode is None:
          raise ValueError('nepa_mantis_conv_mode must be set when nepa_static_tokenizer="mantis_tokgen"')
        allowed_conv_modes = ('patch_embed', 'stride1_pool')
        if self.nepa_mantis_conv_mode not in allowed_conv_modes:
          raise ValueError(
            f'nepa_mantis_conv_mode must be one of {sorted(allowed_conv_modes)}, got {self.nepa_mantis_conv_mode!r}')
        if self.nepa_mantis_conv_mode == 'patch_embed' and self.nepa_mantis_kernel_size != self.patch_size:
          raise ValueError(
            'nepa_mantis_conv_mode="patch_embed" requires nepa_mantis_kernel_size==patch_size, '
            f'got nepa_mantis_kernel_size={self.nepa_mantis_kernel_size}, patch_size={self.patch_size}')

        if not isinstance(self.nepa_mantis_use_diff_branch, bool):
          raise ValueError(
            'nepa_mantis_use_diff_branch must be a bool, '
            f'got {type(self.nepa_mantis_use_diff_branch).__name__}')

        if self.nepa_mantis_instance_norm_mode is None:
          raise ValueError(
            'nepa_mantis_instance_norm_mode must be set when nepa_static_tokenizer="mantis_tokgen"')
        allowed_norm_modes = ('per_channel', 'global_over_channels')
        if self.nepa_mantis_instance_norm_mode not in allowed_norm_modes:
          raise ValueError(
            'nepa_mantis_instance_norm_mode must be one of '
            f'{sorted(allowed_norm_modes)}, got {self.nepa_mantis_instance_norm_mode!r}')
        if self.nepa_mantis_instance_norm_eps is None:
          raise ValueError(
            'nepa_mantis_instance_norm_eps must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_instance_norm_eps, (int, float)):
          raise ValueError(
            'nepa_mantis_instance_norm_eps must be a float, '
            f'got {type(self.nepa_mantis_instance_norm_eps).__name__}')
        if self.nepa_mantis_instance_norm_eps <= 0:
          raise ValueError(
            'nepa_mantis_instance_norm_eps must be > 0, '
            f'got {self.nepa_mantis_instance_norm_eps}')

        if self.nepa_mantis_stats_mode is None:
          raise ValueError('nepa_mantis_stats_mode must be set when nepa_static_tokenizer="mantis_tokgen"')
        allowed_stats_modes = ('flatten_project', 'per_channel_slots')
        if self.nepa_mantis_stats_mode not in allowed_stats_modes:
          raise ValueError(
            f'nepa_mantis_stats_mode must be one of {sorted(allowed_stats_modes)}, got {self.nepa_mantis_stats_mode!r}')
        if self.nepa_mantis_stats_dim is None:
          raise ValueError('nepa_mantis_stats_dim must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_stats_dim, int):
          raise ValueError(f'nepa_mantis_stats_dim must be int, got {type(self.nepa_mantis_stats_dim).__name__}')
        if self.nepa_mantis_stats_dim < 1:
          raise ValueError(f'nepa_mantis_stats_dim must be >= 1, got {self.nepa_mantis_stats_dim}')

        if self.nepa_mantis_scalar_scales is None:
          raise ValueError('nepa_mantis_scalar_scales must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_scalar_scales, (list, tuple)):
          raise ValueError(
            'nepa_mantis_scalar_scales must be a list/tuple of floats, '
            f'got {type(self.nepa_mantis_scalar_scales).__name__}')
        if not self.nepa_mantis_scalar_scales:
          raise ValueError('nepa_mantis_scalar_scales must be non-empty')
        for idx, scale in enumerate(self.nepa_mantis_scalar_scales):
          if not isinstance(scale, (int, float)):
            raise ValueError(f'nepa_mantis_scalar_scales[{idx}] must be a float, got {scale!r}')
          if scale <= 0:
            raise ValueError(f'nepa_mantis_scalar_scales[{idx}] must be > 0, got {scale}')

        if self.nepa_mantis_scalar_hidden_dim is None:
          raise ValueError('nepa_mantis_scalar_hidden_dim must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_scalar_hidden_dim, int):
          raise ValueError(
            f'nepa_mantis_scalar_hidden_dim must be int, got {type(self.nepa_mantis_scalar_hidden_dim).__name__}')
        if self.nepa_mantis_scalar_hidden_dim < 1:
          raise ValueError(f'nepa_mantis_scalar_hidden_dim must be >= 1, got {self.nepa_mantis_scalar_hidden_dim}')
        if self.nepa_mantis_scalar_epsilon is None:
          raise ValueError('nepa_mantis_scalar_epsilon must be set when nepa_static_tokenizer="mantis_tokgen"')
        if not isinstance(self.nepa_mantis_scalar_epsilon, (int, float)):
          raise ValueError(
            'nepa_mantis_scalar_epsilon must be a float, '
            f'got {type(self.nepa_mantis_scalar_epsilon).__name__}')
        if self.nepa_mantis_scalar_epsilon <= 0:
          raise ValueError(f'nepa_mantis_scalar_epsilon must be > 0, got {self.nepa_mantis_scalar_epsilon}')
      if self.nepa_use_dynamic_chunking:
        if self.nepa_static_tokenizer != 'conv_patch_embed':
          raise ValueError(
            'nepa_static_tokenizer must be "conv_patch_embed" when nepa_use_dynamic_chunking=True')
        if self.nepa_chunk_router_mode is None:
          raise ValueError('nepa_chunk_router_mode must be set when nepa_use_dynamic_chunking=True')
        if self.nepa_chunk_router_mode not in ('causal', 'lookahead'):
          raise ValueError(
            'nepa_chunk_router_mode must be "causal" or "lookahead", '
            f'got {self.nepa_chunk_router_mode!r}')
        if self.nepa_chunk_boundary_threshold is None:
          raise ValueError('nepa_chunk_boundary_threshold must be set when nepa_use_dynamic_chunking=True')
        if not 0 < self.nepa_chunk_boundary_threshold < 1:
          raise ValueError(
            'nepa_chunk_boundary_threshold must be in (0, 1), '
            f'got {self.nepa_chunk_boundary_threshold}')
        if self.nepa_chunk_ratio_target is None:
          raise ValueError('nepa_chunk_ratio_target must be set when nepa_use_dynamic_chunking=True')
        if self.nepa_chunk_ratio_target <= 1:
          raise ValueError(
            'nepa_chunk_ratio_target must be > 1, '
            f'got {self.nepa_chunk_ratio_target}')
        if self.nepa_chunk_ratio_loss_scale is None:
          raise ValueError('nepa_chunk_ratio_loss_scale must be set when nepa_use_dynamic_chunking=True')
        if self.nepa_chunk_ratio_loss_scale < 0:
          raise ValueError(
            'nepa_chunk_ratio_loss_scale must be >= 0, '
            f'got {self.nepa_chunk_ratio_loss_scale}')
        if self.nepa_chunk_max_tokens is None:
          raise ValueError('nepa_chunk_max_tokens must be set when nepa_use_dynamic_chunking=True')
        if self.nepa_chunk_max_tokens < 2:
          raise ValueError(
            'nepa_chunk_max_tokens must be >= 2, '
            f'got {self.nepa_chunk_max_tokens}')
        _validate_mamba_settings('nepa_use_dynamic_chunking=True')
      elif self.nepa_static_tokenizer == 'mamba_static_patches':
        _validate_mamba_settings('nepa_static_tokenizer="mamba_static_patches"')
      elif self.nepa_static_tokenizer == 'mamba_topk_tokens':
        _validate_mamba_settings('nepa_static_tokenizer="mamba_topk_tokens"')
    else:
      allowed_masking = {'remove', 'zero', 'mask'}
      if self.masking_strategy not in allowed_masking:
        raise ValueError(
          f'masking_strategy must be one of {sorted(allowed_masking)}, got {self.masking_strategy}')
      if self.use_lejepa:
        if self.num_views < 2:
          raise ValueError(f'LeJEPA requires num_views >= 2, got {self.num_views}')
        if self.loss_scale_sigreg is None:
          raise ValueError('LeJEPA requires loss_scale_sigreg to be set')
        if self.loss_scale_inv is None:
          raise ValueError('LeJEPA requires loss_scale_inv to be set')
      if self.loss_scale_inv_patch is not None and self.masking_strategy != 'mask':
        raise ValueError(
          'loss_scale_inv_patch requires masking_strategy="mask" to preserve patch tokens')
    if self.nepa_use_dynamic_chunking and not self.use_nepa:
      raise ValueError('nepa_use_dynamic_chunking requires use_nepa=True')

    # Projector backend validation (shared by JEPA/NEPA/LeJEPA).
    allowed_proj_backends = ("mlp", "invertible_flow")
    if not isinstance(self.proj_backend, str):
      raise ValueError(f'proj_backend must be a str, got {type(self.proj_backend).__name__}')
    if self.proj_backend not in allowed_proj_backends:
      raise ValueError(
        f'proj_backend must be one of {sorted(allowed_proj_backends)}, got {self.proj_backend!r}')
    if self.proj_backend == "invertible_flow":
      if not self.use_projector:
        raise ValueError('proj_backend="invertible_flow" requires use_projector=true (otherwise it is unused).')
      if self.proj_dim < 2:
        raise ValueError(
          'proj_backend="invertible_flow" requires proj_dim >= 2 for non-degenerate coupling layers, '
          f'got proj_dim={self.proj_dim}')
      if self.proj_flow_n_layers < 0:
        raise ValueError(f'proj_flow_n_layers must be >= 0, got {self.proj_flow_n_layers}')
      if self.proj_flow_hidden_dim < 1:
        raise ValueError(f'proj_flow_hidden_dim must be >= 1, got {self.proj_flow_hidden_dim}')
      if self.proj_flow_n_blocks < 1:
        raise ValueError(f'proj_flow_n_blocks must be >= 1, got {self.proj_flow_n_blocks}')
      if self.proj_flow_s_max <= 0:
        raise ValueError(f'proj_flow_s_max must be > 0, got {self.proj_flow_s_max}')
      if self.proj_flow_mask_mode not in ("alternating", "half"):
        raise ValueError(
          'proj_flow_mask_mode must be "alternating" or "half", '
          f'got {self.proj_flow_mask_mode!r}')
      if self.proj_flow_permute_mode not in ("none", "per_layer", "per_pair"):
        raise ValueError(
          'proj_flow_permute_mode must be "none", "per_layer", or "per_pair", '
          f'got {self.proj_flow_permute_mode!r}')
      if not isinstance(self.proj_flow_permute_seed, int):
        raise ValueError(
          'proj_flow_permute_seed must be an int, '
          f'got {type(self.proj_flow_permute_seed).__name__}')
      if self.proj_flow_permute_seed < 0:
        raise ValueError(f'proj_flow_permute_seed must be >= 0, got {self.proj_flow_permute_seed}')
