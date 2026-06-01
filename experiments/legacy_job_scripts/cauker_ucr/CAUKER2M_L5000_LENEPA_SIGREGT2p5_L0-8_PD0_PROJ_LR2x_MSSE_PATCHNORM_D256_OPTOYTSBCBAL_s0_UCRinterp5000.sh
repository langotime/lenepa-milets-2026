#!/usr/bin/env bash
set -euo pipefail

cd /mnt/t0-train-shared/lejepa/ecg_jepa

export HF_HOME=/mnt/t0-train-shared/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/t0-train-shared/.cache/huggingface/datasets
mkdir -p "$HF_DATASETS_CACHE"

/usr/bin/time -f "elapsed=%E" uv run python -m pretrain \
  --config-name ViTXS_cauker2m \
  offline_probe_dataset=toyts_basic_components_balanced \
  "run_name=CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_OPTOYTSBCBAL_s0_UCRinterp5000" \
  out=/mnt/t0-train-shared/lejepa/pretrain \
  seed=0 \
  stop_after_steps=20000 \
  ucr.resize_mode=interp512 \
  ucr.resize_target_length=5000 \
  +data_preprocess_normalize=false \
  +nepa_patch_embed_scalar_stats_mode=patch_norm \
  +nepa_patch_embed_cnn_dim=192 \
  dim=256 \
  cauker.parquet_path=/mnt/t0-train-shared/lejepa/data/cauker_repo/CauKer2M_L5000.parquet \
  channel_size=5000 \
  ema_encoder=false \
  proj_dim=64 \
  proj_hidden_dim=1536 \
  proj_num_layers=1 \
  nepa_sigreg_scale=null \
  nepa_sigreg_layers=[] \
  nepa_sigreg_rep_layers=[] \
  nepa_sigreg_innov_layers=[] \
  nepa_sigreg_batch_time_layers=[] \
  nepa_sigreg_scale_time=2.5 \
  offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8] \
  learning_rate=0.0002 \
  final_learning_rate=0.0002
