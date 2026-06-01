#!/usr/bin/env bash
set -euo pipefail

cd /mnt/t0-train-shared/lejepa/ecg_jepa

/usr/bin/time -f "elapsed=%E" uv run python -m pretrain \
  --config-name ViTXS_ptbxl \
  "run_name=PTBXL_LENEPA_SIGREG_BRI20_L0-8_PD0_PROJ_s1" \
  out=/mnt/t0-train-shared/lejepa/pretrain \
  seed=1 \
  steps=20000 \
  stop_after_steps=20000 \
  offline_probe_eval_interval=1000 \
  +offline_probe_eval_at_step0=true \
  learning_rate_warmup_steps=1000 \
  learning_rate=0.0001 \
  weight_decay=1.0e-2 \
  final_weight_decay=1.0e-1 \
  proj_dim=64 \
  proj_hidden_dim=1536 \
  proj_num_layers=1 \
  +data.ptb-xl=/mnt/t0-train-shared/lejepa/data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.npy \
  ptb_xl_data_dir=/mnt/t0-train-shared/lejepa/data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
  probe_eval_interval=20000 \
  objective=nepa_sigreg \
  ema_encoder=false \
  pred_depth=0 \
  nepa_layer_output_norm=none \
  nepa_sigreg_scale=20 \
  nepa_sigreg_layers=[0,8] \
  nepa_sigreg_scale_time=null \
  nepa_sigreg_time_layers=[] \
  nepa_sigreg_rep_scale=20 \
  nepa_sigreg_rep_layers=[8] \
  nepa_sigreg_rep_use_projector=true \
  nepa_sigreg_innov_scale=20 \
  nepa_sigreg_innov_layers=[0,8] \
  nepa_sigreg_scale_batch_time=null \
  nepa_sigreg_batch_time_layers=[] \
  offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8] \
  use_projector=true
