#!/usr/bin/env bash
set -euo pipefail

cd /mnt/t0-train-shared/lejepa/ecg_jepa

/usr/bin/time -f "elapsed=%E" uv run python -m pretrain \
  --config-name ViTXS_ptbxl \
  "run_name=PTBXL_NEPA_STOPGRAD_MEAN_NOPROJ_s0" \
  out=/mnt/t0-train-shared/lejepa/pretrain \
  seed=0 \
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
  objective=nepa_stopgrad \
  ema_encoder=true \
  nepa_rep_pooling=mean \
  nepa_layer_output_norm=none \
  offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8] \
  use_projector=false \
  nepa_sigreg_rep_use_projector=false
