#!/usr/bin/env bash
set -euo pipefail

cd /mnt/t0-train-shared/lejepa/ecg_jepa

/usr/bin/time -f "elapsed=%E" uv run python -m pretrain \
  --config-name ViTXS_toyts \
  "run_name=AIONO_JEPA_MEAN_s7" \
  dataset=toyts_basic_components \
  offline_probe_dataset=same_as_train \
  toyts.basic_components.num_enabled=2 \
  out=/mnt/t0-train-shared/lejepa/pretrain \
  seed=7 \
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
  objective=jepa \
  use_mean_pooling=true \
  +offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  +offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8]
