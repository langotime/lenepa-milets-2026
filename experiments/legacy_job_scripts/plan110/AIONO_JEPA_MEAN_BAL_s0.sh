#!/usr/bin/env bash
set -euo pipefail

cd /mnt/t0-train-shared/lejepa/ecg_jepa

/usr/bin/time -f "elapsed=%E" uv run python -m pretrain \
  --config-name ViTXS_toyts \
  "run_name=AIONO_JEPA_MEAN_BAL_s0" \
  dataset=toyts_basic_components \
  offline_probe_dataset=same_as_train \
  toyts.basic_components.num_enabled=2 \
  "+toyts.basic_components.component_weights_by_key={constant:1.0,gaussian_noise:1.0,uniform_noise:1.0,random_walk_noise:1.0,linear_trend:1.0,quadratic_trend:1.0,log_trend:1.0,sigmoid_trend:1.0,sine:1.0,sawtooth:1.0,square:1.0,spike:1.0,level_change:1.0,gaussian:1.0}" \
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
  objective=jepa \
  use_mean_pooling=true \
  +offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  +offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8]
