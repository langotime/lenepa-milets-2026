# LeNEPA MiLeTS 2026 Reproduction

This repository is a clean extraction of the code, experiment manifests, result tables, and paper assets used for the LeNEPA MiLeTS/KDD 2026 submission.

## What Is Included

- LeNEPA/NEPA/JEPA training code needed to reproduce the paper runs.
- Paper configs for PTB-XL, Aionoscope basic components, and CauKer2M.
- Machine-readable tables under `results/tables/`.
- Paper source and the figures/tables used by the submitted manuscript under `paper/`.

Raw datasets, checkpoints, W&B run directories, VM logs, Hydra multiruns, and local outputs are intentionally not included.

## Setup

Use Python 3.12 and `uv`:

```bash
cd /mnt/t0-train-shared/lenepa-milets-2026
uv sync --group dev
```

The `aiono` dependency currently points to the local sibling checkout `../aionoscope/aionoscope` in `pyproject.toml`. For public release, replace that source with the published Aionoscope/Aiono package URL or version pin.

Useful environment variables:

```bash
export OUTPUT_DIR=/path/to/pretrain_outputs
export PTBXL_ROOT=/path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
export PTBXL_NPY=${PTBXL_ROOT}.npy
export CAUKER_PARQUET=/path/to/CauKer2M_L5000.parquet
export UCR_ROOT=/path/to/UCRArchive_2018
```

## Data

PTB-XL is not vendored. Build the NumPy dump once:

```bash
uv run python -m scripts.dump_data --data-dir "$PTBXL_ROOT" --dataset ptb-xl
```

Aionoscope data is generated online by the `aiono` dependency; no static data file is required.

CauKer2M and UCR are not vendored. The CauKer run expects a parquet file via `CAUKER_PARQUET`; UCR evaluation expects the UCR Archive 2018 root via `UCR_ROOT`.

## Main Runs

PTB-XL LeNEPA, seed 0:

```bash
uv run python -m pretrain \
  --config-name ViTXS_ptbxl \
  run_name=ptbxl_lenepa_sigregt20_seed0 \
  out="$OUTPUT_DIR" \
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
  "+data.ptb-xl=$PTBXL_NPY" \
  ptb_xl_data_dir="$PTBXL_ROOT" \
  probe_eval_interval=20000 \
  objective=nepa_sigreg \
  ema_encoder=false \
  pred_depth=0 \
  nepa_layer_output_norm=none \
  nepa_sigreg_scale=null \
  nepa_sigreg_layers=[] \
  nepa_sigreg_scale_time=20 \
  nepa_sigreg_time_layers=[0,8] \
  nepa_sigreg_rep_scale=null \
  nepa_sigreg_rep_layers=[] \
  nepa_sigreg_innov_scale=null \
  nepa_sigreg_innov_layers=[] \
  nepa_sigreg_scale_batch_time=null \
  nepa_sigreg_batch_time_layers=[] \
  offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8] \
  use_projector=true
```

Aionoscope LeNEPA, seed 0:

```bash
uv run python -m pretrain \
  --config-name ViTXS_aiono \
  run_name=aionoscope_lenepa_sigregt20_seed0 \
  dataset=aiono_basic_components_imbalanced \
  offline_probe_dataset=same_as_train \
  out="$OUTPUT_DIR" \
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
  objective=nepa_sigreg \
  ema_encoder=false \
  pred_depth=0 \
  nepa_layer_output_norm=none \
  nepa_sigreg_scale=null \
  nepa_sigreg_layers=[] \
  nepa_sigreg_scale_time=20 \
  nepa_sigreg_time_layers=[0,8] \
  nepa_sigreg_rep_scale=null \
  nepa_sigreg_rep_layers=[] \
  nepa_sigreg_innov_scale=null \
  nepa_sigreg_innov_layers=[] \
  nepa_sigreg_scale_batch_time=null \
  nepa_sigreg_batch_time_layers=[] \
  offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8] \
  offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8] \
  use_projector=true
```

CauKer2M to UCR frozen-encoder run:

```bash
uv run python -m pretrain \
  --config-name ViTXS_cauker2m \
  offline_probe_dataset=aiono_basic_components_balanced \
  run_name=cauker2m_lenepa_sigregt2p5_ucr_seed0 \
  out="$OUTPUT_DIR" \
  seed=0 \
  stop_after_steps=20000 \
  ucr.archive_root="$UCR_ROOT" \
  ucr.resize_mode=interp512 \
  ucr.resize_target_length=5000 \
  +data_preprocess_normalize=false \
  +nepa_patch_embed_scalar_stats_mode=patch_norm \
  +nepa_patch_embed_cnn_dim=192 \
  dim=256 \
  cauker.parquet_path="$CAUKER_PARQUET" \
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
```

The broader matrix is summarized in `experiments/manifests/main_matrix.csv`. Submitted paper summaries use seeds `0..4` for the PTB-XL/Aionoscope fixed-horizon matrix and seed `0` for the CauKer-to-UCR frozen-encoder check.

## Rebuilding Results

The current paper tables/figures are already checked in. W&B histories are not part of this repository.

To rebuild the UCR layer profile from local `ucr_step*.json` files:

```bash
uv run python paper/scripts/extract_ucr_layer_profiles.py \
  --run-dir results/ucr_benchmark \
  --out-dir paper
```

To build the paper PDF:

```bash
cd paper
latexmk -pdf paper_lenepa.tex
```

## Repository Hygiene

Do not commit raw data, checkpoints, W&B run directories, Hydra outputs, VM logs, or local `pretrain/` outputs. The `.gitignore` is set up for those paths.
