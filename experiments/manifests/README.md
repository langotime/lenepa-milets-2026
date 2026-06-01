# Experiment Manifests

`main_matrix.csv` is the paper-facing summary of the PTB-XL/Diag experiment matrix.

`legacy_name_map.csv` maps historical run/config names (`AIONO`, `toyts`) to the names used in the paper and current configs.

`cauker_ucr.yaml` captures the single CauKer2M to UCR run used for the frozen-encoder check.

The shell scripts under `experiments/legacy_job_scripts/` are provenance copies. They intentionally preserve old absolute paths from `/mnt/t0-train-shared/lejepa/` and are not expected to run unchanged in this cleaned repository.
