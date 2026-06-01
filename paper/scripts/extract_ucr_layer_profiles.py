#!/usr/bin/env python3
"""Extract UCR per-layer metrics from a local W&B run and plot layer profiles."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_DIR = PAPER_DIR.parent
DEFAULT_RUN_DIR = REPO_DIR / "results/ucr_benchmark"


def _step_from_path(path: Path) -> int:
  match = re.fullmatch(r"ucr_step(\d+)\.json", path.name)
  if match is None:
    raise ValueError(f"Unexpected UCR file name: {path}")
  return int(match.group(1))


def _load_records(run_dir: Path) -> list[dict[str, object]]:
  files = sorted(run_dir.glob("ucr_step*.json"), key=_step_from_path)
  if not files:
    raise FileNotFoundError(f"No ucr_step*.json files found in {run_dir}")

  records: list[dict[str, object]] = []
  for path in files:
    payload = json.loads(path.read_text())
    by_layer = payload.get("mean_acc_by_layer")
    if not isinstance(by_layer, dict) or not by_layer:
      raise ValueError(f"Missing mean_acc_by_layer in {path}")

    layer_values = {int(layer): float(acc) for layer, acc in by_layer.items()}
    best_layer = max(layer_values, key=layer_values.get)
    records.append({
        "step": int(payload["log_step"]),
        "best_layer": best_layer,
        "best_acc": layer_values[best_layer],
        "num_datasets": int(payload["num_datasets"]),
        "resize_mode": str(payload.get("resize_mode", "")),
        "classifier": str(payload.get("classifier", "")),
        "layers": layer_values,
    })
  return records


def _write_long_csv(records: list[dict[str, object]], out_csv: Path) -> None:
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  with out_csv.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "step",
        "layer",
        "mean_acc",
        "mean_acc_pct",
        "best_layer_at_step",
        "best_acc_at_step",
        "best_acc_at_step_pct",
        "num_datasets",
        "resize_mode",
        "classifier",
    ])
    for record in records:
      for layer, acc in sorted(record["layers"].items()):
        writer.writerow([
            record["step"],
            layer,
            f"{acc:.10f}",
            f"{100.0 * acc:.4f}",
            record["best_layer"],
            f"{record['best_acc']:.10f}",
            f"{100.0 * float(record['best_acc']):.4f}",
            record["num_datasets"],
            record["resize_mode"],
            record["classifier"],
        ])


def _write_profile_csv(records: list[dict[str, object]], out_csv: Path) -> None:
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  layers = sorted(records[0]["layers"])
  best_record = max(records, key=lambda record: float(record["best_acc"]))
  final_record = max(records, key=lambda record: int(record["step"]))
  step0_record = min(records, key=lambda record: int(record["step"]))

  max_by_layer = {
      layer: max(float(record["layers"][layer]) for record in records)
      for layer in layers
  }
  max_step_by_layer = {
      layer: max(records, key=lambda record: float(record["layers"][layer]))["step"]
      for layer in layers
  }

  with out_csv.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "layer",
        "step0_acc_pct",
        f"best_checkpoint_step{best_record['step']}_acc_pct",
        f"final_step{final_record['step']}_acc_pct",
        "max_over_steps_acc_pct",
        "max_over_steps_step",
    ])
    for layer in layers:
      writer.writerow([
          layer,
          f"{100.0 * float(step0_record['layers'][layer]):.4f}",
          f"{100.0 * float(best_record['layers'][layer]):.4f}",
          f"{100.0 * float(final_record['layers'][layer]):.4f}",
          f"{100.0 * max_by_layer[layer]:.4f}",
          max_step_by_layer[layer],
      ])


def _plot_layer_profile(records: list[dict[str, object]], out_base: Path) -> None:
  out_base.parent.mkdir(parents=True, exist_ok=True)
  layers = sorted(records[0]["layers"])
  best_record = max(records, key=lambda record: float(record["best_acc"]))
  final_record = max(records, key=lambda record: int(record["step"]))
  step0_record = min(records, key=lambda record: int(record["step"]))
  max_by_layer = [
      100.0 * max(float(record["layers"][layer]) for record in records)
      for layer in layers
  ]

  series = [
      ("step 0", step0_record, "#8b8b8b", "o", "--"),
      (f"step {best_record['step']:,} (best checkpoint)", best_record, "#126a8a", "o", "-"),
      (f"step {final_record['step']:,} (final)", final_record, "#d15f02", "s", "-"),
  ]

  plt.figure(figsize=(7.0, 4.2))
  for label, record, color, marker, linestyle in series:
    values = [100.0 * float(record["layers"][layer]) for layer in layers]
    plt.plot(
        layers,
        values,
        label=label,
        color=color,
        marker=marker,
        linewidth=2.0,
        markersize=5,
        linestyle=linestyle,
    )

  plt.plot(
      layers,
      max_by_layer,
      label="best over checkpoints per layer",
      color="#3a923a",
      marker="^",
      linewidth=1.8,
      markersize=5,
      alpha=0.9,
  )

  best_layer = int(best_record["best_layer"])
  best_acc_pct = 100.0 * float(best_record["best_acc"])
  plt.scatter([best_layer], [best_acc_pct], color="#08384f", s=58, zorder=5)
  plt.annotate(
      f"peak L{best_layer}: {best_acc_pct:.2f}%",
      xy=(best_layer, best_acc_pct),
      xytext=(best_layer + 0.35, best_acc_pct + 0.9),
      arrowprops={"arrowstyle": "->", "color": "#08384f", "lw": 1.0},
      fontsize=9,
  )

  plt.xticks(layers)
  all_values = [
      100.0 * float(record["layers"][layer])
      for record in records
      for layer in layers
  ]
  plt.ylim(min(all_values) - 1.0, max(all_values) + 2.3)
  plt.xlabel("Encoder layer")
  plt.ylabel("UCR-128 mean accuracy (%)")
  plt.title("LeNEPA UCR layer profile: middle layers dominate", fontsize=14, pad=12)
  plt.grid(True, axis="y", alpha=0.25)
  plt.legend(frameon=False, fontsize=8, loc="lower right")
  plt.tight_layout()

  for suffix in (".pdf", ".png", ".svg"):
    plt.savefig(out_base.with_suffix(suffix), dpi=240)
  plt.close()


def _plot_heatmap(records: list[dict[str, object]], out_base: Path) -> None:
  out_base.parent.mkdir(parents=True, exist_ok=True)
  layers = sorted(records[0]["layers"])
  steps = [int(record["step"]) for record in records]
  data = np.array([
      [100.0 * float(record["layers"][layer]) for layer in layers]
      for record in records
  ])

  fig, ax = plt.subplots(figsize=(7.2, 5.0))
  image = ax.imshow(data, aspect="auto", origin="lower", cmap="YlGnBu")
  ax.set_xticks(np.arange(len(layers)))
  ax.set_xticklabels(layers)
  ax.set_yticks(np.arange(len(steps)))
  ax.set_yticklabels([f"{step // 1000}k" if step else "0" for step in steps])
  ax.set_xlabel("Encoder layer")
  ax.set_ylabel("Pretraining step")
  ax.set_title("UCR-128 accuracy by layer and checkpoint")

  best_record = max(records, key=lambda record: float(record["best_acc"]))
  best_step_index = steps.index(int(best_record["step"]))
  best_layer_index = layers.index(int(best_record["best_layer"]))
  ax.scatter([best_layer_index], [best_step_index], color="#d95f02", s=50, marker="*", zorder=3)

  cbar = fig.colorbar(image, ax=ax)
  cbar.set_label("Mean accuracy (%)")
  fig.tight_layout()

  for suffix in (".pdf", ".png", ".svg"):
    fig.savefig(out_base.with_suffix(suffix), dpi=240)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
  parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
  parser.add_argument("--prefix", default="ucr_layer_profile_lenepa_seqnorm")
  args = parser.parse_args()

  records = _load_records(args.run_dir)
  out_dir = args.out_dir
  _write_long_csv(records, out_dir / "tables" / f"{args.prefix}.csv")
  _write_profile_csv(records, out_dir / "tables" / f"{args.prefix}_summary.csv")
  _plot_layer_profile(records, out_dir / "figures" / args.prefix)
  _plot_heatmap(records, out_dir / "figures" / f"{args.prefix}_heatmap")

  best_record = max(records, key=lambda record: float(record["best_acc"]))
  final_record = max(records, key=lambda record: int(record["step"]))
  print(f"loaded {len(records)} UCR checkpoints from {args.run_dir}")
  print(
      "best checkpoint: "
      f"step={best_record['step']} layer={best_record['best_layer']} "
      f"acc={100.0 * float(best_record['best_acc']):.4f}%"
  )
  print(
      "final checkpoint: "
      f"step={final_record['step']} layer={final_record['best_layer']} "
      f"acc={100.0 * float(final_record['best_acc']):.4f}%"
  )


if __name__ == "__main__":
  main()
