import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path, default='checkpoint path')
args = parser.parse_args()

# NOTE: PyTorch >= 2.6 defaults `weights_only=True`, which breaks loading
# our local training checkpoints (they include non-tensor metadata).
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
config = checkpoint["config"]
model = {
  param: weights
  for param, weights in checkpoint["model"].items()
  if param.startswith("target_encoder")
}
min_checkpoint_path = f"{args.checkpoint.stem}.min{args.checkpoint.suffix}"
torch.save({
  "model": model,
  "config": config,
}, min_checkpoint_path)
