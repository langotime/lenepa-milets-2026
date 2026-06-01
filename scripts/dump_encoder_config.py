import argparse
import dataclasses
from os import path

import torch
import yaml

from configs.encoder import EncoderConfig


parser = argparse.ArgumentParser(description='Dump an EncoderConfig YAML from a training checkpoint.')
parser.add_argument('--chkpt', required=True, help='Path to a .pt checkpoint containing a "config" dict')
parser.add_argument('--out', required=True, help='Output path for the EncoderConfig YAML')
args = parser.parse_args()

chkpt_path = path.normpath(args.chkpt)
if not path.isfile(chkpt_path):
  raise ValueError(f'Checkpoint file does not exist: {chkpt_path}')
if path.splitext(chkpt_path)[1] != '.pt':
  raise ValueError(f'Expected a .pt checkpoint, got: {chkpt_path}')

out_path = path.normpath(args.out)
out_dir = path.dirname(out_path) or '.'
if not path.isdir(out_dir):
  raise ValueError(f'Output directory does not exist: {out_dir}')
if path.exists(out_path):
  raise ValueError(f'Output file already exists: {out_path}')

# NOTE: PyTorch >= 2.6 defaults `weights_only=True`, which breaks loading
# our local training checkpoints (they include non-tensor metadata).
chkpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)
if not isinstance(chkpt, dict):
  raise ValueError(f'Checkpoint must be a dict, got {type(chkpt)} from {chkpt_path}')
config = chkpt.get('config')
if config is None:
  raise ValueError(f'Checkpoint does not contain a "config" key: {chkpt_path}')
if not isinstance(config, dict):
  raise ValueError(
    f'Checkpoint "config" must be a dict, got {type(config)} from {chkpt_path}')

allowed_keys = {field.name for field in dataclasses.fields(EncoderConfig)}
filtered = {key: config[key] for key in allowed_keys if key in config}
missing_keys = sorted(allowed_keys - filtered.keys())
if missing_keys:
  raise ValueError(
    f'Checkpoint config is missing required encoder keys: {missing_keys}. '
    f'Provide a checkpoint that contains the full encoder config (e.g. from pretrain/finetune).')

if isinstance(filtered.get('channels'), tuple):
  filtered['channels'] = list(filtered['channels'])

with open(out_path, 'w') as fh:
  yaml.safe_dump(filtered, fh, sort_keys=False)

print(f'Wrote encoder config to {out_path}')
