"""Reproducibility helpers (opt-in).

This module implements *optional* fixed seeding for LeJEPA training runs.

Critical design constraint:
- When `seed is None`, callers must not alter RNG state or DataLoader behavior.
"""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch

__all__ = [
  'derive_seed',
  'seed_everything',
  'seed_worker_init_fn',
  'torch_generator',
]


def derive_seed(*, seed: int, salt: str) -> int:
  """Derive a deterministic 32-bit seed from a base seed and a salt string.

  This avoids using Python's built-in `hash()` (which is randomized per process).

  Args:
    seed: Base seed (must be a non-negative int).
    salt: Context string (e.g., "ptbxl_train_loader" or f"offline_probe_step={step}").

  Returns:
    A derived non-negative int seed in `[0, 2**32 - 1]`.
  """
  # Step 1: validate inputs.
  if not isinstance(seed, int):
    raise TypeError(f'seed must be an int, got {type(seed).__name__}')
  if seed < 0:
    raise ValueError(f'seed must be >= 0, got {seed}')
  if not isinstance(salt, str) or not salt:
    raise ValueError('salt must be a non-empty string')

  # Step 2: combine base seed with a stable CRC32 of the salt.
  salt_crc = zlib.crc32(salt.encode('utf-8')) & 0xFFFF_FFFF
  return int((seed ^ salt_crc) & 0xFFFF_FFFF)


def seed_everything(seed: int) -> None:
  """Seed Python, NumPy and PyTorch RNGs.

  Args:
    seed: Seed value (must be a non-negative int).
  """
  # Step 1: validate inputs.
  if not isinstance(seed, int):
    raise TypeError(f'seed must be an int, got {type(seed).__name__}')
  if seed < 0:
    raise ValueError(f'seed must be >= 0, got {seed}')

  # Step 2: seed RNGs used across the training stack.
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def seed_worker_init_fn(worker_id: int) -> None:
  """Seed NumPy/Python RNGs inside a PyTorch DataLoader worker.

  This function is intended to be passed as `DataLoader(..., worker_init_fn=...)`.
  It uses the PyTorch worker seed so that:
  - shuffling order (torch RNG) and
  - NumPy-based transforms (e.g., random ECG crops)
  are deterministic when the DataLoader is constructed with a deterministic generator.

  Args:
    worker_id: DataLoader worker id (unused; required by PyTorch API).
  """
  # Step 1: derive a per-worker seed from PyTorch's worker seed.
  _ = worker_id
  worker_seed = int(torch.initial_seed() % 2**32)

  # Step 2: seed worker-local RNGs.
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
  """Create a CPU torch.Generator seeded deterministically.

  Args:
    seed: Seed value (must be a non-negative int).

  Returns:
    A CPU `torch.Generator` instance.
  """
  # Step 1: validate inputs.
  if not isinstance(seed, int):
    raise TypeError(f'seed must be an int, got {type(seed).__name__}')
  if seed < 0:
    raise ValueError(f'seed must be >= 0, got {seed}')

  # Step 2: build the generator used by DataLoader shuffling.
  generator = torch.Generator()
  generator.manual_seed(seed)
  return generator

