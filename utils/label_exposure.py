"""Label exposure tracking utilities.

This module implements lightweight trackers for multi-label exposure curves:
- cumulative exposure on an incoming stream of targets, and
- unique-sample exposure for probe optimization loops (to avoid repeat inflation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
  'LabelExposureSnapshot',
  'LabelExposureTracker',
  'UniqueSampleLabelExposureTracker',
]


@dataclass(frozen=True)
class LabelExposureSnapshot:
  """JSON-serializable snapshot of label exposure state."""

  step: int
  cumulative_positives: list[int]
  first_positive_step: list[int | None]
  coverage_count: int
  coverage_fraction: float


class LabelExposureTracker:
  """Track cumulative label exposure for a multi-label target stream.

  Given targets $Y_t \\in \\{0,1\\}^{B\\times C}$ at step $t$, the tracker maintains:
  - cumulative positives per class: $E_t[c] = \\sum_{s\\le t}\\sum_i Y_s[i,c]$
  - first positive step per class: $F[c] = \\min\\{t : E_t[c] > 0\\}$
  - coverage: $\\mathrm{cov}_t = \\frac{1}{C}\\sum_c \\mathbb{1}[E_t[c] > 0]$
  """

  def __init__(self, *, class_names: list[str]) -> None:
    """Initialize the tracker for a fixed set of class names."""
    # Step 1: validate class metadata.
    if not isinstance(class_names, list) or not class_names:
      raise ValueError('class_names must be a non-empty list of strings.')
    if len(set(class_names)) != len(class_names):
      raise ValueError('class_names must be unique.')
    for name in class_names:
      if not isinstance(name, str) or not name.strip():
        raise ValueError(f'class_names must contain non-empty strings, got {name!r}')

    # Step 2: allocate tracking buffers.
    self.class_names = list(class_names)
    self.num_classes = len(self.class_names)
    self._cumulative_positives = torch.zeros((self.num_classes,), dtype=torch.int64)  # [C]
    self._first_positive_step: list[int | None] = [None] * self.num_classes  # len=C

  @property
  def cumulative_positives(self) -> torch.Tensor:
    """Return cumulative positives per class as an int64 tensor `[C]` on CPU."""
    return self._cumulative_positives

  @property
  def first_positive_step(self) -> list[int | None]:
    """Return the first-positive step per class (length `C`)."""
    return list(self._first_positive_step)

  def update(self, targets: torch.Tensor, *, step: int) -> None:
    """Update exposure counts from a target batch.

    Args:
      targets: Multi-label targets as a tensor `[B, C]` (any device/dtype).
      step: Global step index (must be a positive int).
    """
    # Step 1: validate inputs.
    if not isinstance(step, int) or step <= 0:
      raise ValueError(f'step must be a positive int, got {step!r}')
    if not isinstance(targets, torch.Tensor):
      raise TypeError(f'targets must be a torch.Tensor, got {type(targets).__name__}')
    if targets.dim() != 2:
      raise ValueError(f'targets must be 2D [B, C], got {tuple(targets.shape)}')
    if int(targets.size(1)) != self.num_classes:
      raise ValueError(
        'targets class dimension mismatch. '
        f'Got C={int(targets.size(1))}, expected num_classes={self.num_classes}.')

    # Step 2: compute per-class positive counts for the batch.
    # targets: [B, C]
    batch_pos = (targets.detach() > 0).to(torch.int64).sum(dim=0)  # [C]
    batch_pos_cpu = batch_pos.cpu()  # [C]

    # Step 3: update cumulative counts and first-positive steps.
    was_zero = self._cumulative_positives == 0  # [C]
    self._cumulative_positives += batch_pos_cpu  # [C]
    newly_positive = was_zero & (batch_pos_cpu > 0)  # [C]
    if bool(newly_positive.any().item()):
      for class_index in newly_positive.nonzero(as_tuple=False).flatten().tolist():
        if self._first_positive_step[class_index] is None:
          self._first_positive_step[class_index] = step

  def coverage_count(self) -> int:
    """Return how many classes have been seen positive at least once."""
    return int((self._cumulative_positives > 0).sum().item())

  def snapshot(self, *, step: int) -> LabelExposureSnapshot:
    """Return a JSON-friendly snapshot of the current exposure state."""
    # Step 1: validate step.
    if not isinstance(step, int) or step <= 0:
      raise ValueError(f'step must be a positive int, got {step!r}')

    # Step 2: compute coverage metrics.
    coverage_count = self.coverage_count()
    coverage_fraction = float(coverage_count / max(1, self.num_classes))

    # Step 3: pack buffers as Python-native values.
    return LabelExposureSnapshot(
      step=int(step),
      cumulative_positives=[int(v) for v in self._cumulative_positives.tolist()],
      first_positive_step=list(self._first_positive_step),
      coverage_count=int(coverage_count),
      coverage_fraction=float(coverage_fraction),
    )


class UniqueSampleLabelExposureTracker(LabelExposureTracker):
  """Label exposure tracker that counts each sample at most once.

  This is intended for offline probe optimization loops that repeatedly shuffle
  and cycle through a cached feature dataset. We track per-sample indices and
  only update exposure counts when an index is first seen.
  """

  def __init__(self, *, class_names: list[str], num_samples: int) -> None:
    """Initialize the unique-sample exposure tracker.

    Args:
      class_names: Classification target names (length `C`).
      num_samples: Total number of unique samples in the probe training split.
    """
    # Step 1: validate sample count.
    if not isinstance(num_samples, int) or num_samples <= 0:
      raise ValueError(f'num_samples must be a positive int, got {num_samples!r}')
    super().__init__(class_names=class_names)

    # Step 2: allocate a seen-mask for sample indices.
    self.num_samples = int(num_samples)
    self._seen = torch.zeros((self.num_samples,), dtype=torch.bool)  # [N]
    self._unique_seen = 0
    self.coverage_count_by_step: list[int] = []
    self.unique_samples_seen_by_step: list[int] = []

  @property
  def unique_samples_seen(self) -> int:
    """Return the number of unique indices observed so far."""
    return int(self._unique_seen)

  def update_unique(self, sample_indices: torch.Tensor, targets: torch.Tensor, *, step: int) -> int:
    """Update exposure using only newly-seen sample indices.

    Args:
      sample_indices: Sample ids as a tensor `[B]` on CPU (dtype int64/long).
      targets: Targets aligned with sample_indices as a tensor `[B, C]` (CPU or CUDA).
      step: Probe optimization step (must be a positive int).

    Returns:
      The number of newly-seen samples in this update.
    """
    # Step 1: validate inputs.
    if not isinstance(step, int) or step <= 0:
      raise ValueError(f'step must be a positive int, got {step!r}')
    if not isinstance(sample_indices, torch.Tensor):
      raise TypeError(
        f'sample_indices must be a torch.Tensor, got {type(sample_indices).__name__}')
    if sample_indices.dim() != 1:
      raise ValueError(f'sample_indices must be 1D [B], got {tuple(sample_indices.shape)}')
    if sample_indices.device.type != 'cpu':
      raise ValueError('sample_indices must be on CPU.')
    if sample_indices.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
      raise ValueError(f'sample_indices must be an integer tensor, got {sample_indices.dtype}')
    if not isinstance(targets, torch.Tensor):
      raise TypeError(f'targets must be a torch.Tensor, got {type(targets).__name__}')
    if targets.dim() != 2:
      raise ValueError(f'targets must be 2D [B, C], got {tuple(targets.shape)}')
    if int(targets.size(0)) != int(sample_indices.size(0)):
      raise ValueError(
        'targets/sample_indices batch mismatch. '
        f'Got B_targets={int(targets.size(0))} vs B_indices={int(sample_indices.size(0))}.')
    if int(targets.size(1)) != self.num_classes:
      raise ValueError(
        'targets class dimension mismatch. '
        f'Got C={int(targets.size(1))}, expected num_classes={self.num_classes}.')

    # Step 2: select newly-seen indices for this batch.
    # sample_indices: [B]
    sample_indices = sample_indices.to(torch.int64)  # [B]
    if int(sample_indices.min().item()) < 0 or int(sample_indices.max().item()) >= self.num_samples:
      raise ValueError(
        'sample_indices out of bounds. '
        f'Expected indices in [0, {self.num_samples - 1}], got min={int(sample_indices.min().item())}, '
        f'max={int(sample_indices.max().item())}.')
    new_mask = ~self._seen[sample_indices]  # [B]
    new_count = int(new_mask.sum().item())
    if new_count > 0:
      new_indices = sample_indices[new_mask]  # [B_new]
      self._seen[new_indices] = True  # [N]
      self._unique_seen += new_count
      new_targets = targets[new_mask]  # [B_new, C]
      super().update(new_targets, step=step)

    # Step 3: record per-step curves (coverage and unique sample count).
    self.coverage_count_by_step.append(self.coverage_count())
    self.unique_samples_seen_by_step.append(int(self._unique_seen))
    return new_count
