"""Batch-level label presence and gap statistics.

This module implements a lightweight tracker for minibatch composition diagnostics
for multi-label targets.

Given targets $Y_t \\in \\{0,1\\}^{B\\times C}$ at step $t$, define label presence:
$$I_t[c] = \\mathbb{1}\\left[\\sum_i Y_t[i,c] > 0\\right].$$

We track, for each class $c$:
- the last step where $I_t[c]=1$,
- gap statistics (mean/max) between consecutive steps with $I_t[c]=1$,
and a few cheap per-step scalars (coverage and rare-label coverage).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
  'BatchLabelStatsSnapshot',
  'BatchLabelStatsTracker',
]


@dataclass(frozen=True)
class BatchLabelStatsSnapshot:
  """JSON-serializable snapshot of per-step batch label statistics."""

  step: int
  coverage_count: int
  rare_label_coverage_count: int
  rare_positive_instances: int


class BatchLabelStatsTracker:
  """Track batch-level label presence/gaps for a multi-label stream.

  This tracker is intended to be updated once per training step with targets
  $Y_t \\in \\{0,1\\}^{B\\times C}$.
  """

  def __init__(
    self,
    *,
    class_names: list[str],
    train_positive_counts: list[int],
    rare_class_indices: list[int],
    rare_bottom_k: int,
  ) -> None:
    """Initialize the tracker for a fixed label space and a fixed rare set.

    Args:
      class_names: Classification target names (length `C`).
      train_positive_counts: Per-class positive counts on the *train* split (length `C`).
      rare_class_indices: Indices of the rare-class subset (must be unique, within `[0, C)`).
      rare_bottom_k: The K used to define the rare set (recorded for reporting).
    """
    # Step 1: validate metadata.
    if not isinstance(class_names, list) or not class_names:
      raise ValueError('class_names must be a non-empty list of strings.')
    if len(set(class_names)) != len(class_names):
      raise ValueError('class_names must be unique.')
    for name in class_names:
      if not isinstance(name, str) or not name.strip():
        raise ValueError(f'class_names must contain non-empty strings, got {name!r}')

    if not isinstance(train_positive_counts, list) or not train_positive_counts:
      raise ValueError('train_positive_counts must be a non-empty list of ints.')
    for value in train_positive_counts:
      if not isinstance(value, int):
        raise ValueError(
          'train_positive_counts must contain ints, '
          f'got {type(value).__name__}')
      if value < 0:
        raise ValueError(f'train_positive_counts must be non-negative, got {value}')

    num_classes = len(class_names)
    if len(train_positive_counts) != num_classes:
      raise ValueError(
        'train_positive_counts length mismatch. '
        f'Got {len(train_positive_counts)} vs C={num_classes}.')

    if not isinstance(rare_bottom_k, int) or rare_bottom_k <= 0:
      raise ValueError(f'rare_bottom_k must be a positive int, got {rare_bottom_k!r}')

    if not isinstance(rare_class_indices, list) or not rare_class_indices:
      raise ValueError('rare_class_indices must be a non-empty list of ints.')
    if len(set(rare_class_indices)) != len(rare_class_indices):
      raise ValueError(f'rare_class_indices must be unique, got {rare_class_indices}')
    for index in rare_class_indices:
      if not isinstance(index, int):
        raise ValueError(
          'rare_class_indices must contain ints, '
          f'got {type(index).__name__}')
      if index < 0 or index >= num_classes:
        raise ValueError(
          f'rare_class_indices must be in [0, {num_classes - 1}], got {index}')

    # Step 2: store metadata.
    self.class_names = list(class_names)
    self.num_classes = num_classes
    self.train_positive_counts = list(train_positive_counts)
    self.rare_class_indices = list(rare_class_indices)
    self.rare_class_names = [self.class_names[index] for index in self.rare_class_indices]
    self.rare_bottom_k = int(rare_bottom_k)

    # Step 3: allocate tracking buffers on CPU.
    self._last_seen_step: list[int | None] = [None] * self.num_classes  # len=C
    self._gap_sum = torch.zeros((self.num_classes,), dtype=torch.int64)  # [C]
    self._gap_count = torch.zeros((self.num_classes,), dtype=torch.int64)  # [C]
    self._gap_max = torch.full((self.num_classes,), -1, dtype=torch.int64)  # [C]
    self._last_update_step: int | None = None

  def update(self, targets: torch.Tensor, *, step: int) -> BatchLabelStatsSnapshot:
    """Update statistics from a single labeled batch.

    Args:
      targets: Multi-label targets as a tensor `[B, C]` (any device/dtype).
      step: Global step index (must be a strictly increasing positive int).

    Returns:
      BatchLabelStatsSnapshot for this step.
    """
    # Step 1: validate inputs.
    if not isinstance(step, int) or step <= 0:
      raise ValueError(f'step must be a positive int, got {step!r}')
    if self._last_update_step is not None and step <= self._last_update_step:
      raise ValueError(
        'step must be strictly increasing for BatchLabelStatsTracker. '
        f'Got step={step} after last_step={self._last_update_step}. '
        'Fix: call update() once per training step.')
    if not isinstance(targets, torch.Tensor):
      raise TypeError(f'targets must be a torch.Tensor, got {type(targets).__name__}')
    if targets.dim() != 2:
      raise ValueError(f'targets must be 2D [B, C], got {tuple(targets.shape)}')
    if int(targets.size(1)) != self.num_classes:
      raise ValueError(
        'targets class dimension mismatch. '
        f'Got C={int(targets.size(1))}, expected num_classes={self.num_classes}.')
    self._last_update_step = int(step)

    # Step 2: compute per-class presence and positive-instance counts.
    # targets: [B, C]
    targets_detached = targets.detach()
    presence = (targets_detached > 0).any(dim=0).to(torch.bool).cpu()  # [C]
    positive_instances = (targets_detached > 0).to(torch.int64).sum(dim=0).cpu()  # [C]

    # Step 3: update per-class gap statistics for present labels.
    present_indices = presence.nonzero(as_tuple=False).flatten().tolist()
    for class_index in present_indices:
      last_seen = self._last_seen_step[class_index]
      if last_seen is not None:
        gap = int(step - last_seen)
        if gap <= 0:
          raise ValueError(
            'Non-positive gap encountered in BatchLabelStatsTracker. '
            f'class={class_index} step={step} last_seen={last_seen}')
        self._gap_sum[class_index] += gap  # [C]
        self._gap_count[class_index] += 1  # [C]
        if gap > int(self._gap_max[class_index].item()):
          self._gap_max[class_index] = gap  # [C]
      self._last_seen_step[class_index] = int(step)

    # Step 4: compute scalar diagnostics.
    coverage_count = int(presence.sum().item())
    rare_coverage_count = int(presence[self.rare_class_indices].sum().item())
    rare_positive_instances = int(positive_instances[self.rare_class_indices].sum().item())

    return BatchLabelStatsSnapshot(
      step=int(step),
      coverage_count=int(coverage_count),
      rare_label_coverage_count=int(rare_coverage_count),
      rare_positive_instances=int(rare_positive_instances),
    )

  def gap_summary(self) -> dict[str, object]:
    """Return a JSON-friendly summary of per-class gap statistics."""
    # Step 1: convert buffers to Python-native values.
    gap_sum = self._gap_sum.tolist()  # len=C
    gap_count = self._gap_count.tolist()  # len=C
    gap_max = self._gap_max.tolist()  # len=C

    # Step 2: compute mean/max gaps (null when undefined).
    mean_gap: list[float | None] = []
    max_gap: list[int | None] = []
    for total, count, max_value in zip(gap_sum, gap_count, gap_max):
      if count <= 0:
        mean_gap.append(None)
        max_gap.append(None)
        continue
      mean_gap.append(float(total / count))
      max_gap.append(int(max_value))

    return {
      'class_names': list(self.class_names),
      'train_positive_counts': list(self.train_positive_counts),
      'rare_bottom_k': int(self.rare_bottom_k),
      'rare_class_indices': list(self.rare_class_indices),
      'rare_class_names': list(self.rare_class_names),
      'gap_count': [int(v) for v in gap_count],
      'mean_gap': mean_gap,
      'max_gap': max_gap,
      'last_seen_step': list(self._last_seen_step),
    }

