from __future__ import annotations

import numpy as np
import torch

from data import transforms


class PreprocessECG:  # called once when loading the data
  def __init__(self, *, mean_std, resample_ratio, channel_order, normalize=True):
    if normalize and mean_std is None:
      raise ValueError('mean_std must be provided when normalize=True')
    self.mean_std = mean_std
    self.resample_ratio = resample_ratio
    self.channel_order = channel_order
    self.normalize = normalize

  def __call__(self, x):
    transforms.interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      channel_size, num_channels = x.shape
      channel_size = int(self.resample_ratio * channel_size)
      x = transforms.resample(x, channel_size)
    if self.normalize:
      mean, std = self.mean_std
      transforms.normalize_(x, mean_std=(mean, std))
      x.clip(-5, 5, out=x)
    x = x[:, self.channel_order]
    return x


class TransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size=None, end_truncate=0):
    self.crop_size = crop_size
    self.end_truncate = end_truncate

  def __call__(self, x):
    if self.end_truncate > 0:
      x = x[:-self.end_truncate]  # truncate end before crop
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()  # [C, L]
    return x


class EvalTransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size: int | None = None, crop_stride: int | None = None):
    self.crop_size = crop_size
    self.crop_stride = crop_stride or crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
      x = np.swapaxes(x, 1, 2)  # channels first
    else:
      x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()  # [V, C, L] or [C, L]
    return x


def strided_crops(x, size: int, stride: int):  # x: (L, C)
  channel_size, num_channels = x.shape
  crop_starts = range(0, channel_size - size + 1, stride)
  num_crops = len(crop_starts)
  x_ = np.empty((num_crops, size, num_channels), dtype=x.dtype)
  for i, start in enumerate(crop_starts):
    x_[i] = x[start:start + size]
  return x_
