"""Wristband Gaussian loss (vendored from ml-tidbits).

This module vendors the minimal subset of `EmbedModels.py` needed for Plan 113:
  - `EpsForDtype`
  - `W2ToStandardNormalSq`
  - `S_LossComponents`
  - `C_WristbandGaussianLoss`

The math is kept identical to the source implementation; only style/typing/comments
are adapted to match this repository.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

__all__ = [
  "EpsForDtype",
  "W2ToStandardNormalSq",
  "S_LossComponents",
  "C_WristbandGaussianLoss",
]


def EpsForDtype(dtype: torch.dtype, large: bool = False) -> float:
  """Return a small epsilon suitable for `dtype`.

  Args:
    dtype: Torch dtype (must be floating point).
    large: If True, return $\\sqrt{\\epsilon}$ instead of $\\epsilon$.

  Returns:
    Float epsilon value.
  """
  eps = torch.finfo(dtype).eps
  return math.sqrt(eps) if large else float(eps)


def W2ToStandardNormalSq(x: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
  r"""Squared 2-Wasserstein distance to $\mathcal{N}(0, I)$ for a Gaussian fit.

  For samples $x$ with mean $\mu$ and covariance eigenvalues $\lambda_i$:
  $$
  W_2^2 = \|\mu\|^2 + \sum_i (\sqrt{\lambda_i} - 1)^2
  $$

  Shapes:
    - Input: `x[..., B, d]` where `B` is the number of samples, `d` feature dim.
    - Output:
      - `(...)` when `reduction="none"`,
      - scalar otherwise.

  Args:
    x: Sample tensor with shape `(..., B, d)`.
    reduction: One of `"none"`, `"mean"`, `"sum"`.
  """
  if x.ndim < 2:
    raise ValueError(f"Expected x.ndim>=2 with shape (..., B, d), got {tuple(x.shape)}")
  b = int(x.shape[-2])
  d = int(x.shape[-1])
  if b < 2:
    raise ValueError("Need B>=2 for covariance (denominator B-1).")

  work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
  xw = x.to(dtype=work_dtype)

  mu = xw.mean(dim=-2, keepdim=True)  # (..., 1, d)
  xc = xw - mu  # (..., B, d)
  mu2 = mu.squeeze(-2).square().sum(dim=-1)  # (...,)
  denom = float(b - 1)

  # Choose smaller PSD matrix to eigendecompose.
  if d <= b:
    m = (xc.transpose(-1, -2) @ xc) / denom  # (..., d, d)
    m_dim = d
  else:
    m = (xc @ xc.transpose(-1, -2)) / denom  # (..., B, B)
    m_dim = b

  m = 0.5 * (m + m.transpose(-1, -2))  # (..., m_dim, m_dim) symmetrize

  m_fp32 = m.to(dtype=torch.float32)  # (..., m_dim, m_dim)
  eig = torch.linalg.eigvalsh(m_fp32).to(dtype=m.dtype).clamp_min(0.0)  # (..., m_dim)
  sqrt_eig = torch.sqrt(eig + EpsForDtype(eig.dtype))  # (..., m_dim)
  bw2 = (sqrt_eig - 1.0).square().sum(dim=-1)  # (...,)

  # If we used Gram (m_dim=B<d), Sigma has (d-m_dim) extra zeros, each adds 1.
  if d > m_dim:
    bw2 = bw2 + (d - m_dim)

  loss = mu2 + bw2  # (...,)

  if reduction == "none":
    return loss
  if reduction == "mean":
    return loss.mean()
  if reduction == "sum":
    return loss.sum()
  raise ValueError("reduction must be one of {'none','mean','sum'}")


class S_LossComponents(NamedTuple):
  """Loss components returned by `C_WristbandGaussianLoss`."""

  total: torch.Tensor
  rep: torch.Tensor
  rad: torch.Tensor
  ang: torch.Tensor
  mom: torch.Tensor


class C_WristbandGaussianLoss:
  r"""Wristband repulsion loss encouraging $x \\sim \\mathcal{N}(0, I)$.

  This loss maps each sample to a wristband representation `(u, t)`:
    - $u = x / \|x\|$ (unit direction)
    - $t = \\mathrm{gammainc}(d/2, \|x\|^2/2)$ (CDF-transformed radius; uniform under the null)

  It computes a reflecting-kernel repulsion term in `(u, t)` space (O(N^2)),
  plus optional radial-uniformity, angular-uniformity, and moment penalties.

  Notes:
    - This class is intentionally NOT an `nn.Module` (no parameters).
    - Computation uses the dtype/device of the input tensor; internal work is promoted
      to float32 for fp16/bf16 inputs.
  """

  def __init__(
    self,
    *,
    beta: float = 8.0,
    alpha: float | None = None,
    angular: str = "chordal",
    reduction: str = "per_point",
    lambda_rad: float = 0.1,
    lambda_ang: float = 0.0,
    moment: str = "w2",
    lambda_mom: float = 1.0,
    calibration_shape: tuple[int, int] | None = None,
    calibration_reps: int = 1024,
    calibration_device: str | torch.device = "cpu",
    calibration_dtype: torch.dtype = torch.float32,
  ) -> None:
    if beta <= 0:
      raise ValueError("beta must be > 0")
    if angular not in ("chordal", "geodesic"):
      raise ValueError("angular must be 'chordal' or 'geodesic'")
    if reduction not in ("per_point", "global"):
      raise ValueError("reduction must be 'per_point' or 'global'")
    if moment not in ("mu_only", "kl_diag", "kl_full", "jeff_diag", "jeff_full", "w2"):
      raise ValueError("moment must be 'mu_only', 'kl_diag', 'kl_full', 'jeff_diag', 'jeff_full' or 'w2'")

    self.beta = float(beta)
    self.angular = str(angular)
    self.reduction = str(reduction)

    if alpha is None:
      if angular == "chordal":
        alpha = math.sqrt(1.0 / 12.0)
      else:
        alpha = math.sqrt(2.0 / (3.0 * math.pi * math.pi))
    self.alpha = float(alpha)
    self.beta_alpha2 = self.beta * (self.alpha * self.alpha)

    self.lambda_rad = float(lambda_rad)
    self.lambda_ang = float(lambda_ang)
    self.moment = str(moment)
    self.lambda_mom = float(lambda_mom)
    self.eps = 1e-12
    self.clamp_cos = 1e-6

    # Calibration stats (identity transform when not calibrated).
    self.mean_rep = self.mean_rad = self.mean_ang = self.mean_mom = 0.0
    self.std_rep = self.std_rad = self.std_ang = self.std_mom = 1.0
    self.std_total = 1.0

    if calibration_shape is not None:
      self._calibrate(calibration_shape, calibration_reps, calibration_device, calibration_dtype)

  def _calibrate(self, shape: tuple[int, int], reps: int, device: object, dtype: torch.dtype) -> None:
    """Estimate null distribution mean/std for each component via Monte-Carlo."""
    n, d = shape
    if n < 2 or d < 1 or reps < 2:
      return

    sum_rep = sum_rad = sum_ang = sum_mom = 0.0
    sum2_rep = sum2_rad = sum2_ang = sum2_mom = 0.0
    all_rep: list[float] = []
    all_rad: list[float] = []
    all_ang: list[float] = []
    all_mom: list[float] = []

    with torch.no_grad():
      for _ in range(int(reps)):
        x_gauss = torch.randn(int(n), int(d), device=device, dtype=dtype)  # [N, D]
        comp = self._Compute(x_gauss)  # scalars

        f_rep, f_rad, f_ang, f_mom = float(comp.rep), float(comp.rad), float(comp.ang), float(comp.mom)
        sum_rep += f_rep
        sum2_rep += f_rep * f_rep
        all_rep.append(f_rep)
        sum_rad += f_rad
        sum2_rad += f_rad * f_rad
        all_rad.append(f_rad)
        sum_ang += f_ang
        sum2_ang += f_ang * f_ang
        all_ang.append(f_ang)
        sum_mom += f_mom
        sum2_mom += f_mom * f_mom
        all_mom.append(f_mom)

    reps_f = float(reps)
    bessel = reps_f / (reps_f - 1.0)

    self.mean_rep = sum_rep / reps_f
    self.mean_rad = sum_rad / reps_f
    self.mean_ang = sum_ang / reps_f
    self.mean_mom = sum_mom / reps_f

    var_rep = (sum2_rep / reps_f - self.mean_rep * self.mean_rep) * bessel
    var_rad = (sum2_rad / reps_f - self.mean_rad * self.mean_rad) * bessel
    var_ang = (sum2_ang / reps_f - self.mean_ang * self.mean_ang) * bessel
    var_mom = (sum2_mom / reps_f - self.mean_mom * self.mean_mom) * bessel

    eps_cal = float(EpsForDtype(dtype, True))
    self.std_rep = math.sqrt(max(var_rep, eps_cal))
    self.std_rad = math.sqrt(max(var_rad, eps_cal))
    self.std_ang = math.sqrt(max(var_ang, eps_cal))
    self.std_mom = math.sqrt(max(var_mom, eps_cal))

    # Std of the weighted total (for final normalization).
    sum_total = sum2_total = 0.0
    for i in range(int(reps)):
      t_rep = (all_rep[i] - self.mean_rep) / self.std_rep
      t_rad = self.lambda_rad * (all_rad[i] - self.mean_rad) / self.std_rad
      t_ang = self.lambda_ang * (all_ang[i] - self.mean_ang) / self.std_ang
      t_mom = self.lambda_mom * (all_mom[i] - self.mean_mom) / self.std_mom
      total = t_rep + t_rad + t_ang + t_mom
      sum_total += total
      sum2_total += total * total

    mean_total = sum_total / reps_f
    var_total = (sum2_total / reps_f - mean_total * mean_total) * bessel
    self.std_total = math.sqrt(max(var_total, eps_cal))

  def _Compute(self, x: torch.Tensor) -> S_LossComponents:
    """Compute raw (uncalibrated) components for `x[..., N, D]`."""
    # x: (..., N, D) where N is #samples, D feature dim.
    if x.ndim < 2:
      raise ValueError(f"Expected x.ndim>=2 with shape (..., N, D), got {tuple(x.shape)}")

    n = int(x.shape[-2])
    d = int(x.shape[-1])
    batch_shape = x.shape[:-2]

    if n < 2 or d < 1:
      z = x.sum(dim=(-2, -1)) * 0.0  # (...) scalar-like
      return S_LossComponents(z, z, z, z, z)

    wdtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    xw = x.to(wdtype)  # (..., N, D)
    n_f, d_f = float(n), float(d)
    beta, eps = self.beta, self.eps

    mu = xw.mean(dim=-2)  # (..., D)
    xc = xw - mu[..., None, :]  # (..., N, D)

    # ---- moment penalty ----
    mom_pen = xw.new_zeros(batch_shape)  # (...)
    if self.lambda_mom != 0.0:
      if self.moment == "w2":
        mom_pen = W2ToStandardNormalSq(xw, reduction="none") / d_f  # (...)
      elif self.moment == "jeff_diag":
        var = xc.square().sum(dim=-2) / (n_f - 1.0)  # (..., D)
        v = var + eps  # (..., D)
        inv_v = v.reciprocal()  # (..., D)
        mu2 = mu.square()  # (..., D)
        mom_pen = 0.25 * (v + inv_v + mu2 + mu2 * inv_v - 2.0).mean(dim=-1)  # (...)
      elif self.moment == "jeff_full":
        eps_cov = max(eps, 1e-6) if wdtype == torch.float32 else max(eps, float(torch.finfo(wdtype).eps))
        cov = (xc.transpose(-1, -2) @ xc) / (n_f - 1.0)  # (..., D, D)
        eye = torch.eye(d, device=xw.device, dtype=wdtype)  # [D, D]
        cov = cov + eps_cov * eye  # (..., D, D)
        chol, _info = torch.linalg.cholesky_ex(cov)  # (..., D, D)
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
        inv_cov = torch.cholesky_solve(eye, chol)  # (..., D, D)
        tr_inv = inv_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
        mu_col = mu[..., :, None]  # (..., D, 1)
        sol_mu = torch.cholesky_solve(mu_col, chol)  # (..., D, 1)
        mu_inv_mu = (mu_col * sol_mu).sum(dim=(-2, -1))  # (...)
        mu2_sum = mu.square().sum(dim=-1)  # (...)
        mom_pen = 0.25 * (tr + tr_inv + mu2_sum + mu_inv_mu - 2.0 * d_f) / d_f  # (...)
      elif self.moment == "mu_only":
        mom_pen = mu.square().mean(dim=-1)  # (...)
      elif self.moment == "kl_diag":
        var = xc.square().sum(dim=-2) / (n_f - 1.0)  # (..., D)
        mom_pen = 0.5 * (var + mu.square() - 1.0 - torch.log(var + eps)).mean(dim=-1)  # (...)
      else:  # "kl_full"
        eye = torch.eye(d, device=xw.device, dtype=wdtype)  # [D, D]
        cov = (xc.transpose(-1, -2) @ xc) / n_f + eps * eye  # (..., D, D)
        chol, _info = torch.linalg.cholesky_ex(cov)  # (..., D, D)
        diag = chol.diagonal(dim1=-2, dim2=-1)  # (..., D)
        logdet = 2.0 * torch.log(diag).sum(dim=-1)  # (...)
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
        mu2_sum = mu.square().sum(dim=-1)  # (...)
        mom_pen = 0.5 * (tr + mu2_sum - d_f - logdet) / d_f  # (...)

    # ---- wristband map (u, t) ----
    s = xw.square().sum(dim=-1).clamp_min(eps)  # (..., N)
    u = xw * torch.rsqrt(s)[..., :, None]  # (..., N, D)
    a_df = s.new_tensor(0.5 * d_f)  # []
    t = torch.special.gammainc(a_df, 0.5 * s).clamp(eps, 1.0 - eps)  # (..., N)

    # ---- radial 1D W2^2 on t vs Unif(0,1) ----
    rad_loss = xw.new_zeros(batch_shape)  # (...)
    if self.lambda_rad != 0.0:
      t_sorted, _ = torch.sort(t, dim=-1)  # (..., N)
      q = (torch.arange(n, device=xw.device, dtype=wdtype) + 0.5) / n_f  # [N]
      rad_loss = 12.0 * (t_sorted - q).square().mean(dim=-1)  # (...)

    # ---- angular kernel exponent ----
    g = (u @ u.transpose(-1, -2)).clamp(-1.0, 1.0)  # (..., N, N)

    if self.angular == "chordal":
      e_ang = (2.0 * self.beta_alpha2) * (g - 1.0)  # (..., N, N)
    else:
      theta = torch.acos(g.clamp(-1.0 + self.clamp_cos, 1.0 - self.clamp_cos))  # (..., N, N)
      ang2 = theta.square()  # (..., N, N)
      ang2 = ang2 - torch.diag_embed(ang2.diagonal(dim1=-2, dim2=-1))  # (..., N, N) zero diag
      e_ang = -self.beta_alpha2 * ang2  # (..., N, N)

    # ---- optional angular-only uniformity ----
    ang_loss = xw.new_zeros(batch_shape)  # (...)
    if self.lambda_ang != 0.0:
      if self.reduction == "per_point":
        row_sum = torch.exp(e_ang).sum(dim=-1) - 1.0  # (..., N)
        mean_k = row_sum / (n_f - 1.0)  # (..., N)
        ang_loss = torch.log(mean_k + eps).mean(dim=-1) / beta  # (...)
      else:
        total = torch.exp(e_ang).sum(dim=(-2, -1)) - n_f  # (...)
        mean_k = total / (n_f * (n_f - 1.0))  # (...)
        ang_loss = torch.log(mean_k + eps) / beta  # (...)

    # ---- 3-image reflected kernel for joint (u, t) repulsion ----
    tc = t[..., :, None]  # (..., N, 1)
    tr = t[..., None, :]  # (..., 1, N)
    diff0 = tc - tr  # (..., N, N)
    diff1 = tc + tr  # (..., N, N)
    diff2 = diff1 - 2.0  # (..., N, N)

    if self.reduction == "per_point":
      row_sum = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-beta)).sum(dim=-1)  # (..., N)
      row_sum += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-beta)).sum(dim=-1)  # (..., N)
      row_sum += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-beta)).sum(dim=-1)  # (..., N)
      row_sum -= 1.0  # remove only the real self term (=1), keep diagonal mirror terms
      mean_k = row_sum / (3.0 * n_f - 1.0)  # (..., N)
      rep_loss = torch.log(mean_k + eps).mean(dim=-1) / beta  # (...)
    else:
      total = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-beta)).sum(dim=(-2, -1))  # (...)
      total += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-beta)).sum(dim=(-2, -1))  # (...)
      total += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-beta)).sum(dim=(-2, -1))  # (...)
      total -= n_f  # remove n real-self terms (=1)
      mean_k = total / (3.0 * n_f * n_f - n_f)  # (...)
      rep_loss = torch.log(mean_k + eps) / beta  # (...)

    # Return components; `total` is a dummy placeholder at this stage (as in source).
    return S_LossComponents(rep_loss, rep_loss, rad_loss, ang_loss, mom_pen)

  def __call__(self, x: torch.Tensor) -> S_LossComponents:
    """Compute calibrated wristband loss for `x[..., N, D]`."""
    comp = self._Compute(x)

    norm_rep = (comp.rep - self.mean_rep) / self.std_rep
    norm_rad = (comp.rad - self.mean_rad) / self.std_rad
    norm_ang = (comp.ang - self.mean_ang) / self.std_ang
    norm_mom = (comp.mom - self.mean_mom) / self.std_mom

    total = (norm_rep + self.lambda_rad * norm_rad + self.lambda_ang * norm_ang + self.lambda_mom * norm_mom) / self.std_total

    return S_LossComponents(total.mean(), norm_rep.mean(), norm_rad.mean(), norm_ang.mean(), norm_mom.mean())
