# Copyright 2023 solo-learn development team.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Source (upstream):
# https://raw.githubusercontent.com/YilunKuang/rectified-lp-jepa/refs/heads/main/solo/losses/rectified_lpjepa.py
#
# NOTE:
# - This module is vendored from the official Rectified LpJEPA implementation to ensure
#   behavior matches the paper/code exactly.
# - The upstream file contains additional wrappers/utilities that this repo does not use
#   (two-view wrappers, invariance loss, distributed gather helpers, W&B metrics).
#   Those are intentionally removed here to keep dependencies minimal.

from __future__ import annotations

import math

import torch
from torch.distributions.laplace import Laplace
from torch.nn import functional as F

__all__ = [
    "apply_feature_activation",
    "determine_sigma_for_lp_dist",
    "diag_update_for_features",
    "l0_sparsity_metric",
    "l1_sparsity_metric",
    "reprelu",
    "sample_lp_distribution",
    "sample_unit_sphere_projections",
    "sliced_wasserstein_distance_batched",
    "sliced_wasserstein_distance_for_one_view",
    "swd_mean_over_views",
]


def reprelu(values: torch.Tensor) -> torch.Tensor:
    """RepReLU activation from the Rectified LpJEPA paper.

    Forward pass equals ReLU, while backward pass equals GeLU:

    $$
    \\operatorname{RepReLU}(z):=\\operatorname{ReLU}(z)\\cdot\\mathrm{detach}()
      + \\operatorname{GeLU}(z) - \\operatorname{GeLU}(z)\\cdot\\mathrm{detach}()
    $$

    Args:
        values: Tensor of any shape.

    Returns:
        Tensor of the same shape and dtype as `values`.
    """
    relu_detached = F.relu(values).detach()  # [*]
    gelu = F.gelu(values)  # [*]
    return relu_detached + gelu - gelu.detach()  # [*]


def l0_sparsity_metric(features: torch.Tensor) -> torch.Tensor:
    """Compute mean fraction of non-zero entries per sample.

    Args:
        features: [B, D] feature matrix.

    Returns:
        Scalar float32 tensor in [0, 1] (fraction of non-zero entries).
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    D = features.size(1)
    nonzero_per_sample = (features != 0).sum(dim=1).to(dtype=torch.float32) / float(D)  # [B]
    return nonzero_per_sample.mean()  # []


def l1_sparsity_metric(features: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute the L1 sparsity metric used by the official Rectified LpJEPA code.

    Metric:
    $$
    m_{\\ell_1}(z) = \\frac{1}{D} \\left(\\frac{\\|z\\|_1}{\\|z\\|_2 + \\epsilon}\\right)^2
    $$
    averaged over samples.

    Args:
        features: [B, D] feature matrix.
        eps: Numerical stability epsilon.

    Returns:
        Scalar float32 tensor.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    D = features.size(1)
    features_f32 = features.to(dtype=torch.float32)  # [B, D]
    l1_norms = torch.linalg.norm(features_f32, ord=1, dim=1)  # [B]
    l2_norms = torch.linalg.norm(features_f32, ord=2, dim=1)  # [B]
    per_sample = (1.0 / float(D)) * (l1_norms / (l2_norms + eps)).square()  # [B]
    return per_sample.mean()  # []


def sample_unit_sphere_projections(
    *,
    num_projections: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample random projection vectors and normalize to unit $\\ell_2$ norm.

    Args:
        num_projections: Number of projection vectors $N$.
        dim: Feature dimension $D$.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Projection matrix `[N, D]` with unit-normalized rows.
    """
    if num_projections < 1:
        raise ValueError(f"num_projections must be >= 1, got {num_projections}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    projections = torch.randn((num_projections, dim), device=device, dtype=dtype)  # [N, D]
    denom = projections.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)  # [N, 1]
    return projections / denom  # [N, D]


def apply_feature_activation(features: torch.Tensor, *, feature_activation: str) -> torch.Tensor:
    """Apply the rectifying activation configured for RDMReg.

    Args:
        features: Tensor of any shape.
        feature_activation: "none", "relu" or "reprelu".

    Returns:
        Tensor of the same shape and dtype as `features`.
    """
    if feature_activation == "none":
        return features  # [*]
    if feature_activation == "relu":
        return F.relu(features)  # [*]
    if feature_activation == "reprelu":
        return reprelu(features)  # [*]
    raise ValueError(
        'feature_activation must be "none", "relu" or "reprelu", '
        f"got {feature_activation!r}"
    )


def swd_mean_over_views(
    *,
    features_by_view: torch.Tensor,
    mask_by_view: torch.Tensor | None,
    projection_vectors: torch.Tensor,
    target_dist: str,
    mean_shift_value: float,
    lp_norm_parameter: float,
    chosen_sigma: float,
    feature_activation: str,
) -> torch.Tensor:
    """Compute mean SWD over a set of "views", ignoring empty masked views.

    This mirrors the existing SIGReg usage pattern: a stability loss is applied to a
    tensor `[V, B, D]` (or reshaped into that form), averaged over the view dimension.

    Args:
        features_by_view: `[V, B, D]` tensor.
        mask_by_view: Optional `[V, B]` boolean mask selecting valid samples per view.
        projection_vectors: `[N, D]` projection matrix.
        target_dist: `"rectified_lp_distribution"` or `"lp_distribution"`.
        mean_shift_value: Target mean shift $\\mu$.
        lp_norm_parameter: Target generalized Gaussian $p$.
        chosen_sigma: Target scale $\\sigma$.
        feature_activation: `"none"`, `"relu"` or `"reprelu"`.

    Returns:
        Scalar SWD loss averaged over valid views.
    """
    if features_by_view.dim() != 3:
        raise ValueError(f"features_by_view must be [V, B, D], got {tuple(features_by_view.shape)}")
    if mask_by_view is not None:
        if mask_by_view.dtype is not torch.bool:
            raise ValueError(f"mask_by_view must be bool, got {mask_by_view.dtype}")
        if mask_by_view.shape != features_by_view.shape[:2]:
            raise ValueError(
                "mask_by_view must match features_by_view first two dims, "
                f"features_by_view.shape={tuple(features_by_view.shape)}, mask_by_view.shape={tuple(mask_by_view.shape)}"
            )

    if projection_vectors.dim() != 2:
        raise ValueError(f"projection_vectors must be [N, D], got {tuple(projection_vectors.shape)}")
    _, _, D = features_by_view.shape
    if projection_vectors.size(1) != D:
        raise ValueError(
            "projection_vectors second dim must match features dim, "
            f"features_by_view.shape={tuple(features_by_view.shape)}, projection_vectors.shape={tuple(projection_vectors.shape)}"
        )

    projection_vectors_f32 = projection_vectors.to(dtype=torch.float32)  # [N, D]

    # Step 1: apply activation in a vectorized manner over all views.
    activated_by_view = apply_feature_activation(
        features_by_view,
        feature_activation=feature_activation,
    )  # [V, B, D]

    # Step 2: fast path (mask=None): compute per-view SWD in a single batched call.
    if mask_by_view is None:
        activated_by_view_f32 = activated_by_view.to(dtype=torch.float32)  # [V, B, D]
        losses_by_view = sliced_wasserstein_distance_batched(
            activated_by_view_f32,
            projection_vectors_f32,
            target_dist=target_dist,
            mean_shift_value=float(mean_shift_value),
            lp_norm_parameter=float(lp_norm_parameter),
            chosen_sigma=float(chosen_sigma),
        )  # [V]
        return losses_by_view.mean()  # []

    # Step 3: masked path (correctness-first): keep per-view indexing.
    losses_view: list[torch.Tensor] = []
    V = features_by_view.size(0)
    for view_idx in range(V):
        feats = activated_by_view[view_idx]  # [B, D]
        mask_v = mask_by_view[view_idx]  # [B]
        if not mask_v.any():
            continue
        feats = feats[mask_v]  # [Bv, D]
        feats_f32 = feats.to(dtype=torch.float32)  # [Bv, D]
        losses_view.append(
            sliced_wasserstein_distance_for_one_view(
                feats_f32,
                projection_vectors_f32,
                target_dist=target_dist,
                mean_shift_value=float(mean_shift_value),
                lp_norm_parameter=float(lp_norm_parameter),
                chosen_sigma=float(chosen_sigma),
            )
        )  # []
    if not losses_view:
        raise ValueError("mask_by_view results in zero valid views for SWD")
    return torch.stack(losses_view, dim=0).mean()  # []


def diag_update_for_features(
    *,
    diag_key_prefix: str,
    features_2d: torch.Tensor,
    feature_activation: str,
) -> dict[str, torch.Tensor]:
    """Build sparsity diagnostics (Appendix B.4) for a `[B, D]` feature matrix.

    The metrics are computed on activated (rectified) features to match the
    regularized distribution.
    """
    if features_2d.dim() != 2:
        raise ValueError(f"features_2d must be [B, D], got {tuple(features_2d.shape)}")
    activated = apply_feature_activation(features_2d, feature_activation=feature_activation)  # [B, D]
    return {
        f"{diag_key_prefix}_l0_frac_nonzero": l0_sparsity_metric(activated.detach()),
        f"{diag_key_prefix}_l1_metric": l1_sparsity_metric(activated.detach()),
    }


def _sample_product_laplace(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    *,
    loc: float = 0.0,
    scale: float = 1 / math.sqrt(2),
) -> torch.Tensor:
    """Sample from the product Laplace distribution directly on the target device."""
    loc_t = torch.tensor(loc, device=device, dtype=dtype)  # []
    scale_t = torch.tensor(scale, device=device, dtype=dtype)  # []
    laplace_dist = Laplace(loc=loc_t, scale=scale_t)
    return laplace_dist.sample(shape)  # [*shape]


def sample_lp_distribution(
    *,
    shape: tuple[int, ...],
    p: float,
    loc: float = 0.0,
    scale: float = 1.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample from the Generalized Gaussian distribution $GN_p(\\mu, \\sigma)$."""
    if p == 1.0:
        return _sample_product_laplace(shape, device, dtype, loc=loc, scale=scale)
    if p == 2.0:
        return loc + scale * torch.randn(shape, device=device, dtype=dtype)  # [*shape]

    sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5)  # [*shape]
    sign = 2 * sign - 1  # [*shape]
    concentration = torch.tensor(1.0 / p, device=device, dtype=dtype)  # []
    rate = torch.tensor(1.0, device=device, dtype=dtype)  # []
    gamma = torch.distributions.Gamma(concentration=concentration, rate=rate)
    g = gamma.sample(shape)  # [*shape]
    x = sign * (p * g).pow(1.0 / p)  # [*shape]
    return loc + scale * x  # [*shape]


def determine_sigma_for_lp_dist(p: float) -> float:
    """Return $\\sigma$ so that $GN_p(0, \\sigma)$ has unit variance (upstream closed form)."""
    sigma = (math.gamma(1 / p) ** (1 / 2)) / ((p ** (1 / p)) * (math.gamma(3 / p) ** (1 / 2)))
    return sigma


def sliced_wasserstein_distance_batched(
    features_by_view: torch.Tensor,
    projection_vectors: torch.Tensor,
    *,
    target_dist: str,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float | None = None,
) -> torch.Tensor:
    """Compute per-view sliced Wasserstein-2 distance for a `[V, B, D]` tensor.

    This is equivalent to calling `sliced_wasserstein_distance_for_one_view` for each
    view and stacking the results, but it avoids the Python loop.

    For each view $v$:
    $$
    P_v = X_v C^\\top \\in \\mathbb{R}^{B\\times N},\\quad
    Q_v = Y_v C^\\top \\in \\mathbb{R}^{B\\times N}
    $$
    where $X_v$ is the feature matrix and $Y_v$ are i.i.d. samples from the target
    distribution. The loss is
    $$
    \\mathrm{SWD}(v) = \\frac{1}{N}\\sum_{n=1}^{N} \\frac{1}{B}\\left\\|P_v^{\\uparrow}[:,n]
      - Q_v^{\\uparrow}[:,n]\\right\\|_2^2,
    $$
    where $\\uparrow$ denotes sorting along the sample dimension.

    Args:
        features_by_view: `[V, B, D]` feature tensor (ideally float32).
        projection_vectors: `[N, D]` projection matrix (ideally float32).
        target_dist: `"rectified_lp_distribution"` or `"lp_distribution"`.
        mean_shift_value: Target mean shift $\\mu$.
        lp_norm_parameter: Target generalized Gaussian $p$.
        chosen_sigma: Target scale $\\sigma$ (required; no silent defaults).

    Returns:
        Per-view SWD losses `[V]` (float32).
    """
    if features_by_view.dim() != 3:
        raise ValueError(f"features_by_view must be [V, B, D], got {tuple(features_by_view.shape)}")
    if projection_vectors.dim() != 2:
        raise ValueError(f"projection_vectors must be [N, D], got {tuple(projection_vectors.shape)}")
    if chosen_sigma is None:
        raise ValueError("chosen_sigma must be provided (no silent defaults).")

    V, B, D = features_by_view.shape
    if projection_vectors.size(1) != D:
        raise ValueError(
            "projection_vectors second dim must match features dim, "
            f"features_by_view.shape={tuple(features_by_view.shape)}, projection_vectors.shape={tuple(projection_vectors.shape)}"
        )

    # Step 1: project features onto random directions.
    projected_features = torch.matmul(features_by_view, projection_vectors.T)  # [V, B, N]

    # Step 2: sample targets (i.i.d. per view) and project them onto the same directions.
    if target_dist == "rectified_lp_distribution":
        target_samples = torch.relu(
            sample_lp_distribution(
                shape=(V, B, D),
                p=lp_norm_parameter,
                loc=mean_shift_value,
                scale=float(chosen_sigma),
                device=features_by_view.device,
                dtype=features_by_view.dtype,
            )
        )  # [V, B, D]
    elif target_dist == "lp_distribution":
        target_samples = sample_lp_distribution(
            shape=(V, B, D),
            p=lp_norm_parameter,
            loc=mean_shift_value,
            scale=float(chosen_sigma),
            device=features_by_view.device,
            dtype=features_by_view.dtype,
        )  # [V, B, D]
    else:
        raise ValueError(f"Unsupported target_dist: {target_dist}")

    projected_targets = torch.matmul(target_samples, projection_vectors.T)  # [V, B, N]

    # Step 3: sort projections along the sample dimension (per view and per projection).
    sorted_features, _ = torch.sort(projected_features, dim=1)  # [V, B, N]
    sorted_targets, _ = torch.sort(projected_targets, dim=1)  # [V, B, N]

    # Step 4: compute mean squared distance (W2) along the sample axis and average projections.
    wasserstein_1d = torch.mean((sorted_features - sorted_targets) ** 2, dim=1)  # [V, N]
    return torch.mean(wasserstein_1d, dim=1)  # [V]


def sliced_wasserstein_distance_for_one_view(
    features: torch.Tensor,
    projection_vectors: torch.Tensor,
    target_dist: str,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float | None = None,
) -> torch.Tensor:
    """Compute sliced Wasserstein-2 distance between projected features and target samples."""
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    if projection_vectors.dim() != 2:
        raise ValueError(f"projection_vectors must be [N, D], got {tuple(projection_vectors.shape)}")
    if features.size(1) != projection_vectors.size(1):
        raise ValueError(
            "projection_vectors second dim must match features dim, "
            f"features.shape={tuple(features.shape)}, projection_vectors.shape={tuple(projection_vectors.shape)}"
        )
    if chosen_sigma is None:
        raise ValueError("chosen_sigma must be provided (no silent defaults).")

    B, D = features.shape
    projected_features = torch.matmul(features, projection_vectors.T)  # [B, N]

    if target_dist == "rectified_lp_distribution":
        target_samples = torch.relu(
            sample_lp_distribution(
                shape=(B, D),
                p=lp_norm_parameter,
                loc=mean_shift_value,
                scale=chosen_sigma,
                device=features.device,
                dtype=features.dtype,
            )
        )  # [B, D]
    elif target_dist == "lp_distribution":
        target_samples = sample_lp_distribution(
            shape=(B, D),
            p=lp_norm_parameter,
            loc=mean_shift_value,
            scale=chosen_sigma,
            device=features.device,
            dtype=features.dtype,
        )  # [B, D]
    else:
        raise ValueError(f"Unsupported target_dist: {target_dist}")

    projected_targets = torch.matmul(target_samples, projection_vectors.T)  # [B, N]

    sorted_features, _ = torch.sort(projected_features, dim=0)  # [B, N]
    sorted_targets, _ = torch.sort(projected_targets, dim=0)  # [B, N]

    wasserstein_1d = torch.mean((sorted_features - sorted_targets) ** 2, dim=0)  # [N]
    return torch.mean(wasserstein_1d)  # []
