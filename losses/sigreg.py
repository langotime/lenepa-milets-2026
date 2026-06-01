"""SIGReg stability loss.

This file lives in `losses/` so both SIGReg and RDMReg are co-located.
"""

from __future__ import annotations

import torch

__all__ = ["SIGReg"]


class SIGReg(torch.nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Computes an Epps-Pulley statistic to measure deviation from $\\mathcal{N}(0, I)$.
    Uses random 1D projections to approximate multivariate characteristic functions.
    """

    def __init__(self, knots: int = 17, normalize: bool = False, *, seed: int | None = None):
        """Initialize SIGReg.

        Args:
            knots: Number of knots for the characteristic function integral.
            normalize: If True, normalize embeddings per-view before scoring.
            seed: Optional RNG seed to make SIGReg projection sampling independent from the
                global torch RNG stream (opt-in). When provided, SIGReg maintains a private
                `torch.Generator` on the active device and draws random projections from it.
        """
        super().__init__()
        self.normalize = normalize
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"seed must be an int or null, got {type(seed).__name__}")
            if seed < 0:
                raise ValueError(f"seed must be >= 0, got {seed}")
        self.seed = seed
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

        t = torch.linspace(0, 3, knots, dtype=torch.float32)  # [T]
        dt = 3 / (knots - 1)  # []

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)  # [T]
        weights[[0, -1]] = dt  # [T]

        window = torch.exp(-t.square() / 2.0)  # [T]

        self.register_buffer("t", t)  # [T]
        self.register_buffer("phi", window)  # [T]
        self.register_buffer("weights", weights * window)  # [T]

    def _projection_generator(self, device: torch.device) -> torch.Generator:
        """Return a private generator for projection sampling on the given device."""
        if self.seed is None:
            raise ValueError("SIGReg seed is null; projection generator is unavailable")
        if self._generator is None or self._generator_device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.seed))
            self._generator = generator
            self._generator_device = device
        return self._generator

    def forward(self, proj: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: embeddings [V, B, D] or [B, D]
            mask: optional bool mask [V, B] or non-negative float weights [V, B]

        Returns:
            Scalar tensor loss.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # [1, B, D]
        V, B, D = proj.shape

        sample_weights: torch.Tensor | None = None
        if mask is not None:
            if mask.shape != (V, B):
                raise ValueError(f"mask shape must be (V, B)=({V}, {B}), got {tuple(mask.shape)}")
            if mask.dtype is torch.bool:
                if not mask.any():
                    raise ValueError("mask must include at least one valid element")
                sample_weights = mask.to(dtype=torch.float32)  # [V, B]
            elif mask.dtype.is_floating_point:
                if (mask < 0).any():
                    raise ValueError("mask weights must be non-negative")
                if mask.sum() <= 0:
                    raise ValueError("mask must include at least one positive weight")
                sample_weights = mask.to(dtype=torch.float32)  # [V, B]
            else:
                raise ValueError(f"mask must be bool or floating point weights, got {mask.dtype}")

        if self.normalize:
            if sample_weights is None:
                mu = proj.mean(dim=1, keepdim=True)  # [V, 1, D]
                sigma = proj.std(dim=1, keepdim=True).clamp_min(1e-6)  # [V, 1, D]
            else:
                weights_f = sample_weights.to(dtype=proj.dtype).unsqueeze(-1)  # [V, B, 1]
                denom = weights_f.sum(dim=1, keepdim=True)  # [V, 1, 1]
                if (denom <= 0).any():
                    raise ValueError("mask results in zero total weight for SIGReg normalization")
                mu = (proj * weights_f).sum(dim=1, keepdim=True) / denom  # [V, 1, D]
                diff = (proj - mu) * weights_f  # [V, B, D]
                var = (diff * diff).sum(dim=1, keepdim=True) / denom  # [V, 1, D]
                sigma = var.sqrt().clamp_min(1e-6)  # [V, 1, D]
            proj = (proj - mu) / sigma  # [V, B, D]

        M = 256
        if self.seed is None:
            A = torch.randn(D, M, device=proj.device)  # [D, M]
        else:
            A = torch.randn(
                D,
                M,
                device=proj.device,
                generator=self._projection_generator(proj.device),
            )  # [D, M]
        A = A.div_(A.norm(p=2, dim=0))  # [D, M]

        x_t = (proj @ A).unsqueeze(-1) * self.t  # [V, B, M, T]

        if sample_weights is None:
            cos_mean = x_t.cos().mean(dim=-3)  # [V, M, T]
            sin_mean = x_t.sin().mean(dim=-3)  # [V, M, T]
            err = (cos_mean - self.phi).square() + sin_mean.square()  # [V, M, T]
            view_weights = None
        else:
            weights_f = sample_weights.to(dtype=x_t.dtype).unsqueeze(-1).unsqueeze(-1)  # [V, B, 1, 1]
            denom = weights_f.sum(dim=-3)  # [V, 1, 1]
            if (denom <= 0).any():
                raise ValueError("mask results in zero total weight for SIGReg")
            cos_sum = (x_t.cos() * weights_f).sum(dim=-3)  # [V, M, T]
            sin_sum = (x_t.sin() * weights_f).sum(dim=-3)  # [V, M, T]
            cos_mean = cos_sum / denom  # [V, M, T]
            sin_mean = sin_sum / denom  # [V, M, T]
            err = (cos_mean - self.phi).square() + sin_mean.square()  # [V, M, T]
            view_weights = sample_weights.sum(dim=1) > 0  # [V]

        statistic = err @ self.weights  # [V, M]

        if view_weights is None:
            return statistic.mean()  # []
        view_weights_f = view_weights.to(dtype=statistic.dtype)  # [V]
        denom = view_weights_f.sum() * statistic.size(1)  # []
        if denom <= 0:
            raise ValueError("mask results in zero valid views for SIGReg")
        return (statistic * view_weights_f.unsqueeze(-1)).sum() / denom  # []
