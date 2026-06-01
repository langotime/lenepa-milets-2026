"""Loss functions used across LeNEPA models.

This package keeps "regularization / stability" losses (e.g., SIGReg, RDMReg)
in a single import location to avoid circular dependencies with model code.
"""

from .rectified_lpjepa import (  # noqa: F401
    determine_sigma_for_lp_dist,
    diag_update_for_features,
    l0_sparsity_metric,
    l1_sparsity_metric,
    reprelu,
    sample_lp_distribution,
    sample_unit_sphere_projections,
    sliced_wasserstein_distance_batched,
    sliced_wasserstein_distance_for_one_view,
    swd_mean_over_views,
)
from .sigreg import SIGReg  # noqa: F401

__all__ = [
    "SIGReg",
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
