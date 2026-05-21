from __future__ import annotations

from typing import Optional

import gpytorch as gpth
import torch as th


class SimpleGP(gpth.models.ExactGP):
    mean_module: gpth.means.Mean
    covar_module: gpth.kernels.Kernel
    likelihood: gpth.likelihoods.GaussianLikelihood

    def __init__(
        self,
        mean_module: gpth.means.Mean,
        covar_module: gpth.kernels.Kernel,
        train_inputs: Optional[th.Tensor],
        train_targets: Optional[th.Tensor],
        likelihood: gpth.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, xs: th.Tensor) -> gpth.distributions.MultivariateNormal:
        return gpth.distributions.MultivariateNormal(
            self.mean_module(xs), self.covar_module(xs)
        )
