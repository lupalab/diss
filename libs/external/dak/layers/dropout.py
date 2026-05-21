import gpytorch
import torch
import torch.nn as nn

from ..utils.sparse_design.design_class import HyperbolicCrossDesign
from .activation import InducedPriorUnit


class KernelRandomFeature(nn.Module):
    def __init__(
        self,
        in_features,
        induced_level,
        kernel,
        design_class=HyperbolicCrossDesign,
        grid_bounds=(-1.0, 1.0),
    ):
        super(KernelRandomFeature, self).__init__()

        # Saturation layer
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            grid_bounds[0], grid_bounds[1]
        )

        # InducedGaussianUnit
        self.gp_activation = InducedPriorUnit(
            in_features=in_features,
            induced_level=induced_level,
            kernel=kernel,
            design_class=design_class,
            grid_bounds=grid_bounds,
        )

        num_inducing = self.gp_activation.num_inducing
        self.out_features = num_inducing * in_features
        self.phi_weight = nn.Parameter(
            torch.Tensor(in_features, num_inducing)
        )  # [num_out, D, M]
        self.phi_weight.data.normal_(mean=0, std=0.1)

    def forward(self, x):
        x = self.scale_to_bounds(x)
        x = self.gp_activation(x)
        z = torch.randn_like(x, device=x.device)  # sample from normal noise
        x = x * z + x * self.phi_weight
        return x


class KernelDropout(nn.Module):
    def __init__(
        self,
        in_features,
        induced_level,
        kernel,
        design_class=HyperbolicCrossDesign,
        grid_bounds=(-1.0, 1.0),
    ):
        super(KernelDropout, self).__init__()

        # Saturation layer
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            grid_bounds[0], grid_bounds[1]
        )

        # InducedGaussianUnit
        self.gp_activation = InducedPriorUnit(
            in_features=in_features,
            induced_level=induced_level,
            kernel=kernel,
            design_class=design_class,
            grid_bounds=grid_bounds,
        )

        num_inducing = self.gp_activation.num_inducing
        self.out_features = num_inducing * in_features
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.scale_to_bounds(x)
        x = self.gp_activation(x)
        x = self.dropout(x)
        return x
