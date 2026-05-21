from __future__ import annotations

import math
from typing import Any, Optional

import torch as th
from disslib.estimators import StructureRewardEstBase


class OneToManyStructureZeroOneRewardEst(StructureRewardEstBase):
    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        fraction_training_data_per_split: float,
        n_mimic_models: int,
        mimic_xgbc_kwargs: dict[str, Any] = dict(),
        base_xgbc_kwargs: dict[str, Any] = dict(),
        alpha: float = 0.0,
        xs_etrain: Optional[th.Tensor] = None,
        ys_etrain: Optional[th.Tensor] = None,
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_ctx_covs=n_ctx_covs,
            n_bdms_per_fcomb=n_bdms_per_fcomb,
            fraction_training_data_per_split=fraction_training_data_per_split,
            n_mimic_models=n_mimic_models,
            n_base_models=1,
            mimic_xgbc_kwargs=mimic_xgbc_kwargs,
            base_xgbc_kwargs=base_xgbc_kwargs,
            alpha=alpha,
            xs_etrain=xs_etrain,
            ys_etrain=ys_etrain,
            rseed=rseed,
        )

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        inputs = inputs.to(self.device)
        mimic_outs_l, base_outs_l = self._forward(inputs)
        outs: th.Tensor = th.stack(
            [
                -(
                    (
                        th.argmax(mo, dim=1).flatten()
                        != th.argmax(base_outs_l[0], dim=1).flatten()
                    ).to(dtype=th.float32)
                )
                for mo in mimic_outs_l
            ],
            # [
            #     -th.nn.functional.cross_entropy(
            #         torch.distributions.utils.probs_to_logits(mo),
            #         base_outs_l[0],
            #         reduction="none",
            #     )
            #     for mo in mimic_outs_l
            # ],
            dim=1,
        ).to(device=inputs.device, dtype=th.float32)
        # TODO use decompose inputs
        # _, xacts = th.chunk(inputs, chunks=2, dim=1)
        _, cinds, _ = self.decompose_inputs(inputs)
        outs = outs - (self.alpha * th.sum(cinds, dim=1))[:, None]
        return outs

    def sample_posterior(
        self, forward_outs: th.Tensor, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        n: int = len(forward_outs)
        n_samps: int = math.prod(sample_shape)
        idxs: th.Tensor = th.randint(
            0, forward_outs.shape[1], (n, n_samps), dtype=th.long, device=self.device
        )
        outs: th.Tensor = th.gather(forward_outs, dim=1, index=idxs).reshape(
            (n, *sample_shape)
        )
        return outs

    def get_posterior_covariance(self, forward_outs: th.Tensor) -> th.Tensor:
        assert len(self.mimic_models) > 1
        # forward_outs (bsz, n_splits)
        mus: th.Tensor = th.mean(forward_outs, dim=1, keepdim=True)
        fouts_minus_mus: th.Tensor = forward_outs - mus
        # (bsz, bsz)
        covars: th.Tensor = (1 / (len(self.mimic_models) - 1)) * (
            fouts_minus_mus @ fouts_minus_mus.T
        )
        return covars
