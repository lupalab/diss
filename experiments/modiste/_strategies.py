from __future__ import annotations

from abc import ABC, abstractmethod

import lightning as pl
import torch as th

from disslib import Env

from _estimators import ModisteRewardEstBase


class OptStrat(ABC):
    support_lazy_fit: bool = False

    env: Env
    n_queries: int

    def __init__(self, env: Env, n_queries: int) -> None:
        super().__init__()
        self.env = env
        self.n_queries = n_queries

    @abstractmethod
    def suggest_next_queries(
        self, score_est: ModisteRewardEstBase, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Suggest next query point

        Args:
            score_est (ModisteRewardEstBase): score estimator
            plf (pl.Fabric): lightning fabric object

        Returns:
            th.Tensor: (n_queries, n_covs) suggested contexts
            th.Tensor: (n_queries,) suggested actions
            th.Tensor: (n_queries,) expected improvement taking suggested context-action pairs
        """


class RandomOptStrat(OptStrat):
    support_lazy_fit = True

    def suggest_next_queries(
        self, score_est: ModisteRewardEstBase, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # (n, n_covs)
        ctxs, _ctxs_info = self.env.get_ctxs(self.n_queries)
        # (n, n_avail_acts, n_act_feats)
        xacts, acts = self.env.get_avail_actions(ctxs)
        # (n, )
        aidxs: th.Tensor = th.randint(0, xacts.shape[1], (len(ctxs),), dtype=th.long)
        # (n, )
        best_acts = th.gather(
            acts, dim=1, index=aidxs[:, None, None].expand(-1, -1, acts.shape[2])
        )[:, 0]
        # (n, )
        imps: th.Tensor = th.inf * th.ones((len(ctxs),), dtype=th.float32)
        return ctxs, best_acts, imps


class ModisteOptStrat(OptStrat):
    epsilon: float

    def __init__(self, env: Env, n_queries: int, epsilon: float) -> None:
        super().__init__(env, n_queries)
        self.epsilon = epsilon

    @th.no_grad()
    def suggest_next_queries(
        self, score_est: ModisteRewardEstBase, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        score_est.eval()
        ctxs, _ctxs_info = self.env.get_ctxs(self.n_queries)
        bsz: int = len(ctxs)
        xacts, acts = self.env.get_avail_actions(ctxs)
        n_acts_avail: int = xacts.shape[1]
        ctxs_: th.Tensor = ctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # NOTE old impl. does not batch over input
        # # (bsz * n_acts, n_covs + n_act_feats)
        # inputs: th.Tensor = th.cat((ctxs_, xacts), dim=2).flatten(0, 1)
        # (bsz, n_acts)
        # outputs: th.Tensor = th.unflatten(
        #     score_est(inputs), dim=0, sizes=(bsz, n_acts_avail)
        # ).to(device="cpu")
        # NOTE new impl. that batches over input
        # (bsz * n_acts, n_covs + n_act_feats)
        inputs: th.Tensor = th.cat((ctxs_, xacts), dim=2).flatten(0, 1)
        # (bsz, n_acts)
        outputs: th.Tensor = th.unflatten(
            th.cat(
                [
                    score_est(_binps.to(device=plf.device)).to(device="cpu")
                    # TODO need to consider specifying batch size in constructor
                    for _binps in th.split(
                        inputs, split_size_or_sections=max(n_acts_avail, bsz), dim=0
                    )
                ],
                dim=0,
            ),
            0,
            (bsz, n_acts_avail),
        )
        # epsilon-greedy select action
        aidxs_knn: th.Tensor = th.argmax(outputs, dim=1)
        aidxs_rand: th.Tensor = th.randint(
            0, n_acts_avail, (self.n_queries,), dtype=th.long
        )
        aidxs: th.Tensor = th.where(
            th.rand((bsz,)) < self.epsilon, aidxs_rand, aidxs_knn
        )
        best_acts: th.Tensor = th.gather(
            acts, dim=1, index=aidxs[:, None, None].expand(-1, -1, acts.shape[2])
        )[:, 0]
        # meaningless expected improvements
        imps: th.Tensor = th.inf * th.ones((self.n_queries,), dtype=th.float32)
        return ctxs, best_acts, imps
