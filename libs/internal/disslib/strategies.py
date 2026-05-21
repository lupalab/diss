from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

import lightning as pl
import sklearn.cluster as skl_cluster
import tensordict as thd
import torch as th

if TYPE_CHECKING:
    from . import Env, RewardEst
    from .smart_fitters import SmartFitter


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
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        """Suggest next query point

        Args:
            reward_est (RewardEst): score estimator
            plf (pl.Fabric): lightning fabric object

        Returns:
            th.Tensor: (n_queries, n_covs) suggested contexts
            th.Tensor: (n_queries,) suggested actions
            th.Tensor: (n_queries,) expected improvement taking suggested context-action pairs
            thd.TensorDict: (n_queries,) addtional context information
        """


class RandomOptStrat(OptStrat):
    support_lazy_fit = True

    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        # (n, n_covs)
        ctxs, ctxs_info = self.env.get_ctxs(self.n_queries)
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
        return ctxs, best_acts, imps, ctxs_info


class TS(OptStrat):
    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = self.env.get_ctxs(self.n_queries)
        n: int = len(ctxs)
        xacts, acts = self.env.get_avail_actions(ctxs)
        n_acts_avail: int = xacts.shape[1]
        ctxs_: th.Tensor = ctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        inputs: th.Tensor = (
            th.cat((ctxs_, xacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        outs: th.Tensor | tuple[th.Tensor, ...] = reward_est(inputs)
        # (bsz, n_avail_acts)
        psamps: th.Tensor = (
            reward_est.sample_posterior(outs)
            .unflatten(0, (n, n_acts_avail))
            .to(device="cpu")
        )
        # (bsz, )
        psamps, aidxs = th.max(psamps, dim=1)
        imps: th.Tensor = th.inf * th.ones((self.n_queries,), dtype=th.float32)
        best_acts: th.Tensor = th.gather(
            acts,
            dim=1,
            index=aidxs[:, None, None].expand(-1, -1, xacts.shape[2]),
        )[:, 0]
        return ctxs, best_acts, imps, ctxs_info


class UCB(OptStrat):
    # Gaussian Process Optimization in the Bandit Setting:
    # No Regret and Experimental Design
    # Niranjan Srinivas, Andreas Krause, Sham M. Kakade, Matthias Seeger
    # https://icml.cc/Conferences/2010/papers/422.pdf
    alpha: float

    def __init__(self, env: Env, n_queries: int, alpha=2.0) -> None:
        super().__init__(env, n_queries)
        self.alpha = alpha

    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = self.env.get_ctxs(self.n_queries)
        n: int = len(ctxs)
        xacts, acts = self.env.get_avail_actions(ctxs)
        n_acts_avail: int = xacts.shape[1]
        ctxs_: th.Tensor = ctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        inputs: th.Tensor = (
            th.cat((ctxs_, xacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        outs: th.Tensor | tuple[th.Tensor, ...] = reward_est(inputs)
        # (bsz * n_avail_acts)
        pmeans: th.Tensor = reward_est.get_posterior_mean(outs)
        pstds: th.Tensor = reward_est.get_posterior_std(outs)
        # (bsz, n_avail_acts)
        pucbs: th.Tensor = (pmeans + self.alpha * pstds).unflatten(0, (n, n_acts_avail))
        # (bsz, )
        pucbs, aidxs = th.max(pucbs, dim=1)
        imps: th.Tensor = th.inf * th.ones((self.n_queries,), dtype=th.float32)
        best_acts: th.Tensor = th.gather(
            acts,
            dim=1,
            index=aidxs[:, None, None].expand(-1, -1, xacts.shape[2]),
        )[:, 0]
        return ctxs, best_acts, imps, ctxs_info


class ChooseMultiQueryMethod(ABC):
    def __call__(
        self,
        ctxs: th.Tensor,
        acts: th.Tensor,
        imps: th.Tensor,
        ctxs_info: thd.TensorDict,
        n_queries: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        return self.choose(ctxs, acts, imps, ctxs_info, n_queries)

    @abstractmethod
    def choose(
        self,
        ctxs: th.Tensor,
        acts: th.Tensor,
        imps: th.Tensor,
        ctxs_info: thd.TensorDict,
        n_queries: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]: ...


class ChooseGreedily(ChooseMultiQueryMethod):
    def choose(
        self,
        ctxs: th.Tensor,
        acts: th.Tensor,
        imps: th.Tensor,
        ctxs_info: thd.TensorDict,
        n_queries: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        n_queries = min(len(ctxs), n_queries)
        # sort items according to improvements made
        best_imps, imps_idxs = th.sort(imps, descending=True)
        # choose top n_queries ctx-act pairs
        best_imps, imps_idxs = best_imps[:n_queries], imps_idxs[:n_queries]
        imps_idxs = imps_idxs.to(device="cpu")
        best_ctxs: th.Tensor = ctxs[imps_idxs]
        best_acts: th.Tensor = acts[imps_idxs]
        best_ctxs_info: thd.TensorDict = ctxs_info[imps_idxs]
        return best_ctxs, best_acts, best_imps, best_ctxs_info


_ClusterAlgType = TypeVar("_ClusterAlgType", bound=skl_cluster.KMeans)
# _ClusterAlgType = TypeVar(
#     "_ClusterAlgType", bound=skl_cluster.KMeans | skle_cluster.KMedoids
# )


class _ChooseSimMatchBase(Generic[_ClusterAlgType], ChooseMultiQueryMethod):
    @abstractmethod
    def _make_cluster_alg(self, *args, **kwargs) -> _ClusterAlgType: ...

    def choose(
        self,
        ctxs: th.Tensor,
        acts: th.Tensor,
        imps: th.Tensor,
        ctxs_info: thd.TensorDict,
        n_queries: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        n_queries = min(len(ctxs), n_queries)
        # sort items according to improvements made
        imps_sorted, imps_idxs = th.sort(imps, descending=True)
        ctxs_sorted: th.Tensor = ctxs[imps_idxs]
        acts_sorted: th.Tensor = acts[imps_idxs]
        ctxs_info_sorted: thd.TensorDict = ctxs_info[imps_idxs]
        # no need to run kmeans if bsz is already less than n_queries
        if n_queries == len(ctxs):
            return ctxs_sorted, acts_sorted, imps_sorted, ctxs_info_sorted
        # run kmeans
        kms: _ClusterAlgType = self._make_cluster_alg(n_clusters=n_queries)
        ctxs_cids: th.Tensor = th.as_tensor(
            kms.fit_predict(ctxs_sorted.numpy(force=True)), device=ctxs_sorted.device
        )
        # fall back to greedy select if there exists empty cluster
        if len(th.unique(ctxs_cids)) != n_queries:
            return (
                ctxs_sorted[:n_queries],
                acts_sorted[:n_queries],
                imps_sorted[:n_queries],
                ctxs_info_sorted[:n_queries],
            )
        # diversity-aware choose top n_queries
        best_ctxs: th.Tensor = th.empty((n_queries, ctxs.shape[1]), device=ctxs.device)
        best_acts: th.Tensor = th.empty((n_queries,), dtype=th.long, device=acts.device)
        best_imps: th.Tensor = th.empty((n_queries,), device=imps.device)
        best_ctxs_info_l: list[thd.TensorDict] = [
            thd.make_tensordict({}) for _ in range(n_queries)
        ]
        # keep track of context from which cluster has already been added to best-series tensor
        cid_set: set[int] = set()
        for _cid, _ctx, _act, _imp, _ctx_info in zip(
            ctxs_cids, ctxs_sorted, acts_sorted, imps_sorted, ctxs_info_sorted
        ):
            _cid = int(_cid.item())
            # continue if a context from _cid is already included
            if _cid in cid_set:
                continue
            i: int = len(cid_set)
            best_ctxs[i] = _ctx
            best_acts[i] = _act
            best_imps[i] = _imp
            best_ctxs_info_l[i] = _ctx_info
            cid_set.add(_cid)
        best_ctxs_info: thd.TensorDict = thd.stack(
            best_ctxs_info_l, dim=0
        )  # type:ignore
        return best_ctxs, best_acts, best_imps, best_ctxs_info


class ChooseSimMatchKMeans(_ChooseSimMatchBase[skl_cluster.KMeans]):
    def _make_cluster_alg(self, *args, **kwargs) -> skl_cluster.KMeans:
        kms: skl_cluster.KMeans = skl_cluster.KMeans(*args, **kwargs)
        return kms


# class ChooseSimMatchKMedoid(_ChooseSimMatchBase[skle_cluster.KMedoids]):
#     def _make_cluster_alg(self, *args, **kwargs) -> skle_cluster.KMedoids:
#         return skle_cluster.KMedoids(*args, **kwargs)


class _ContextActionJointOptStrat(OptStrat):
    n_ctxs: int
    bsz: int

    multi_query_method: ChooseMultiQueryMethod

    def __init__(
        self,
        env: Env,
        n_queries: int,
        n_ctxs: int,
        bsz: int,
        multi_query_method: ChooseMultiQueryMethod,
    ) -> None:
        super().__init__(env, n_queries)
        self.n_ctxs = n_ctxs
        self.bsz = bsz
        self.multi_query_method = multi_query_method

    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = self.env.get_ctxs(self.n_ctxs)
        xacts, acts = self.env.get_avail_actions()
        n_ctxs: int = len(ctxs)
        n_acts_avail: int = len(xacts)
        utilities: th.Tensor = th.empty(
            (n_ctxs, n_acts_avail), dtype=th.float32, device=plf.device
        )
        for _bidxs in th.split(
            th.cartesian_prod(th.arange(n_ctxs), th.arange(n_acts_avail)),
            split_size_or_sections=self.bsz,
        ):
            utilities[_bidxs[:, 0], _bidxs[:, 1]] = self.compute_utility(
                ctxs[_bidxs[:, 0]], xacts[_bidxs[:, 1]], reward_est, plf
            )
        best_ctxs, best_acts, best_utilities, best_ctxs_info = self.multi_query_method(
            ctxs=ctxs[:, None, :].expand(-1, n_acts_avail, -1).flatten(0, 1),
            acts=acts[None, :, :].expand(n_ctxs, -1, -1).flatten(0, 1),
            imps=utilities.flatten(0, 1),
            ctxs_info=ctxs_info.expand(n_acts_avail, n_ctxs)
            .permute(1, 0)
            .flatten(0, 1),
            n_queries=self.n_queries,
        )
        return best_ctxs, best_acts, best_utilities, best_ctxs_info

    @abstractmethod
    def compute_utility(
        self,
        ctxs: th.Tensor,
        xacts: th.Tensor,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> th.Tensor:
        pass


class TSContextActionJointOptStrat(_ContextActionJointOptStrat):
    def compute_utility(
        self,
        ctxs: th.Tensor,
        xacts: th.Tensor,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> th.Tensor:
        reward_est.eval().to(device=plf.device)
        # (bsz, n_covs + n_act_feats)
        inputs: th.Tensor = th.cat((ctxs, xacts), dim=1).to(device=plf.device)
        outs: th.Tensor | tuple[th.Tensor, ...] = reward_est(inputs)
        # (bsz,)
        psamps: th.Tensor = reward_est.sample_posterior(outs)
        return psamps


class UCBContextActionJointOptStrat(_ContextActionJointOptStrat):
    alpha: float

    def __init__(
        self,
        env: Env,
        n_queries: int,
        n_ctxs: int,
        bsz: int,
        multi_query_method: ChooseMultiQueryMethod,
        alpha: float = 2.0,
    ) -> None:
        super().__init__(env, n_queries, n_ctxs, bsz, multi_query_method)
        self.alpha = alpha

    def compute_utility(
        self,
        ctxs: th.Tensor,
        xacts: th.Tensor,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> th.Tensor:
        reward_est.eval().to(device=plf.device)
        # (bsz, n_covs + n_act_feats)
        inputs: th.Tensor = th.cat((ctxs, xacts), dim=1).to(device=plf.device)
        outs: th.Tensor | tuple[th.Tensor, ...] = reward_est(inputs)
        # (bsz,)
        pmeans: th.Tensor = reward_est.get_posterior_mean(outs)
        pstds: th.Tensor = reward_est.get_posterior_std(outs)
        # (bsz,)
        pucbs: th.Tensor = pmeans + self.alpha * pstds
        return pucbs


class ProfileOptStratSuggestNextQueryMethod(ABC):
    @abstractmethod
    def __call__(
        self,
        opt_strat: ProfileOptStrat,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        pass


class VanillaProfileOptSuggestNextQuery(ProfileOptStratSuggestNextQueryMethod):
    choose_multi_query_method: ChooseMultiQueryMethod

    def __init__(
        self,
        choose_multi_query_method: ChooseMultiQueryMethod = ChooseGreedily(),
    ) -> None:
        super().__init__()
        self.choose_multi_query_method = choose_multi_query_method

    def __call__(
        self,
        opt_strat: ProfileOptStrat,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = opt_strat.env.get_ctxs(opt_strat.n_ctxs)
        best_acts_l: list[th.Tensor] = list()
        imps_l: list[th.Tensor] = list()
        for bctxs in th.split(ctxs, opt_strat.ctxs_bsz):
            bbest_acts, bimps = opt_strat._get_ctxs_improvements(bctxs, reward_est, plf)
            best_acts_l.append(bbest_acts)
            imps_l.append(bimps)
        # (self.n_ctxs, )
        acts: th.Tensor = th.cat(best_acts_l, dim=0)
        imps: th.Tensor = th.cat(imps_l, dim=0)
        # # sort items according to improvements made
        # best_imps, imps_idxs = th.sort(imps, descending=True)
        # # choose top n_queries ctx-act pairs
        # n_queries: int = min(len(ctxs), self.n_queries)
        # best_imps, imps_idxs = best_imps[:n_queries], imps_idxs[:n_queries]
        # best_ctxs: th.Tensor = ctxs[imps_idxs]
        # best_acts: th.Tensor = acts[imps_idxs]
        best_ctxs, best_acts, best_imps, best_ctxs_info = (
            self.choose_multi_query_method(
                ctxs, acts, imps, ctxs_info, opt_strat.n_queries
            )
        )
        return best_ctxs, best_acts, best_imps, best_ctxs_info


class LoopBatchedProfileOptSuggestNextQuery(ProfileOptStratSuggestNextQueryMethod):
    def __call__(
        self,
        opt_strat: ProfileOptStrat,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = opt_strat.env.get_ctxs(opt_strat.n_ctxs)
        n_queries: int = min(len(ctxs), opt_strat.n_queries)
        n_covs: int = ctxs.shape[1]
        best_ctxs: th.Tensor = th.empty(
            (n_queries, n_covs), dtype=th.float32, device=ctxs.device
        )
        best_acts: th.Tensor = th.empty((n_queries,), dtype=th.long, device=ctxs.device)
        best_imps: th.Tensor = th.empty(
            (n_queries,), dtype=th.float32, device=ctxs.device
        )
        best_ctxs_info: thd.TensorDict = thd.make_tensordict(
            {_k: th.empty_like(_v) for _k, _v in ctxs_info[0].items()}
        )
        for i in range(n_queries):
            _best_acts_l: list[th.Tensor] = list()
            _imps_l: list[th.Tensor] = list()
            for bctxs in th.split(ctxs, opt_strat.ctxs_bsz):
                bbest_acts, bimps = opt_strat._get_ctxs_improvements(
                    bctxs, reward_est, plf
                )
                _best_acts_l.append(bbest_acts)
                _imps_l.append(bimps)
            # (self.n_ctxs, )
            _acts: th.Tensor = th.cat(_best_acts_l, dim=0)
            _imps: th.Tensor = th.cat(_imps_l, dim=0)
            _best_ctxs, _best_acts, _best_imps, _best_ctxs_info = ChooseGreedily()(
                ctxs, _acts, _imps, ctxs_info, 1
            )
            best_ctxs[i] = _best_ctxs[0]
            best_acts[i] = _best_acts[0]
            best_imps[i] = _best_imps[0]
            best_ctxs_info[i] = _best_ctxs_info[0]
        return best_ctxs, best_acts, best_imps, best_ctxs_info


class RSampleProfileOptSuggestNextQuery(ProfileOptStratSuggestNextQueryMethod):
    rsample_size: int | None

    def __init__(self, rsample_size: Optional[int] = None) -> None:
        super().__init__()
        self.rsample_size = rsample_size

    def __call__(
        self,
        opt_strat: ProfileOptStrat,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = opt_strat.env.get_ctxs(opt_strat.n_ctxs)
        n_queries: int = min(len(ctxs), opt_strat.n_queries)
        n_covs: int = ctxs.shape[1]
        best_ctxs: th.Tensor = th.empty(
            (n_queries, n_covs), dtype=th.float32, device=ctxs.device
        )
        best_acts: th.Tensor = th.empty((n_queries,), dtype=th.long, device=ctxs.device)
        best_imps: th.Tensor = th.empty(
            (n_queries,), dtype=th.float32, device=ctxs.device
        )
        best_ctxs_info: thd.TensorDict = thd.make_tensordict(
            {_k: th.empty_like(_v) for _k, _v in ctxs_info[0].items()}
        )
        for i in range(n_queries):
            _idxs: th.Tensor = th.randint(
                0,
                len(ctxs),
                (len(ctxs) if self.rsample_size is None else self.rsample_size,),
                dtype=th.long,
                device=ctxs.device,
            )
            _best_acts_l: list[th.Tensor] = list()
            _imps_l: list[th.Tensor] = list()
            for bctxs in th.split(ctxs[_idxs], opt_strat.ctxs_bsz):
                bbest_acts, bimps = opt_strat._get_ctxs_improvements(
                    bctxs, reward_est, plf
                )
                _best_acts_l.append(bbest_acts)
                _imps_l.append(bimps)
            # (self.n_ctxs, )
            _acts: th.Tensor = th.cat(_best_acts_l, dim=0)
            _imps: th.Tensor = th.cat(_imps_l, dim=0)
            _best_ctxs, _best_acts, _best_imps, _best_ctxs_info = ChooseGreedily()(
                ctxs[_idxs], _acts, _imps, ctxs_info[_idxs], 1
            )
            best_ctxs[i] = _best_ctxs[0]
            best_acts[i] = _best_acts[0]
            best_imps[i] = _best_imps[0]
            best_ctxs_info[i] = _best_ctxs_info[0]
        return best_ctxs, best_acts, best_imps, best_ctxs_info


class ProfileOptStrat(OptStrat, ABC):
    n_ctxs: int
    ctxs_bsz: int

    suggest_next_queries_method: ProfileOptStratSuggestNextQueryMethod

    def __init__(
        self,
        env: Env,
        n_queries: int,
        n_ctxs: int,
        ctxs_bsz: int,
        suggest_next_queries_method: ProfileOptStratSuggestNextQueryMethod = VanillaProfileOptSuggestNextQuery(),
    ) -> None:
        super().__init__(env, n_queries)
        self.n_ctxs = n_ctxs
        self.ctxs_bsz = ctxs_bsz
        self.suggest_next_queries_method = suggest_next_queries_method

    @th.no_grad()
    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        return self.suggest_next_queries_method(self, reward_est, plf)

    @abstractmethod
    def _get_ctxs_improvements(
        self, bctxs: th.Tensor, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor]:
        pass


class MTSPM(ProfileOptStrat):
    def _get_ctxs_improvements(
        self, bctxs: th.Tensor, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor]:
        reward_est.eval().to(plf.device)
        bsz: int = len(bctxs)
        bxacts, bacts = self.env.get_avail_actions(bctxs)
        n_acts_avail: int = bxacts.shape[1]
        bctxs_: th.Tensor = bctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        binputs: th.Tensor = (
            th.cat((bctxs_, bxacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        bouts: th.Tensor | tuple[th.Tensor, ...] = reward_est(binputs)
        # get posteior mean and one posterior sample
        # (bsz, n_avail_acts)
        bpms: th.Tensor = (
            reward_est.get_posterior_mean(bouts)
            .unflatten(0, (bsz, n_acts_avail))
            .to(device="cpu")
        )
        bpsamps: th.Tensor = (
            reward_est.sample_posterior(bouts)
            .unflatten(0, (bsz, n_acts_avail))
            .to(device="cpu")
        )
        # compute improvement
        # (bsz, )
        bbest_pms: th.Tensor = th.max(
            # cap the greatest posterior mean value by the greatest observation so far
            th.minimum(
                bpms, th.broadcast_to(th.max(reward_est.train_targets), bpms.shape)
            ),
            dim=1,
        )[0]
        bbest_psamps, bbest_aidxs = th.max(bpsamps, dim=1)
        bimps: th.Tensor = bbest_psamps - bbest_pms
        # from actions indices to acts
        bbest_acts: th.Tensor = th.gather(
            bacts,
            dim=1,
            index=bbest_aidxs[:, None, None].expand(-1, -1, bxacts.shape[2]),
        )[:, 0]
        return bbest_acts, bimps


class PEI(ProfileOptStrat):
    def _get_ctxs_improvements(
        self, bctxs: th.Tensor, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor]:
        reward_est.eval().to(plf.device)
        bsz: int = len(bctxs)
        bxacts, bacts = self.env.get_avail_actions(bctxs)
        n_acts_avail: int = bxacts.shape[1]
        bctxs_: th.Tensor = bctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        binputs: th.Tensor = (
            th.cat((bctxs_, bxacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        bouts: th.Tensor | tuple[th.Tensor, ...] = reward_est(binputs)
        # get posteior mean and one posterior sample
        # (bsz, n_avail_acts)
        bpms: th.Tensor = (
            reward_est.get_posterior_mean(bouts)
            .unflatten(0, (bsz, n_acts_avail))
            .to(device="cpu")
        )
        bpvars: th.Tensor = (
            reward_est.get_posterior_covariance(bouts)
            .diagonal()
            .unflatten(0, (bsz, n_acts_avail))
            .to(device="cpu")
        )
        bpstds: th.Tensor = th.sqrt(bpvars)
        # compute improvement
        # (bsz, 1)
        bbest_pms: th.Tensor = th.max(
            # cap the greatest posterior mean value by the greatest observation so far
            th.minimum(
                bpms, th.broadcast_to(th.max(reward_est.train_targets), bpms.shape)
            ),
            dim=1,
            keepdim=True,
        )[0]
        # (bsz, n_avail_acts)
        bnorm_diffs: th.Tensor = (bpms - bbest_pms) / (bpstds + 1e-5)
        norm_rv = th.distributions.Normal(th.tensor(0.0), th.tensor(1.0))
        beis: th.Tensor = bpstds * (
            bnorm_diffs * norm_rv.cdf(bnorm_diffs)
            + th.exp(norm_rv.log_prob(bnorm_diffs))
        )
        bbest_eis, bbest_aidxs = th.max(beis, dim=1)
        # from actions indices to acts
        bbest_acts: th.Tensor = th.gather(
            bacts,
            dim=1,
            index=bbest_aidxs[:, None, None].expand(-1, -1, bxacts.shape[2]),
        )[:, 0]
        return bbest_acts, bbest_eis


class MUCB(ProfileOptStrat):
    # Gaussian Process Optimization in the Bandit Setting:
    # No Regret and Experimental Design
    # Niranjan Srinivas, Andreas Krause, Sham M. Kakade, Matthias Seeger
    # https://icml.cc/Conferences/2010/papers/422.pdf
    n_ctxs: int
    choose_multi_query_method: ChooseMultiQueryMethod
    alpha = 2.0

    def __init__(
        self,
        env: Env,
        n_queries: int,
        n_ctxs: int,
        ctxs_bsz: int,
        suggest_next_queries_method: ProfileOptStratSuggestNextQueryMethod,
        alpha=2.0,
    ) -> None:
        super().__init__(env, n_queries, n_ctxs, ctxs_bsz, suggest_next_queries_method)
        self.alpha = alpha

    def _get_ctxs_improvements(
        self,
        bctxs: th.Tensor,
        reward_est: RewardEst | SmartFitter,
        plf: pl.Fabric,
    ) -> tuple[th.Tensor, th.Tensor]:
        reward_est.eval().to(plf.device)
        bsz: int = len(bctxs)
        bxacts, bacts = self.env.get_avail_actions(bctxs)
        n_acts_avail: int = bxacts.shape[1]
        bctxs_: th.Tensor = bctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        binputs: th.Tensor = (
            th.cat((bctxs_, bxacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        bouts: th.Tensor | tuple[th.Tensor, ...] = reward_est(binputs)
        # get posteior mean and one posterior sample
        # (bsz, n_avail_acts)
        bpmeans: th.Tensor = reward_est.get_posterior_mean(bouts).unflatten(
            0, (bsz, n_acts_avail)
        )
        bpstds: th.Tensor = reward_est.get_posterior_std(bouts).unflatten(
            0, (bsz, n_acts_avail)
        )
        bpucbs: th.Tensor = bpmeans + self.alpha * bpstds
        # compute improvement
        bbest_pucbs, bbest_aidxs = th.max(bpucbs, dim=1)
        # from actions indices to acts
        bbest_acts: th.Tensor = th.gather(
            bacts,
            dim=1,
            index=bbest_aidxs[:, None, None]
            .expand(-1, -1, bxacts.shape[2])
            .to(device="cpu"),
        )[:, 0]
        return bbest_acts, bbest_pucbs


class EI(OptStrat):
    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = self.env.get_ctxs(self.n_queries)
        n: int = len(ctxs)
        xacts, acts = self.env.get_avail_actions(ctxs)
        n_acts_avail: int = xacts.shape[1]
        ctxs_: th.Tensor = ctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz * n_avail_acts, n_covs + n_act_feats)
        inputs: th.Tensor = (
            th.cat((ctxs_, xacts), dim=2).flatten(0, 1).to(device=plf.device)
        )
        outs: th.Tensor | tuple[th.Tensor, ...] = reward_est(inputs)
        # get posteior mean and one posterior sample
        # (bsz, n_avail_acts)
        pms: th.Tensor = (
            reward_est.get_posterior_mean(outs)
            .unflatten(0, (n, n_acts_avail))
            .to(device="cpu")
        )
        pvars: th.Tensor = (
            reward_est.get_posterior_covariance(outs)
            .diagonal()
            .unflatten(0, (n, n_acts_avail))
            .to(device="cpu")
        )
        # (bsz, 1)
        best_pms: th.Tensor = th.max(pms, dim=1, keepdim=True)[0]
        norm_diffs: th.Tensor = (pms - best_pms) / (pvars + 1e-5)
        norm_rv = th.distributions.Normal(th.tensor(0.0), th.tensor(1.0))
        beis: th.Tensor = (
            norm_diffs + norm_rv.cdf(norm_diffs) + th.exp(norm_rv.log_prob(norm_diffs))
        )
        best_eis, best_aidxs = th.max(beis, dim=1)
        # from actions indices to acts
        best_acts: th.Tensor = th.gather(
            acts,
            dim=1,
            index=best_aidxs[:, None, None].expand(-1, -1, xacts.shape[2]),
        )[:, 0]
        return ctxs, best_acts, best_eis, ctxs_info


class REVI(OptStrat):
    @staticmethod
    def knowledge_gradient(means: th.Tensor, sigmas: th.Tensor) -> th.Tensor:
        """Calculate knowledge gradient.

        Args:
            means: Tensor of shape (n_acts,)
            sigmas: Tensor of shape (n_acts,) - vector of uncertainties for each action

        Returns:
            Tensor of shape (n_acts,) containing improvement values
        """
        # Ensure inputs are at least 1D
        means = means.view(-1)  # Reshape to 1D
        sigmas = sigmas.view(-1)  # Reshape to 1D
        n_acts = means.shape[0]
        sorted_idx = th.argsort(sigmas)
        means = means[sorted_idx]
        sigmas = sigmas[sorted_idx]
        means = means - means.max()
        # Initialize improvements tensor
        improvements = th.zeros(n_acts, device=means.device)
        vertices = [0, 1]
        # z_points = [float("-inf"), (means[0] - means[1]) / (sigmas[1] - sigmas[0])]
        z_points = [-9e9, (means[0] - means[1]) / (sigmas[1] - sigmas[0] + 1e-10)]
        for i in range(2, len(means)):
            while len(vertices) >= 2:
                j = vertices[-1]
                z = (means[i] - means[j]) / (sigmas[j] - sigmas[i] + 1e-10)
                if z > z_points[-1]:
                    break
                vertices.pop()
                z_points.pop()
            vertices.append(i)
            z_points.append(z)
        # z_points.append(float("inf"))
        z_points.append(9e9)
        norm = th.distributions.Normal(0, 1)
        z_tensor = th.tensor(z_points, device=means.device)
        cdfs = norm.cdf(z_tensor)
        pdfs = norm.log_prob(z_tensor).exp()
        # Calculate improvement for each vertex
        for i, idx in enumerate(vertices):
            cdf_diff = cdfs[i + 1] - cdfs[i]
            pdf_diff = pdfs[i] - pdfs[i + 1]
            improvements[idx] = means[idx] * cdf_diff + sigmas[idx] * pdf_diff
        # Unsort the improvements to match original ordering
        inv_sort = th.argsort(sorted_idx)
        return improvements[inv_sort]

    def suggest_next_queries(
        self, reward_est: RewardEst | SmartFitter, plf: pl.Fabric
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, thd.TensorDict]:
        ctxs, ctxs_info = self.env.get_ctxs(self.n_queries)  # (batch_size, n_covs)
        # (batch_size, n_acts_avail, n_act_feats), (batch_size, n_acts_avail, n_act_feats)
        xacts, acts = self.env.get_avail_actions(ctxs)
        batch_size = ctxs.shape[0]
        n_acts_avail = xacts.shape[1]
        ctxs_expanded = ctxs[:, None, :].expand(
            -1, n_acts_avail, -1
        )  # (batch_size, n_acts_avail, n_covs)
        inputs = th.cat((ctxs_expanded, xacts), dim=2).to(
            device=plf.device
        )  # (batch_size, n_acts_avail, n_covs + n_act_feats)
        flattened_inputs = inputs.reshape(
            -1, inputs.shape[-1]
        )  # (batch_size * n_acts_avail, n_covs + n_act_feats)
        outs = reward_est(flattened_inputs)
        means = reward_est.get_posterior_mean(outs).reshape(
            batch_size, n_acts_avail
        )  # (batch_size, n_acts_avail)
        vars_ = (
            reward_est.get_posterior_covariance(outs)
            .diagonal()
            .reshape(batch_size, n_acts_avail)
        )  # (batch_size, n_acts_avail)
        vars_ = vars_ + 1e-5
        best_acts = th.empty(batch_size, dtype=th.long, device=plf.device)
        best_improvements = th.empty(batch_size, device=plf.device)
        for i in range(batch_size):
            sigmas = th.sqrt(vars_[i])  # (n_acts_avail,)
            batch_means = means[i]  # (n_acts_avail,)
            # Calculate knowledge gradient
            improvements = self.knowledge_gradient(
                batch_means, sigmas
            )  # Returns tensor of shape (n_acts_avail,)
            # Store best action and its improvement
            best_act_idx = th.argmax(improvements)
            best_improvements[i] = improvements[int(best_act_idx.item())]
            best_acts[i] = best_act_idx.item()
        # Get selected actions using gather (still on GPU)
        best_acts_for_gather = best_acts[:, None, None].expand(-1, -1, xacts.shape[2])
        selected_acts = th.gather(
            acts.to(plf.device), dim=1, index=best_acts_for_gather
        )[:, 0]
        # Move results back to CPU for dataset operations
        ctxs_cpu = ctxs
        selected_acts_cpu = selected_acts.cpu()
        best_improvements_cpu = best_improvements.cpu()
        return ctxs_cpu, selected_acts_cpu, best_improvements_cpu, ctxs_info
