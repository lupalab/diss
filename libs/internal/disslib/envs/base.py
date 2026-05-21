from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Final, Optional, Self, Sequence

import mymodels
import numpy as np
import tensordict as thd
import torch as th
import torch.distributions.utils

from . import utils


class Env(ABC):
    n_covs: int

    _acts_avail: th.Tensor | None

    def __init__(
        self,
        n_covs: int,
    ) -> None:
        super().__init__()
        self.n_covs = n_covs
        self._acts_avail = None

    def get_reward_log_keys(self) -> Sequence[str]:
        return tuple()

    def get_init_avail_actions(
        self, ctxs: Optional[th.Tensor], generator: Optional[np.random.Generator]
    ) -> tuple[th.Tensor, th.Tensor]:
        """get initial available actions

        This method return "`n_acts_avail`" not the entire action space; indices to the first or second dimension might not return the actual action!

        Args:
            ctxs (Optional[th.Tensor], optional): (bsz, n_covs) contexts of interest; if none, just return action space
            generator (Optional[th.Tensor, optional]): numpy random generator use to genreate initial available actions.

        Returns:
            th.Tensor: (bsz, n_acts_avail, n_act_feats) action features; if ctxs is none, return tensor has shape (n_acts_avail, n_act_feats)
            th.Tensor: (bsz, n_acts_avail, n_act_feats) actions; if ctxs is none, return tensor has shape (n_acts_avail, n_act_feats).
        """
        return self.get_avail_actions(ctxs)

    def set_avail_actions(self, acts_avail: th.Tensor) -> Self:
        self._acts_avail = acts_avail
        return self

    @abstractmethod
    def get_avail_actions(
        self, ctxs: Optional[th.Tensor] = None
    ) -> tuple[th.Tensor, th.Tensor]:
        """get available actions

        This method return "`n_acts_avail`" not the entire action space; indices to the first or second dimension might not return the actual action!

        Args:
            ctxs (Optional[th.Tensor], optional): (bsz, n_covs) contexts of interest; if none, just return action space

        Returns:
            th.Tensor: (bsz, n_acts_avail, n_act_feats) action features; if ctxs is none, return tensor has shape (n_acts_avail, n_act_feats)
            th.Tensor: (bsz, n_acts_avail, n_act_feats) actions; if ctxs is none, return tensor has shape (n_acts_avail, n_act_feats).
        """

    @abstractmethod
    def decompose_acts(self, acts: th.Tensor) -> thd.TensorDict:
        """decompose action features.

        Args:
            xacts (th.Tensor): (bsz, n_act_feats)

        Returns:
            thd.TensorDict: (bsz, ) must include `"fcinds"` of type `th.Tensor` and shape `(bsz, n_covs)`
        """

    def get_init_ctxs(
        self, n_ctxs: int, generator: Optional[np.random.Generator]
    ) -> tuple[th.Tensor, thd.TensorDict]:
        """get initial contexts

        Args:
            n_ctxs (int): number of contexts to sample from the environment
            generator (Optional[np.random.Generator], optional): numpy random generator use to genreate initial context.

        Returns:
            th.Tensor: (n_ctxs, n_covs) contexts
            thd.TensorDict: environment specific TensorDict
        """
        return self.get_ctxs(n_ctxs)

    @abstractmethod
    def get_ctxs(self, n_ctxs: int) -> tuple[th.Tensor, thd.TensorDict]:
        """get contexts

        Args:
            n_ctxs (int): number of contexts to sample from the environment

        Returns:
            th.Tensor: (n_ctxs, n_covs) contexts
            thd.TensorDict: environment specific TensorDict
        """

    @abstractmethod
    def compute_rewards(
        self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
    ) -> tuple[th.Tensor, thd.TensorDict]:
        """compute reward of given ctx-action pair

        Args:
            ctxs (th.Tensor): (n_ctxs, n_covs) contexts
            acts (th.Tensor): (n_ctxs, n_act_feats) action to be taken
            infos (thd.TensorDict): (n_ctxs, ) additional information for computing rewards

        Returns:
            th.Tensor: (n_ctxs, ) reward of the given context taking selected action.
            thd.TensorDict: additional information
        """

    @abstractmethod
    def compute_optimal_rewards(self, ctxs: th.Tensor) -> th.Tensor:
        """compute optimal rewards for the given context

        Args:
            ctxs (th.Tensor): (n_ctxs, n_covs) contexts

        Returns:
            th.Tensor: (n_ctxs, ) rewards taking optimal actions
        """

    @abstractmethod
    def has_next(self) -> bool:
        """whether the environment has next contexts to evaluate

        Returns:
            bool: if there are still some contexts left to be evaluated
        """

    def reset(self) -> None:
        """reset the environment"""
        return


class _TensorDictEnvBase(Env, th.nn.Module, ABC):
    n_bdms_per_fcomb: int
    alpha: float

    bdm: mymodels.classifiers.SubsetFeatureClassifier
    is_train: Final[bool]

    data: thd.TensorDict
    min_features: int
    max_features: int
    n_total_acts: int

    _bincount_fcombs_l: list[int]

    _n_loaded: int

    _idxs: list[int] | None

    def __init__(
        self,
        data: thd.TensorDict,
        bdm: mymodels.classifiers.SubsetFeatureClassifier,
        alpha: float,
        is_train: bool,
        n_acts_avail: Optional[int],
        min_features: int,
        max_features: int | None,
    ) -> None:
        super().__init__(
            n_covs=data["xs"].shape[1],
        )
        self.n_bdms_per_fcomb = bdm.n_bdms_per_fcomb
        self.alpha = alpha
        self.data = data
        self.bdm = bdm
        self.is_train = is_train
        self._ctx_to_idx = {tuple(_x.tolist()): _i for _i, _x in enumerate(data["xs"])}
        self._n_loaded = 0
        self._idxs = None
        # if maximum available action is not configured, use all features
        # maximum available action shall not exceed len(self.act_to_comb)
        self.min_features = min_features
        self.max_features = self.n_covs if max_features is None else max_features
        self._bincount_fcombs_l = [
            math.comb(self.n_covs, i)
            for i in range(self.min_features, self.max_features + 1)
        ]
        self.n_total_acts = sum(self._bincount_fcombs_l)
        self.n_acts_avail = (
            self.n_total_acts
            if n_acts_avail is None or n_acts_avail > self.n_total_acts
            else n_acts_avail
        )

    def get_init_avail_actions(
        self, ctxs: th.Tensor | None, generator: np.random.Generator | None
    ) -> tuple[th.Tensor, th.Tensor]:
        if self._acts_avail is not None:
            xacts: th.Tensor = self._acts_avail.to(dtype=th.float32)
            acts: th.Tensor = self._acts_avail.to(dtype=th.long)
            if ctxs is not None:
                n: int = ctxs.shape[0]
                xacts = xacts[None, :, :].expand(n, -1, -1)
                acts = acts[None, :, :].expand(n, -1, -1)
            return xacts, acts
        # entire action space is always avaialable; no need to sample subset of action space
        if self.n_acts_avail == self.n_total_acts:
            return self.get_avail_actions(ctxs)
        # need to sample subset of action with fixed random generator
        if ctxs is None:
            # sample subset of action
            xacts, acts = self._sample_subset_of_actions(1, generator)
            return xacts[0], acts[0]
        xacts, acts = self._sample_subset_of_actions(len(ctxs), generator)
        return xacts, acts

    def get_avail_actions(
        self, ctxs: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor]:
        if self._acts_avail is not None:
            xacts: th.Tensor = self._acts_avail.to(dtype=th.float32)
            acts: th.Tensor = self._acts_avail.to(dtype=th.long)
            if ctxs is not None:
                n: int = ctxs.shape[0]
                xacts = xacts[None, :, :].expand(n, -1, -1)
                acts = acts[None, :, :].expand(n, -1, -1)
            return xacts, acts
        if self.n_acts_avail == self.n_total_acts:
            xacts: th.Tensor = utils.make_full_action_features(
                n_covs=self.n_covs,
                min_features=self.min_features,
                max_features=self.max_features,
                n_bdms_per_fcomb=self.n_bdms_per_fcomb,
            )[1]
            acts: th.Tensor = xacts.to(dtype=th.long)
            if ctxs is not None:
                n: int = ctxs.shape[0]
                xacts = xacts[None, :, :].expand(n, -1, -1)
                acts = acts[None, :, :].expand(n, -1, -1)
            return xacts, acts
        # need to sample subset of action
        if ctxs is None:
            # sample subset of action
            xacts, acts = self._sample_subset_of_actions(1)
            return xacts[0], acts[0]
        xacts, acts = self._sample_subset_of_actions(len(ctxs))
        return xacts, acts

    def decompose_acts(self, acts: th.Tensor) -> thd.TensorDict:
        if self.n_bdms_per_fcomb == 1:
            # feature indicator vector
            fcinds: th.Tensor = acts.to(dtype=th.long)
            # expert indicator vectors
            exinds: th.Tensor = th.zeros(
                (len(acts),), dtype=th.long, device=acts.device
            )
            return thd.make_tensordict(
                {"fcinds": fcinds, "exinds": exinds}
            ).auto_batch_size_(1)
        fcinds: th.Tensor = acts[:, : self.n_covs].to(dtype=th.long)
        exinds: th.Tensor = acts[:, self.n_covs :].to(dtype=th.long)
        return thd.make_tensordict(
            {"fcinds": fcinds, "exinds": exinds}
        ).auto_batch_size_(1)

    # def compute_rewards(
    #     self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
    # ) -> tuple[th.Tensor, thd.TensorDict]:
    #     assert self._idxs is not None
    #     ctxs, ys = self._get_verify_ctxs_ys(ctxs)
    #     pyhats: th.Tensor = self.bdm.predict_proba(ctxs, acts)
    #     cels: th.Tensor = th.nn.functional.cross_entropy(
    #         torch.distributions.utils.probs_to_logits(pyhats), ys, reduction="none"
    #     )
    #     # n_labels: int = pyhats.shape[1]
    #     # rewards: th.Tensor = (
    #     #     -th.nn.functional.cross_entropy(
    #     #         torch.distributions.utils.probs_to_logits(pyhats), ys, reduction="none"
    #     #     )
    #     #     if n_labels > 2
    #     #     else self._custom_bce(pyhats, ys)
    #     # )
    #     fcinds: th.Tensor = self.decompose_acts(acts)["fcinds"]
    #     rewards = -cels - self.alpha * th.sum(fcinds, dim=1)
    #     info = thd.TensorDict(
    #         {
    #             "pyhats": pyhats,
    #             "ys": ys,
    #             "cels": cels,
    #         }
    #     ).auto_batch_size_(1)
    #     return rewards, info

    # def _custom_bce(self, pyhats: th.Tensor, ys: th.Tensor) -> th.Tensor:
    #     pyhats = th.clip(pyhats, 1e-6, 1.0 - 1e-6)
    #     assert th.all(th.greater(pyhats, 0.0))
    #     assert th.all(th.less(pyhats, 1.0))
    #     llike: th.Tensor = th.where(
    #         ys.to(dtype=th.bool), th.log(pyhats[:, 1]), th.log(1 - pyhats[:, 1])
    #     )
    #     return llike

    def compute_optimal_rewards(self, ctxs: th.Tensor) -> th.Tensor:
        return th.inf * th.ones((len(ctxs),), device=ctxs.device)

    def get_init_ctxs(
        self, n_ctxs: int, generator: Optional[np.random.Generator]
    ) -> tuple[th.Tensor, thd.TensorDict]:
        return self._get_ctxs_train(n_ctxs, generator)

    def get_ctxs(self, n_ctxs: int) -> tuple[th.Tensor, thd.TensorDict]:
        if self.is_train:
            return self._get_ctxs_train(n_ctxs)
        return self._get_ctxs_eval(n_ctxs)

    def has_next(self) -> bool:
        if self.is_train:
            return self._has_next_train()
        return self._has_next_eval()

    def reset(self) -> None:
        if self.is_train:
            self._reset_train()
        else:
            self._reset_eval()
        super().reset()

    # train
    def _get_ctxs_train(
        self, n_ctxs: int, generator: Optional[np.random.Generator] = None
    ) -> tuple[th.Tensor, thd.TensorDict]:
        self._n_loaded = self._n_loaded + n_ctxs
        self._idxs = (
            generator.integers(0, len(self.data), (n_ctxs,), dtype=np.int64).tolist()
            if generator is not None
            else th.randint(0, len(self.data), (n_ctxs,), dtype=th.long).tolist()
        )
        assert self._idxs is not None
        ctxs: th.Tensor = self.data[self._idxs]["xs"]
        # ctxs: th.Tensor = th.stack([self.data[i][0] for i in self._idxs])
        infos: thd.TensorDict = self.data[self._idxs]
        return ctxs, infos

    def _has_next_train(self) -> bool:
        return self._n_loaded < len(self.data)

    def _reset_train(self) -> None:
        self._n_loaded = 0
        self._idxs = None

    # eval
    def _get_ctxs_eval(self, n_ctxs: int) -> tuple[th.Tensor, thd.TensorDict]:
        start_idx: int = self._end_idx
        end_idx: int = min(start_idx + n_ctxs, len(self.data))
        # ctxs = th.stack([self._dataset[i][0] for i in range(start_idx, end_idx)])
        self._idxs = list(range(start_idx, end_idx))
        ctxs: th.Tensor = self.data[self._idxs]["xs"]
        # ctxs = th.stack([self._dataset[i][0] for i in self._idxs])
        self._end_idx = end_idx
        infos: thd.TensorDict = self.data[self._idxs]
        return ctxs, infos

    def _has_next_eval(self) -> bool:
        return self._end_idx < len(self.data)

    def _reset_eval(self) -> None:
        self._end_idx = 0
        self._idxs = None

    # protected
    def _sample_subset_of_actions(
        self, bsz: int, generator: Optional[np.random.Generator] = None
    ) -> tuple[th.Tensor, th.Tensor]:
        """Generate a subset of actions

        precondition: self.n_acts_avail should never be greater than or equal to self._n_total_acts

        Args:
            bsz (int): the batch size
            generator (Optional[np.random.Generator], optional): random generator used to generate teh mask. Defaults to None.

        Returns:
            th.Tensor: (bsz, n_acts_avail, n_act_features) the actions features
            th.Tensor: (bsz, n_acts_avail, n_act_features) the actions
        """
        # (bsz, n_acts_avail,)
        # if acts_full is available, just sample from it
        assert not self.n_acts_avail >= self.n_total_acts
        if self.n_bdms_per_fcomb == 1:
            return self._runtime_sample_subset_of_actions_single_expert(bsz, generator)
        return self._runtime_sample_subset_of_actions_multi_expert(bsz, generator)

    def _runtime_sample_subset_of_actions_single_expert(
        self, bsz: int, generator: Optional[np.random.Generator]
    ) -> tuple[th.Tensor, th.Tensor]:
        assert not self.n_acts_avail >= self.n_total_acts
        generator = np.random.default_rng() if generator is None else generator
        # generate current batch of action on the fly
        # for each bin, figure out how many action i'm gonna draw from each n_covs choose 'k'
        bincount_fcombs: th.Tensor = th.as_tensor(
            self._bincount_fcombs_l, dtype=th.long
        )
        ps: th.Tensor = bincount_fcombs / th.sum(bincount_fcombs).to(dtype=th.float64)
        nfcomb_from_each_binned_fcombs: th.Tensor = th.as_tensor(
            generator.multinomial(n=self.n_acts_avail, pvals=ps.numpy(force=True)),
            dtype=th.long,
        )
        # in case number of actions in any of the bin exceeds maximum number of actions
        _curr_bincounts: th.Tensor = nfcomb_from_each_binned_fcombs
        while th.any(_curr_bincounts > bincount_fcombs):
            _tmp_ps: th.Tensor = th.where(
                _curr_bincounts >= bincount_fcombs, 0, bincount_fcombs - _curr_bincounts
            ).to(dtype=th.float64)
            _tmp_ps = _tmp_ps / th.sum(_tmp_ps)
            _realloc_counts: th.Tensor = th.where(
                _curr_bincounts > bincount_fcombs, _curr_bincounts - bincount_fcombs, 0
            )
            _tmp_bincounts: th.Tensor = th.as_tensor(
                generator.multinomial(
                    n=int(th.sum(_realloc_counts).item()),
                    pvals=_tmp_ps.numpy(force=True),
                ),
                dtype=th.long,
            )
            _curr_bincounts = _curr_bincounts - _realloc_counts + _tmp_bincounts
        nfcomb_from_each_binned_fcombs = _curr_bincounts
        # make unique feature combination
        fcomb_sets_by_bins: list[set[tuple[int, ...]]] = [
            set() for _ in bincount_fcombs
        ]
        for _k, (_count, _fcomb_set) in enumerate(
            zip(nfcomb_from_each_binned_fcombs, fcomb_sets_by_bins)
        ):
            if _count == 0:
                continue
            while len(_fcomb_set) < _count:
                _fcomb: tuple[int, ...] = tuple(
                    sorted(
                        generator.choice(
                            self.n_covs, size=(_k + 1,), replace=False
                        ).tolist()
                    )
                )
                if _fcomb not in _fcomb_set:
                    _fcomb_set.add(_fcomb)
        # from fcomb to act
        acts: th.Tensor = th.zeros((self.n_acts_avail, self.n_covs), dtype=th.long)
        fcombs_l: list[tuple[int, ...]] = [
            _fc for _fcs in fcomb_sets_by_bins for _fc in _fcs
        ]
        assert len(acts) == len(fcombs_l)
        for _i, _fcomb in enumerate(fcombs_l):
            acts[_i, _fcomb] = 1
        # expand with batch size
        acts = acts[None, :, :].expand(bsz, -1, -1)
        return acts.to(dtype=th.float32), acts

    def _runtime_sample_subset_of_actions_multi_expert(
        self, bsz: int, generator: Optional[np.random.Generator]
    ) -> tuple[th.Tensor, th.Tensor]:
        assert not self.n_acts_avail >= self.n_total_acts
        generator = np.random.default_rng() if generator is None else generator
        # generate current batch of action on the fly
        # for each bin, figure out how many action i'm gonna draw from each n_covs choose 'k'
        bincount_fcombs: th.Tensor = th.as_tensor([self._bincount_fcombs_l]).expand(
            self.n_bdms_per_fcomb, -1
        )
        ps: th.Tensor = bincount_fcombs.to(dtype=th.float64)
        ps: th.Tensor = bincount_fcombs.flatten() / th.sum(bincount_fcombs)
        # nfcomb_from_each_binned_fcombs: th.Tensor = th.bincount(
        #     th.multinomial(ps, self.n_acts_avail, replacement=True),
        #     minlength=len(bincount_fcombs),
        # )
        nfcomb_from_each_binned_fcombs: th.Tensor
        try:
            nfcomb_from_each_binned_fcombs = th.as_tensor(
                generator.multinomial(n=self.n_acts_avail, pvals=ps.numpy(force=True)),
                dtype=th.long,
            ).unflatten(0, (self.n_bdms_per_fcomb, -1))
        except Exception:
            nfcomb_from_each_binned_fcombs = th.bincount(
                th.multinomial(
                    input=ps, num_samples=self.n_acts_avail, replacement=True
                ),
                minlength=len(ps),
            ).unflatten(0, (self.n_bdms_per_fcomb, -1))
        # in case number of actions in any of the bin exceeds maximum number of actions
        for _eid, (_bincount_fcombs, _curr_bincounts) in enumerate(
            zip(bincount_fcombs, nfcomb_from_each_binned_fcombs)
        ):
            while th.any(_curr_bincounts > _bincount_fcombs):
                _tmp_ps: th.Tensor = th.where(
                    _curr_bincounts >= _bincount_fcombs,
                    0,
                    _bincount_fcombs - _curr_bincounts,
                ).to(dtype=th.float64)
                _tmp_ps = _tmp_ps / th.sum(_tmp_ps)
                _realloc_counts: th.Tensor = th.where(
                    _curr_bincounts > _bincount_fcombs,
                    _curr_bincounts - _bincount_fcombs,
                    0,
                )
                _tmp_bincounts: th.Tensor = th.as_tensor(
                    generator.multinomial(
                        n=int(th.sum(_realloc_counts).item()),
                        pvals=_tmp_ps.numpy(force=True),
                    ),
                    dtype=th.long,
                )
                _curr_bincounts = _curr_bincounts - _realloc_counts + _tmp_bincounts
            nfcomb_from_each_binned_fcombs[_eid] = _curr_bincounts
        # make unique feature combination
        fcomb_sets_by_bins: list[list[set[tuple[int, ...]]]] = [
            [set() for _ in nfcomb_from_each_binned_fcombs[_eid]]
            for _eid in range(self.n_bdms_per_fcomb)
        ]
        for _eid in range(self.n_bdms_per_fcomb):
            for _k, (_count, _fcomb_set) in enumerate(
                zip(nfcomb_from_each_binned_fcombs[_eid], fcomb_sets_by_bins[_eid])
            ):
                if _count == 0:
                    continue
                while len(_fcomb_set) < _count:
                    _fcomb: tuple[int, ...] = tuple(
                        sorted(
                            generator.choice(
                                self.n_covs, size=(_k + 1,), replace=False
                            ).tolist()
                        )
                    )
                    if _fcomb not in _fcomb_set:
                        _fcomb_set.add(_fcomb)
        # from fcomb to act
        acts: th.Tensor = th.zeros(
            (self.n_acts_avail, self.n_covs + self.n_bdms_per_fcomb), dtype=th.long
        )
        start_idx: int = 0
        for _eid in range(self.n_bdms_per_fcomb):
            fcombs_l: list[tuple[int, ...]] = [
                _fc for _fcs in fcomb_sets_by_bins[_eid] for _fc in _fcs
            ]
            for _i, _fcomb in enumerate(fcombs_l):
                acts[start_idx + _i, _fcomb] = 1
                acts[start_idx + _i, self.n_covs :][_eid] = 1
            start_idx = start_idx + len(fcombs_l)
        # expand with batch size
        acts = acts[None, :, :].expand(bsz, -1, -1)
        return acts.to(dtype=th.float32), acts

    def _get_verify_ctxs_ys(self, ctxs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        assert self._idxs is not None
        idxs: th.Tensor = th.as_tensor(self._idxs, dtype=th.long)
        if len(idxs) != len(ctxs) or not th.allclose(
            self.data[idxs]["xs"].to(device=ctxs.device), ctxs
        ):
            idxs = th.as_tensor(
                [self._ctx_to_idx[tuple(ctx.tolist())] for ctx in ctxs], dtype=th.long
            )
        # if len(idxs) != len(ctxs) or not th.allclose(
        #     th.stack([self._dataset[i][0] for i in self._idxs]), ctxs
        # ):
        #     idxs = th.as_tensor(
        #         [self._ctx_to_idx[tuple(ctx.tolist())] for ctx in ctxs], dtype=th.long
        #     )
        ys: th.Tensor = self.data[idxs]["ys"].to(device=ctxs.device)
        # ys: th.Tensor = th.stack([self._dataset[i][2] for i in idxs])
        return ctxs, ys


class TensorDictEnv(_TensorDictEnvBase):
    def __init__(
        self,
        data: thd.TensorDict,
        bdm: mymodels.classifiers.SubsetFeatureClassifier,
        alpha: float,
        is_train: bool,
        n_acts_avail: Optional[int],
        min_features: int,
        max_features: int | None,
    ) -> None:
        super().__init__(
            data=data,
            bdm=bdm,
            alpha=alpha,
            is_train=is_train,
            n_acts_avail=n_acts_avail,
            min_features=min_features,
            max_features=max_features,
        )

    def compute_rewards(
        self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
    ) -> tuple[th.Tensor, thd.TensorDict]:
        assert self._idxs is not None
        ctxs, ys = self._get_verify_ctxs_ys(ctxs)
        pyhats: th.Tensor = self.bdm.predict_proba(ctxs, acts)
        cels: th.Tensor = th.nn.functional.cross_entropy(
            torch.distributions.utils.probs_to_logits(pyhats), ys, reduction="none"
        )
        fcinds: th.Tensor = self.decompose_acts(acts)["fcinds"]
        rewards = -cels - self.alpha * th.sum(fcinds, dim=1)
        info = thd.TensorDict(
            {
                "pyhats": pyhats,
                "ys": ys,
                "cels": cels,
            }
        ).auto_batch_size_(1)
        return rewards, info
