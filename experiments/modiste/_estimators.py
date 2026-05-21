from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Literal, Optional

import lightning as pl
import tensordict as thd
import torch as th


# TODO set train data directly save infos thd.TensorDict instances
class ModisteRewardEstBase(th.nn.Module, ABC):
    _enable_lazy_fit: bool = True

    n_ctx_covs: int
    n_bdms_per_fcomb: int

    _train_inputs: th.Tensor | None
    _train_targets: th.Tensor | None
    _ys: th.Tensor | None
    _pyhats: th.Tensor | None
    _xact_to_aidxs: defaultdict[tuple[int, ...], list[int]]

    _dummy: th.Tensor

    @property
    def device(self):
        return self._dummy.device

    @property
    def train_inputs(self) -> th.Tensor:
        assert self._train_inputs is not None
        return self._train_inputs

    @property
    def train_targets(self) -> th.Tensor:
        assert self._train_targets is not None
        return self._train_targets

    @property
    def ys(self) -> th.Tensor:
        assert self._ys is not None
        return self._ys

    @property
    def pyhats(self) -> th.Tensor:
        assert self._pyhats is not None
        return self._pyhats

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
    ) -> None:
        super().__init__()
        self.n_ctx_covs = n_ctx_covs
        self.n_bdms_per_fcomb = n_bdms_per_fcomb
        self._train_inputs = None
        self._train_targets = None
        self._ys = None
        self._pyhats = None
        self.register_buffer("_dummy", th.empty(()))
        self._xact_to_aidxs = defaultdict(list)

    def set_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[thd.TensorDict] = None,
    ):
        assert len(inputs) == len(targets.flatten())
        self._train_inputs = inputs.clone().to(device="cpu")
        self._train_targets = targets.clone().flatten().to(device="cpu")
        if infos is not None:
            self._ys = infos["ys"].clone().to(device="cpu")
            self._pyhats = infos["pyhats"].clone().to(device="cpu")
        # record indices of ctx-act pairs that have the same action
        # set xact to nidxs
        # self._xact_to_aidxs = defaultdict(list)
        # TODO use decompose inputs
        # _, xacts = th.chunk(inputs, chunks=2, dim=1)
        _, cinds, exinds = self.decompose_inputs(inputs)
        xacts: th.Tensor = (
            cinds
            if self.n_bdms_per_fcomb == 1
            else th.cat((cinds, exinds), dim=1).to(dtype=th.long)
        )
        uxacts_t, invidxs_t = th.unique(xacts, dim=0, return_inverse=True)
        for i, xact_t in enumerate(uxacts_t):
            xact: tuple[int, ...] = tuple(xact_t.tolist())
            self._xact_to_aidxs[xact].extend(
                th.argwhere(invidxs_t == i).flatten().tolist()
            )
        return

    def add_to_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[thd.TensorDict] = None,
    ):
        # add xact to nidxs
        start_idx: int = len(self.train_inputs)
        # TODO decompose inputs
        _, xacts = th.chunk(inputs, chunks=2, dim=1)
        xacts = xacts.to(dtype=th.long)
        uxacts_t, invidxs_t = th.unique(xacts, dim=0, return_inverse=True)
        for i, xact_t in enumerate(uxacts_t):
            xact: tuple[int, ...] = tuple(xact_t.tolist())
            self._xact_to_aidxs[xact].extend(
                (th.argwhere(invidxs_t == i) + start_idx).flatten().tolist()
            )
        # update train_inputs and train_targets
        self._train_inputs = th.cat((self.train_inputs, inputs.to(device="cpu")), dim=0)
        self._train_targets = th.cat(
            (self.train_targets, targets.to(device="cpu")), dim=0
        )
        if infos is not None:
            self._ys = th.cat((self.ys, infos["ys"].to(device="cpu")), dim=0)
            self._pyhats = th.cat(
                (self.pyhats, infos["pyhats"].to(device="cpu")), dim=0
            )
        return

    def decompose_inputs(
        self, inputs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.n_bdms_per_fcomb == 1:
            ctxs, cinds = th.chunk(inputs, chunks=2, dim=1)
            exinds: th.Tensor = th.empty(
                (len(ctxs)), dtype=th.float32, device=inputs.device
            )
            return ctxs, cinds, exinds
        ctxs: th.Tensor = inputs[:, : self.n_ctx_covs]
        cinds: th.Tensor = inputs[:, self.n_ctx_covs : 2 * self.n_ctx_covs]
        exinds: th.Tensor = inputs[:, 2 * self.n_ctx_covs :]
        return ctxs, cinds, exinds

    @abstractmethod
    def forward(self, inputs: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def fit_(self, plf: pl.Fabric) -> dict[str, float]: ...

    def get_extra_state(self) -> dict[str, Any]:
        extra_state: dict[str, Any] = {
            "train_inputs": self._train_inputs,
            "train_targets": self._train_targets,
            "ys": self._ys,
            "pyhats": self._pyhats,
        }
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        self._train_inputs = state["train_inputs"]
        self._train_targets = state["train_targets"]
        self._ys = state["ys"]
        self._pyhats = state["pyhats"]


class ModisteKNNRewardEst(ModisteRewardEstBase):
    n_neighbors: int
    init_strat: Literal["min", "mean"]

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        n_neighbors: int,
        init_strat: Literal["min", "mean"] = "min",
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        self.n_neighbors = n_neighbors
        self.init_strat = init_strat

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        bsz: int = len(inputs)
        train_inputs: th.Tensor = self.train_inputs.to(device=self.device)
        train_targets: th.Tensor = self.train_targets.to(device=self.device)
        # initialize outputs to be the mean
        outputs: th.Tensor = th.empty(
            (len(inputs),), dtype=th.float32, device=self.device
        )
        if self.init_strat == "mean":
            outputs.fill_(th.mean(train_targets))
        elif self.init_strat == "min":
            outputs.fill_(th.min(train_targets))
        else:
            raise ValueError("self.init_strat must be either one of min or mean.")
        # TODO
        # tctxs, _ = th.chunk(train_inputs, chunks=2, dim=1)
        # xacts = th.chunk(inputs, chunks=2, dim=1)
        tctxs, _, _ = self.decompose_inputs(train_inputs)
        ctxs, cinds, exinds = self.decompose_inputs(inputs)
        xacts: th.Tensor = (
            cinds
            if self.n_bdms_per_fcomb == 1
            else th.cat((cinds, exinds), dim=1).to(dtype=th.long)
        )
        uxacts_t, invidxs_t = th.unique(xacts, dim=0, return_inverse=True)
        xacts_l: list[tuple[int, ...]] = [tuple(xact.tolist()) for xact in uxacts_t]
        for i, xact in enumerate(xacts_l):
            if self.device.type == "cuda":
                th.cuda.empty_cache()
            # indices of inputs that have the same action
            idxs: th.Tensor = th.argwhere(invidxs_t == i).flatten()
            # indices to training inputs correspond to current xact
            aidxs_l: list[int] = self._xact_to_aidxs[xact]
            if len(aidxs_l) == 0:
                continue
            n_neighbors: int = min(self.n_neighbors, len(aidxs_l))
            aidxs: th.Tensor = th.as_tensor(aidxs_l, dtype=th.long, device=self.device)
            # compute distance of current context to training contexts
            # (len(_ctxs), 1, n_covs)
            _ctxs: th.Tensor = ctxs[idxs][:, None, :]
            # (len(_ctxs), len(aidxs), n_covs)
            _tctxs: th.Tensor = tctxs[aidxs][None, :, :].expand(len(_ctxs), -1, -1)
            # (len(_ctxs), len(aidxs))
            _, didxs_sorted = th.sort(
                th.cat(
                    [
                        th.cdist(_ctxs, _btctxs)[:, 0, :]
                        for _btctxs in th.split(_tctxs, bsz, dim=1)
                    ],
                    dim=1,
                ),
                dim=1,
            )
            # _, didxs_sorted = th.sort(th.cdist(_ctxs, _tctxs)[:, 0, :], dim=1)
            # estimated rewards
            # (len(_ctxs), )
            outputs[idxs] = th.mean(
                th.gather(
                    train_targets[aidxs][None, :].expand(len(_ctxs), -1),
                    dim=1,
                    index=didxs_sorted[:, :n_neighbors],
                ),
                dim=1,
            )
        return outputs

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        return dict()


class ModisteUnifiedKNNRewardEst(ModisteRewardEstBase):
    n_neighbors: int
    alpha: th.Tensor
    beta: th.Tensor

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        n_neighbors: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        # assert n_experts_per_act == 1
        self.n_neighbors = n_neighbors
        assert alpha > 0 and beta > 0
        self.register_buffer("alpha", th.tensor(alpha, dtype=th.float32))
        self.register_buffer("beta", th.tensor(beta, dtype=th.float32))

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        train_targets: th.Tensor = self.train_targets.to(device=self.device)
        # compute distnce among contexts and actions
        ds: th.Tensor = self._compute_distances(inputs)
        _, didxs_sorted = th.sort(ds, dim=1)
        # compute ouputs
        n_neighbors: int = min(self.n_neighbors, len(train_targets))
        outputs: th.Tensor = th.mean(
            th.gather(
                train_targets[None, :].expand(len(inputs), -1),
                dim=1,
                index=didxs_sorted[:, :n_neighbors],
            ),
            dim=1,
        )
        return outputs

    def _compute_distances(self, inputs: th.Tensor) -> th.Tensor:
        bsz: int = len(inputs)
        train_inputs: th.Tensor = self.train_inputs.to(device=self.device)
        # TODO use decompose inputs
        # ctxs, xacts = th.chunk(inputs, chunks=2, dim=1)
        ctxs, cinds, exinds = self.decompose_inputs(inputs)
        xacts: th.Tensor = (
            cinds if self.n_bdms_per_fcomb == 1 else th.cat((cinds, exinds), dim=1)
        )
        ctxs_: th.Tensor = ctxs[:, None, :]
        xacts_: th.Tensor = xacts[:, None, :]
        # TODO use decompose inputs
        # tctxs, txacts = th.chunk(train_inputs, chunks=2, dim=1)
        tctxs, tcinds, texinds = self.decompose_inputs(train_inputs)
        txacts: th.Tensor = (
            tcinds if self.n_bdms_per_fcomb == 1 else th.cat((tcinds, texinds), dim=1)
        )
        tctxs_: th.Tensor = tctxs[None, :, :].expand(len(ctxs), -1, -1)
        txacts_: th.Tensor = txacts[None, :, :].expand(len(ctxs), -1, -1)
        # (len(ctxs), len(train_inputs))
        ds_ctx: th.Tensor = th.cat(
            [
                th.cdist(ctxs_, _btctxs)[:, 0, :]
                for _btctxs in th.split(tctxs_, bsz, dim=1)
            ],
            dim=1,
        )
        # ds_ctx: th.Tensor = th.cdist(ctxs_, tctxs_)[:, 0, :]
        if self.device.type == "cuda":
            th.cuda.empty_cache()
        ds_xact: th.Tensor = th.cat(
            [
                th.cdist(xacts_, _btxacts)[:, 0, :]
                for _btxacts in th.split(txacts_, bsz, dim=1)
            ],
            dim=1,
        )
        # ds_xact: th.Tensor = th.cdist(xacts_, txacts_)[:, 0, :]
        if self.device.type == "cuda":
            th.cuda.empty_cache()
        # combine distances and identify neighbors
        ds: th.Tensor = 1 / self.alpha * ds_ctx + 1 / self.beta * ds_xact
        return ds

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        return dict()
