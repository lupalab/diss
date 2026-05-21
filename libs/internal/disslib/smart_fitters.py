from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import lightning as pl
import mymodels
import numpy as np
import torch as th
import torch.distributions.utils
import xgboost as xgbst

from . import estimators

if TYPE_CHECKING:
    from .strategies import OptStrat

EstT = TypeVar("EstT", bound=estimators.RewardEst)
OutputT = TypeVar("OutputT")


class SmartFitter(th.nn.Module, Generic[EstT, OutputT], ABC):
    reward_est: EstT

    @property
    def train_inputs(self) -> th.Tensor:
        return self.reward_est.train_inputs

    @property
    def train_targets(self) -> th.Tensor:
        return self.reward_est.train_targets

    @property
    def ys(self) -> th.Tensor:
        return self.reward_est.ys

    @property
    def pyhats(self) -> th.Tensor:
        return self.reward_est.pyhats

    def __init__(self, reward_est: EstT) -> None:
        super().__init__()
        self.reward_est = reward_est

    @abstractmethod
    def is_compatible(self, strat: OptStrat) -> bool: ...

    @abstractmethod
    def forward(self, inputs: th.Tensor) -> OutputT:
        pass

    @abstractmethod
    def get_posterior_mean(self, forward_outs: OutputT) -> th.Tensor: ...

    @abstractmethod
    def get_posterior_std(self, forward_outs: OutputT) -> th.Tensor: ...

    @abstractmethod
    def get_posterior_covariance(self, forward_outs: OutputT) -> th.Tensor: ...

    @abstractmethod
    def sample_posterior(
        self, forward_outs: OutputT, sample_shape: th.Size = th.Size()
    ) -> th.Tensor: ...

    @abstractmethod
    def smart_fit_(self, plf: pl.Fabric) -> dict[str, float]:
        pass

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        return self.reward_est.fit_(plf)


class XGBoostRewardEstFitOnlyOne(SmartFitter[estimators.XGBoostRewardEst, th.Tensor]):
    def __init__(self, reward_est: estimators.XGBoostRewardEst) -> None:
        super().__init__(reward_est)
        self._m = xgbst.XGBRegressor(**reward_est.xgbr_kwargs)

    def is_compatible(self, strat: OptStrat) -> bool:
        from . import strategies

        return isinstance(
            strat, (strategies.RandomOptStrat, strategies.TS, strategies.MTSPM)
        )

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        _inputs: np.ndarray = inputs.numpy(force=True)
        outs_l: list[th.Tensor] = [
            th.as_tensor(self._m.predict(_inputs), device=self.reward_est.device)
        ]
        outs: th.Tensor = th.stack(outs_l, dim=1)
        return outs

    def get_posterior_mean(self, forward_outs: th.Tensor) -> th.Tensor:
        return forward_outs[:, 0]

    def get_posterior_std(self, forward_outs: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def get_posterior_covariance(self, forward_outs: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def sample_posterior(
        self, forward_outs: th.Tensor, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        if len(sample_shape) == 0:
            return forward_outs[:, 0]
        orig_shape = forward_outs[:, 0].shape
        outs: th.Tensor = forward_outs[:, 0]
        for _ in sample_shape:
            outs = outs[None]
        outs = outs.expand((*sample_shape, *orig_shape))
        return outs

    def smart_fit_(self, plf: pl.Fabric) -> dict[str, float]:
        rsquares_l: list[float] = list()
        mses_l: list[float] = list()
        n_data: int = math.ceil(
            len(self.reward_est.train_targets)
            * self.reward_est.fraction_training_data_per_split
        )
        idxs: th.Tensor = th.randint(
            0,
            len(self.reward_est.train_targets),
            (n_data,),
            dtype=th.long,
            generator=self.reward_est._rg,
        )
        xs: th.Tensor | np.ndarray = self.reward_est.train_inputs[idxs,]
        ys: th.Tensor | np.ndarray = self.reward_est.train_targets[idxs]
        if (
            self.reward_est.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(xs.storage())
            < th.cuda.get_device_properties(self.reward_est.device).total_memory
        ):
            self._m.set_params(device=str(self.reward_est.device))
            xs = mymodels.utils.to_cp_or_np(xs.to(device=self.reward_est.device))
            ys = mymodels.utils.to_cp_or_np(ys.to(device=self.reward_est.device))
        else:
            self._m.set_params(device="cpu")
            xs = xs.numpy(force=True)
            ys = ys.numpy(force=True)
        self._m.fit(xs, ys)
        train_inputs: np.ndarray = self.reward_est.train_inputs.numpy(force=True)
        train_targets: np.ndarray = self.reward_est.train_targets.numpy(force=True)
        rsquares_l.append(self._m.score(train_inputs, train_targets))
        mses_l.append(
            th.nn.functional.mse_loss(
                th.as_tensor(self._m.predict(train_inputs), dtype=th.float32),
                self.reward_est.train_targets,
            ).item()
        )
        metrics = {
            "est_rsquared": th.mean(th.as_tensor(rsquares_l)).item(),
            "est_mse": th.mean(th.as_tensor(mses_l)).item(),
        }
        return metrics


class StructureRewardEstFitOnlyOne(
    SmartFitter[estimators.StructureRewardEstBase, th.Tensor]
):
    _base_xgbc: xgbst.XGBClassifier
    _mimic_xgbc: xgbst.XGBClassifier

    def __init__(self, reward_est: estimators.StructureRewardEstBase) -> None:
        super().__init__(reward_est)
        self._base_xgbc = xgbst.XGBClassifier(**reward_est.base_xgbc_kwargs)
        self._mimic_xgbc = xgbst.XGBClassifier(**reward_est.mimic_xgbc_kwargs)

    def is_compatible(self, strat: OptStrat) -> bool:
        from . import strategies

        return isinstance(
            strat, (strategies.RandomOptStrat, strategies.TS, strategies.MTSPM)
        )

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        inputs = inputs.to(self.reward_est.device)
        mimic_outs_l, base_outs_l = self._forward(inputs)
        outs: th.Tensor = th.stack(
            [
                -th.nn.functional.cross_entropy(
                    torch.distributions.utils.probs_to_logits(mimic_outs_l[0]),
                    base_outs_l[0],
                    reduction="none",
                )
            ],
            dim=1,
        ).to(device=inputs.device)
        # TODO use decompose inputs
        # _, xacts = th.chunk(inputs, chunks=2, dim=1)
        _, cinds, _ = self.reward_est.decompose_inputs(inputs)
        outs = outs - (self.reward_est.alpha * th.sum(cinds, dim=1))[:, None]
        return outs

    def get_posterior_mean(self, forward_outs: th.Tensor) -> th.Tensor:
        return self.reward_est.get_posterior_mean(forward_outs)

    def get_posterior_covariance(self, forward_outs: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def get_posterior_std(self, forward_outs: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def sample_posterior(
        self, forward_outs: th.Tensor, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        if len(sample_shape) == 0:
            return forward_outs[:, 0]
        orig_shape = forward_outs[:, 0].shape
        outs: th.Tensor = forward_outs[:, 0]
        for _ in sample_shape:
            outs = outs[None]
        outs = outs.expand((*sample_shape, *orig_shape))
        return outs

    def _forward(self, inputs: th.Tensor) -> tuple[list[th.Tensor], list[th.Tensor]]:
        mimic_outs_l: list[th.Tensor] = self._forward_mimic(inputs)
        base_outs_l: list[th.Tensor] = self._forward_base(inputs)
        return mimic_outs_l, base_outs_l

    def _forward_mimic(self, inputs: th.Tensor) -> list[th.Tensor]:
        # TODO use decompose inputs
        # xs, inds = th.chunk(inputs, 2, dim=1)
        # inputs_: th.Tensor = th.cat((xs * inds, inds), dim=1)
        xs, cinds, exinds = self.reward_est.decompose_inputs(inputs)
        inputs_: th.Tensor = (
            th.cat((xs * cinds, cinds), dim=1)
            if self.reward_est.n_bdms_per_fcomb == 1
            else th.cat((xs * cinds, cinds, exinds), dim=1)
        )
        inputs_n: np.ndarray = inputs_.numpy(force=True)
        mimic_outs_l: list[th.Tensor] = [
            th.as_tensor(
                self._mimic_xgbc.predict_proba(inputs_n),
                dtype=th.float32,
                device=self.reward_est.device,
            )
        ]
        return mimic_outs_l

    def _forward_base(self, inputs: th.Tensor) -> list[th.Tensor]:
        # TODO use decompose inputs
        # inputs_n: np.ndarray = th.chunk(inputs, 2, dim=1)[0].numpy(force=True)
        ctxs_n: np.ndarray = self.reward_est.decompose_inputs(inputs)[0].numpy(
            force=True
        )
        base_outs_l: list[th.Tensor] = [
            th.as_tensor(
                self._base_xgbc.predict_proba(ctxs_n),
                dtype=th.float32,
                device=self.reward_est.device,
            )
        ]
        return base_outs_l

    def smart_fit_(self, plf: pl.Fabric) -> dict[str, float]:
        self._smart_fit_base()
        self._smart_fit_mimic()
        self.eval()
        mse_loss: th.Tensor
        with th.no_grad():
            mse_loss = th.nn.functional.mse_loss(
                self.get_posterior_mean(self(self.reward_est.train_inputs)),
                self.reward_est.train_targets.to(device=self.reward_est.device),
            )
        metrics_d: dict[str, float] = {"est_mse": mse_loss.item()}
        return metrics_d

    def _smart_fit_base(self):
        assert self.reward_est.xs_etrain is not None
        assert self.reward_est.ys_etrain is not None
        # TODO use decompose inputs
        # ctxs, _ = th.chunk(self.train_inputs, chunks=2, dim=1)
        ctxs: th.Tensor = self.reward_est.decompose_inputs(
            self.reward_est.train_inputs
        )[0]
        # get rid of repeat elements
        tmp_t: th.Tensor = th.unique(
            th.cat((ctxs, self.reward_est.ys[:, None]), dim=1), dim=0
        )
        xs: th.Tensor | np.ndarray = th.cat(
            (tmp_t[:, :-1], self.reward_est.xs_etrain), dim=0
        )
        ys: th.Tensor | np.ndarray = th.cat(
            (tmp_t[:, -1].to(dtype=th.long), self.reward_est.ys_etrain), dim=0
        )
        # with repeated elements in the set
        # xs: th.Tensor = th.cat((ctxs, self.xs_etrain), dim=0)
        # ys: th.Tensor = th.cat((self.ys, self.ys_etrain), dim=0)
        n_data: int = math.ceil(
            len(xs) * self.reward_est.fraction_training_data_per_split
        )
        idxs: th.Tensor = th.randint(
            0, len(xs), (n_data,), dtype=th.long, generator=self.reward_est._rg
        )
        _xs: np.ndarray
        _ys: np.ndarray
        if (
            self.reward_est.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(xs.storage())
            < th.cuda.get_device_properties(self.reward_est.device).total_memory
        ):
            self._base_xgbc.set_params(device=str(self.reward_est.device))
            _xs = mymodels.utils.to_cp_or_np(xs[idxs].to(device=self.reward_est.device))
            _ys = mymodels.utils.to_cp_or_np(ys[idxs].to(device=self.reward_est.device))
        else:
            self._base_xgbc.set_params(device="cpu")
            _xs = xs[idxs].numpy(force=True)
            _ys = ys[idxs].numpy(force=True)
        self._base_xgbc.fit(_xs, _ys)
        return

    def _smart_fit_mimic(self):
        n_labels: int = self.reward_est.pyhats.shape[1]
        n_data: int = math.ceil(
            len(self.reward_est.train_targets)
            * self.reward_est.fraction_training_data_per_split
        )
        idxs: th.Tensor = th.randint(
            0,
            len(self.reward_est.train_targets),
            (n_data,),
            dtype=th.long,
            generator=self.reward_est._rg,
        )
        # TODO decompose inputs
        # _xs: th.Tensor = self.train_inputs[idxs][:, None, :].expand(
        #     -1, n_labels, -1
        # )
        # _ctxs, _xacts = th.chunk(_xs, chunks=2, dim=2)
        _tctxs, _tcinds, _texinds = self.reward_est.decompose_inputs(
            self.reward_est.train_inputs[idxs]
        )
        _xs: th.Tensor | np.ndarray = (
            th.cat((_tctxs * _tcinds, _tcinds), dim=1)
            if self.reward_est.n_bdms_per_fcomb == 1
            else th.cat((_tctxs * _tcinds, _tcinds, _texinds), dim=1)
        )
        _xs = _xs[:, None, :].expand(-1, n_labels, -1)
        _ys: th.Tensor | np.ndarray = th.arange(0, n_labels, dtype=th.long)
        _ys = _ys[None, :].expand(n_data, -1)
        _xs = _xs.flatten(0, 1)
        _ys = _ys.flatten(0, 1)
        _weight = self.reward_est.pyhats[idxs].flatten(0, 1)
        if (
            self.reward_est.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(_xs.storage())
            < th.cuda.get_device_properties(self.reward_est.device).total_memory
        ):
            self._mimic_xgbc.set_params(device=str(self.reward_est.device))
            _xs = mymodels.utils.to_cp_or_np(_xs.to(device=self.reward_est.device))
            _ys = mymodels.utils.to_cp_or_np(_ys.to(device=self.reward_est.device))
            _weight = mymodels.utils.to_cp_or_np(
                _weight.to(device=self.reward_est.device)
            )
        else:
            self._mimic_xgbc.set_params(device="cpu")
            _xs = _xs.numpy(force=True)
            _ys = _ys.numpy(force=True)
            _weight = _weight.numpy(force=True)
        self._mimic_xgbc.fit(_xs, _ys, sample_weight=_weight)
        return
