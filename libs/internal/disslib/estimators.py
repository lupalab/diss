from __future__ import annotations

import math
import os
import sys
import tempfile as tmpf
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Generic, Iterable, Literal, Optional, TypeVar

import dak
import gpytorch as gpth
import lab
import lightning as pl
import mylib
import mymodels
import neuralprocesses.torch as nps
import numpy as np
import sklearn.exceptions as skl_exceptions
import tensordict as thd
import torch as th
import torch.distributions.utils
import torch.utils.data as th_data
import tqdm.auto as tqdm
import xgboost as xgbst

OutputT = TypeVar("OutputT")
ParamsT = (
    Iterable[th.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, th.Tensor]]
)


class RewardEst(th.nn.Module, ABC, Generic[OutputT]):
    _enable_lazy_fit: bool = True

    n_ctx_covs: int
    n_bdms_per_fcomb: int
    _train_inputs: th.Tensor | None
    _train_targets: th.Tensor | None
    _infos: thd.TensorDict | None

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
    def infos(self) -> thd.TensorDict:
        assert self._infos is not None
        return self._infos

    @property
    def ys(self) -> th.Tensor:
        return self.infos["ys"]

    @property
    def pyhats(self) -> th.Tensor:
        return self.infos["pyhats"]

    def __init__(self, n_ctx_covs: int, n_bdms_per_fcomb: int) -> None:
        super().__init__()
        self.n_ctx_covs = n_ctx_covs
        self.n_bdms_per_fcomb = n_bdms_per_fcomb
        self._train_inputs = None
        self._train_targets = None
        self._infos = None
        self.register_buffer("_dummy", th.empty(()))

    def set_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[thd.TensorDict] = None,
    ) -> None:
        assert len(inputs) == len(targets.flatten())
        self._train_inputs = inputs.to(device="cpu").clone()
        self._train_targets = targets.to(device="cpu").clone().flatten()
        if infos is not None:
            assert len(inputs) == len(infos)
            self._infos = infos.to(device="cpu").clone()

    def add_to_train_data_(
        self,
        inputs: th.Tensor,
        targets: th.Tensor,
        infos: Optional[thd.TensorDict] = None,
    ) -> None:
        # if nothing to concatenate to
        if self._train_inputs is None or self._train_targets is None:
            self.set_train_data_(inputs, targets, infos)
            return
        # concat. to previous data
        new_inputs = th.cat((self.train_inputs, inputs.to(device="cpu")), dim=0)
        new_targets = th.cat((self.train_targets, targets.to(device="cpu")), dim=0)
        new_infos: Optional[thd.TensorDict] = None
        if infos is not None and self._infos is not None:
            new_infos = thd.cat((self.infos, infos.to(device="cpu")), dim=0)
            # new_ys = th.cat((self.ys, infos["ys"].to(device="cpu")), dim=0)
            # new_pyhats = th.cat((self.pyhats, infos["pyhats"].to(device="cpu")), dim=0)
            # infos = thd.make_tensordict(
            #     {"pyhats": new_pyhats, "ys": new_ys}
            # ).auto_batch_size_(1)
        self.set_train_data_(new_inputs, new_targets, new_infos)

    def initialize(self, xs: th.Tensor, ys: th.Tensor, *args, **kwargs):
        pass

    def decompose_inputs(
        self, inputs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """decompose inputs into ctxs, cinds, exinds

        Args:
            inputs (th.Tensor): (bsz, n_covs)

        Returns:
            th.Tensor: (bsz, n_ctx_covs) contexts
            th.Tensor: (bsz, n_ctx_covs) contexts indicator masks
            th.Tensor: (bsz, ) expert indicator masks
        """
        if self.n_bdms_per_fcomb == 1:
            ctxs, fcinds = th.chunk(inputs, chunks=2, dim=1)
            exinds: th.Tensor = th.empty(
                (len(ctxs)), dtype=th.float32, device=inputs.device
            )
            return ctxs, fcinds, exinds
        ctxs: th.Tensor = inputs[:, : self.n_ctx_covs]
        fcinds: th.Tensor = inputs[:, self.n_ctx_covs : 2 * self.n_ctx_covs]
        exinds: th.Tensor = inputs[:, 2 * self.n_ctx_covs :]
        return ctxs, fcinds, exinds

    @abstractmethod
    def forward(self, inputs: th.Tensor) -> OutputT: ...

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
    def fit_(self, plf: pl.Fabric) -> dict[str, float]: ...

    def get_extra_state(self) -> dict[str, Any]:
        extra_state: dict[str, Any] = {
            "train_inputs": self._train_inputs,
            "train_targets": self._train_targets,
            # "ys": self._ys,
            # "pyhats": self._pyhats,
            "infos": self._infos,
        }
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        self._train_inputs = state["train_inputs"]
        self._train_targets = state["train_targets"]
        # self._ys = state["ys"]
        # self._pyhats = state["pyhats"]
        self._infos = state["infos"]


class XGBoostRewardEst(RewardEst[th.Tensor]):
    fraction_training_data_per_split: float
    n_splits: int
    xgbr_kwargs: dict[str, Any]

    _models: list[xgbst.XGBRegressor]
    _rg: th.Generator

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        fraction_training_data_per_split: float,
        n_splits: int,
        xgbr_kwargs: dict[str, Any] = dict(),
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        self.fraction_training_data_per_split = fraction_training_data_per_split
        self.n_splits = n_splits
        self.xgbr_kwargs = xgbr_kwargs
        self._models = [xgbst.XGBRegressor(**xgbr_kwargs) for _ in range(n_splits)]
        self._rg = th.Generator()
        if rseed is not None:
            self._rg.manual_seed(rseed)

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        _inputs: np.ndarray
        if (
            self.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(inputs.storage())
            < th.cuda.get_device_properties(self.device).total_memory
        ):
            [m.set_params(device=str(self.device)) for m in self._models]
            _inputs = mymodels.utils.to_cp_or_np(inputs.to(device=self.device))
        else:
            [m.set_params(device="cpu") for m in self._models]
            _inputs = inputs.numpy(force=True)
        outs_l: list[th.Tensor] = [
            th.as_tensor(m.predict(_inputs), device=self.device) for m in self._models
        ]
        outs: th.Tensor = th.stack(outs_l, dim=1)
        return outs

    def get_posterior_mean(self, forward_outs: th.Tensor) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        return th.mean(forward_outs, dim=1)

    def get_posterior_std(self, forward_outs: th.Tensor) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        # (bsz)
        stds: th.Tensor = th.std(forward_outs, dim=1)
        return stds

    def get_posterior_covariance(self, forward_outs: th.Tensor) -> th.Tensor:
        assert self.n_splits > 1
        # forward_outs (bsz, n_splits)
        mus: th.Tensor = th.mean(forward_outs, dim=1, keepdim=True)
        fouts_minus_mus: th.Tensor = forward_outs - mus
        # (bsz, bsz)
        covars: th.Tensor = (1 / (self.n_splits - 1)) * (
            fouts_minus_mus @ fouts_minus_mus.T
        )
        return covars

    def sample_posterior(
        self, forward_outs: th.Tensor, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        n: int = len(forward_outs)
        n_samps: int = math.prod(sample_shape)
        idxs: th.Tensor = th.randint(
            0, len(self._models), (n, n_samps), dtype=th.long, device=self.device
        )
        outs: th.Tensor = th.gather(forward_outs, dim=1, index=idxs).reshape(
            (n, *sample_shape)
        )
        return outs

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        self.train()
        rsquares_l: list[float] = list()
        mses_l: list[float] = list()
        for m in tqdm.tqdm(
            self._models,
            desc="fit-reward_est",
            total=len(self._models),
            dynamic_ncols=True,
            leave=False,
        ):
            n_data: int = math.ceil(
                len(self.train_targets) * self.fraction_training_data_per_split
            )
            idxs: th.Tensor = th.randint(
                0, len(self.train_targets), (n_data,), dtype=th.long, generator=self._rg
            )
            xs: th.Tensor | np.ndarray = self.train_inputs[idxs,]
            ys: th.Tensor | np.ndarray = self.train_targets[idxs]
            if (
                self.device.type == "cuda"
                and xgbst.build_info()["USE_CUDA"]
                and sys.getsizeof(xs.storage())
                < th.cuda.get_device_properties(self.device).total_memory
            ):
                m.set_params(device=str(self.device))
                xs = mymodels.utils.to_cp_or_np(xs.to(device=self.device))
                ys = mymodels.utils.to_cp_or_np(ys.to(device=self.device))
            else:
                m.set_params(device="cpu")
                xs = xs.numpy(force=True)
                ys = ys.numpy(force=True)
            m.fit(xs, ys)
            train_inputs: np.ndarray = self.train_inputs.numpy(force=True)
            train_targets: np.ndarray = self.train_targets.numpy(force=True)
            rsquares_l.append(m.score(train_inputs, train_targets))
            mses_l.append(
                th.nn.functional.mse_loss(
                    th.as_tensor(m.predict(train_inputs), dtype=th.float32),
                    self.train_targets,
                ).item()
            )
        metrics = {
            "est_rsquared": th.mean(th.as_tensor(rsquares_l)).item(),
            "est_mse": th.mean(th.as_tensor(mses_l)).item(),
        }
        return metrics

    def get_extra_state(self) -> Any:
        extra_state: dict[str, Any] = super().get_extra_state()
        model_states_l: list[list[str]] = list()
        extra_state.update(
            {
                "model_states_l": model_states_l,
                "fraction_training_data_per_split": self.fraction_training_data_per_split,
                "n_splits": self.n_splits,
            }
        )
        try:
            with tmpf.TemporaryDirectory() as td:
                for i, model in enumerate(self._models):
                    p = os.path.join(td, f"m{i}.json")
                    model.save_model(p)
                    with open(p, mode="r") as f:
                        model_states: list[str] = f.readlines()
                        model_states_l.append(model_states)
        except skl_exceptions.NotFittedError:
            pass
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        self.fraction_training_data_per_split = state[
            "fraction_training_data_per_split"
        ]
        self.n_splits = state["n_splits"]
        self._models.clear()
        with tmpf.TemporaryDirectory() as td:
            for i, model_states in enumerate(state["model_states_l"]):
                p = os.path.join(td, f"m{i}.json")
                with open(p, mode="w") as f:
                    f.writelines(model_states)
                model = xgbst.XGBRegressor()
                model.load_model(p)
                self._models.append(model)
        return


class StructureRewardEstBase(RewardEst[th.Tensor]):
    alpha: th.Tensor
    fraction_training_data_per_split: float
    mimic_xgbc_kwargs: dict[str, Any]
    base_xgbc_kwargs: dict[str, Any]
    mimic_models: list[xgbst.XGBClassifier]
    base_models: list[xgbst.XGBClassifier]
    xs_etrain: th.Tensor | None
    ys_etrain: th.Tensor | None
    _rg: th.Generator

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        fraction_training_data_per_split: float,
        n_mimic_models: int,
        n_base_models: int,
        mimic_xgbc_kwargs: dict[str, Any] = dict(),
        base_xgbc_kwargs: dict[str, Any] = dict(),
        alpha: float = 0.0,
        xs_etrain: Optional[th.Tensor] = None,
        ys_etrain: Optional[th.Tensor] = None,
        rseed: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        self.register_buffer("alpha", th.tensor(alpha))
        self.fraction_training_data_per_split = fraction_training_data_per_split
        self.xs_etrain = xs_etrain
        self.ys_etrain = ys_etrain
        self.mimic_xgbc_kwargs = mimic_xgbc_kwargs
        self.base_xgbc_kwargs = base_xgbc_kwargs
        self.mimic_models = [
            xgbst.XGBClassifier(**mimic_xgbc_kwargs) for _ in range(n_mimic_models)
        ]
        self.base_models = [
            xgbst.XGBClassifier(**base_xgbc_kwargs) for _ in range(n_base_models)
        ]
        self._rg = th.Generator()
        if rseed is not None:
            self._rg.manual_seed(rseed)

    def initialize(self, xs: th.Tensor, ys: th.Tensor):
        self.xs_etrain = xs.cpu()
        self.ys_etrain = ys.cpu()
        return

    def get_posterior_mean(self, forward_outs: th.Tensor) -> th.Tensor:
        outs: th.Tensor = th.mean(forward_outs, dim=1)
        return outs

    def get_posterior_std(self, forward_outs: th.Tensor) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        # (bsz)
        stds: th.Tensor = th.std(forward_outs, dim=1)
        return stds

    def _forward(self, inputs: th.Tensor) -> tuple[list[th.Tensor], list[th.Tensor]]:
        mimic_outs_l: list[th.Tensor] = self._forward_mimic(inputs)
        base_outs_l: list[th.Tensor] = self._forward_base(inputs)
        return mimic_outs_l, base_outs_l

    def _forward_mimic(self, inputs: th.Tensor) -> list[th.Tensor]:
        # TODO use decompose inputs
        # xs, inds = th.chunk(inputs, 2, dim=1)
        # inputs_: th.Tensor = th.cat((xs * inds, inds), dim=1)
        xs, fcinds, exinds = self.decompose_inputs(inputs)
        inputs_: th.Tensor | np.ndarray = (
            th.cat((xs * fcinds, fcinds), dim=1)
            if self.n_bdms_per_fcomb == 1
            else th.cat((xs * fcinds, fcinds, exinds), dim=1)
        )
        if (
            self.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(inputs_.storage())
            < th.cuda.get_device_properties(self.device).total_memory
        ):
            [m.set_params(device=str(self.device)) for m in self.mimic_models]
            inputs_ = mymodels.utils.to_cp_or_np(inputs_.to(device=self.device))
        else:
            [m.set_params(device="cpu") for m in self.mimic_models]
            inputs_ = inputs_.numpy(force=True)
        mimic_outs_l: list[th.Tensor] = [
            th.as_tensor(m.predict_proba(inputs_), dtype=th.float32, device=self.device)
            for m in self.mimic_models
        ]
        return mimic_outs_l

    def _forward_base(self, inputs: th.Tensor) -> list[th.Tensor]:
        # TODO use decompose inputs
        # inputs_n: np.ndarray = th.chunk(inputs, 2, dim=1)[0].numpy(force=True)
        ctxs: th.Tensor | np.ndarray = self.decompose_inputs(inputs)[0]
        if (
            self.device.type == "cuda"
            and xgbst.build_info()["USE_CUDA"]
            and sys.getsizeof(inputs.storage())
            < th.cuda.get_device_properties(self.device).total_memory
        ):
            [m.set_params(device=str(self.device)) for m in self.base_models]
            ctxs = mymodels.utils.to_cp_or_np(ctxs.to(device=self.device))
        else:
            [m.set_params(device="cpu") for m in self.base_models]
            ctxs = ctxs.numpy(force=True)
        base_outs_l: list[th.Tensor] = [
            th.as_tensor(m.predict_proba(ctxs), dtype=th.float32, device=self.device)
            for m in self.base_models
        ]
        return base_outs_l

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        self.train()
        self._fit_base()
        self._fit_mimic()
        self.eval()
        mse_loss: th.Tensor
        with th.no_grad():
            mse_loss = th.nn.functional.mse_loss(
                self.get_posterior_mean(self(self.train_inputs)),
                self.train_targets.to(device=self.device),
            )
        metrics_d: dict[str, float] = {"est_mse": mse_loss.item()}
        return metrics_d

    def _fit_base(self):
        assert self.xs_etrain is not None
        assert self.ys_etrain is not None
        # TODO use decompose inputs
        # ctxs, _ = th.chunk(self.train_inputs, chunks=2, dim=1)
        ctxs: th.Tensor = self.decompose_inputs(self.train_inputs)[0]
        # get rid of repeat elements
        tmp_t: th.Tensor = th.unique(th.cat((ctxs, self.ys[:, None]), dim=1), dim=0)
        xs: th.Tensor | np.ndarray = th.cat((tmp_t[:, :-1], self.xs_etrain), dim=0)
        ys: th.Tensor | np.ndarray = th.cat(
            (tmp_t[:, -1].to(dtype=th.long), self.ys_etrain), dim=0
        )
        if len(self.base_models) == 1:
            if (
                self.device.type == "cuda"
                and xgbst.build_info()["USE_CUDA"]
                and sys.getsizeof(xs.storage())
                < th.cuda.get_device_properties(self.device).total_memory
            ):
                self.base_models[0].set_params(device=str(self.device))
                xs = mymodels.utils.to_cp_or_np(xs.to(device=self.device))
                ys = mymodels.utils.to_cp_or_np(ys.to(device=self.device))
            else:
                self.base_models[0].set_params(device="cpu")
                xs = xs.numpy(force=True)
                ys = ys.numpy(force=True)
            self.base_models[0].fit(xs, ys)
            return
        for bm in self.base_models:
            n_data: int = math.ceil(len(xs) * self.fraction_training_data_per_split)
            idxs: th.Tensor = th.randint(
                0, len(xs), (n_data,), dtype=th.long, generator=self._rg
            )
            _xs: np.ndarray
            _ys: np.ndarray
            if (
                self.device.type == "cuda"
                and xgbst.build_info()["USE_CUDA"]
                and sys.getsizeof(xs.storage())
                < th.cuda.get_device_properties(self.device).total_memory
            ):
                bm.set_params(device=str(self.device))
                _xs = mymodels.utils.to_cp_or_np(xs[idxs].to(device=self.device))
                _ys = mymodels.utils.to_cp_or_np(ys[idxs].to(device=self.device))
            else:
                bm.set_params(device="cpu")
                _xs = xs[idxs].numpy(force=True)
                _ys = ys[idxs].numpy(force=True)
            bm.fit(_xs, _ys)
        return

    def _fit_mimic(self):
        n_labels: int = self.pyhats.shape[1]
        for m in self.mimic_models:
            n_data: int = math.ceil(
                len(self.train_targets) * self.fraction_training_data_per_split
            )
            idxs: th.Tensor = th.randint(
                0, len(self.train_targets), (n_data,), dtype=th.long, generator=self._rg
            )
            # TODO decompose inputs
            # _xs: th.Tensor = self.train_inputs[idxs][:, None, :].expand(
            #     -1, n_labels, -1
            # )
            # _ctxs, _xacts = th.chunk(_xs, chunks=2, dim=2)
            _tctxs, _tcinds, _texinds = self.decompose_inputs(self.train_inputs[idxs])
            _xs: th.Tensor | np.ndarray = (
                th.cat((_tctxs * _tcinds, _tcinds), dim=1)
                if self.n_bdms_per_fcomb == 1
                else th.cat((_tctxs * _tcinds, _tcinds, _texinds), dim=1)
            )
            _xs = _xs[:, None, :].expand(-1, n_labels, -1)
            _ys: th.Tensor | np.ndarray = th.arange(0, n_labels, dtype=th.long)
            _ys = _ys[None, :].expand(n_data, -1)
            _xs = _xs.flatten(0, 1)
            _ys = _ys.flatten(0, 1)
            _weight = self.pyhats[idxs].flatten(0, 1)
            if (
                self.device.type == "cuda"
                and xgbst.build_info()["USE_CUDA"]
                and sys.getsizeof(_xs.storage())
                < th.cuda.get_device_properties(self.device).total_memory
            ):
                m.set_params(device=str(self.device))
                _xs = mymodels.utils.to_cp_or_np(_xs.to(device=self.device))
                _ys = mymodels.utils.to_cp_or_np(_ys.to(device=self.device))
                _weight = mymodels.utils.to_cp_or_np(_weight.to(device=self.device))
            else:
                m.set_params(device="cpu")
                _xs = _xs.numpy(force=True)
                _ys = _ys.numpy(force=True)
                _weight = _weight.numpy(force=True)
            m.fit(_xs, _ys, sample_weight=_weight)
        return

    def get_extra_state(self) -> Any:
        extra_state: dict[str, Any] = super().get_extra_state()
        base_model_states_l: list[list[str]] = list()
        mimic_model_states_l: list[list[str]] = list()
        extra_state.update(
            {
                "base_model_states_l": base_model_states_l,
                "mimic_model_states_l": mimic_model_states_l,
                "fraction_training_data_per_split": self.fraction_training_data_per_split,
            }
        )
        try:
            with tmpf.TemporaryDirectory() as td:
                for i, model in enumerate(self.base_models):
                    p = os.path.join(td, f"base_m{i}.json")
                    model.save_model(p)
                    with open(p, mode="r") as f:
                        model_states: list[str] = f.readlines()
                        base_model_states_l.append(model_states)
                for i, model in enumerate(self.mimic_models):
                    p = os.path.join(td, f"mimic_m{i}.json")
                    model.save_model(p)
                    with open(p, mode="r") as f:
                        model_states: list[str] = f.readlines()
                        mimic_model_states_l.append(model_states)
        except skl_exceptions.NotFittedError:
            pass
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        self.fraction_training_data_per_split = state[
            "fraction_training_data_per_split"
        ]
        self.base_models.clear()
        self.mimic_models.clear()
        with tmpf.TemporaryDirectory() as td:
            for i, model_states in enumerate(state["base_model_states_l"]):
                p = os.path.join(td, f"base_m{i}.json")
                with open(p, mode="w") as f:
                    f.writelines(model_states)
                model = xgbst.XGBClassifier()
                model.load_model(p)
                self.base_models.append(model)
            for i, model_states in enumerate(state["mimic_model_states_l"]):
                p = os.path.join(td, f"mimic_m{i}.json")
                with open(p, mode="w") as f:
                    f.writelines(model_states)
                model = xgbst.XGBClassifier()
                model.load_model(p)
                self.mimic_models.append(model)
        return


class ZipStructureRewardEst(StructureRewardEstBase):
    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        fraction_training_data_per_split: float,
        n_models: int,
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
            n_mimic_models=n_models,
            n_base_models=n_models,
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
                -th.nn.functional.cross_entropy(
                    torch.distributions.utils.probs_to_logits(mo), bo, reduction="none"
                )
                for mo, bo in zip(mimic_outs_l, base_outs_l)
            ],
            dim=1,
        ).to(device=inputs.device)
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


class OneToManyStructureRewardEst(StructureRewardEstBase):
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
                -th.nn.functional.cross_entropy(
                    torch.distributions.utils.probs_to_logits(mo),
                    base_outs_l[0],
                    reduction="none",
                )
                for mo in mimic_outs_l
            ],
            dim=1,
        ).to(device=inputs.device)
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


class NeuralProcessRewardEst(RewardEst[thd.TensorDict]):
    gnp: th.nn.Module | nps.Model
    make_opt_fn: Callable[[ParamsT], th.optim.Optimizer]
    n_fit_iter: int
    fit_bsz: int
    n_fit_iter_init: int

    _mu: th.Tensor
    _sigma: th.Tensor

    opt: th.optim.Optimizer
    _opt_step: int
    _is_first_fit: bool

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        make_opt_fn: Callable[[ParamsT], th.optim.Optimizer],
        n_fit_iter: int,
        fit_bsz: int,
        n_fit_iter_init: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        # self.gnp = nps.construct_convgnp(
        #     dim_x=2 * n_ctx_covs, dim_y=1, dtype=th.float32
        # )
        self.gnp = nps.construct_gnp(dim_x=2 * n_ctx_covs, dim_y=1, dtype=th.float32)
        self.make_opt_fn = make_opt_fn
        self.n_fit_iter = n_fit_iter
        self.fit_bsz = fit_bsz
        self.n_fit_iter_init = (
            n_fit_iter if n_fit_iter_init is None else n_fit_iter_init
        )
        self.register_buffer("_mu", th.tensor(0.0))
        self.register_buffer("_sigma", th.tensor(1.0))
        self.opt = self.make_opt_fn(self.gnp.parameters())  # type:ignore
        self._opt_step = 0
        self._is_first_fit = True

    def forward(self, inputs: th.Tensor) -> thd.TensorDict:
        device: th.device = inputs.device
        # (1, 2 * n_ctx_covs, n_data)
        xcs: th.Tensor = self.train_inputs.T[None, :, :].to(device=device)
        # (1, 1, n_data)
        ycs: th.Tensor = self.train_targets[None, None, :].to(device=device)
        # (1, 2 * n_ctx_covs, n)
        xts: th.Tensor = inputs.T[None, :, :].to(device=device)
        with lab.on_device(str(device)):
            mus, variances, yhats_noiseless, yhats_noisy = nps.predict(
                self.gnp, xcs, ycs, xts, batch_size=len(inputs), num_samples=1
            )
        outs: thd.TensorDict = thd.make_tensordict(
            {
                "inputs": inputs.to(device=device),
                "xcs": xcs.to(device=device),
                "ycs": ycs.to(device=device),
                "xts": xts.to(device=device),
                "means": mus.to(device=device),
                "variances": variances.to(device=device),
                "yhats_noiseless": yhats_noiseless.to(device=device),
                "yhats_noisy": yhats_noisy.to(device=device),
            },
            device=device,
        )
        return outs

    def get_posterior_mean(self, forward_outs: thd.TensorDict) -> th.Tensor:
        means: th.Tensor = forward_outs["means"][0, 0, :]
        means = means * self._sigma + self._mu
        return means

    def get_posterior_covariance(self, forward_outs: thd.TensorDict) -> th.Tensor:
        raise NotImplementedError

    def get_posterior_std(self, forward_outs: thd.TensorDict) -> th.Tensor:
        variances: th.Tensor = forward_outs["variances"][0, 0, :].T
        # TODO feel like this is wrong
        variances = self._sigma * variances
        return variances

    def sample_posterior(
        self, forward_outs: thd.TensorDict, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        n: int = len(forward_outs["inputs"])
        n_samps: int = math.prod(sample_shape)
        if n_samps == 1:
            yhats_smp: th.Tensor = forward_outs["yhats_noiseless"][
                :, 0, 0, :
            ].T.reshape((n, *sample_shape))
            return yhats_smp
        with lab.on_device(str(self.device)):
            yhats_smp: th.Tensor = nps.predict(
                self.gnp,
                forward_outs["xcs"],
                forward_outs["ycs"],
                forward_outs["xts"],
                batch_size=len(forward_outs["inputs"]),
                num_samples=1,
            )[2]
        yhats_smp = yhats_smp[:, 0, 0, :].T.reshape((n, *sample_shape))
        return yhats_smp

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        gnp = self.gnp.to(device=plf.device)
        opt = self.opt
        bsz: int = self.fit_bsz
        bsz = min(bsz, math.floor(len(self.train_inputs) / 2))
        # prep training inputs
        xs: th.Tensor = self.train_inputs.to(self.device)
        ys: th.Tensor = self.train_targets.to(self.device)
        sigma, mu = th.std_mean(ys)
        self._sigma.copy_(sigma)
        self._mu.copy_(mu)
        ys_stdzd: th.Tensor = (ys - self._mu) / self._sigma
        # optimize nnet
        n_iter: int = self.n_fit_iter
        if self._is_first_fit:
            n_iter = self.n_fit_iter_init
            self._opt_step = 0
            self._is_first_fit = False
        if n_iter == 0:
            self.eval()
            return dict()
        pbar = tqdm.trange(n_iter, leave=False, dynamic_ncols=True)
        for _ in pbar:
            _bidxs: th.Tensor = th.multinomial(
                th.ones(len(xs), dtype=th.float32),
                num_samples=bsz * 2,
                replacement=False,
            )
            gnp.train()
            _xcs = xs[None, _bidxs[:bsz], :].permute(0, 2, 1).to(device=plf.device)
            _ycs = ys_stdzd[None, None, _bidxs[:bsz]].to(device=plf.device)
            _xts = xs[None, _bidxs[bsz:], :].permute(0, 2, 1).to(device=plf.device)
            _yts = ys_stdzd[None, None, _bidxs[bsz:]].to(device=plf.device)
            with lab.on_device(str(plf.device)):
                _bloss: th.Tensor = -th.mean(
                    nps.loglik(
                        gnp, _xcs, _ycs, _xts, _yts, normalize=True, batch_size=bsz
                    )
                )
            if th.any(th.isnan(_bloss)):
                _metrics_d = {"bloss": _bloss.item()}
                pbar.set_postfix(_metrics_d)
                plf.log_dict(
                    mylib.utils.add_prefix_to_dict(_metrics_d, "train-gnp"),
                    self._opt_step,
                )
                pbar.close()
                raise ValueError("nan encountered during training")
            opt.zero_grad()
            # th.nn.utils.clip_grad_norm_(gnp.parameters(), max_norm=1)
            _bloss.backward()
            opt.step()
            with th.no_grad() as _, lab.on_device(str(self.device)) as _:
                gnp.eval()
                _yhats: th.Tensor = nps.predict(
                    gnp,
                    self.train_inputs.T[None, :, :].to(device=plf.device),
                    self.train_targets[None, None, :].to(device=plf.device),
                    _xcs,
                    batch_size=bsz,
                    num_samples=1,
                )[2]
                _bmse: th.Tensor = th.nn.functional.mse_loss(
                    _yhats.flatten(), _ycs.flatten()
                )
            metrics_d = {"bloss": _bloss.item(), "bmse": _bmse.item()}
            pbar.set_postfix(metrics_d)
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(metrics_d, "train-gnp"), self._opt_step
            )
            self._opt_step = self._opt_step + 1
        pbar.close()
        self.eval()
        return dict()


class DeepAdditiveKernelRewardEst(RewardEst[thd.TensorDict]):
    nnet: th.nn.Module
    n_mc: int
    make_opt_fn: Callable[[ParamsT], th.optim.Optimizer]
    n_fit_iter: int
    fit_bsz: int
    n_fit_iter_init: int

    _mu: th.Tensor
    _sigma: th.Tensor

    opt: th.optim.Optimizer
    _opt_step: int
    _is_first_fit: bool

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        make_feature_extractor_fn: Callable[[int, int], th.nn.Module],
        kernel: th.nn.Module,
        n_mc: int,
        make_opt_fn: Callable[[ParamsT], th.optim.Optimizer],
        n_fit_iter: int,
        fit_bsz: int,
        n_fit_iter_init: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        feature_extractor = make_feature_extractor_fn(2 * n_ctx_covs, 2)
        # additive markov gp
        amk = dak.layers.activation.Amk1d(
            in_features=2,
            n_level=3,
            input_lb=-1.0,
            input_ub=1.0,
            kernel=kernel,
        )
        gp = dak.layers.linear.LinearFlipout(amk.out_features, out_features=1)
        self.nnet = th.nn.Sequential(
            feature_extractor,
            dak.layers.functional.ScaleToBounds(-1.0, 1.0),
            amk,
            th.nn.Flatten(start_dim=1),
            gp,
        )
        self.n_mc = n_mc
        self.make_opt_fn = make_opt_fn
        self.n_fit_iter = n_fit_iter
        self.fit_bsz = fit_bsz
        self.n_fit_iter_init = (
            n_fit_iter if n_fit_iter_init is None else n_fit_iter_init
        )
        self.register_buffer("_mu", th.tensor(0.0))
        self.register_buffer("_sigma", th.tensor(1.0))
        self.opt = self.make_opt_fn(self.nnet.parameters())  # type:ignore
        self._opt_step = 0
        self._is_first_fit = True

    def forward(self, inputs: th.Tensor) -> thd.TensorDict:
        outs: thd.TensorDict = thd.stack(
            [
                (lambda outs: thd.make_tensordict({"yhats": outs[0], "kl": outs[1]}))(
                    self.nnet(inputs)
                )
                for _ in range(self.n_mc)
            ]
        ).to(device=self.device)
        return outs

    def get_posterior_mean(self, forward_outs: thd.TensorDict) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        mus: th.Tensor = th.mean(forward_outs["yhats"], dim=0)[:, 0]
        mus = mus * self._sigma + self._mu
        return mus

    def get_posterior_std(self, forward_outs: thd.TensorDict) -> th.Tensor:
        # forward_outs (bsz, n_splits)
        # (bsz)
        stds: th.Tensor = th.std(forward_outs["yhats"], dim=0)[:, 0]
        stds = stds * self._sigma
        return stds

    def get_posterior_covariance(self, forward_outs: thd.TensorDict) -> th.Tensor:
        raise NotImplementedError

    def sample_posterior(
        self, forward_outs: thd.TensorDict, sample_shape: th.Size = th.Size()
    ) -> th.Tensor:
        n: int = forward_outs["yhats"].shape[1]
        n_samps: int = math.prod(sample_shape)
        idxs: th.Tensor = th.randint(
            0, forward_outs.shape[0], (n_samps, n), dtype=th.long, device=self.device
        )
        outs: th.Tensor = th.gather(
            forward_outs["yhats"][:, :, 0], dim=0, index=idxs
        ).reshape((n, *sample_shape))
        outs = outs * self._sigma + self._mu
        return outs

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        bsz: int = self.fit_bsz
        bsz = min(bsz, math.floor(len(self.train_inputs) / 2))
        # prep training inputs
        xs: th.Tensor = self.train_inputs.to(self.device)
        ys: th.Tensor = self.train_targets.to(self.device)
        sigma, mu = th.std_mean(ys)
        self._sigma.copy_(sigma)
        self._mu.copy_(mu)
        ys_stdzd: th.Tensor = (ys - self._mu) / self._sigma
        # optimize nnet
        n_iter: int = self.n_fit_iter
        if self._is_first_fit:
            n_iter = self.n_fit_iter_init
            self._opt_step = 0
            self._is_first_fit = False
        if n_iter == 0:
            self.eval()
            return dict()
        nnet = self.nnet.to(device=plf.device)
        opt = self.opt
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
        tloader = th_data.DataLoader(
            th_data.TensorDataset(xs, ys_stdzd[:, None]),
            batch_size=self.fit_bsz,
            shuffle=True,
        )
        pbar = tqdm.trange(n_iter, desc="fit-dak", leave=False, dynamic_ncols=True)
        for _ in pbar:
            _bmetrics_l = list()
            _bpbar = tqdm.tqdm(tloader, desc="batch", leave=False, dynamic_ncols=True)
            for _btxs, _btys in _bpbar:
                nnet.train()
                _btxs: th.Tensor = _btxs.to(device=plf.device)
                _btys: th.Tensor = _btys.to(device=plf.device)
                _bouts: thd.TensorDict = self(_btxs)
                _byhats: th.Tensor = _bouts["yhats"]
                _bkls: th.Tensor = _bouts["kl"]
                _bmse: th.Tensor = th.nn.functional.mse_loss(
                    th.mean(_byhats, dim=0), _btys
                )
                _bloss: th.Tensor = _bmse + th.mean(_bkls, dim=0) / len(_btxs)
                _bmetrics_d = {"bloss": _bloss.item(), "bmse": _bmse.item()}
                _bmetrics_l.append(thd.make_tensordict(_bmetrics_d))
                _bpbar.set_postfix(_bmetrics_d)
                plf.log_dict(
                    mylib.utils.add_prefix_to_dict(_bmetrics_d, "train-dak"),
                    step=self._opt_step,
                )
                if th.any(th.isnan(_bmse)):
                    pbar.close()
                    _bpbar.close()
                    raise ValueError("nan encountered during training")
                opt.zero_grad()
                # th.nn.utils.clip_grad_norm_(nnet.parameters(), max_norm=1)
                _bloss.backward()
                opt.step()
                self._opt_step = self._opt_step + 1
            _bpbar.close()
            lr_scheduler.step()
            _bmetrics_d = {
                f"{_k}-mean-eob": _v.item()
                for _k, _v in thd.stack(_bmetrics_l, dim=0).mean().items()
            }
            pbar.set_postfix(_bmetrics_d)
            plf.log_dict(
                mylib.utils.add_prefix_to_dict(_bmetrics_d, "train-dak"),
                step=self._opt_step,
            )
        pbar.close()
        self.eval()
        return dict()


class GaussianProcessRewardEst(RewardEst[gpth.distributions.MultivariateNormal]):
    _enable_lazy_fit = False

    _GP = mymodels.gaussian_process.SimpleGP

    n_fit_iter: int
    n_fit_iter_init: int
    make_opt_fn: Callable[[ParamsT], th.optim.Optimizer]

    _gp_est: _GP
    _is_first_fit: bool
    _opt_step: int
    _mu: th.Tensor
    _sigma: th.Tensor
    _opt: th.optim.Optimizer

    def __init__(
        self,
        n_ctx_covs: int,
        n_bdms_per_fcomb: int,
        mean_module: gpth.means.Mean,
        covar_module: gpth.kernels.Kernel,
        make_opt_fn: Callable[[ParamsT], th.optim.Optimizer],
        n_fit_iter: int,
        n_fit_iter_init: Optional[int] = None,
    ) -> None:
        super().__init__(n_ctx_covs=n_ctx_covs, n_bdms_per_fcomb=n_bdms_per_fcomb)
        self._gp_est = GaussianProcessRewardEst._GP(
            mean_module, covar_module, None, None, gpth.likelihoods.GaussianLikelihood()
        )
        self.n_fit_iter = n_fit_iter
        self.n_fit_iter_init = (
            n_fit_iter if n_fit_iter_init is None else n_fit_iter_init
        )
        self.make_opt_fn = make_opt_fn
        self._is_first_fit = True
        self._opt_step = 0
        self.register_buffer("_mu", th.tensor(0.0))
        self.register_buffer("_sigma", th.tensor(1.0))
        self._opt = make_opt_fn(self._gp_est.parameters())

    def forward(self, inputs: th.Tensor) -> gpth.distributions.MultivariateNormal:
        return self._gp_est(inputs)

    def get_posterior_mean(
        self, forward_outs: gpth.distributions.MultivariateNormal
    ) -> th.Tensor:
        return (forward_outs.mean * self._sigma) + self._mu

    def get_posterior_std(
        self, forward_outs: gpth.distributions.MultivariateNormal
    ) -> gpth.Tensor:
        return forward_outs.stddev * self._sigma

    def get_posterior_covariance(
        self, forward_outs: gpth.distributions.MultivariateNormal
    ) -> th.Tensor:
        return (self._sigma**2) * forward_outs.covariance_matrix

    def sample_posterior(
        self,
        forward_outs: gpth.distributions.MultivariateNormal,
        sample_shape: th.Size = th.Size(),
    ) -> th.Tensor:
        return (forward_outs.sample(sample_shape) * self._sigma) + self._mu

    def fit_(self, plf: pl.Fabric) -> dict[str, float]:
        self.train().to(device=plf.device)
        gp_est = self._gp_est
        xs: th.Tensor = self.train_inputs.to(self.device)
        ys: th.Tensor = self.train_targets.to(self.device)
        sigma, mu = th.std_mean(ys)
        self._sigma.copy_(sigma)
        self._mu.copy_(mu)
        zs = (ys - self._mu) / self._sigma
        gp_est.set_train_data(xs, zs, strict=False)
        assert gp_est.train_inputs is not None
        assert gp_est.train_targets is not None
        mll = gpth.mlls.ExactMarginalLogLikelihood(gp_est.likelihood, gp_est)
        n_iter: int = self.n_fit_iter
        if self._is_first_fit:
            n_iter = self.n_fit_iter_init
            self._opt_step = 0
            self._is_first_fit = False
        if n_iter == 0:
            self.eval()
            return dict()
        pbar = tqdm.trange(n_iter, leave=False, dynamic_ncols=True)
        for _ in pbar:
            self._opt.zero_grad()
            posterior = gp_est(xs)
            nll: th.Tensor = -mll(posterior, zs)
            plf.log_dict(
                {
                    "train_gp/nll": nll.item(),
                    "train_gp/mse_zs": gpth.metrics.mean_squared_error(
                        posterior, zs
                    ).item(),
                },
                step=self._opt_step,
            )
            pbar.set_postfix({"nll": -nll.item()})
            if th.any(th.isnan(nll)):
                pbar.close()
                raise ValueError("nan encountered during training")
            nll.backward()
            th.nn.utils.clip_grad_norm_(gp_est.parameters(), max_norm=1)
            self._opt.step()
            self._opt_step = self._opt_step + 1
        pbar.close()
        mse: float
        with th.no_grad():
            mse = th.nn.functional.mse_loss(
                self.get_posterior_mean(self(xs)), ys
            ).item()
        metrics_d = {"est_mse": mse}
        return metrics_d


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
