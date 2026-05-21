from __future__ import annotations

import os
from typing import Any, Optional

import disslib
import mydatasets
import mymodels
import sklearn.compose as skl_compose
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm


def make_overload_components(
    sig_mult: float,
    bias_level: float,
    min_temp: float = 1.0,
    bias_mult: float = 5.0,
    vdata_seed: Optional[int] = None,
) -> tuple[
    disslib.Env,
    Optional[disslib.Env],
    Optional[disslib.Env],
    thd.TensorDict,
    thm.MetricCollection,
]:
    tdata, tmmcdata, bdmtdata, vdata, _, _ = load_data(vdata_seed=vdata_seed)
    bdm = mymodels.classifiers.SubsetFeatureOverloadClassifierWrapper(
        mymodels.classifiers.SubsetFeatureNadarayaWatsonClassifier(
            xs_train=bdmtdata["xs"].numpy(force=True),  # type:ignore
            ys_train=bdmtdata["ys"].numpy(force=True),  # type:ignore
            sig_mult=sig_mult,
        ),
        bias_level=bias_level,
        min_temp=min_temp,
        bias_mult=bias_mult,
    )
    tenv = disslib.envs.base.TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=0.0,
        is_train=True,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    venv = disslib.envs.base.TensorDictEnv(
        data=vdata,
        bdm=bdm,
        alpha=0.0,
        is_train=False,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    n_labels: int = len(th.unique(tdata["ys"]))
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    return tenv, venv, None, tmmcdata, metrics_func


def make_simplicity_components(
    sig_mult: float,
    bias_level: float,
    poison_feature_idx: int,
    xgbc_kwargs: dict[str, Any] = dict(),
    vdata_seed: Optional[int] = None,
) -> tuple[
    disslib.Env,
    Optional[disslib.Env],
    Optional[disslib.Env],
    thd.TensorDict,
    thm.MetricCollection,
]:
    tdata, tmmcdata, bdmtdata, vdata, _, _ = load_data(vdata_seed=vdata_seed)
    bdm = mymodels.classifiers.SubsetFeatureSimplicityBiasClassifierWrapper(
        mymodels.classifiers.SubsetFeatureNadarayaWatsonClassifier(
            xs_train=bdmtdata["xs"].numpy(force=True),  # type:ignore
            ys_train=bdmtdata["ys"].numpy(force=True),  # type:ignore
            sig_mult=sig_mult,
        ),
        bias_level=bias_level,
        poison_feature_idx=poison_feature_idx,
        xgbc_kwargs=xgbc_kwargs,
    )
    tenv = disslib.envs.base.TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=0.0,
        is_train=True,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    venv = disslib.envs.base.TensorDictEnv(
        data=vdata,
        bdm=bdm,
        alpha=0.0,
        is_train=False,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    n_labels: int = len(th.unique(tdata["ys"]))
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    return tenv, venv, None, tmmcdata, metrics_func


def make_interp_components(
    alpha: float,
    max_features: int = 3,
    to_cache_model: bool = True,
    lrc_kwars: dict[str, Any] = dict(),
    vdata_seed: Optional[int] = None,
) -> tuple[
    disslib.Env,
    Optional[disslib.Env],
    Optional[disslib.Env],
    thd.TensorDict,
    thm.MetricCollection,
]:
    tdata, tmmcdata, bdmtdata, vdata, _, _ = load_data(vdata_seed=vdata_seed)
    bdm = mymodels.classifiers.SubsetFeatureLogisticRegressionClassifier(
        xs_train=bdmtdata["xs"].numpy(force=True),  # type:ignore
        ys_train=bdmtdata["ys"].numpy(force=True),  # type:ignore
        to_cache_model=to_cache_model,
        lrc_kwargs=lrc_kwars,
    )
    tenv = disslib.envs.base.TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=alpha,
        is_train=True,
        n_acts_avail=500,
        min_features=1,
        max_features=max_features,
    )
    venv = disslib.envs.base.TensorDictEnv(
        data=vdata,
        bdm=bdm,
        alpha=alpha,
        is_train=False,
        n_acts_avail=500,
        min_features=1,
        max_features=max_features,
    )
    n_labels: int = len(th.unique(tdata["ys"]))
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    return tenv, venv, None, tmmcdata, metrics_func


def make_multiple_expert_components(
    n_experts: int,
    sig_mult: float,
    kmeans_kwargs: dict[str, Any] = dict(),
    vdata_seed: Optional[int] = None,
) -> tuple[
    disslib.Env,
    Optional[disslib.Env],
    Optional[disslib.Env],
    thd.TensorDict,
    thm.MetricCollection,
]:
    tdata, tmmcdata, bdmtdata, vdata, _, _ = load_data(vdata_seed=vdata_seed)
    bdm = mymodels.classifiers.SubsetFeatureMultiExpertNadarayaWatsonClassifier(
        n_bdms_per_fcomb=n_experts,
        xs_train=bdmtdata["xs"].numpy(force=True),  # type:ignore
        ys_train=bdmtdata["ys"].numpy(force=True),  # type:ignore
        kmeans_kwargs=kmeans_kwargs,
        sig_mult=sig_mult,
    )
    tenv = disslib.envs.base.TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=0.0,
        is_train=True,
        n_acts_avail=5000,
        min_features=1,
        max_features=None,
    )
    venv = disslib.envs.base.TensorDictEnv(
        data=vdata,
        bdm=bdm,
        alpha=0.0,
        is_train=False,
        n_acts_avail=5000,
        min_features=1,
        max_features=None,
    )
    n_labels: int = len(th.unique(tdata["ys"]))
    metrics_func = thm.MetricCollection(
        {
            "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
            "precision": thm.Precision(task="multiclass", num_classes=n_labels),
            "recall": thm.Recall(task="multiclass", num_classes=n_labels),
            "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
            "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
        }
    )
    return tenv, venv, None, tmmcdata, metrics_func


def load_data(
    vdata_seed: Optional[int] = None,
) -> tuple[
    thd.TensorDict,
    thd.TensorDict,
    thd.TensorDict,
    thd.TensorDict,
    skl_preproc.StandardScaler,
    skl_compose.ColumnTransformer,
]:
    data: thd.TensorDict = th.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(),
            "uci",
            "diabetes-130-ordinal-binary.pt",
        ),
        map_location="cpu",
        weights_only=False,
    )
    # Load raw data with feature_preprocessor for inverse transforms
    raw_data: dict = th.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(),
            "uci",
            "raw-diabetes-130-ordinal-binary.pkl",
        ),
        map_location="cpu",
        weights_only=False,
    )
    feature_preprocessor: skl_compose.ColumnTransformer = raw_data[
        "feature_preprocessor"
    ]
    data["ys"] = data["ys"].flatten()
    data["xs_orig"] = data["xs"]
    stdsclr = skl_preproc.StandardScaler()
    data["xs"] = th.as_tensor(stdsclr.fit_transform(data["xs"]), dtype=th.float32)
    _tidxs, _mmctidxs, _bdmtidxs, _vidxs = [
        th.as_tensor(_d.indices, dtype=th.long)
        for _d in th_data.random_split(
            th_data.TensorDataset(th.arange(len(data))),
            lengths=(0.34, 0.16, 0.16, 0.34),
            generator=th.Generator().manual_seed(279),
        )
    ]
    _vidxs = _vidxs[
        th.multinomial(
            th.ones((len(_vidxs),), dtype=th.float32),
            num_samples=1000,
            generator=th.Generator().manual_seed(
                580 if vdata_seed is None else vdata_seed
            ),
        ).flatten()
    ]
    tdata, mmctdata, bdmtdata, vdata = (
        data[_tidxs],
        data[_mmctidxs],
        data[_bdmtidxs],
        data[_vidxs],
    )
    return tdata, mmctdata, bdmtdata, vdata, stdsclr, feature_preprocessor
