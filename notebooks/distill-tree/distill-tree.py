# %%
from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol

import disslib
import envs
import hydra as hd
import lightning as pl
import matplotlib.pyplot as plt
import mylib
import numpy as np
import sklearn.tree as skl_tree
import tensordict as thd
import torch as th
import torchmetrics as thm
import yellowbrick.cluster as yb_cluster
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import sklearn.cluster as skl_cluster
else:
    # NOTE IF CUML IS USED, PATCH YELLOWBRICK
    # def is_estimator(model):
    # """
    # Determines if a model is an estimator using issubclass and isinstance.

    # Parameters
    # ----------
    # estimator : class or instance
    #     The object to test if it is a Scikit-Learn clusterer, especially a
    #     Scikit-Learn estimator or Yellowbrick visualizer
    # """
    # try:
    #     import cuml
    #     if inspect.isclass(model):
    #         return issubclass(model, (cuml.Base))

    #     return isinstance(model, (cuml.Base))
    # except:
    #     pass

    # if inspect.isclass(model):
    #     return issubclass(model, (BaseEstimator, ContribEstimator))

    # return isinstance(model, (BaseEstimator, ContribEstimator))
    try:
        import cuml
        import cuml.cluster as skl_cluster

        cuml.set_global_output_type("numpy")
    except:
        import sklearn.cluster as skl_cluster


# %%
@dataclass
class MainConf(DictConfig):
    make_envs_func: Any
    reward_est: Any
    strat: Any
    train_conf: TrainConf
    plf: Any


@dataclass
class TrainConf(DictConfig):
    init_capital: int
    n_iter: int
    eval_bsz: int
    eval_every_n_iter: int
    save_ckpt_every_n_iter: int
    init_capital_rseed: Optional[int]
    to_eval_on_train: bool
    to_log_additional_metrics: bool


class MakeEnvsFunc(Protocol):
    _ReturnType = NamedTuple(
        "_ReturnType",
        (
            ("tenv", disslib.Env),
            ("venv", Optional[disslib.Env]),
            ("tstenv", Optional[disslib.Env]),
            ("tmmcdata", thd.TensorDict),
            ("metrics_func", thm.MetricCollection),
        ),
    )

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> MakeEnvsFunc._ReturnType: ...


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


# %%
@th.no_grad()
def gather_actions(
    env: disslib.Env,
    reward_est: disslib.RewardEst,
    plf: pl.Fabric,
    eval_bsz: int,
) -> thd.TensorDict:
    if isinstance(env, th.nn.Module):
        env.eval().to(device=plf.device)
    reward_est.eval().to(device=plf.device)
    metrics_func.to(device=plf.device).reset()
    outs_l: list[th.Tensor] = list()
    env.reset()
    while env.has_next():
        # (bsz, n_covs)
        bctxs, bctxs_info = env.get_ctxs(eval_bsz)
        bctxs = bctxs.to(device=plf.device)
        bctxs_info = bctxs_info.to(device=plf.device)
        # (bsz, n_acts_avail, n_act_feats), (bsz, n_acts_avail)
        bxacts, bacts_avail = env.get_avail_actions(bctxs)
        bxacts = bxacts.to(device=plf.device)
        bacts_avail = bacts_avail.to(device=plf.device)
        # shape info
        bsz: int = bxacts.shape[0]
        n_acts_avail: int = bxacts.shape[1]
        n_act_feats: int = bxacts.shape[2]
        # compute score for each ctx-act pair
        # (bsz, n_acts_avail, n_covs)
        bctxs_ = bctxs[:, None, :].expand(-1, n_acts_avail, -1)
        # (bsz, n_acts_avail, n_covs + n_act_feats)
        binputs: th.Tensor = th.cat((bctxs_, bxacts), dim=2).flatten(0, 1)
        # (bsz, n_acts_avail)
        bpms: th.Tensor = th.unflatten(
            reward_est.get_posterior_mean(reward_est(binputs.to(plf.device))),
            dim=0,
            sizes=(bsz, n_acts_avail),
        )
        # choose which action to take
        # (bsz,)
        btargets_est, baidxs = th.max(bpms, dim=1)
        # (bsz, n_act_feats)
        bacts: th.Tensor = th.gather(
            bacts_avail,
            dim=1,
            index=baidxs[:, None, None].expand(-1, -1, n_act_feats),
        )[:, 0]
        bouts = thd.make_tensordict(
            {"ctxs": bctxs, "acts": bacts}, batch_size=(bsz,)
        ).to(device="cpu")
        outs_l.append(bouts)
    outs: thd.TensorDict = thd.cat(outs_l)
    return outs


@th.no_grad()
def eval_env_with_dtc(
    env: disslib.Env,
    dtc: skl_tree.DecisionTreeClassifier,
    centroids_b: th.Tensor,
    plf: pl.Fabric,
    eval_bsz: int,
    metrics_func: thm.MetricCollection,
) -> dict[str, float]:
    if isinstance(env, th.nn.Module):
        env.eval().to(device=plf.device)
    n_selected_l: list[th.Tensor] = list()
    rewards_l: list[th.Tensor] = list()
    regrets_l: list[th.Tensor] = list()
    metrics_func.to(device=plf.device).reset()
    env.reset()
    while env.has_next():
        # (bsz, n_covs)
        bctxs, bctxs_info = env.get_ctxs(eval_bsz)
        bctxs = bctxs.to(device=plf.device)
        bctxs_info = bctxs_info.to(device=plf.device)
        # (bsz, n_acts_avail, n_act_feats), (bsz, n_acts_avail)
        bsz: int = bctxs.shape[0]
        bxacts: th.Tensor = (
            centroids_b[None, :, :]
            .expand(bsz, -1, -1)
            .to(dtype=th.float32, device=plf.device)
        )
        bacts_avail: th.Tensor = (
            centroids_b[None, :, :]
            .expand(bsz, -1, -1)
            .to(dtype=th.float32, device=plf.device)
        )
        # shape info
        n_acts_avail: int = bxacts.shape[1]
        n_act_feats: int = bxacts.shape[2]
        # decide which cluster to use
        # (bsz,)
        baidxs: th.Tensor = th.as_tensor(
            dtc.predict(bctxs.numpy(force=True)), dtype=th.int64, device=plf.device
        )
        # (bsz, n_act_feats)
        bacts: th.Tensor = th.gather(
            bacts_avail,
            dim=1,
            index=baidxs[:, None, None].expand(-1, -1, n_act_feats),
        )[:, 0]
        # collect rewards
        # (bsz,)
        brewards, binfo = env.compute_rewards(bctxs, bacts, bctxs_info)
        bfcinds = env.decompose_acts(
            th.gather(
                bxacts, dim=1, index=baidxs[:, None, None].expand(-1, -1, n_act_feats)
            )[:, 0, :]
        )["fcinds"]
        bn_selected: th.Tensor = th.sum(bfcinds, dim=1)
        bbest_rewards: th.Tensor = env.compute_optimal_rewards(bctxs)
        bregrets: th.Tensor = bbest_rewards - brewards
        # metrics of current selection
        metrics_func.update(
            binfo["pyhats"][:, :, None].to(device=plf.device),
            binfo["ys"][:, None].to(device=plf.device),
        )
        n_selected_l.append(bn_selected)
        # record metrics
        rewards_l.append(brewards.to(device="cpu"))
        regrets_l.append(bregrets.to(device="cpu"))
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    # compute average metrics
    reward: th.Tensor = th.mean(th.cat(rewards_l, dim=0))
    regret: th.Tensor = th.mean(th.cat(regrets_l, dim=0))
    n_selected: th.Tensor = th.mean(th.cat(n_selected_l, dim=0).to(dtype=th.float32))
    metrics_d.update(
        {
            "reward": reward.item(),
            "regret": regret.item(),
            "n_selected": n_selected.item(),
        }
    )
    return metrics_d


# %%
feature_names = envs.uci_health.bar_crawl.FEATURE_NAMES

# %%
run_p: str = (
    "experiments/sequential-bandits/outputs/diab130-vllm-distill/20260305_065808/3"
)
run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)
cfg: MainConf = OmegaConf.load(
    os.path.join(run_p, ".hydra", "config.yaml")
)  # type: ignore
output_p: str = os.path.join("outputs", "bar")
figname_prefix: str = "overload"
os.makedirs(output_p, exist_ok=True)

# %%
# make components used in the experiment
make_envs_func: MakeEnvsFunc = hd.utils.call(cfg.make_envs_func, _partial_=True)
tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
n_covs: int = tenv.n_covs
# make reward estimator
reward_est: disslib.RewardEst = hd.utils.instantiate(
    cfg.reward_est,
    n_ctx_covs=tenv.n_covs,
    n_bdms_per_fcomb=(
        tenv.n_bdms_per_fcomb  # type: ignore
        if hasattr(tenv, "n_bdms_per_fcomb")
        else 1
    ),
)
if isinstance(reward_est, disslib.estimators.StructureRewardEstBase):
    reward_est.initialize(mmctdata["xs"], mmctdata["ys"])
ckpt_p: str = os.path.join(run_p, "checkpoints")
ckpt = th.load(
    os.path.join(ckpt_p, "itr_end.ckpt"), map_location="cpu", weights_only=False
)
reward_est.load_state_dict(ckpt)

# %%
plf: pl.Fabric = pl.Fabric(accelerator="cpu")

# %%
assert venv is not None
outs: thd.TensorDict = gather_actions(
    env=tenv, reward_est=reward_est, plf=plf, eval_bsz=cfg.train_conf.eval_bsz
)
acts: th.Tensor = outs["acts"]

# %%
kms = skl_cluster.KMeans(n_clusters=100, random_state=279).fit(acts.numpy(force=True))
cs = th.as_tensor(kms.predict(acts.numpy(force=True)))
cids: th.Tensor = th.unique(cs)

# %%
centroids: th.Tensor = th.stack(
    [th.mean(acts[cs == _cid].to(dtype=th.float32), dim=0) for _cid in cids], dim=0
)
centroids_b: th.Tensor = centroids > 0.5

# %%
dtc_kwargs = {
    "max_depth": 5,
    "splitter": "best",
    "criterion": "log_loss",
    "random_state": 279,
    "ccp_alpha": 0.03,
}
dtc = skl_tree.DecisionTreeClassifier(**dtc_kwargs).fit(
    outs["ctxs"].numpy(force=True), cs.numpy(force=True)
)

# %%
metrics_d: dict[str, Any] = eval_env_with_dtc(
    env=venv,
    dtc=dtc,
    centroids_b=centroids_b,
    plf=plf,
    eval_bsz=cfg.train_conf.eval_bsz,
    metrics_func=metrics_func,
)

# %%
