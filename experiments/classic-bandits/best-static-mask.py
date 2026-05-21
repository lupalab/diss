from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import traceback
from typing import TYPE_CHECKING, Any, Optional

import disslib
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import tensordict as thd
import torch as th
import torchmetrics as thm
import train as _train
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

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


OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


@dataclass
class MainConf:
    train_exp: Optional[MakeTemplateExpConf]
    train_run: Optional[str]


@dataclass
class MakeTemplateExpConf:
    exp_p: str
    run_id: int


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
def compute_pms(
    env: disslib.Env,
    reward_est: disslib.RewardEst,
    plf: pl.Fabric,
    eval_bsz: int,
) -> thd.TensorDict:
    if isinstance(env, th.nn.Module):
        env.eval().to(device=plf.device)
    reward_est.eval().to(device=plf.device)
    outs_l: list[thd.TensorDict] = list()
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
        bouts = thd.make_tensordict({"ctxs": bctxs, "pms": bpms}, batch_size=(bsz,)).to(
            device="cpu"
        )
        outs_l.append(bouts)
    outs: thd.TensorDict = thd.cat(outs_l, dim=0)
    return outs


@th.no_grad()
def eval_env_with_static_policy(
    env: disslib.Env,
    static_act: th.Tensor,
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
        # (bsz, n_act_feats)
        bacts: th.Tensor = static_act[None, :].expand(bsz, -1).to(device=plf.device)
        # collect rewards
        # (bsz,)
        brewards, binfo = env.compute_rewards(bctxs, bacts, bctxs_info)
        bn_selected: th.Tensor = th.sum(bacts, dim=1)
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


def _get_run_dir(cfg: MainConf) -> str:
    if hasattr(cfg, "train_run") and cfg.train_run is not None:
        return cfg.train_run
    assert hasattr(cfg, "train_exp") and cfg.train_exp is not None
    return os.path.join(cfg.train_exp.exp_p, str(cfg.train_exp.run_id))


def main(cfg: MainConf, logger: logging.Logger):
    output_dir: str = HydraConfig.get().runtime.output_dir
    train_run_p: str = _get_run_dir(cfg)
    train_run_cfg: _train.MainConf = OmegaConf.load(
        os.path.join(
            mylib.utils.get_project_root_dir(), train_run_p, ".hydra", "config.yaml"
        )
    )  # type:ignore
    OmegaConf.save(
        train_run_cfg, os.path.join(output_dir, ".hydra", "train_run_cfg.yaml")
    )
    # make components used in the experiment
    make_envs_func: _train.MakeEnvsFunc = hd.utils.call(
        train_run_cfg.make_envs_func, _partial_=True
    )
    tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
    # make reward estimator
    reward_est: disslib.RewardEst = hd.utils.instantiate(
        train_run_cfg.reward_est,
        n_ctx_covs=tenv.n_covs,
        n_bdms_per_fcomb=(
            tenv.n_bdms_per_fcomb  # type:ignore
            if hasattr(tenv, "n_bdms_per_fcomb")
            else 1
        ),
    )
    if isinstance(reward_est, disslib.estimators.StructureRewardEstBase):
        reward_est.initialize(mmctdata["xs"], mmctdata["ys"])
    ckpt_p: str = os.path.join(
        mylib.utils.get_project_root_dir(), train_run_p, "checkpoints"
    )
    ckpt = th.load(
        os.path.join(ckpt_p, "itr_end.ckpt"), map_location="cpu", weights_only=False
    )
    reward_est.load_state_dict(ckpt)
    # configure plf and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")
    plf: pl.Fabric = hd.utils.instantiate(train_run_cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger],
        plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
    )
    ckpt_p: str = os.path.join(tfb_logger.log_dir, "checkpoints")
    # most frequent feature mask
    acts_avail: th.Tensor = tenv.get_avail_actions()[1]
    tenv.set_avail_actions(acts_avail)
    outs: thd.TensorDict = compute_pms(
        env=tenv,
        reward_est=reward_est,
        plf=plf,
        eval_bsz=train_run_cfg.train_conf.eval_bsz,
    )
    static_act: th.Tensor = acts_avail[
        int(th.argmax(th.mean(outs["pms"], dim=0)).item())
    ]
    th.save(
        {
            "compute-pms-outputs": outs,
            "static-act": static_act,
        },
        os.path.join(output_dir, "output.ckpt"),
    )
    # tmetrics_d: dict[str, Any] = eval_env_with_static_policy(
    #     env=tenv,
    #     static_act=static_act,
    #     plf=plf,
    #     eval_bsz=train_run_cfg.train_conf.eval_bsz,
    #     metrics_func=metrics_func,
    # )
    # plf.log_dict(mylib.utils.add_prefix_to_dict(tmetrics_d, "eval_train"))
    if venv is not None:
        vmetrics_d: dict[str, Any] = eval_env_with_static_policy(
            env=venv,
            static_act=static_act,
            plf=plf,
            eval_bsz=train_run_cfg.train_conf.eval_bsz,
            metrics_func=metrics_func,
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "eval_val"))
    if tstenv is not None:
        tstmetrics_d: dict[str, Any] = eval_env_with_static_policy(
            env=tstenv,
            static_act=static_act,
            plf=plf,
            eval_bsz=train_run_cfg.train_conf.eval_bsz,
            metrics_func=metrics_func,
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(tstmetrics_d, "eval_test"))
    # logger flush record and close
    tfb_logger.finalize("success")
    csv_logger.finalize("success")


if __name__ == "__main__":

    @hd.main(version_base=None)
    def _main(cfg: MainConf):
        logger = logging.getLogger(HydraConfig.get().job.name)
        try:
            main(cfg, logger)
        except Exception as e:
            logger.error(e, exc_info=True, stack_info=True)
            traceback.print_exception(e)

    _main()
