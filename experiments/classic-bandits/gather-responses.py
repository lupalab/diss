from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Optional

import disslib
import hydra as hd
import lightning as pl
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import tensordict as thd
import torch as th
import train as _train
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


@dataclass
class MainConf:
    train_exp: Optional[ExpConf]
    train_run: Optional[str]


@dataclass
class ExpConf:
    exp_p: str
    run_id: int


@th.no_grad()
def gather_responses(
    env: disslib.Env,
    reward_est: disslib.RewardEst,
    plf: pl.Fabric,
    eval_bsz: int,
) -> thd.TensorDict:
    if isinstance(env, th.nn.Module):
        env.eval().to(device=plf.device)
    reward_est.eval().to(device=plf.device)
    resps_l: list[thd.TensorDict] = list()
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
        # collect rewards
        # (bsz,)
        brewards, binfo = env.compute_rewards(bctxs, bacts, bctxs_info)
        bfcinds = env.decompose_acts(
            th.gather(
                bxacts, dim=1, index=baidxs[:, None, None].expand(-1, -1, n_act_feats)
            )[:, 0, :]
        )["fcinds"]
        bn_selected: th.Tensor = th.sum(bfcinds, dim=1)
        # make responses
        bresps: thd.TensorDict = binfo.clone()
        bresps["rewards"] = brewards
        bresps["num-feautre-selected"] = bn_selected
        resps_l.append(bresps)
    resps: thd.TensorDict = thd.cat(resps_l, dim=0)
    return resps


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
    )  # type: ignore
    OmegaConf.save(
        train_run_cfg, os.path.join(output_dir, ".hydra", "train_run_cfg.yaml")
    )
    # make components used in the experiment
    make_envs_func: _train.MakeEnvsFunc = hd.utils.call(
        train_run_cfg.make_envs_func, _partial_=True
    )
    tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
    assert venv is not None
    # make reward estimator
    reward_est: disslib.RewardEst = hd.utils.instantiate(
        train_run_cfg.reward_est,
        n_ctx_covs=tenv.n_covs,
        n_bdms_per_fcomb=(
            tenv.n_bdms_per_fcomb  # type: ignore
            if hasattr(tenv, "n_bdms_per_fcomb")
            else 1
        ),
    )
    if isinstance(reward_est, disslib.estimators.StructureRewardEstBase):
        reward_est.initialize(mmctdata["xs"], mmctdata["ys"])
    ckpt_p: str = os.path.join(
        mylib.utils.get_project_root_dir(), train_run_p, "checkpoints"
    )
    try:
        ckpt = th.load(
            os.path.join(ckpt_p, "itr_end.ckpt"), map_location="cpu", weights_only=False
        )
    except FileNotFoundError as e:
        logger.error(e)
        return
    reward_est.load_state_dict(ckpt)
    # configure plf and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    # tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")
    plf: pl.Fabric = hd.utils.instantiate(train_run_cfg.plf, _partial_=True)(
        plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
    )
    # gather predictions
    resps: thd.TensorDict = gather_responses(
        env=venv,
        reward_est=reward_est,
        plf=plf,
        eval_bsz=train_run_cfg.train_conf.eval_bsz,
    )
    th.save(resps, os.path.join(output_dir, "resps.pt"))


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
