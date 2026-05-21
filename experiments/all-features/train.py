from __future__ import annotations

import logging
import os
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Protocol

import disslib
import hydra as hd
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import lightning.fabric.plugins.environments as plf_plugins_envs
import mylib
import tensordict as thd
import torch as th
import torchmetrics as thm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass
class MainConf(DictConfig):
    make_envs_func: Any
    eval_bsz: int
    plf: Any


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
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


@th.no_grad()
def eval_env_all_features(
    env: disslib.Env, plf: pl.Fabric, eval_bsz: int, metrics_func: thm.MetricCollection
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
        bxacts, bacts_avail = env.get_avail_actions(bctxs)
        bsz: int = bxacts.shape[0]
        # n_acts_avail: int = bxacts.shape[1]
        n_act_feats: int = bxacts.shape[2]
        # (bsz, n_act_feats)
        bacts: th.Tensor = th.ones((bsz, n_act_feats), dtype=th.long, device=plf.device)
        # collect rewards
        # (bsz,)
        brewards, binfo = env.compute_rewards(bctxs, bacts, bctxs_info)
        bfcinds = env.decompose_acts(bacts)["fcinds"]
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


def main(cfg: MainConf, logger: logging.Logger):
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make components used in the experiment
    make_envs_func: MakeEnvsFunc = hd.utils.call(cfg.make_envs_func, _partial_=True)
    tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
    # configure plf and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger],
        plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
    )
    # tmetrics_d: dict[str, Any] = eval_env_all_features(
    #     env=tenv,
    #     plf=plf,
    #     eval_bsz=cfg.eval_bsz,
    #     metrics_func=metrics_func,
    # )
    # plf.log_dict(mylib.utils.add_prefix_to_dict(tmetrics_d, "eval_train"))
    if venv is not None:
        vmetrics_d: dict[str, Any] = eval_env_all_features(
            env=venv,
            plf=plf,
            eval_bsz=cfg.eval_bsz,
            metrics_func=metrics_func,
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(vmetrics_d, "eval_val"))
    if tstenv is not None:
        tstmetrics_d: dict[str, Any] = eval_env_all_features(
            env=tstenv,
            plf=plf,
            eval_bsz=cfg.eval_bsz,
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
