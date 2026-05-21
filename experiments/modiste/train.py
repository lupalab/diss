from __future__ import annotations

import gc
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
import numpy as np
import tensordict as thd
import torch as th
import torchmetrics as thm
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from _estimators import ModisteRewardEstBase
from _strategies import OptStrat


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
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls)
)


@th.no_grad()
def eval_env(
    env: disslib.Env,
    score_est: ModisteRewardEstBase,
    plf: pl.Fabric,
    eval_bsz: int,
    metrics_func: thm.MetricCollection,
    to_log_additioanl_metrics: bool,
) -> dict[str, float]:
    if isinstance(env, th.nn.Module):
        env.eval().to(device=plf.device)
    score_est.eval().to(device=plf.device)
    n_selected_l: list[th.Tensor] = list()
    rewards_l: list[th.Tensor] = list()
    regrets_l: list[th.Tensor] = list()
    mses_rand_l: list[th.Tensor] = list()
    mses_sel_l: list[th.Tensor] = list()
    metrics_func.reset()
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
        # (bsz * n_acts_avail, n_covs + n_act_feats)
        binputs: th.Tensor = th.cat((bctxs_, bxacts), dim=2).flatten(0, 1)
        # (bsz, n_acts)
        # NOTE new impl. that batch over inputs
        bpms: th.Tensor = th.unflatten(
            th.cat(
                [
                    score_est(_binps.to(device=plf.device)).to(device="cpu")
                    for _binps in th.split(
                        binputs, split_size_or_sections=eval_bsz, dim=0
                    )
                ],
                dim=0,
            ),
            dim=0,
            sizes=(bxacts.shape[0], bxacts.shape[1]),
        )
        # choose which action to take
        # (bsz,)
        btargets_est, baidxs = th.max(bpms, dim=1)
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
        # mse of selected actions
        bmses_sel: th.Tensor = th.nn.functional.mse_loss(
            btargets_est, brewards, reduction="none"
        )
        # metrics of current selection
        metrics_func.update(
            binfo["pyhats"][:, :, None].to(device="cpu"),
            binfo["ys"][:, None].to(device="cpu"),
        )
        n_selected_l.append(bn_selected)
        # record metrics
        rewards_l.append(brewards.to(device="cpu"))
        regrets_l.append(bregrets.to(device="cpu"))
        mses_sel_l.append(bmses_sel.to(device="cpu"))
        if to_log_additioanl_metrics:
            # TODO bacts seems off with indices
            # mse of same ctx but random action
            # choose a random action from current available actions
            baidxs = th.randint(
                0, n_acts_avail, (len(bctxs),), dtype=th.long, device=plf.device
            )
            # (bsz, n_acts_feats)
            bacts = th.gather(
                bacts_avail,
                dim=1,
                index=baidxs[:, None, None].expand(-1, -1, n_act_feats),
            )[:, 0]
            brewards_rand: th.Tensor = env.compute_rewards(bctxs, bacts, bctxs_info)[0]
            bmses_rand: th.Tensor = th.nn.functional.mse_loss(
                th.gather(bpms, dim=1, index=baidxs[:, None]).flatten(),
                brewards_rand,
                reduction="none",
            )
            mses_rand_l.append(bmses_rand.to(device="cpu"))
    metrics_d: dict[str, float] = {
        k: v.item() for k, v in metrics_func.compute().items()
    }
    metrics_func.reset()
    # compute average metrics
    reward: th.Tensor = th.mean(th.cat(rewards_l, dim=0))
    regret: th.Tensor = th.mean(th.cat(regrets_l, dim=0))
    mse_sel: th.Tensor = th.mean(th.cat(mses_sel_l, dim=0))
    n_selected: th.Tensor = th.mean(th.cat(n_selected_l, dim=0).to(dtype=th.float32))
    metrics_d.update(
        {
            "reward": reward.item(),
            "regret": regret.item(),
            "mse_selected": mse_sel.item(),
            "n_selected": n_selected.item(),
        }
    )
    if to_log_additioanl_metrics:
        mse_rand: th.Tensor = th.mean(th.cat(mses_rand_l, dim=0))
        metrics_d.update({"mse": mse_rand.item()})
    return metrics_d


def make_init_queries(
    train_env: disslib.Env,
    score_est: ModisteRewardEstBase,
    init_capital: int,
    plf: pl.Fabric,
    rseed: Optional[int] = None,
) -> dict[str, float]:
    if isinstance(train_env, th.nn.Module):
        train_env.eval().to(device=plf.device)
    score_est.to(plf.device)
    train_env.reset()
    # one round of query
    generator: np.random.Generator | None = (
        np.random.default_rng(rseed) if rseed is not None else None
    )
    # (init_capital, n_covs)
    ctxs, ctxs_info = train_env.get_init_ctxs(init_capital, generator=generator)
    ctxs = ctxs.to(device=plf.device)
    ctxs_info = ctxs_info.to(device=plf.device)
    # (init_capital, n_acts_avail, n_act_feats) (init_capital, n_acts_avail)
    xacts, acts_avail = train_env.get_init_avail_actions(ctxs, generator)
    xacts = xacts.to(device=plf.device)
    acts_avail = acts_avail.to(device=plf.device)
    bsz: int = xacts.shape[0]
    n_acts_avail: int = xacts.shape[1]
    n_act_feats: int = xacts.shape[2]
    # (init_capital, )
    aidxs: th.Tensor = (
        th.as_tensor(
            generator.integers(0, n_acts_avail, (bsz,)),
            dtype=th.long,
            device=plf.device,
        )
        if generator is not None
        else th.randint(0, n_acts_avail, (bsz,), dtype=th.long, device=plf.device)
    )
    # (init_capital, )
    acts: th.Tensor = th.gather(
        acts_avail, dim=1, index=aidxs[:, None, None].expand(-1, -1, n_act_feats)
    )[:, 0]
    # (init_capital, n_act_feats)
    xacts: th.Tensor = th.gather(
        xacts, dim=1, index=aidxs[:, None, None].expand(-1, -1, n_act_feats)
    )[:, 0, :]
    # (init_capital, n_covs + n_act_feats)
    init_inputs: th.Tensor = th.cat((ctxs, xacts), dim=1)
    init_targets, init_infos = train_env.compute_rewards(ctxs, acts, ctxs_info)
    score_est.set_train_data_(init_inputs, init_targets, init_infos)
    # fit score estimator
    init_metrics = score_est.fit_(plf)
    init_metrics.update({"n_obs": len(score_est.train_inputs)})
    return init_metrics


def run_trial(
    score_est: ModisteRewardEstBase, strat: OptStrat, plf: pl.Fabric
) -> dict[str, float]:
    score_est.eval().to(device=plf.device)
    strat.env.reset()
    # determine next batch of queries
    bctxs, bacts, bimps = strat.suggest_next_queries(score_est, plf)
    # evaluate suggested ctx-act pair
    bxacts: th.Tensor = strat.env.decompose_acts(bacts)["fcinds"]
    binputs: th.Tensor = th.cat((bctxs, bxacts), dim=1).to(device=plf.device)
    # get ctxs_info for compute_rewards
    _ctxs, bctxs_info = strat.env.get_ctxs(len(bctxs))
    bctxs_info = bctxs_info.to(device=plf.device)
    btargets, binfos = strat.env.compute_rewards(bctxs, bacts, bctxs_info)
    btargets = btargets.to(device=plf.device)
    # compute actual improvement
    bpms: th.Tensor = score_est(binputs)
    brimps: th.Tensor = btargets - bpms
    brimps = brimps.to(device="cpu")
    # add queries to estimator
    score_est.add_to_train_data_(binputs, btargets, binfos)
    metrics_d: dict[str, float] = {
        "imp_avg": th.mean(bimps).item(),
        "imp_max": th.max(bimps).item(),
        "imp_min": th.min(bimps).item(),
        "rimp_avg": th.mean(brimps).item(),
        "rimp_max": th.max(brimps).item(),
        "rimp_min": th.min(brimps).item(),
        "n_obs": len(score_est.train_inputs),
    }
    if not strat.support_lazy_fit:
        fit_metrics_d: dict[str, float] = score_est.fit_(plf)
        metrics_d.update(fit_metrics_d)
    return metrics_d


def maximize(
    train_env: disslib.Env,
    score_est: ModisteRewardEstBase,
    strat: OptStrat,
    init_capital: int,
    n_iter: int,
    metrics_func: thm.MetricCollection,
    plf: pl.Fabric,
    val_env: Optional[disslib.Env] = None,
    test_env: Optional[disslib.Env] = None,
    eval_bsz: int = 1,
    eval_every_n_iter: int = 1,
    ckpt_p: Optional[str] = None,
    save_ckpt_every_n_iter: int = 1,
    init_capital_rseed: Optional[int] = None,
    to_eval_on_train: bool = False,
    to_log_additional_metrics: bool = False,
    gc_every_n_iter: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    # make initial queries
    score_est.to(plf.device)
    # run trials
    pbar = tqdm.trange(n_iter)
    for itr in pbar:
        try:
            metrics_d: dict[str, float]
            if itr == 0:
                metrics_d = make_init_queries(
                    train_env,
                    score_est,
                    init_capital,
                    plf,
                    init_capital_rseed,
                )
            else:
                metrics_d = run_trial(score_est=score_est, strat=strat, plf=plf)
            if (
                strat.support_lazy_fit
                and score_est._enable_lazy_fit
                and ((itr + 1) == n_iter or itr % eval_every_n_iter == 0)
                and (itr != 0)
            ):
                metrics_d.update(score_est.fit_(plf))
            plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "train"), itr)
            # evaluate
            if itr % eval_every_n_iter == 0 or ((itr + 1) == n_iter):
                if to_eval_on_train:
                    # evaluate on train set
                    metrics_d = eval_env(
                        env=train_env,
                        score_est=score_est,
                        metrics_func=metrics_func,
                        plf=plf,
                        eval_bsz=eval_bsz,
                        to_log_additioanl_metrics=to_log_additional_metrics,
                    )
                    plf.log_dict(
                        mylib.utils.add_prefix_to_dict(metrics_d, "train"), itr
                    )
                # evaluate on validation set
                if val_env is not None:
                    metrics_d = eval_env(
                        env=val_env,
                        score_est=score_est,
                        metrics_func=metrics_func,
                        plf=plf,
                        eval_bsz=eval_bsz,
                        to_log_additioanl_metrics=to_log_additional_metrics,
                    )
                    plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "val"), itr)
            # save ckpt if needed
            if ckpt_p is not None and (
                (itr % save_ckpt_every_n_iter == 0) or ((itr + 1) == n_iter)
            ):
                plf.save(
                    os.path.join(ckpt_p, f"itr_{itr}.ckpt"), score_est.state_dict()
                )
            if gc_every_n_iter is not None and (itr + 1) % gc_every_n_iter == 0:
                gc.collect()
        except (KeyboardInterrupt, RuntimeError) as e:
            traceback.print_exception(e)
            if logger is not None:
                logger.error(e, exc_info=True, stack_info=True)
            break
    pbar.close()
    if ckpt_p is not None:
        plf.save(os.path.join(ckpt_p, "itr_end.ckpt"), score_est.state_dict())
    # evaluate on test set
    if test_env is not None:
        metrics_d: dict[str, float] = eval_env(
            env=test_env,
            score_est=score_est,
            metrics_func=metrics_func,
            plf=plf,
            eval_bsz=eval_bsz,
            to_log_additioanl_metrics=to_log_additional_metrics,
        )
        plf.log_dict(mylib.utils.add_prefix_to_dict(metrics_d, "test"), n_iter)
    return


def main(cfg: MainConf, logger: logging.Logger):
    output_dir: str = HydraConfig.get().runtime.output_dir
    # make components used in the experiment
    make_envs_func: MakeEnvsFunc = hd.utils.call(cfg.make_envs_func, _partial_=True)
    tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
    # construct score estimator
    score_est: ModisteRewardEstBase = hd.utils.instantiate(
        cfg.reward_est,
        n_ctx_covs=tenv.n_covs,
        n_bdms_per_fcomb=(
            tenv.n_bdms_per_fcomb  # type:ignore
            if hasattr(tenv, "n_bdms_per_fcomb")
            else 1
        ),
    )
    # construct optimization strategy
    strat: OptStrat = hd.utils.instantiate(cfg.strat, env=tenv)
    # configure plf and ckpt path
    os.makedirs(output_dir, exist_ok=True)
    tfb_logger = plf_loggers.TensorBoardLogger(root_dir=output_dir, name="", version="")
    csv_logger = plf_loggers.CSVLogger(root_dir=output_dir, name="", version="")
    plf: pl.Fabric = hd.utils.instantiate(cfg.plf, _partial_=True)(
        loggers=[tfb_logger, csv_logger],
        plugins=[plf_plugins_envs.LightningEnvironment()],  # type: ignore
    )
    ckpt_p: str = os.path.join(tfb_logger.log_dir, "checkpoints")
    # train selector
    maximize(
        train_env=tenv,
        score_est=score_est,
        strat=strat,
        init_capital=cfg.train_conf.init_capital,
        n_iter=cfg.train_conf.n_iter,
        metrics_func=metrics_func,
        plf=plf,
        val_env=venv,
        test_env=tstenv,
        eval_bsz=cfg.train_conf.eval_bsz,
        eval_every_n_iter=cfg.train_conf.eval_every_n_iter,
        ckpt_p=ckpt_p,
        save_ckpt_every_n_iter=cfg.train_conf.save_ckpt_every_n_iter,
        init_capital_rseed=cfg.train_conf.init_capital_rseed,
        to_eval_on_train=cfg.train_conf.to_eval_on_train,
        to_log_additional_metrics=cfg.train_conf.to_log_additional_metrics,
        gc_every_n_iter=100,
        logger=logger,
    )
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
