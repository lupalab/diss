from __future__ import annotations

import itertools as itrtls
import math
from typing import Optional

import tensordict as thd
import torch as th


def make_feature_masks(
    n_covs: int,
    n_masks: int,
    min_features: int,
    max_features: Optional[int],
    generator: Optional[th.Generator],
) -> th.Tensor:
    """make random subset feature masks

    Args:
        n_covs (int): number of covariates
        n_masks (int): number of masks
        min_features (int): minimum number of features enabled in each mask
        max_features (Optional[int]): maximum number of features enabled in each masks; `max_features = n_covs` if `None`

    Returns:
        th.Tensor: (n_masks_actual, n_covs) feature masks
    """
    max_features = n_covs if max_features is None else max_features
    bincnt_fcs_l: list[int] = [
        # in order to accomondate for init_fidx,
        # both n_covs and i is one less than desired n_feats
        min(math.comb(n_covs, i), th.iinfo(th.long).max)
        for i in range(min_features, max_features + 1)
    ]
    n_masks = min(n_masks, sum(bincnt_fcs_l))
    bincnt_fcs: th.Tensor = th.as_tensor(bincnt_fcs_l, dtype=th.long)
    ps: th.Tensor = th.ones_like(bincnt_fcs, dtype=th.float32)
    nfc_from_each_binned_fcs: th.Tensor = th.bincount(
        th.multinomial(ps, n_masks, replacement=True, generator=generator),
        minlength=len(bincnt_fcs),
    )
    # in case number of actions in any of the bin exceeds maximum number of actions
    _curr_bincnts: th.Tensor = nfc_from_each_binned_fcs
    while th.any(_curr_bincnts > bincnt_fcs):
        _tmp_ps: th.Tensor = th.where(_curr_bincnts >= bincnt_fcs, 0.0, 1.0)
        _realloc_cnts: th.Tensor = th.where(
            _curr_bincnts > bincnt_fcs, _curr_bincnts - bincnt_fcs, 0
        )
        _tmp_bincnts: th.Tensor = th.bincount(
            th.multinomial(
                _tmp_ps,
                int(th.sum(_realloc_cnts).item()),
                replacement=True,
                generator=generator,
            ),
            minlength=len(bincnt_fcs),
        )
        _curr_bincnts = _curr_bincnts - _realloc_cnts + _tmp_bincnts
    nfc_from_each_binned_fcs = _curr_bincnts
    # make unique feature combination
    fcs_sets_by_bins: list[set[tuple[int, ...]]] = [set() for _ in bincnt_fcs]
    for _k, (_count, _fcs_set) in enumerate(
        zip(nfc_from_each_binned_fcs, fcs_sets_by_bins)
    ):
        if _count == 0:
            continue
        _nfeats: int = _k + min_features
        while len(_fcs_set) < _count:
            _fc_l: list[int] = th.multinomial(
                th.ones((n_covs,)), num_samples=_nfeats, generator=generator
            ).tolist()
            _fc_l.sort()
            # ensure _ctmpl_fcs are all unique entries
            _fc = tuple(_fc_l)
            if _fc not in _fcs_set:
                _fcs_set.add(_fc)
    # from fcomb to act
    fms: th.Tensor = th.zeros((n_masks, n_covs), dtype=th.long)
    fcs_l: list[tuple[int, ...]] = [_fc for _fcs in fcs_sets_by_bins for _fc in _fcs]
    assert len(fms) == len(fcs_l)
    for _i, _fc in enumerate(fcs_l):
        fms[_i, _fc] = 1
    return fms


def make_cact_fcomb_map(
    n_covs: int,
    min_features: int,
    max_features: Optional[int],
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """make comb-action to feature-combination map

    Args:
        n_covs (int): number of covariate
        min_features (int): minimum number of selected features
        max_features (Optional[int]): maximum number of selected features

    Returns:
        list[tuple[int, ...]]: combination action to feature combination
        dict[tuple[int, ...], int]: feature combination to action combination
    """
    max_features = n_covs if max_features is None else max_features
    assert 0 < min_features and min_features <= n_covs
    assert min_features < max_features and max_features <= n_covs
    cact_to_fcomb = list(
        itrtls.chain(
            *[
                itrtls.combinations(range(n_covs), i)
                for i in range(min_features, max_features + 1)
            ]
        )
    )
    fcomb_to_cact = {fcomb: cact for cact, fcomb in enumerate(cact_to_fcomb)}
    return cact_to_fcomb, fcomb_to_cact


def make_full_action_features(
    n_covs: int,
    min_features: int,
    max_features: Optional[int],
    n_bdms_per_fcomb: int,
) -> tuple[th.Tensor, th.Tensor, list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """
    Generates full action feature representations for all possible feature combinations.

    This function creates indicator tensors for all possible combinations of features (covariates)
    given the specified constraints, and optionally augments them for multiple experts per combination.

    Args:
        n_covs (int): Number of covariates (features).
        min_features (int): Minimum number of features in a combination.
        max_features (Optional[int]): Maximum number of features in a combination. If None, no upper limit is applied.
        n_bdms_per_fcomb (int): Number of experts (or bandits) per feature combination.

    Returns:
        tuple:
            - fcinds_f (th.Tensor): Tensor of shape (num_combinations, n_covs) indicating feature combinations.
            - xacts_f (th.Tensor): Tensor of shape (num_combinations * n_bdms_per_fcomb, n_covs + n_bdms_per_fcomb)
              if n_bdms_per_fcomb > 1, otherwise same as fcinds_f.
            - cact_to_fcomb (list[tuple[int, ...]]): List mapping action indices to feature combinations.
            - fcomb_to_cact (dict[tuple[int, ...], int]): Dictionary mapping feature combinations to action indices.

    Notes:
        - The function relies on `make_cact_fcomb_map` to enumerate all valid feature combinations.
        - If `n_bdms_per_fcomb > 1`, the action features are augmented with one-hot expert indicators.
    """
    cact_to_fcomb, fcomb_to_cact = make_cact_fcomb_map(
        n_covs=n_covs, min_features=min_features, max_features=max_features
    )
    # make all combination action features
    fcinds_f = th.zeros((len(cact_to_fcomb), n_covs), dtype=th.float32)
    for cact, fcomb in enumerate(cact_to_fcomb):
        fcinds_f[cact, fcomb] = 1.0
    # make all expert combination features
    xacts_f: th.Tensor = fcinds_f
    if n_bdms_per_fcomb > 1:
        # (n_experts_per_comb, len(cact_to_fcomb), n_covs)
        fcinds: th.Tensor = fcinds_f[None, :, :].expand(n_bdms_per_fcomb, -1, -1)
        exinds: th.Tensor = th.eye(n_bdms_per_fcomb, dtype=th.float32)[
            :, None, :
        ].expand(-1, len(cact_to_fcomb), -1)
        # (n_experts_per_comb * len(cact_to_fcomb), n_covs + n_experts_per_comb)
        xacts_f = th.cat((fcinds, exinds), dim=2).flatten(0, 1)
    # (
    #   all possible feature combination indicator,
    #   all possible combinations,
    #   action index to feature combination,
    #   feature combination to action index,
    # )
    return fcinds_f, xacts_f, cact_to_fcomb, fcomb_to_cact


def split_expert_policy_train_data(
    data: thd.TensorDict,
    rseed: int = 42,
) -> tuple[thd.TensorDict, thd.TensorDict]:
    """split data into expert train data and policy train data

    Args:
        data (th.TensorDict): dataset
        rseed (int, optional): random seed. Defaults to 42.

    Returns:
        tuple[th.TensorDict, th.TensorDict]: expert train data, policy train data
    """
    rprm = th.randperm(len(data), generator=th.Generator().manual_seed(rseed))
    extdata: thd.TensorDict = data[rprm[: len(rprm) // 2]]
    tdata = data[rprm[len(rprm) // 2 :]]
    return extdata, tdata


def cat_mask(xs: th.Tensor, fms: th.Tensor, dim: int) -> th.Tensor:
    """mask feature then concatenate the mask along `dim`

    Args:
        xs (th.Tensor): input feature values
        fms (th.Tensor): feature masks
        dim (int): the dimension where the feature mask will be concatenated to

    Returns:
        th.Tensor: masked features with its featue mask concatenated along `dim`
    """
    return th.cat((xs * fms, fms), dim=dim)
