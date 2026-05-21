from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Final, Optional

import disslib
import httpx
import mydatasets
import mylib
import mymodels
import numpy as np
import openai
import pandas as pd
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import tqdm.auto as tqdm

try:
    from . import _common
except ImportError:
    import _common


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
    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
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
    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
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
    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
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
    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
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


def make_vllm_diss_components(
    n_neighs: int,
    server_url: str,
    api_key: str,
    vdata_seed: Optional[int] = None,
):
    class _TensorDictEnv(disslib.envs.base.TensorDictEnv):
        def compute_rewards(
            self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
        ) -> tuple[th.Tensor, thd.TensorDict]:
            assert self._idxs is not None
            ctxs, ys = self._get_verify_ctxs_ys(ctxs)
            pyhats: th.Tensor = self.bdm.predict_proba(ctxs, acts)
            maes: th.Tensor = th.nn.functional.l1_loss(
                pyhats[:, 1], ys, reduction="none"
            )
            fcinds: th.Tensor = self.decompose_acts(acts)["fcinds"]
            rewards = -maes - self.alpha * th.sum(fcinds, dim=1)
            info = thd.TensorDict(
                {
                    "pyhats": pyhats,
                    "ys": ys,
                    "maes": maes,
                }
            ).auto_batch_size_(1)
            return rewards, info

    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
    bdm = MedGemmaVLLMSubsetFeatureClassifier(
        n_neighs=n_neighs,
        tdata=bdmtdata,
        server_url=server_url,
        stdsclr=stdsclr,
        api_key=api_key,
    )
    tenv = _TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=0.0,
        is_train=True,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    venv = _TensorDictEnv(
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


def make_vllm_zero_one_diss_components(
    n_neighs: int,
    server_url: str,
    api_key: str,
    vdata_seed: Optional[int] = None,
):
    class _TensorDictEnv(disslib.envs.base._TensorDictEnvBase):
        def compute_rewards(
            self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
        ) -> tuple[th.Tensor, thd.TensorDict]:
            assert self._idxs is not None
            ctxs, ys = self._get_verify_ctxs_ys(ctxs)
            pyhats: th.Tensor = self.bdm.predict_proba(ctxs, acts)
            zero_one_losses: th.Tensor = (
                th.argmax(pyhats, dim=1).flatten() != ys.flatten()
            ).to(dtype=th.float32)
            fcinds: th.Tensor = self.decompose_acts(acts)["fcinds"]
            rewards = -zero_one_losses - self.alpha * th.sum(fcinds, dim=1)
            info = thd.TensorDict(
                {
                    "pyhats": pyhats,
                    "ys": ys,
                    "zero-one-losses": zero_one_losses,
                }
            ).auto_batch_size_(1)
            return rewards, info

    tdata, tmmcdata, bdmtdata, vdata, stdsclr = load_data(vdata_seed=vdata_seed)
    bdm = MedGemmaVLLMSubsetFeatureClassifier(
        n_neighs=n_neighs,
        tdata=bdmtdata,
        server_url=server_url,
        stdsclr=stdsclr,
        api_key=api_key,
    )
    tenv = _TensorDictEnv(
        data=tdata,
        bdm=bdm,
        alpha=0.0,
        is_train=True,
        n_acts_avail=500,
        min_features=1,
        max_features=None,
    )
    venv = _TensorDictEnv(
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


def load_data(
    vdata_seed: Optional[int],
) -> tuple[
    thd.TensorDict,
    thd.TensorDict,
    thd.TensorDict,
    thd.TensorDict,
    skl_preproc.StandardScaler,
]:
    data: thd.TensorDict = th.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(),
            "uci",
            "cdc-diabetes.pt",
        ),
        map_location="cpu",
        weights_only=False,
    )
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
    assert len(th.unique(tdata["ys"])) == 2
    assert len(th.unique(mmctdata["ys"])) == 2
    assert len(th.unique(bdmtdata["ys"])) == 2
    assert len(th.unique(vdata["ys"])) == 2
    return tdata, mmctdata, bdmtdata, vdata, stdsclr


# ============================================================================
# Constants (CDC Diabetes Dataset)
# ============================================================================
FEATURE_NAMES: Final[list[str]] = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
# Binary features (14 features)
_BINARY_FEATURES: Final[list[str]] = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
# Ordinal features (7 features)
_ORDINAL_FEATURES: Final[list[str]] = [
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "Education",
    "Income",
]
RESPONSE_COLUMNS: Final[list[str]] = [
    "diabetes_risk",
    "confidence",
    "screening_recommendation",
]


# ============================================================================
# Feature Preprocessing (Dataset-Specific)
# ============================================================================
def inverse_transform_xs(
    xs: th.Tensor,
    stdsclr: skl_preproc.StandardScaler | None = None,
) -> pd.DataFrame:
    """Convert encoded tensor to readable DataFrame (numeric format).

    Format: feature_name_{ordinal_value}
    Example: "Age_9", "HighBP_1"

    Args:
        xs: Feature tensor [N, F] - in standardized space if stdsclr is provided
        stdsclr: Optional StandardScaler to inverse standardization back to original feature space

    Returns:
        DataFrame with feature_name_{value} format
    """
    xs_numpy = xs.numpy(force=True)
    # Track NaN positions (from feature masking)
    nan_mask = np.isnan(xs_numpy)

    # Inverse StandardScaler if provided to revert features back to original feature space
    if stdsclr is not None:
        # Replace NaN with 0 temporarily for inverse transform
        xs_temp = np.where(nan_mask, 0.0, xs_numpy)
        xs_ordinal = stdsclr.inverse_transform(xs_temp)
        # Restore NaN positions
        xs_ordinal = np.where(nan_mask, np.nan, xs_ordinal)
        # Round to nearest integer for ordinal values
        xs_ordinal = np.where(np.isnan(xs_ordinal), np.nan, np.round(xs_ordinal))
    else:
        xs_ordinal = xs_numpy

    xs_df = pd.DataFrame(xs_ordinal, columns=FEATURE_NAMES)

    # Convert all to numeric codes
    for col in xs_df.columns:
        xs_df[col] = xs_df[col].apply(
            lambda x: (
                f"{col}_{int(x)}"
                if not (np.isnan(x) if isinstance(x, float) else False)
                else "Unknown"
            )
        )

    return xs_df


def dataframe_to_json(df: pd.DataFrame, include_outcome: bool = False) -> dict:
    """Convert patient dataframe to simple flat JSON (numeric format).

    For binary features, adds suffix: feature_name_{0/1}
    For ordinal features, adds suffix: feature_name_{ordinal_value}

    Args:
        df: DataFrame with feature columns
        include_outcome: Whether to include diabetes_outcome

    Returns:
        Dict with patient data in flat JSON format
    """
    if len(df) == 1:
        row = df.iloc[0]
        patient_data = {}

        # Add all features as flat key-value pairs
        for col in df.columns:
            if (
                col != "diabetes_outcome"
                and pd.notna(row[col])
                and row[col] != "Unknown"
            ):
                patient_data[col] = row[col]

        # Add outcome if requested
        if include_outcome and "diabetes_outcome" in df.columns:
            patient_data["diabetes_outcome"] = int(row["diabetes_outcome"])

        return {"patient": patient_data}
    else:
        # Similar cases
        similar_cases = []
        for idx, row in df.iterrows():
            case_json = dataframe_to_json(
                pd.DataFrame([row]), include_outcome=include_outcome
            )
            similar_cases.append(case_json)
        return {"similar_cases": similar_cases}


# ============================================================================
# JSON Extraction (Dataset-Specific - depends on prompt format)
# ============================================================================
def extract_json_from_response(text: str) -> dict:
    """Extract JSON with "values" wrapper from model response.

    Matches notebook approach: Look for {"values": {...}} pattern.

    Args:
        text: Raw model response

    Returns:
        Parsed JSON dict, or default dict on failure
    """
    # Clean up markdown and special tokens
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<unused\d+>.*?</unused\d+>", "", text)
    text = re.sub(r"<unused\d+>\w+", "", text)
    # Try to find JSON with "values" wrapper
    try:
        json_match = re.search(
            r'\{[^}]*"values"\s*:\s*\{[^}]*\}[^}]*\}', text, re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: try to find complete JSON object
    start = text.find("{")
    if start != -1:
        brace_count = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    # Return default on failure
    return {
        "values": {"diabetes_risk": 2, "confidence": 0, "screening_recommendation": ""}
    }


# ============================================================================
# Prompt Generation (Dataset-Specific)
# ============================================================================
def _make_prompt(
    ctx: th.Tensor,
    act: th.Tensor,
    extdata: thd.TensorDict,
    n_neighs: int,
    stdsclr: skl_preproc.StandardScaler | None = None,
) -> list[dict[str, str]]:
    """Generate prompt for MedGemma (CDC diabetes risk task).

    CRITICAL FIX: Instructions in USER message, not system message!
    MedGemma ignores system messages, so all format requirements must be
    in the user message where the model will actually see them.

    Args:
        ctx: Context tensor (single patient features) [F] - in standardized space
        act: Active features mask [F]
        extdata: External data (training set) for k-NN - in standardized space
        n_neighs: Number of nearest neighbors to include
        stdsclr: Optional StandardScaler for inverse transform to original feature space

    Returns:
        List of message dicts with 'role' and 'content' keys

    Note:
        k-NN computation is performed in standardized space for consistency.
        Feature values in prompts are inverse-transformed back to original space
        for human-readable LLM input.
    """
    # Apply mask
    ctxs = _common.apply_feature_mask(ctx[None, :], act[None, :])
    active_indices = (
        _common.get_active_feature_indices(act) if act is not None else None
    )

    # Get k nearest neighbors (if n_neighs > 0)
    if n_neighs > 0:
        knn_indices = _common.get_knn_indices(
            query_features=ctxs,
            training_features=extdata["xs"],
            n_neighbors=n_neighs,
            active_feature_indices=active_indices,
        )
        knndata = extdata[knn_indices]
        knnctxs = _common.apply_feature_mask(
            knndata["xs"], act[None, :].expand_as(knndata["xs"])
        )

        # Convert to JSON (inverse transform to original feature space for prompts)
        knnctxs_df = inverse_transform_xs(knnctxs, stdsclr)
        knnctxs_df.dropna(axis=1, inplace=True)
        knnresps_df = pd.DataFrame(
            knndata["ys"].numpy(force=True), columns=["diabetes_outcome"]
        )
        knndata_df = pd.concat((knnctxs_df, knnresps_df), axis=1)
        knn_json = dataframe_to_json(knndata_df, include_outcome=True)
        knndata_str = json.dumps(knn_json, indent=2)
    else:
        # Zero-shot mode: no k-NN examples
        knndata_str = None

    ctxs_df = inverse_transform_xs(ctxs, stdsclr).dropna(axis=1)
    patient_json = dataframe_to_json(ctxs_df, include_outcome=False)
    ctx_str = json.dumps(patient_json, indent=2)

    # Build similar cases section (only if n_neighs > 0)
    if knndata_str is not None:
        similar_cases_section = f"""
Similar Cases (k-NN):
{knndata_str}

"""
    else:
        similar_cases_section = ""

    # CRITICAL: Instructions in USER message, not system!
    messages = [
        {
            "role": "system",
            "content": """You are a medical AI assistant specializing in diabetes risk assessment.

Analyze the patient's health indicator codes and similar cases to provide:
1. Diabetes risk score (0-4: 0=very low, 1=low, 2=moderate, 3=high, 4=very high)
2. Confidence in assessment (0-4: 0=very unsure, 1=unsure, 2=moderate, 3=confident, 4=very confident)
3. Screening recommendation (brief text)

Feature encoding:
- Binary features (suffix _0 or _1): 0=No, 1=Yes
- Age: 1-13 (1=18-24 through 13=80+)
- GenHlth: 1-5 (1=Excellent through 5=Poor)
- Education: 1-6 (1=Never attended through 6=College graduate)
- Income: 1-8 (1=<$10k through 8=$75k+)
- BMI: Body Mass Index (continuous)
- MentHlth/PhysHlth: Days in past 30 (0-30)
- Sex: 0=Female, 1=Male

Output ONLY valid JSON in this exact format:
{
  "values": {
    "diabetes_risk": <0-4>,
    "confidence": <0-4>,
    "screening_recommendation": "<recommendation>"
  }
}""",
        },
        {
            "role": "user",
            "content": f"""Patient Codes:
{ctx_str}

{similar_cases_section}CRITICAL: You MUST output this EXACT format:

Assess diabetes risk based on the encoded health indicators.
Provide assessment in the JSON format specified.""",
        },
    ]

    return messages


# ============================================================================
# MedGemma Human Simulator (CDC Diabetes)
# ============================================================================
class MedGemmaVLLMSubsetFeatureClassifier(mymodels.classifiers.SubsetFeatureClassifier):
    """MedGemma + vLLM Human Simulator (Server Mode) for CDC diabetes risk assessment.

    Connects to a running vLLM server via OpenAI-compatible API.
    Start the server first: python scripts/vllm_server.py

    Args:
        n_neighs: Number of nearest neighbors to include in prompts
        tdata: Training data TensorDict
        server_url: vLLM server URL (e.g., "http://localhost:8000/v1")
        stdsclr: Optional StandardScaler for feature normalization
        api_key: API key for server authentication (default: "not-needed")
        verbose: Print progress information

    Example:
        >>> # First start server: python scripts/vllm_server.py
        >>> classifier = MedGemmaVLLMSubsetFeatureClassifier(
        ...     n_neighs=3,
        ...     tdata=tdata,
        ...     server_url="http://localhost:8000/v1",
        ... )
    """

    n_neighs: int
    stdsclr: skl_preproc.StandardScaler | None
    tdata_: thd.TensorDict
    server_url: str
    _client: openai.AsyncOpenAI
    _model_id: str

    def __init__(
        self,
        n_neighs: int,
        tdata: thd.TensorDict,
        server_url: str,
        stdsclr: Optional[skl_preproc.StandardScaler] = None,
        api_key: str = "not-needed",
    ):
        super().__init__(
            n_bdms_per_fcomb=1,
            xs_train=tdata["xs"],
            ys_train=tdata["ys"],
        )
        self.n_neighs = n_neighs
        self.stdsclr = stdsclr
        self.server_url = server_url

        # Create sync client with longer timeout for model detection
        client = openai.OpenAI(base_url=server_url, api_key=api_key, timeout=30.0)

        # Validate server connection with retries
        max_retries = 10
        retry_delay = random.randint(1, 10)
        for attempt in range(max_retries):
            try:
                logging.debug(
                    f"Attempting to connect to vLLM server at {server_url} (attempt {attempt + 1}/{max_retries})"
                )
                models = client.models.list()
                if not models.data:
                    raise RuntimeError(
                        f"No models available on vLLM server at {server_url}"
                    )
                self._model_id = models.data[0].id
                logging.debug(
                    f"Successfully connected to vLLM server. Model ID: {self._model_id}"
                )
                break
            except openai.APIConnectionError as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to vLLM server at {server_url} after {max_retries} attempts. "
                        f"Please ensure the server is running and accessible. "
                        f"Original error: {e}"
                    ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error while connecting to vLLM server at {server_url}: {e}"
                ) from e

        client.close()

        # Create async client with connection pooling to prevent file descriptor exhaustion
        # Limit max connections to prevent "Too many open files" error
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=50,  # Keep max 50 connections alive
                max_connections=100,  # Hard limit on total connections
            ),
            timeout=httpx.Timeout(9999.0),
        )
        self._client = openai.AsyncOpenAI(
            base_url=server_url,
            api_key=api_key,
            timeout=9999.0,
            http_client=http_client,
        )
        # Query server for available model
        self.tdata_ = tdata

    def set_tdata(self, tdata: thd.TensorDict):
        """Set training data for k-NN retrieval."""
        self.tdata_ = tdata

    def forward(
        self, xs: th.Tensor, fms: th.Tensor, save_prompts_to: str | None = None
    ) -> thd.TensorDict:
        """Forward pass - query MedGemma for diabetes risk predictions.

        Uses the same implementation as query() but with stored n_neighs and tdata.

        Args:
            xs: Input features [N, F]
            fms: Feature masks [N, F] (optional, defaults to all 1s if None)
            save_prompts_to: Optional directory to save sample prompts

        Returns:
            TensorDict with keys: risk, confidence, probability
        """
        if self.tdata_ is None:
            raise RuntimeError("Training data not set. Call set_tdata() first.")
        # Delegate to query method
        return self.query(
            xs=xs,
            fms=fms,
            infos=thd.TensorDict(),  # Not used
            extdata=self.tdata_,
            n_neighs=self.n_neighs,
            save_prompts_to=save_prompts_to,
        )

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        """Predict class probabilities.

        Args:
            ctxs: Input features [N, F]
            acts: Feature masks [N, F]

        Returns:
            Tensor of shape (N, 2) with class probabilities on self.device
        """
        outs: thd.TensorDict = self.forward(ctxs, acts)
        # Get probability from query result (already mapped using _common helper)
        pyhats: th.Tensor = th.clamp(outs["probability"], 0.01, 0.99).to(
            device="cpu", dtype=th.float32
        )
        # Return shape (N, 2) tensor on self.device
        pyhats = th.cat((1 - pyhats[:, None], pyhats[:, None]), dim=1).to(
            device=self.device
        )
        return pyhats

    def query(
        self,
        xs: th.Tensor,
        fms: th.Tensor,
        infos: thd.TensorDict,
        extdata: thd.TensorDict,
        n_neighs: int,
        save_prompts_to: Optional[str] = None,
    ) -> thd.TensorDict:
        """Query vLLM with batch of patients.

        Args:
            xs: Patient features (B, F)
            fms: Feature masks (B, F)
            infos: Additional info (not used)
            extdata: Training data for k-NN
            n_neighs: Number of neighbors to retrieve
            save_prompts_to: Optional directory to save prompts

        Returns:
            TensorDict with "risk", "confidence", and "probability" predictions
        """
        # Build all prompts
        prompts = []
        for i in range(len(xs)):
            messages = _make_prompt(
                xs[i],
                fms[i],
                extdata,
                n_neighs=n_neighs,
                stdsclr=self.stdsclr,
            )
            prompts.append(messages)
        # Save prompts if requested
        if save_prompts_to:
            os.makedirs(save_prompts_to, exist_ok=True)
            prompts_file = os.path.join(save_prompts_to, "cdc-prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(prompts[:5], f, indent=2)  # Save first 5
            logging.debug(f"Saved prompts to: {prompts_file}")

        # Query vLLM server via async OpenAI client for batching
        async def _query_single(
            messages: list[dict[str, str]], max_retries: int = 3
        ) -> str:
            """Query single prompt with retry logic for connection errors."""
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    completion = await self._client.chat.completions.create(
                        model=self._model_id,
                        messages=messages,  # type: ignore
                        temperature=0.1,
                        top_p=0.85,
                        max_tokens=3072,
                    )
                    return completion.choices[0].message.content or ""
                except openai.APIConnectionError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = random.randint(3**attempt, 5**attempt)
                        logging.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(
                            f"Failed to query after {max_retries} attempts: {e}"
                        )
                except Exception as e:
                    logging.error(f"Unexpected error during query: {e}")
                    raise

            # If we exit the loop without returning, raise the last connection error
            raise RuntimeError(
                f"vLLM server connection lost during inference at {self.server_url}. "
                f"Please check if the server is still running. Original error: {last_error}"
            ) from last_error

        async def _query_all() -> list[str]:
            # Limit concurrent connections to prevent "Too many open files" error
            # This semaphore ensures we never exceed 50 concurrent HTTP requests
            semaphore = asyncio.Semaphore(50)

            async def _query_with_limit(messages: list[dict[str, str]]) -> str:
                """Query with semaphore to limit concurrent connections."""
                async with semaphore:
                    return await _query_single(messages, 10)

            tasks = [_query_with_limit(m) for m in prompts]
            # Use tqdm_asyncio for progress tracking with concurrent tasks
            return await tqdm.asyncio_tqdm.gather(  # type: ignore
                *tasks, desc="openai", dynamic_ncols=True, leave=False
            )

        # Run async queries (batched by vLLM server's continuous batching)
        try:
            response_texts: list[str] = asyncio.run(_query_all())
        except RuntimeError as e:
            # Re-raise with more context if it's our custom error
            if "vLLM server connection lost" in str(e):
                raise
            raise RuntimeError(
                f"Failed to complete batch inference. This likely means the vLLM server at {self.server_url} "
                f"stopped responding or crashed during processing. Original error: {e}"
            ) from e
        except openai.APIConnectionError as e:
            raise RuntimeError(
                f"Lost connection to vLLM server at {self.server_url} during batch inference. "
                f"The server may have crashed or become unresponsive. Original error: {e}"
            ) from e
        # Save prompts if requested
        if save_prompts_to:
            os.makedirs(save_prompts_to, exist_ok=True)
            prompts_file = os.path.join(save_prompts_to, "cdc-responses.txt")
            with open(prompts_file, "w") as f:
                f.writelines(response_texts)
            logging.debug(f"saved responses to: {prompts_file}")
        # Parse responses
        risks = list()
        confidences = list()
        recommendations = list()
        json_errors = 0
        sample_outputs = []
        for i, response_text in enumerate(
            tqdm.tqdm(
                response_texts, desc="parse resp.", dynamic_ncols=True, leave=False
            )
        ):
            # Save first 5 for debugging
            if i < 5:
                sample_outputs.append(response_text)
            try:
                # Extract JSON with "values" wrapper (returns dict or default)
                response_data = extract_json_from_response(response_text)
                # Extract from "values" object
                risk = int(response_data.get("values", {}).get("diabetes_risk", 2))
                confidence = int(response_data.get("values", {}).get("confidence", 0))
                recommendation = response_data.get("values", {}).get(
                    "screening_recommendation", ""
                )
                # Clamp to 0-4 range
                risk = mylib.utils.clamp(risk, 0, 4)
                confidence = mylib.utils.clamp(confidence, 0, 4)
                risks.append(risk)
                confidences.append(confidence)
                recommendations.append(recommendation)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                json_errors += 1
                logging.debug(f"{json_errors}/{len(response_texts)}")
                risks.append(0)
                confidences.append(0)
        # Convert to tensors
        risk_tensor = th.tensor(risks, dtype=th.float32)
        confidence_tensor = th.tensor(confidences, dtype=th.float32)
        # Map to probability [0, 1] (shared function)
        probability_tensor = _common.map_risk_confidence_to_probability(
            risk_tensor, confidence_tensor
        )
        # Return as TensorDict on self.device
        return thd.TensorDict(
            {
                "risk": risk_tensor,
                "confidence": confidence_tensor,
                "probability": probability_tensor,
            },
            batch_size=[len(risks)],
        ).to(device=self.device)

    def __getitem__(self, key: tuple[int, ...]) -> Any:
        raise NotImplementedError()

    def __del__(self):
        """Cleanup async client on deletion to prevent event loop warnings."""
        try:
            if hasattr(self, "_client"):
                # close() is async, need to handle properly in __del__
                try:
                    # Try to get existing event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't use asyncio.run() in a running loop, schedule cleanup
                        loop.create_task(self._client.close())
                    else:
                        # Loop exists but not running, can use it
                        loop.run_until_complete(self._client.close())
                except RuntimeError:
                    # No event loop exists, create one for cleanup
                    asyncio.run(self._client.close())
        except Exception:
            pass


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import tabulate

    def test_medgemma_predict_proba(
        n_samples: int = 50,
        n_neighs: int = 3,
        server_url: str = "http://localhost:8000/v1",
    ) -> dict[str, float]:
        """Test MedGemmaVLLMSubsetFeatureClassifier using predict_proba output.

        Uses torchmetrics with multiclass configuration for binary classification
        since predict_proba returns shape (N, 2).

        Args:
            n_samples: Number of test samples to evaluate
            n_neighs: Number of neighbors for k-NN prompts
            server_url: vLLM server URL

        Returns:
            Dict with metrics: accuracy, precision, recall, f1-score, auroc
        """
        import time

        # Load data
        tdata, _, bdmtdata, vdata, stdsclr = load_data(vdata_seed=42)

        # Sample test data
        test_indices = th.randperm(len(vdata))[:n_samples]
        test_xs = vdata["xs"][test_indices]
        test_ys = vdata["ys"][test_indices].long()

        # Create all-ones feature mask (use all features)
        test_acts = th.ones_like(test_xs, dtype=th.float32)

        # Initialize classifier
        print(f"Initializing MedGemmaVLLMSubsetFeatureClassifier...")
        print(f"  n_neighs: {n_neighs}")
        print(f"  server_url: {server_url}")
        classifier = MedGemmaVLLMSubsetFeatureClassifier(
            n_neighs=n_neighs,
            tdata=bdmtdata,
            server_url=server_url,
            stdsclr=stdsclr,
        )

        # Run prediction
        print(f"\nRunning predict_proba on {n_samples} samples...")
        start_time = time.time()
        pyhats = classifier.predict_proba(test_xs, test_acts)
        elapsed_time = time.time() - start_time

        # Configure metrics for multiclass (since pyhats is (N, 2))
        n_labels = 2
        metrics_func = thm.MetricCollection(
            {
                "acc": thm.Accuracy(task="multiclass", num_classes=n_labels),
                "precision": thm.Precision(task="multiclass", num_classes=n_labels),
                "recall": thm.Recall(task="multiclass", num_classes=n_labels),
                "f1-score": thm.F1Score(task="multiclass", num_classes=n_labels),
                "auroc": thm.AUROC(task="multiclass", num_classes=n_labels),
            }
        )

        # Move to same device
        pyhats_cpu = pyhats.to(device="cpu")
        test_ys_cpu = test_ys.to(device="cpu")

        # Compute metrics
        metrics_result = metrics_func(pyhats_cpu, test_ys_cpu)

        # Prepare results table
        results = {
            "n_samples": n_samples,
            "n_neighs": n_neighs,
            "elapsed_time_secs": elapsed_time,
            "samples_per_sec": n_samples / elapsed_time,
        }
        results.update({k: v.item() for k, v in metrics_result.items()})

        # Print results using tabulate
        print("\n" + "=" * 60)
        print("MedGemma CDC Diabetes Predict Proba Test Results")
        print("=" * 60)

        # Configuration table
        config_table = [
            ["n_samples", n_samples],
            ["n_neighs", n_neighs],
            ["server_url", server_url],
            ["elapsed_time (sec)", f"{elapsed_time:.2f}"],
            ["samples/sec", f"{n_samples / elapsed_time:.2f}"],
        ]
        print("\nConfiguration:")
        print(
            tabulate.tabulate(
                config_table, headers=["Parameter", "Value"], tablefmt="grid"
            )
        )

        # Metrics table
        metrics_table = [
            ["Accuracy", f"{results['acc']:.4f}"],
            ["Precision", f"{results['precision']:.4f}"],
            ["Recall", f"{results['recall']:.4f}"],
            ["F1-Score", f"{results['f1-score']:.4f}"],
            ["AUROC", f"{results['auroc']:.4f}"],
        ]
        print("\nMetrics:")
        print(
            tabulate.tabulate(
                metrics_table, headers=["Metric", "Value"], tablefmt="grid"
            )
        )

        # Sample predictions table
        print("\nSample Predictions (first 10):")
        sample_table = []
        for i in range(min(10, n_samples)):
            sample_table.append(
                [
                    i,
                    test_ys_cpu[i].item(),
                    f"{pyhats_cpu[i, 0].item():.3f}",
                    f"{pyhats_cpu[i, 1].item():.3f}",
                    pyhats_cpu[i].argmax().item(),
                ]
            )
        print(
            tabulate.tabulate(
                sample_table,
                headers=["Idx", "True Label", "P(class=0)", "P(class=1)", "Predicted"],
                tablefmt="grid",
            )
        )

        return results

    def test_medgemma_query_json_errors(
        n_samples: int = 10,
        n_neighs: int = 0,
        server_url: str = "http://localhost:8000/v1",
    ) -> dict[str, float]:
        """Test JSON error tracking by calling query() directly.

        Args:
            n_samples: Number of test samples (keep small for debugging)
            n_neighs: Number of neighbors for k-NN prompts
            server_url: vLLM server URL

        Returns:
            Dict with json_error_rate and json_errors count
        """
        import time

        # Load data
        tdata, _, bdmtdata, vdata, stdsclr = load_data(vdata_seed=42)

        # Sample test data
        test_indices = th.randperm(len(vdata))[:n_samples]
        test_xs = vdata["xs"][test_indices]
        test_ys = vdata["ys"][test_indices].long()

        # Create all-ones feature mask (use all features)
        test_acts = th.ones_like(test_xs, dtype=th.float32)

        # Initialize classifier
        print(f"\n{'='*60}")
        print(f"Testing JSON Error Tracking (query method)")
        print(f"{'='*60}")
        print(f"  n_samples: {n_samples}")
        print(f"  n_neighs: {n_neighs}")
        print(f"  server_url: {server_url}")

        classifier = MedGemmaVLLMSubsetFeatureClassifier(
            n_neighs=n_neighs,
            tdata=bdmtdata,
            server_url=server_url,
            stdsclr=stdsclr,
        )

        # Run query with debug directory
        print(f"\nRunning query on {n_samples} samples (saving prompts to 'debug')...")
        start_time = time.time()
        query_results = classifier.query(
            xs=test_xs,
            fms=test_acts,
            infos=thd.TensorDict(),
            extdata=bdmtdata,
            n_neighs=n_neighs,
            save_prompts_to="debug",
        )
        elapsed_time = time.time() - start_time

        # Print results (JSON errors are logged via logging.debug in query method)
        print(f"\nResults:")
        print(f"  Total samples: {n_samples}")
        print(f"  Elapsed time: {elapsed_time:.2f} sec")
        print(f"  Samples/sec: {n_samples / elapsed_time:.2f}")
        print(f"  Prompts saved to: debug/")
        print(f"  Check debug logs above for JSON error rate")

        return {
            "n_samples": n_samples,
            "elapsed_time_secs": elapsed_time,
        }

    # logging.getLogger().setLevel(logging.DEBUG)

    # Run JSON error tracking test first (small sample)
    test_medgemma_query_json_errors(
        n_samples=10,
        n_neighs=0,
        server_url="http://192.168.1.130:8000/v1",
    )

    # Run full test
    test_medgemma_predict_proba(
        n_samples=1000,
        n_neighs=0,
        server_url="http://192.168.1.130:8000/v1",
    )
