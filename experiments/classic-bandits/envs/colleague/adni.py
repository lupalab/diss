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
    import _common  # type: ignore[no-redef]


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
            xs_train=bdmtdata["xs"].numpy(force=True),  # type: ignore
            ys_train=bdmtdata["ys"].numpy(force=True),  # type: ignore
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
            xs_train=bdmtdata["xs"].numpy(force=True),  # type: ignore
            ys_train=bdmtdata["ys"].numpy(force=True),  # type: ignore
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
        xs_train=bdmtdata["xs"].numpy(force=True),  # type: ignore
        ys_train=bdmtdata["ys"].numpy(force=True),  # type: ignore
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


def make_vllm_diss_components(
    n_neighs: int,
    server_url: str,
    api_key: str,
    vdata_seed: Optional[int] = None,
    semaphore_value: int = 50,
):
    class _TensorDictEnv(disslib.envs.base.TensorDictEnv):
        def compute_rewards(
            self, ctxs: th.Tensor, acts: th.Tensor, infos: thd.TensorDict
        ) -> tuple[th.Tensor, thd.TensorDict]:
            assert self._idxs is not None
            ctxs, ys = self._get_verify_ctxs_ys(ctxs)
            pyhats: th.Tensor = self.bdm.predict_proba(ctxs, acts)
            n_labels = pyhats.shape[1]
            # MAE between predicted class probabilities and one-hot true labels
            ys_onehot = th.nn.functional.one_hot(
                ys.long(), num_classes=n_labels
            ).float()
            maes: th.Tensor = th.nn.functional.l1_loss(
                pyhats, ys_onehot, reduction="none"
            ).mean(dim=1)
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
        semaphore_value=semaphore_value,
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
    saved: dict = th.load(
        os.path.join(
            mydatasets.common.get_datasets_files_root_dir(),
            "colleague",
            "adni.pt",
        ),
        map_location="cpu",
        weights_only=False,
    )
    # Concatenate train + val as the full labelled pool; test is held out
    trdata: thd.TensorDict = saved["train"]
    vrdata: thd.TensorDict = saved["val"]
    tstdata: thd.TensorDict = saved["test"]

    # Combine train+val into a single pool for splitting
    pool: thd.TensorDict = thd.cat((trdata, vrdata))
    pool["ys"] = pool["ys"].flatten()

    # Binarise: CN (0) vs. abnormal/impaired (MCI=1 or AD=2 → 1)
    pool["ys"] = (pool["ys"] > 0).to(dtype=th.long)

    # Fit StandardScaler on the combined pool
    stdsclr = skl_preproc.StandardScaler()
    pool["xs_orig"] = pool["xs"]
    pool["xs"] = th.as_tensor(
        stdsclr.fit_transform(pool["xs"].numpy()), dtype=th.float32
    )

    # Apply same scaler to test set
    tstdata["xs_orig"] = tstdata["xs"]
    tstdata["xs"] = th.as_tensor(
        stdsclr.transform(tstdata["xs"].numpy()), dtype=th.float32
    )
    tstdata["ys"] = (tstdata["ys"].flatten() > 0).to(dtype=th.long)

    # Split pool into train / mmc-train / bdm-train / val subsets
    _tidxs, _mmctidxs, _bdmtidxs, _vidxs = [
        th.as_tensor(_d.indices, dtype=th.long)
        for _d in th_data.random_split(
            th_data.TensorDataset(th.arange(len(pool))),
            lengths=(0.34, 0.16, 0.16, 0.34),
            generator=th.Generator().manual_seed(279),
        )
    ]
    _vidxs = _vidxs[
        th.multinomial(
            th.ones((len(_vidxs),), dtype=th.float32),
            num_samples=min(1000, len(_vidxs)),
            generator=th.Generator().manual_seed(
                580 if vdata_seed is None else vdata_seed
            ),
        ).flatten()
    ]
    tdata, mmctdata, bdmtdata, vdata = (
        pool[_tidxs],
        pool[_mmctidxs],
        pool[_bdmtidxs],
        pool[_vidxs],
    )
    return tdata, mmctdata, bdmtdata, vdata, stdsclr


# ============================================================================
# Constants (ADNI Dataset)
# ============================================================================
# 41 longitudinal features from the ADNI study (feat_list from train_data.npz)
FEATURE_NAMES: Final[list[str]] = [
    "AGE",  # Age at visit
    "PTGENDER",  # Sex (1=Male, 2=Female)
    "PTEDUCAT",  # Years of education
    "PTETHCAT",  # Ethnicity category
    "PTRACCAT",  # Race category
    "PTMARRY",  # Marital status
    "APOE4",  # APOE ε4 allele count (0/1/2) — major AD genetic risk factor
    "FDG",  # FDG-PET mean cortical uptake (brain glucose metabolism)
    "PIB",  # PIB-PET amyloid burden
    "AV45",  # AV45-PET amyloid burden (florbetapir)
    "CDRSB",  # CDR Sum of Boxes (clinical dementia rating)
    "ADAS11",  # ADAS-Cog 11-item score (higher = worse cognition)
    "ADAS13",  # ADAS-Cog 13-item score
    "MMSE",  # Mini-Mental State Examination (higher = better)
    "RAVLT_immediate",  # Rey Auditory Verbal Learning Test — immediate recall
    "RAVLT_forgetting",  # RAVLT — forgetting score
    "RAVLT_perc_forgetting",  # RAVLT — percent forgetting
    "RAVLT_learning",  # RAVLT — learning score
    "FAQ",  # Functional Activities Questionnaire (higher = more impaired)
    "MOCA",  # Montreal Cognitive Assessment
    "EcogPtMem",  # Everyday Cognition (patient) — memory
    "EcogPtLang",  # Everyday Cognition (patient) — language
    "EcogPtVisspat",  # Everyday Cognition (patient) — visuospatial
    "EcogPtPlan",  # Everyday Cognition (patient) — planning
    "EcogPtOrgan",  # Everyday Cognition (patient) — organization
    "EcogPtDivatt",  # Everyday Cognition (patient) — divided attention
    "EcogPtTotal",  # Everyday Cognition (patient) — total
    "EcogSPMem",  # Everyday Cognition (study partner) — memory
    "EcogSPLang",  # Everyday Cognition (study partner) — language
    "EcogSPVisspat",  # Everyday Cognition (study partner) — visuospatial
    "EcogSPPlan",  # Everyday Cognition (study partner) — planning
    "EcogSPOrgan",  # Everyday Cognition (study partner) — organization
    "EcogSPDivatt",  # Everyday Cognition (study partner) — divided attention
    "EcogSPTotal",  # Everyday Cognition (study partner) — total
    "Ventricles",  # Ventricular volume (MRI, mm³)
    "Hippocampus",  # Hippocampal volume (MRI, mm³) — key AD biomarker
    "WholeBrain",  # Whole brain volume (MRI, mm³)
    "Entorhinal",  # Entorhinal cortex volume (MRI, mm³)
    "Fusiform",  # Fusiform gyrus volume (MRI, mm³)
    "MidTemp",  # Middle temporal gyrus volume (MRI, mm³)
    "ICV",  # Intracranial volume (MRI, mm³) — normalization factor
]

# ADNI binary diagnosis labels (y ∈ {0, 1}; MCI and AD merged into "Cognitively Impaired")
DIAGNOSIS_LABELS: Final[dict[int, str]] = {
    0: "CN",  # Cognitively Normal
    1: "Cognitively Impaired",  # MCI or Alzheimer's Disease
}

RESPONSE_COLUMNS: Final[list[str]] = [
    "diagnosis_risk",
    "confidence",
    "clinical_recommendation",
]


# ============================================================================
# Feature Preprocessing (Dataset-Specific)
# ============================================================================
def inverse_transform_xs(
    xs: th.Tensor,
    stdsclr: skl_preproc.StandardScaler | None = None,
) -> pd.DataFrame:
    """Convert encoded tensor to readable DataFrame.

    Args:
        xs: Feature tensor [N, F] - in standardized space if stdsclr is provided
        stdsclr: Optional StandardScaler to inverse standardization

    Returns:
        DataFrame with feature values
    """
    xs_numpy = xs.numpy(force=True)
    nan_mask = np.isnan(xs_numpy)

    if stdsclr is not None:
        xs_temp = np.where(nan_mask, 0.0, xs_numpy)
        xs_vals = stdsclr.inverse_transform(xs_temp)
        xs_vals = np.where(nan_mask, np.nan, xs_vals)
    else:
        xs_vals = xs_numpy

    return pd.DataFrame(xs_vals, columns=FEATURE_NAMES)


def dataframe_to_json(df: pd.DataFrame, include_outcome: bool = False) -> dict:
    """Convert patient dataframe to flat JSON.

    Args:
        df: DataFrame with feature columns
        include_outcome: Whether to include diagnosis_outcome

    Returns:
        Dict with patient data in flat JSON format
    """
    if len(df) == 1:
        row = df.iloc[0]
        patient_data = {}
        for col in df.columns:
            if (
                col != "diagnosis_outcome"
                and pd.notna(row[col])
                and row[col] != "Unknown"
            ):
                try:
                    patient_data[col] = round(float(row[col]), 3)
                except (TypeError, ValueError):
                    patient_data[col] = row[col]
        if include_outcome and "diagnosis_outcome" in df.columns:
            patient_data["diagnosis_outcome"] = DIAGNOSIS_LABELS.get(
                int(row["diagnosis_outcome"]), str(int(row["diagnosis_outcome"]))
            )
        return {"patient": patient_data}
    else:
        similar_cases = []
        for idx, row in df.iterrows():
            case_json = dataframe_to_json(
                pd.DataFrame([row]), include_outcome=include_outcome
            )
            similar_cases.append(case_json)
        return {"similar_cases": similar_cases}


# ============================================================================
# JSON Extraction (Dataset-Specific)
# ============================================================================
def extract_json_from_response(text: str) -> dict:
    """Extract JSON with "values" wrapper from model response.

    Args:
        text: Raw model response

    Returns:
        Parsed JSON dict, or default dict on failure
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<unused\d+>.*?</unused\d+>", "", text)
    text = re.sub(r"<unused\d+>\w+", "", text)
    try:
        json_match = re.search(
            r'\{[^}]*"values"\s*:\s*\{[^}]*\}[^}]*\}', text, re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass
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
    return {
        "values": {
            "diagnosis_risk": 1,
            "confidence": 0,
            "clinical_recommendation": "",
        }
    }


SYSTEM_PROMPT = """You are a medical AI assistant specializing in Alzheimer's disease and cognitive impairment assessment.

Analyze the patient's longitudinal biomarker data and similar cases to provide:
1. Diagnosis risk score (0=CN/cognitively normal, 1=Cognitively Impaired/MCI or Alzheimer's disease)
2. Confidence in assessment (0-4: 0=very unsure, 1=unsure, 2=moderate, 3=confident, 4=very confident)
3. Clinical recommendation (brief text)

The features are standardized longitudinal biomarkers from the ADNI study including:
- Neuroimaging measures (MRI volumes: Hippocampus, Entorhinal, Fusiform, MidTemp, WholeBrain, Ventricles, ICV)
- Cognitive test scores (CDRSB, ADAS11, ADAS13, MMSE, MOCA, RAVLT, FAQ)
- PET imaging (FDG, PIB, AV45 amyloid burden)
- Everyday Cognition scales (EcogPt*, EcogSP*)
- Demographics and genetics (AGE, PTGENDER, PTEDUCAT, APOE4)

Output ONLY valid JSON in this exact format:
{
  "values": {
    "diagnosis_risk": <0 or 1>,
    "confidence": <0-4>,
    "clinical_recommendation": "<recommendation>"
  }
}"""
CONTENT_PROMPT = lambda ctx_str, similar_cases_section: f"""Patient Biomarkers:
{ctx_str}

{similar_cases_section}CRITICAL: You MUST output this EXACT format:

Assess whether the patient is cognitively normal (0) or cognitively impaired/Alzheimer's (1) based on the longitudinal biomarker data.
Provide assessment in the JSON format specified."""


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
    """Generate prompt for MedGemma (ADNI cognitive diagnosis task).

    Args:
        ctx: Context tensor (single patient features) [F] - in standardized space
        act: Active features mask [F]
        extdata: External data (training set) for k-NN
        n_neighs: Number of nearest neighbors to include
        stdsclr: Optional StandardScaler for inverse transform

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    ctxs = _common.apply_feature_mask(ctx[None, :], act[None, :])
    active_indices = (
        _common.get_active_feature_indices(act) if act is not None else None
    )

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
        knnctxs_df = inverse_transform_xs(knnctxs, stdsclr)
        knnctxs_df.dropna(axis=1, inplace=True)
        knnresps_df = pd.DataFrame(
            knndata["ys"].numpy(force=True), columns=["diagnosis_outcome"]
        )
        knndata_df = pd.concat((knnctxs_df, knnresps_df), axis=1)
        knn_json = dataframe_to_json(knndata_df, include_outcome=True)
        knndata_str = json.dumps(knn_json, indent=2)
    else:
        knndata_str = None

    ctxs_df = inverse_transform_xs(ctxs, stdsclr).dropna(axis=1)
    patient_json = dataframe_to_json(ctxs_df, include_outcome=False)
    ctx_str = json.dumps(patient_json, indent=2)

    if knndata_str is not None:
        similar_cases_section = f"""
Similar Cases (k-NN):
{knndata_str}

"""
    else:
        similar_cases_section = ""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": CONTENT_PROMPT(ctx_str, similar_cases_section),
        },
    ]

    return messages


# ============================================================================
# MedGemma Human Simulator (ADNI)
# ============================================================================
class MedGemmaVLLMSubsetFeatureClassifier(mymodels.classifiers.SubsetFeatureClassifier):
    """MedGemma + vLLM Human Simulator (Server Mode) for ADNI cognitive diagnosis.

    Connects to a running vLLM server via OpenAI-compatible API.

    Args:
        n_neighs: Number of nearest neighbors to include in prompts
        tdata: Training data TensorDict
        server_url: vLLM server URL (e.g., "http://localhost:8000/v1")
        stdsclr: Optional StandardScaler for feature normalization
        api_key: API key for server authentication (default: "not-needed")
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
        semaphore_value: int = 50,
    ):
        super().__init__(
            n_bdms_per_fcomb=1,
            xs_train=tdata["xs"],
            ys_train=tdata["ys"],
        )
        self.n_neighs = n_neighs
        self.stdsclr = stdsclr
        self.server_url = server_url

        client = openai.OpenAI(base_url=server_url, api_key=api_key, timeout=300.0)

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
                        f"Original error: {e}"
                    ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error while connecting to vLLM server at {server_url}: {e}"
                ) from e
        try:
            logging.debug("Warming up prefix cache with system prompt...")
            client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": CONTENT_PROMPT("", ""),
                    },
                ],
                max_tokens=1,
                temperature=0.1,
            )
            logging.debug("Prefix cache warmup complete.")
        except Exception as e:
            logging.warning(f"Prefix cache warmup failed (non-fatal): {e}")
        client.close()

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=semaphore_value,
                max_connections=semaphore_value,
            ),
            timeout=httpx.Timeout(9999.0),
        )
        self._client = openai.AsyncOpenAI(
            base_url=server_url,
            api_key=api_key,
            timeout=9999.0,
            http_client=http_client,
        )
        self.tdata_ = tdata
        self.semaphore_value = semaphore_value

    def set_tdata(self, tdata: thd.TensorDict):
        """Set training data for k-NN retrieval."""
        self.tdata_ = tdata

    def forward(
        self, xs: th.Tensor, fms: th.Tensor, save_prompts_to: str | None = None
    ) -> thd.TensorDict:
        """Forward pass - query MedGemma for ADNI cognitive diagnosis predictions."""
        if self.tdata_ is None:
            raise RuntimeError("Training data not set. Call set_tdata() first.")
        return self.query(
            xs=xs,
            fms=fms,
            infos=thd.TensorDict(),
            extdata=self.tdata_,
            n_neighs=self.n_neighs,
            save_prompts_to=save_prompts_to,
        )

    def predict_proba(self, ctxs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        """Predict class probabilities for CN (0) vs. Cognitively Impaired (1).

        Args:
            ctxs: Input features [N, F]
            acts: Feature masks [N, F]

        Returns:
            Tensor of shape (N, 2) with class probabilities [P(CN), P(Cognitively Impaired)]
        """
        outs: thd.TensorDict = self.forward(ctxs, acts)
        # risk is 0 (CN) or 1 (Cognitively Impaired); use confidence to soften
        risk: th.Tensor = outs["risk"].long().clamp(0, 1)
        confidence: th.Tensor = outs["confidence"].float() / 4.0  # [0, 1]
        # P(impaired): confident → risk value; uncertain → 0.5
        p_impaired = confidence * risk.float() + (1 - confidence) * 0.5
        p_impaired = p_impaired.clamp(0.01, 0.99)
        pyhats = th.stack([1 - p_impaired, p_impaired], dim=1)
        return pyhats.to(device=self.device, dtype=th.float32)

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

        if save_prompts_to:
            os.makedirs(save_prompts_to, exist_ok=True)
            prompts_file = os.path.join(save_prompts_to, "adni-prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(prompts[:5], f, indent=2)
            logging.debug(f"Saved prompts to: {prompts_file}")

        async def _query_single(
            messages: list[dict[str, str]], max_retries: int = 3
        ) -> str:
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    completion = await self._client.chat.completions.create(
                        model=self._model_id,
                        messages=messages,  # type: ignore
                        temperature=0.1,
                        top_p=0.85,
                        max_tokens=128,
                    )
                    return completion.choices[0].message.content or ""
                except openai.APIConnectionError as e:
                    last_error = e
                    if attempt < max_retries - 1:
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
            raise RuntimeError(
                f"vLLM server connection lost during inference at {self.server_url}. "
                f"Original error: {last_error}"
            ) from last_error

        async def _query_all() -> list[str]:
            semaphore = asyncio.Semaphore(self.semaphore_value)

            async def _query_with_limit(messages: list[dict[str, str]]) -> str:
                async with semaphore:
                    return await _query_single(messages, 10)

            tasks = [_query_with_limit(m) for m in prompts]
            return await tqdm.asyncio_tqdm.gather(  # type: ignore
                *tasks, desc="openai", dynamic_ncols=True, leave=False
            )

        try:
            response_texts: list[str] = asyncio.run(_query_all())
        except RuntimeError as e:
            if "vLLM server connection lost" in str(e):
                raise
            raise RuntimeError(
                f"Failed to complete batch inference. Original error: {e}"
            ) from e
        except openai.APIConnectionError as e:
            raise RuntimeError(
                f"Lost connection to vLLM server at {self.server_url}. Original error: {e}"
            ) from e

        if save_prompts_to:
            os.makedirs(save_prompts_to, exist_ok=True)
            responses_file = os.path.join(save_prompts_to, "adni-responses.txt")
            with open(responses_file, "w") as f:
                f.writelines(response_texts)
            logging.debug(f"Saved responses to: {responses_file}")

        risks = []
        confidences = []
        recommendations = []
        json_errors = 0
        for i, response_text in enumerate(
            tqdm.tqdm(
                response_texts, desc="parse resp.", dynamic_ncols=True, leave=False
            )
        ):
            try:
                response_data = extract_json_from_response(response_text)
                risk = int(response_data.get("values", {}).get("diagnosis_risk", 1))
                confidence = int(response_data.get("values", {}).get("confidence", 0))
                recommendation = response_data.get("values", {}).get(
                    "clinical_recommendation", ""
                )
                risk = mylib.utils.clamp(
                    risk, 0, 1
                )  # binary: 0=CN, 1=Cognitively Impaired
                confidence = mylib.utils.clamp(confidence, 0, 4)
                risks.append(risk)
                confidences.append(confidence)
                recommendations.append(recommendation)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                json_errors += 1
                logging.debug(f"{json_errors}/{len(response_texts)}")
                risks.append(1)
                confidences.append(0)
                recommendations.append("")

        risk_tensor = th.tensor(risks, dtype=th.float32)
        confidence_tensor = th.tensor(confidences, dtype=th.float32)
        probability_tensor = _common.map_risk_confidence_to_probability(
            risk_tensor, confidence_tensor
        )
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
        """Cleanup async client on deletion."""
        try:
            if hasattr(self, "_client"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._client.close())
                    else:
                        loop.run_until_complete(self._client.close())
                except RuntimeError:
                    asyncio.run(self._client.close())
        except Exception:
            pass


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    import tabulate

    _parser = argparse.ArgumentParser(description="Test MedGemma ADNI classifier")
    _parser.add_argument(
        "--server-url",
        type=str,
        default="http://192.168.1.130:8000/v1",
        help="vLLM server URL (default: http://192.168.1.130:8000/v1)",
    )
    _args = _parser.parse_args()

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
        print(f"Initializing MedGemmaVLLMSubsetFeatureClassifier (ADNI)...")
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
        print("MedGemma ADNI Predict Proba Test Results")
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

        # Sample predictions table (0=CN, 1=Cognitively Impaired)
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
                headers=["Idx", "True Label", "P(CN)", "P(Cog. Impaired)", "Predicted"],
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
            Dict with elapsed_time_secs and n_samples
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
        print(f"Testing JSON Error Tracking (query method) — ADNI")
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

        # Print results
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
        server_url=_args.server_url,
    )

    # Run full test
    test_medgemma_predict_proba(
        n_samples=1000,
        n_neighs=0,
        server_url=_args.server_url,
    )
