from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
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
    from .. import _common
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
            "oai.pt",
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
    pool["ys"] = pool["ys_womac"].flatten()

    # WOMAC is already binarised by process-oai.py:
    #   score ∈ [0, 5)  → 0 (low / no pain)
    #   score ≥ 5       → 1 (significant pain)
    # Distribution (train+val pool): class 0 → 77.5%, class 1 → 22.5%

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
    tstdata["ys"] = tstdata["ys_womac"].flatten()

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
# Constants (OAI Dataset — WOMAC task)
# ============================================================================
# All 27 OAI features (original order, indices 0–26).
# Context descriptors / static (cost 0.3): indices [0,1,2,3,4,5,6,9,10,16]
# Longitudinal (time-varying): indices [7,8,11,12,13,14,15,17–26]
# Source: ACTOR/LAFA paper (ACM-BCB '26), Tables 8 & 9.
OAI_ALL_FEATURE_NAMES: Final[list[str]] = [
    "HISP",  # [0]  Hispanic/Latino ethnicity indicator (context, cost 0.3)
    "RACE",  # [1]  Self-reported race category (context, cost 0.3)
    "SEX",  # [2]  Participant sex (context, cost 0.3)
    "FAMHXKR",  # [3]  Family history of knee replacement surgery (context, cost 0.3)
    "EDCV",  # [4]  Highest grade or year of school completed (context, cost 0.3)
    "AGE",  # [5]  Age (context, cost 0.3)
    "SMOKE",  # [6]  Smoking history/status (context, cost 0.3)
    "DRNKAMT",  # [7]  Current alcohol consumption amount (longitudinal, cost 0.3)
    "DRKMORE",  # [8]  Alcohol-use indicator: heavier/more frequent drinking (longitudinal, cost 0.3)
    "INCOME2",  # [9]  Household income category (context, cost 0.3)
    "MARITST",  # [10] Marital status (context, cost 0.3)
    "BPSYS",  # [11] Systolic blood pressure (longitudinal, cost 0.5)
    "BPDIAS",  # [12] Diastolic blood pressure (longitudinal, cost 0.5)
    "BMI",  # [13] Body mass index (longitudinal, cost 0.5)
    "CEMPLOY",  # [14] Current employment (longitudinal, cost 0.3)
    "CUREMP",  # [15] Currently work for pay (longitudinal, cost 0.3)
    "MEDINS",  # [16] Medical/health insurance status (context, cost 0.3)
    "JSW_1",  # [17] Radiographic knee joint space width — location 1 (longitudinal, cost 0.8)
    "JSW_2",  # [18] Radiographic knee joint space width — location 2 (longitudinal, cost 0.8)
    "JSW_3",  # [19] Radiographic knee joint space width — location 3 (longitudinal, cost 0.8)
    "JSW_4",  # [20] Radiographic knee joint space width — location 4 (longitudinal, cost 0.8)
    "JSW_5",  # [21] Radiographic knee joint space width — location 5 (longitudinal, cost 0.8)
    "JSW_6",  # [22] Radiographic knee joint space width — location 6 (longitudinal, cost 0.8)
    "JSW_7",  # [23] Radiographic knee joint space width — location 7 (longitudinal, cost 0.8)
    "JSW_8",  # [24] Radiographic knee joint space width — location 8 (longitudinal, cost 0.8)
    "JSW_9",  # [25] Radiographic knee joint space width — location 9 (longitudinal, cost 0.8)
    "JSW_10",  # [26] Radiographic knee joint space width — location 10 (longitudinal, cost 0.8)
]

# OAI WOMAC binary labels (y ∈ {0, 1}; binarised at score < 5 vs ≥ 5)
WOMAC_LABELS: Final[dict[int, str]] = {
    0: "Low / No Pain",  # WOMAC total score < 5
    1: "Significant Pain",  # WOMAC total score ≥ 5
}

RESPONSE_COLUMNS: Final[list[str]] = [
    "pain_risk",
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

    return pd.DataFrame(xs_vals, columns=OAI_ALL_FEATURE_NAMES)


def dataframe_to_json(df: pd.DataFrame, include_outcome: bool = False) -> dict:
    """Convert patient dataframe to flat JSON.

    Args:
        df: DataFrame with feature columns
        include_outcome: Whether to include pain_outcome

    Returns:
        Dict with patient data in flat JSON format
    """
    if len(df) == 1:
        row = df.iloc[0]
        patient_data = {}
        for col in df.columns:
            if col != "pain_outcome" and pd.notna(row[col]) and row[col] != "Unknown":
                try:
                    patient_data[col] = round(float(row[col]), 3)
                except (TypeError, ValueError):
                    patient_data[col] = row[col]
        if include_outcome and "pain_outcome" in df.columns:
            patient_data["pain_outcome"] = WOMAC_LABELS.get(
                int(row["pain_outcome"]), str(int(row["pain_outcome"]))
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
    """Extract JSON with "values" wrapper from model response."""
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
            "pain_risk": 1,
            "confidence": 0,
            "clinical_recommendation": "",
        }
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
    """Generate prompt for MedGemma (OAI knee pain WOMAC task).

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
            knndata["ys"].numpy(force=True), columns=["pain_outcome"]
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
            "content": """You are a medical AI assistant specializing in knee osteoarthritis pain assessment using the WOMAC (Western Ontario and McMaster Universities Arthritis Index) scale.

Analyze the patient's longitudinal clinical and imaging data and similar cases to provide:
1. Pain risk score (0=Low/No Pain / WOMAC total score < 5, 1=Significant Pain / WOMAC total score ≥ 5)
2. Confidence in assessment (0-4: 0=very unsure, 1=unsure, 2=moderate, 3=confident, 4=very confident)
3. Clinical recommendation (brief text)

The features are standardized longitudinal measurements from the OAI (Osteoarthritis Initiative) study including:
- Demographics and context: HISP (Hispanic ethnicity), RACE, SEX, FAMHXKR (family history of knee replacement), EDCV (education), AGE, SMOKE (smoking), INCOME2 (income), MARITST (marital status), MEDINS (medical insurance)
- Lifestyle: DRNKAMT (alcohol consumption), DRKMORE (heavier drinking pattern), CEMPLOY (employment), CUREMP (currently working for pay)
- Clinical measurements: BPSYS (systolic BP), BPDIAS (diastolic BP), BMI
- Radiographic imaging: JSW_1 through JSW_10 (fixed-location knee joint space width measurements — key OA biomarkers)

Output ONLY valid JSON in this exact format:
{
  "values": {
    "pain_risk": <0 or 1>,
    "confidence": <0-4>,
    "clinical_recommendation": "<recommendation>"
  }
}""",
        },
        {
            "role": "user",
            "content": f"""Patient Clinical Data:
{ctx_str}

{similar_cases_section}CRITICAL: You MUST output this EXACT format:

Assess whether the patient has low/no knee pain (0, WOMAC total score < 5) or significant knee pain (1, WOMAC total score ≥ 5) based on the longitudinal clinical and imaging data.
Provide assessment in the JSON format specified.""",
        },
    ]

    return messages


# ============================================================================
# MedGemma Human Simulator (OAI — WOMAC)
# ============================================================================
class MedGemmaVLLMSubsetFeatureClassifier(mymodels.classifiers.SubsetFeatureClassifier):
    """MedGemma + vLLM Human Simulator (Server Mode) for OAI knee pain WOMAC assessment.

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
    ):
        super().__init__(
            n_bdms_per_fcomb=1,
            xs_train=tdata["xs"],
            ys_train=tdata["ys"],
        )
        self.n_neighs = n_neighs
        self.stdsclr = stdsclr
        self.server_url = server_url

        client = openai.OpenAI(base_url=server_url, api_key=api_key, timeout=30.0)

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

        client.close()

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
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

    def set_tdata(self, tdata: thd.TensorDict):
        """Set training data for k-NN retrieval."""
        self.tdata_ = tdata

    def forward(
        self, xs: th.Tensor, fms: th.Tensor, save_prompts_to: str | None = None
    ) -> thd.TensorDict:
        """Forward pass - query MedGemma for OAI WOMAC pain predictions."""
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
        """Predict class probabilities for Low/No Pain (0) vs. Significant Pain (1).

        Args:
            ctxs: Input features [N, F]
            acts: Feature masks [N, F]

        Returns:
            Tensor of shape (N, 2) with class probabilities [P(Low Pain), P(Significant Pain)]
        """
        outs: thd.TensorDict = self.forward(ctxs, acts)
        risk: th.Tensor = outs["risk"].long().clamp(0, 1)
        confidence: th.Tensor = outs["confidence"].float() / 4.0  # [0, 1]
        p_pain = confidence * risk.float() + (1 - confidence) * 0.5
        p_pain = p_pain.clamp(0.01, 0.99)
        pyhats = th.stack([1 - p_pain, p_pain], dim=1)
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
        """Query vLLM with batch of patients."""
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
            prompts_file = os.path.join(save_prompts_to, "oai-womac-prompts.json")
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
                        max_tokens=3072,
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
            semaphore = asyncio.Semaphore(50)

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
            responses_file = os.path.join(save_prompts_to, "oai-womac-responses.txt")
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
                risk = int(response_data.get("values", {}).get("pain_risk", 1))
                confidence = int(response_data.get("values", {}).get("confidence", 0))
                recommendation = response_data.get("values", {}).get(
                    "clinical_recommendation", ""
                )
                risk = mylib.utils.clamp(
                    risk, 0, 1
                )  # binary: 0=Low Pain, 1=Significant Pain
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

    _parser = argparse.ArgumentParser(description="Test MedGemma OAI WOMAC classifier")
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
        print(f"Initializing MedGemmaVLLMSubsetFeatureClassifier (OAI WOMAC)...")
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
        print("MedGemma OAI WOMAC Predict Proba Test Results")
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

        # Sample predictions table (0=Low Pain, 1=Significant Pain)
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
                headers=[
                    "Idx",
                    "True Label",
                    "P(Low Pain)",
                    "P(Sig. Pain)",
                    "Predicted",
                ],
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
        print(f"Testing JSON Error Tracking (query method) — OAI WOMAC")
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
