# %%
from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol

import disslib
import envs.uci_health.diabetes130.vllm_env_all_features as diab130_env
import hydra as hd
import lightning as pl
import matplotlib.pyplot as plt
import mylib
import numpy as np
import sklearn.preprocessing as skl_preproc
import sklearn.tree as skl_tree
import tensordict as thd
import torch as th
import torchmetrics as thm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import sklearn.cluster as skl_cluster
else:
    try:
        import cuml
        import cuml.cluster as skl_cluster

        cuml.set_global_output_type("numpy")
    except:
        import sklearn.cluster as skl_cluster

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)


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
    # make_vllm_diss_components returns: (tenv, venv, tstenv, tmmcdata, metrics_func)
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


# %%
# Feature names for the diabetes130 dataset (47 features)
feature_names: list[str] = diab130_env.FEATURE_NAMES

# %%
run_p: str = (
    "experiments/sequential-bandits/outputs/diab130-vllm-distill/20260308_001335/3"
)
run_p = os.path.join(mylib.utils.get_project_root_dir(), run_p)
cfg = OmegaConf.load(os.path.join(run_p, ".hydra", "train_run_cfg.yaml"))  # type:ignore
output_p: str = os.path.join("outputs")
os.makedirs(output_p, exist_ok=True)

# %%
# make components used in the experiment
# make_vllm_diss_components returns: (tenv, venv, tstenv, tmmcdata, metrics_func)
make_envs_func: MakeEnvsFunc = hd.utils.call(cfg.make_envs_func, _partial_=True)
tenv, venv, tstenv, mmctdata, metrics_func = make_envs_func()
# Load the StandardScaler and feature_preprocessor fitted on the diabetes130 dataset
_, _, _, _, stdsclr, feature_preprocessor = diab130_env.load_data()
n_covs: int = tenv.n_covs

# %%
ckpt_p: str = os.path.join(run_p, "checkpoints")
ckpt = th.load(
    os.path.join(run_p, "output.ckpt"), map_location="cpu", weights_only=False
)
# distilled decision tree policy
dtc: skl_tree.DecisionTreeClassifier = ckpt["dtc"]
# feature subset mask (boolean indicator vectors, shape: [n_clusters, n_features])
centroids: th.Tensor = ckpt["centroids_b"]

# %%
# Numeric (passthrough) features — thresholds kept as raw numbers
_NUMERIC_FEATURES: frozenset[str] = frozenset(
    [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
)

# Ordinal-encoded features with known category lists (for inverse-transform to labels)
_AGE_CATEGORIES: list[str] = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
_WEIGHT_CATEGORIES: list[str] = [
    "[0-25)",
    "[25-50)",
    "[50-75)",
    "[75-100)",
    "[100-125)",
    "[125-150)",
    "[150-175)",
    "[175-200)",
    ">200",
]
_MEDICATION_CATEGORIES: list[str] = ["No", "Steady", "Up", "Down"]
_CHANGE_CATEGORIES: list[str] = ["No", "Ch"]
_DIABETES_MED_CATEGORIES: list[str] = ["No", "Yes"]
_MAX_GLU_SERUM_CATEGORIES: list[str] = ["None", "Norm", ">200", ">300"]
_A1C_CATEGORIES: list[str] = ["None", "Norm", ">7", ">8"]

_MEDICATION_FEATURES: frozenset[str] = frozenset(
    [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
)

# Map feature name -> ordered category list (for ordinal-encoded features)
_FEATURE_CATEGORIES: dict[str, list[str]] = {
    "age": _AGE_CATEGORIES,
    "weight": _WEIGHT_CATEGORIES,
    "change": _CHANGE_CATEGORIES,
    "diabetesMed": _DIABETES_MED_CATEGORIES,
    "max_glu_serum": _MAX_GLU_SERUM_CATEGORIES,
    "A1Cresult": _A1C_CATEGORIES,
    **{feat: _MEDICATION_CATEGORIES for feat in _MEDICATION_FEATURES},
}


def _ordinal_idx_to_label(feat_name: str, ordinal_idx: float) -> str:
    """
    Convert an ordinal-encoded integer index back to the original category label.

    For features with known category lists, rounds the float index to the nearest
    integer and looks up the label. Falls back to the numeric string if out of range.
    """
    cats = _FEATURE_CATEGORIES.get(feat_name)
    if cats is None:
        return f"{ordinal_idx:.4f}"
    idx = int(round(ordinal_idx))
    if 0 <= idx < len(cats):
        return cats[idx]
    return f"{ordinal_idx:.4f}"


def extract_decision_rules(
    dtc: skl_tree.DecisionTreeClassifier,
    centroids_b: th.Tensor,
    feature_names: list[str],
    stdsclr: skl_preproc.StandardScaler,
) -> list[dict]:
    """
    Extract decision rules from a fitted DecisionTreeClassifier.

    Each leaf node corresponds to a cluster index (the prediction), which maps
    to a row in `centroids_b` — a boolean indicator vector of which features
    should be enabled.

    Decision node thresholds are inverse-transformed from standardized space
    back to the original feature space using `stdsclr`, and then further
    mapped to human-readable category labels using the known ordinal encodings
    from the diabetes130 preprocessing pipeline.

    Parameters
    ----------
    dtc : DecisionTreeClassifier
        Fitted decision tree classifier.
    centroids_b : th.Tensor
        Boolean indicator tensor of shape (n_clusters, n_features).
        Each row indicates which features are enabled for that cluster.
    feature_names : list[str]
        Names of the input features (in the same order as the training data).
    stdsclr : StandardScaler
        Fitted StandardScaler used to standardize the input features.
        Used to inverse-transform thresholds back to ordinal-encoded space.

    Returns
    -------
    list[dict]
        A list of rule dicts, one per leaf node. Each dict has:
          - "conditions": list of condition strings
          - "cluster_id": int, the predicted cluster index
          - "enabled_features": list[str], feature names enabled by that cluster
          - "n_samples": int, number of training samples reaching this leaf
    """
    tree = dtc.tree_
    n_nodes = tree.node_count  # type: ignore[attr-defined]
    children_left = tree.children_left  # type: ignore[attr-defined]
    children_right = tree.children_right  # type: ignore[attr-defined]
    feature = tree.feature  # type: ignore[attr-defined]
    threshold = tree.threshold  # type: ignore[attr-defined]  # thresholds in standardized space
    n_node_samples = tree.n_node_samples  # type: ignore[attr-defined]
    value = tree.value  # type: ignore[attr-defined]  # shape: (n_nodes, n_outputs, n_classes)

    # Inverse-transform thresholds from standardized space back to ordinal-encoded space.
    # stdsclr.mean_ and stdsclr.scale_ have shape (n_features,).
    # threshold_ordinal[i] = threshold[i] * scale_[feature[i]] + mean_[feature[i]]
    mean_: np.ndarray = stdsclr.mean_  # type: ignore[assignment]
    scale_: np.ndarray = stdsclr.scale_  # type: ignore[assignment]

    centroids_b_np: np.ndarray = centroids_b.numpy(force=True)

    rules: list[dict] = []

    def recurse(node_id: int, conditions: list[str]) -> None:
        # TREE_LEAF sentinel is -1 in sklearn's internal tree representation
        is_leaf = children_left[node_id] == -1

        if is_leaf:
            # The predicted class is the argmax over class counts
            cluster_id = int(np.argmax(value[node_id][0]))
            # Map cluster_id to the enabled features via centroids_b
            enabled_mask: np.ndarray = centroids_b_np[cluster_id]
            enabled_features: list[str] = [
                feature_names[i]
                for i in range(len(feature_names))
                if i < len(enabled_mask) and enabled_mask[i]
            ]
            rules.append(
                {
                    "conditions": list(conditions),
                    "cluster_id": cluster_id,
                    "enabled_features": enabled_features,
                    "n_samples": int(n_node_samples[node_id]),
                }
            )
        else:
            feat_idx = feature[node_id]
            feat_name = feature_names[feat_idx]
            thresh_std = threshold[node_id]
            # Step 1: inverse-transform from standardized → ordinal-encoded space
            thresh_ordinal: float = thresh_std * scale_[feat_idx] + mean_[feat_idx]

            if feat_name in _NUMERIC_FEATURES:
                # Numeric passthrough: threshold is already in original units.
                # Decision tree splits on integer-valued features produce thresholds
                # like X.5, so floor/ceil gives the correct integer boundary.
                left_int = int(np.floor(thresh_ordinal))
                right_int = int(np.ceil(thresh_ordinal))
                left_cond = f"{feat_name} <= {left_int}"
                right_cond = f"{feat_name} >= {right_int}"
            elif feat_name in _FEATURE_CATEGORIES:
                # Ordinal-encoded categorical: map integer index → category label.
                # The split threshold sits between two integer codes, so:
                #   left  (≤ thresh) → categories with code ≤ floor(thresh)
                #   right (> thresh) → categories with code > floor(thresh)
                cats = _FEATURE_CATEGORIES[feat_name]
                left_idx = int(np.floor(thresh_ordinal))
                right_idx = left_idx + 1
                left_label = (
                    cats[left_idx] if 0 <= left_idx < len(cats) else str(left_idx)
                )
                right_label = (
                    cats[right_idx] if 0 <= right_idx < len(cats) else str(right_idx)
                )
                left_cond = f"{feat_name} <= {left_label}"
                right_cond = f"{feat_name} >= {right_label}"
            else:
                # Nominal ordinal-encoded (race, gender, payer_code, etc.):
                # categories are auto-assigned; keep numeric threshold
                left_cond = f"{feat_name} <= {thresh_ordinal:.4f}"
                right_cond = f"{feat_name} > {thresh_ordinal:.4f}"

            recurse(children_left[node_id], conditions + [left_cond])
            recurse(children_right[node_id], conditions + [right_cond])

    recurse(0, [])
    return rules


def format_rules(rules: list[dict]) -> str:
    """
    Format extracted decision rules into a human-readable string.

    Each rule is formatted as a single IF ... AND ... THEN block.

    Parameters
    ----------
    rules : list[dict]
        Output of `extract_decision_rules`.

    Returns
    -------
    str
        Formatted string representation of all rules.
    """
    lines: list[str] = []
    for i, rule in enumerate(rules):
        lines.append(f"Rule {i + 1} (n_samples={rule['n_samples']}):")
        conditions = rule["conditions"]
        if conditions:
            lines.append(f"  IF {conditions[0]}")
            for cond in conditions[1:]:
                lines.append(f"    AND {cond}")
        else:
            lines.append("  IF <root> (no conditions)")
        lines.append(f"  THEN acquire features: {rule['enabled_features']}")
        lines.append(f"       (cluster_id={rule['cluster_id']})")
        lines.append("")
    return "\n".join(lines)


# %%
# Extract and print decision rules
rules: list[dict] = extract_decision_rules(
    dtc=dtc,
    centroids_b=centroids,
    feature_names=feature_names,
    stdsclr=stdsclr,
)

rules_str: str = format_rules(rules)
print(rules_str)

# %%
# Save rules to file
rules_output_p: str = os.path.join(output_p, "decision_rules.txt")
with open(rules_output_p, "w") as f:
    f.write(rules_str)
print(f"Decision rules saved to: {rules_output_p}")

# %%
# Visualize the decision tree classifier for the diabetes130 dataset
# Build leaf labels: map each cluster_id to its enabled features (abbreviated)
cluster_labels: list[str] = []
centroids_b_np: np.ndarray = centroids.numpy(force=True)
for cluster_id in range(len(centroids_b_np)):
    enabled: list[str] = [
        feature_names[i]
        for i in range(len(feature_names))
        if i < centroids_b_np.shape[1] and centroids_b_np[cluster_id, i]
    ]
    cluster_labels.append(
        f"C{cluster_id}\n"
        + "\n".join(enabled[:3])
        + ("..." if len(enabled) > 3 else "")
    )

fig, ax = plt.subplots(figsize=(40, 20))
skl_tree.plot_tree(
    dtc,
    feature_names=feature_names,
    class_names=cluster_labels,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False,
    precision=3,
)
ax.set_title(
    "Decision Tree Classifier — Diabetes 130 Dataset\n"
    "(Leaf nodes show cluster ID and top acquired features)",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
tree_viz_p: str = os.path.join(output_p, "decision_tree_diabetes130.png")
fig.savefig(tree_viz_p, dpi=150, bbox_inches="tight")
plt.show()
print(f"Decision tree visualization saved to: {tree_viz_p}")

# %%
# Also export the tree as a text representation using export_text
tree_text: str = skl_tree.export_text(
    dtc,
    feature_names=feature_names,
    max_depth=10,
    spacing=3,
    decimals=4,
    show_weights=True,
)
tree_text_p: str = os.path.join(output_p, "decision_tree_text_diabetes130.txt")
with open(tree_text_p, "w") as f:
    f.write(tree_text)
print(f"Decision tree text export saved to: {tree_text_p}")
