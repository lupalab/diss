from __future__ import annotations

import logging
import os

import mydatasets
import numpy as np
import pandas as pd
import sklearn.compose as skl_compose
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th
import ucimlrepo


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Preparing Secondary Mushroom dataset")
    data = ucimlrepo.fetch_ucirepo(id=848)

    xs_df: pd.DataFrame = data.data.features.copy()
    ys_df: pd.DataFrame = data.data.targets.copy()

    # NOTE Classify feature types based on UCI documentation
    # The dataset has 20 features total: 2 continuous (metrical) + 18 categorical (nominal)

    # -- continuous features: already numeric, no encoding needed
    _continuous_features = [
        "cap-diameter",
        "stem-height",
    ]

    # -- nominal categorical features: no natural order, will use ordinal encoding
    # Based on UCI documentation, these are all the categorical features
    _nominal_features = [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "does-bruise-or-bleed",
        "gill-attachment",
        "gill-spacing",
        "gill-color",
        "stem-width",
        "stem-root",
        "stem-surface",
        "stem-color",
        "veil-type",
        "veil-color",
        "has-ring",
        "ring-type",
        "spore-print-color",
        "habitat",
        "season",
    ]

    # NOTE Get all feature names from the dataset
    feature_names = xs_df.columns.tolist()
    target_names = ys_df.columns.tolist()

    # NOTE Missing value handling strategy
    # Similar to diabetes-130 dataset, use a sentinel value for missing/unknown values
    # All OrdinalEncoders will map NaN and unseen categories to this value
    _MISSING_SENTINEL: int = -10

    # NOTE common kwargs for all OrdinalEncoders
    _ordinal_kwargs = dict(
        handle_unknown="use_encoded_value",
        unknown_value=_MISSING_SENTINEL,
        encoded_missing_value=_MISSING_SENTINEL,
        dtype=np.int64,
    )

    # NOTE Create column transformer for features
    # Adjust feature lists based on actual column names in the dataset
    _continuous_features_actual = [f for f in _continuous_features if f in feature_names]
    _nominal_features_actual = [
        f for f in feature_names if f not in _continuous_features_actual
    ]

    feature_preprocessor = skl_compose.ColumnTransformer(
        transformers=[
            # NOTE continuous features: passthrough (keep as-is)
            (
                "continuous_passthrough",
                "passthrough",
                _continuous_features_actual,
            ),
            # NOTE nominal categorical features: ordinal encoded (auto categories)
            # Missing values will be encoded as -10
            (
                "nominal_ordinal",
                skl_preproc.OrdinalEncoder(**_ordinal_kwargs),
                _nominal_features_actual,
            ),
        ],
        remainder="drop",
    )

    # NOTE ensure feature names are preserved in output
    feature_preprocessor.set_output(transform="pandas")
    mushroom_xs: pd.DataFrame = feature_preprocessor.fit_transform(xs_df[feature_names])

    # NOTE strip transformer prefix from column names (e.g. "nominal_ordinal__cap-shape" -> "cap-shape")
    # and reorder columns to match feature_names
    mushroom_xs = mushroom_xs.rename(columns=lambda x: str.split(x, "__", 1)[-1])[
        feature_names
    ]

    # NOTE Encode target: class -> binary {e: 0, p: 1}
    # where e=edible, p=poisonous
    target_preprocessor = skl_preproc.OrdinalEncoder(
        categories=[["e", "p"]], dtype=np.int64
    )
    target_preprocessor.set_output(transform="pandas")
    mushroom_ys: pd.DataFrame = target_preprocessor.fit_transform(ys_df[target_names])

    # NOTE Create TensorDict with processed data
    data_thd: thd.TensorDict = thd.make_tensordict(
        {
            "xs": th.as_tensor(mushroom_xs.to_numpy(), dtype=th.float32),
            "ys": th.as_tensor(mushroom_ys.to_numpy(), dtype=th.int64),
        },
        auto_batch_size=True,
    )

    # NOTE Store raw preprocessors and DataFrames for reference
    raw_data_d = {
        "feature_preprocessor": feature_preprocessor,
        "target_preprocessor": target_preprocessor,
        "mushroom_xs_df": mushroom_xs,
        "mushroom_ys_df": mushroom_ys,
    }

    # NOTE Save processed data
    os.makedirs(
        os.path.join(mydatasets.common.get_datasets_files_root_dir(), "uci"),
        exist_ok=True,
    )

    output_path_tensor = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "secondary-mushroom.pt",
    )
    output_path_raw = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "raw-secondary-mushroom.pkl",
    )

    th.save(data_thd, output_path_tensor)
    th.save(raw_data_d, output_path_raw)

    logger.info("Secondary Mushroom xs shape: %s", tuple(data_thd["xs"].shape))
    logger.info("Secondary Mushroom ys shape: %s", tuple(data_thd["ys"].shape))
    logger.info("Exported Secondary Mushroom TensorDict to %s", output_path_tensor)
    logger.info("Exported Secondary Mushroom metadata to %s", output_path_raw)


if __name__ == "__main__":
    main()
