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


feature_names = [
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
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
    "change",
    "diabetesMed",
]
target_names = ["readmitted"]

# NOTE Classify feature types
# -- numeric (passthrough): already numeric, no encoding needed
_numeric_features = [
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
# -- ordinal: age brackets have a natural order
_age_categories = [
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
# -- ordinal: weight brackets have a natural order
_weight_categories = [
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
# -- medication dosage change features (same categories for all)
_medication_features = [
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
# -- nominal categorical features (no natural order -> ordinal encoded)
_nominal_features = [
    "race",
    "gender",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
]

def main() -> None:
    logger.info("Preparing Diabetes 130-US Hospitals dataset")
    data = ucimlrepo.fetch_ucirepo(id=296)

    xs_df: pd.DataFrame = data.data.features.copy()
    ys_df: pd.DataFrame = data.data.targets.copy()

    # NOTE The UCI diabetes dataset uses "?" to denote missing values in columns
    # such as weight, payer_code, medical_specialty, and diag_1/2/3.
    # The ucimlrepo loader already converts "?" to NaN.
    # Strategy: all features use OrdinalEncoder with
    #   handle_unknown="use_encoded_value", unknown_value=-10
    #   so NaN and any unseen categories map to -10.
    _MISSING_SENTINEL: int = -10

    # NOTE diag columns have mixed types (numeric ICD codes + strings); coerce to str
    xs_df["diag_1"] = xs_df["diag_1"].astype(str)
    xs_df["diag_2"] = xs_df["diag_2"].astype(str)
    xs_df["diag_3"] = xs_df["diag_3"].astype(str)
    # NOTE gender has "Unknown/Invalid" which should be treated as missing
    xs_df["gender"] = xs_df["gender"].replace("Unknown/Invalid", np.nan)

    # NOTE common kwargs for all OrdinalEncoders
    _ordinal_kwargs = dict(
        handle_unknown="use_encoded_value",
        unknown_value=_MISSING_SENTINEL,
        encoded_missing_value=_MISSING_SENTINEL,
        dtype=np.int64,
    )
    # Create column transformer for features
    feature_preprocessor = skl_compose.ColumnTransformer(
        transformers=[
            (
                "age_ordinal",
                skl_preproc.OrdinalEncoder(categories=[_age_categories], **_ordinal_kwargs),
                ["age"],
            ),
            (
                "weight_ordinal",
                skl_preproc.OrdinalEncoder(
                    categories=[_weight_categories], **_ordinal_kwargs
                ),
                ["weight"],
            ),
            (
                "medication_ordinal",
                skl_preproc.OrdinalEncoder(
                    categories=[["No", "Steady", "Up", "Down"]]
                    * len(_medication_features),
                    **_ordinal_kwargs,
                ),
                _medication_features,
            ),
            # NOTE change: No=0, Ch=1
            (
                "change_ordinal",
                skl_preproc.OrdinalEncoder(categories=[["No", "Ch"]], **_ordinal_kwargs),
                ["change"],
            ),
            # NOTE diabetesMed: No=0, Yes=1
            (
                "diabetesMed_ordinal",
                skl_preproc.OrdinalEncoder(categories=[["No", "Yes"]], **_ordinal_kwargs),
                ["diabetesMed"],
            ),
            # NOTE max_glu_serum: None=0, Norm=1, >200=2, >300=3
            (
                "max_glu_serum_ordinal",
                skl_preproc.OrdinalEncoder(
                    categories=[["None", "Norm", ">200", ">300"]], **_ordinal_kwargs
                ),
                ["max_glu_serum"],
            ),
            # NOTE A1Cresult: None=0, Norm=1, >7=2, >8=3
            (
                "A1Cresult_ordinal",
                skl_preproc.OrdinalEncoder(
                    categories=[["None", "Norm", ">7", ">8"]], **_ordinal_kwargs
                ),
                ["A1Cresult"],
            ),
            # NOTE nominal features: ordinal encoded (auto categories); missing -> -10
            (
                "nominal_ordinal",
                skl_preproc.OrdinalEncoder(**_ordinal_kwargs),
                _nominal_features,
            ),
            (
                "numeric_passthrough",
                "passthrough",
                _numeric_features,
            ),
        ],
        remainder="drop",
    )

    # NOTE ensure feature names are preserved in output
    feature_preprocessor.set_output(transform="pandas")
    patient_xs: pd.DataFrame = feature_preprocessor.fit_transform(xs_df[feature_names])
    # NOTE strip transformer prefix from column names (e.g. "age_ordinal__age" -> "age")
    # and reorder columns to match feature_names
    patient_xs = patient_xs.rename(columns=lambda x: str.split(x, "__", 1)[-1])[
        feature_names
    ]

    # NOTE Encode target: readmitted -> binary {NO: 0, YES: 1}
    # where YES = (<30 or >30), i.e. any readmission
    # Manually merge classes before encoding
    ys_df_binary = ys_df.copy()
    ys_df_binary["readmitted"] = ys_df_binary["readmitted"].replace(
        {"<30": "YES", ">30": "YES"}
    )

    target_preprocessor = skl_preproc.OrdinalEncoder(
        categories=[["NO", "YES"]], dtype=np.int64
    )
    target_preprocessor.set_output(transform="pandas")
    patient_ys: pd.DataFrame = target_preprocessor.fit_transform(ys_df_binary[target_names])

    data_thd: thd.TensorDict = thd.make_tensordict(
        {
            "xs": th.as_tensor(patient_xs.to_numpy(), dtype=th.float32),
            "ys": th.as_tensor(patient_ys.to_numpy(), dtype=th.int64),
        },
        auto_batch_size=True,
    )
    raw_data_d = {
        "feature_preprocessor": feature_preprocessor,
        "target_preprocessor": target_preprocessor,
        "patient_xs_df": patient_xs,
        "patient_ys_df": patient_ys,
    }

    os.makedirs(
        os.path.join(mydatasets.common.get_datasets_files_root_dir(), "uci"),
        exist_ok=True,
    )
    output_path_tensor = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "diabetes-130-ordinal-binary.pt",
    )
    output_path_raw = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "raw-diabetes-130-ordinal-binary.pkl",
    )
    th.save(data_thd, output_path_tensor)
    th.save(raw_data_d, output_path_raw)

    logger.info("Diabetes 130 xs shape: %s", tuple(data_thd["xs"].shape))
    logger.info("Diabetes 130 ys shape: %s", tuple(data_thd["ys"].shape))
    logger.info("Exported Diabetes 130 TensorDict to %s", output_path_tensor)
    logger.info("Exported Diabetes 130 metadata to %s", output_path_raw)


if __name__ == "__main__":
    main()
