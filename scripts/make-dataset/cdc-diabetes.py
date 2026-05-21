from __future__ import annotations

import logging
import os

import mydatasets
import pandas as pd
import tensordict as thd
import torch as th
import ucimlrepo


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


feature_names = [
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
target_names = ["Diabetes_binary"]

def main() -> None:
    logger.info("Preparing CDC Diabetes dataset")
    data = ucimlrepo.fetch_ucirepo(id=891)

    xs_df: pd.DataFrame = data.data.features.copy()
    ys_df: pd.DataFrame = data.data.targets.copy()

    data_thd: thd.TensorDict = thd.make_tensordict(
        {
            "xs": th.as_tensor(xs_df.to_numpy(), dtype=th.float32),
            "ys": th.as_tensor(ys_df.to_numpy(), dtype=th.int64),
        },
        auto_batch_size=True,
    )

    os.makedirs(
        os.path.join(mydatasets.common.get_datasets_files_root_dir(), "uci"),
        exist_ok=True,
    )
    output_path = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "cdc-diabetes.pt",
    )
    th.save(data_thd, output_path)

    logger.info("CDC Diabetes xs shape: %s", tuple(data_thd["xs"].shape))
    logger.info("CDC Diabetes ys shape: %s", tuple(data_thd["ys"].shape))
    logger.info("Exported CDC Diabetes TensorDict to %s", output_path)


if __name__ == "__main__":
    main()
