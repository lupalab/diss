from __future__ import annotations

import os
from dataclasses import dataclass

import mylib


@dataclass
class DatasetConf:
    _target_: str
    dataset_type: str


def get_datasets_files_root_dir() -> str:
    datasets_files_dir: str = os.path.join(mylib.utils.get_project_root_dir(), "data")
    datasets_files_dir = os.path.abspath(datasets_files_dir)
    if not os.path.exists(datasets_files_dir):
        raise FileNotFoundError("data does not exists")
    return datasets_files_dir
