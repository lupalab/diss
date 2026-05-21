from __future__ import annotations

import os
from typing import Any

import numpy as np
import scipy.io as sp_io
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th

from . import common


def load_matfile_data(name: str) -> dict[str, Any]:
    rdata: dict[str, Any] = sp_io.loadmat(
        os.path.join(common.get_datasets_files_root_dir(), "matfile", f"{name}.mat")
    )
    return rdata


def load_pickle_data(
    name: str, to_normalize: bool = True
) -> tuple[thd.TensorDict, thd.TensorDict, thd.TensorDict]:
    data: dict[str, tuple[np.ndarray, np.ndarray]] = np.load(
        os.path.join(common.get_datasets_files_root_dir(), "pklfile", f"{name}.pkl"),
        allow_pickle=True,
    )
    xst: np.ndarray = data["train"][0]
    yst: np.ndarray = data["train"][1].flatten()
    xsv: np.ndarray = data["valid"][0]
    ysv: np.ndarray = data["valid"][1].flatten()
    xstst: np.ndarray = data["test"][0]
    ystst: np.ndarray = data["test"][1].flatten()
    if to_normalize:
        nmlr = skl_preproc.StandardScaler()
        xst = nmlr.fit_transform(xst)
        xsv = nmlr.transform(xsv)
    tdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xst, dtype=th.float32),
            "ys": th.as_tensor(yst, dtype=th.long),
        }
    ).auto_batch_size_(1)
    vdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xsv, dtype=th.float32),
            "ys": th.as_tensor(ysv, dtype=th.long),
        }
    ).auto_batch_size_(1)
    tstdata = thd.TensorDict(
        {
            "xs": th.as_tensor(xstst, dtype=th.float32),
            "ys": th.as_tensor(ystst, dtype=th.long),
        }
    ).auto_batch_size_(1)
    return tdata, vdata, tstdata
