from __future__ import annotations

from typing import Any, Iterable

import torch as th

ParamsT = (
    Iterable[th.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, th.Tensor]]
)
