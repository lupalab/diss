from __future__ import annotations

import os
from collections import deque
from typing import Any, Optional, Sequence, TypeVar

_T = TypeVar("_T")
_NumberT = TypeVar("_NumberT", int, float)


def add_prefix_to_dict(
    d: dict[str, Any], prefix: str, sep: Optional[str] = "/"
) -> dict[str, Any]:
    """add prefix to dictionary

    Args:
        d (dict[str, Any]): a dictionary
        prefix (str): prefix to be prependes to each key
        sep (Optional[str], optional): seperator between `prefix` and key. Defaults to "/".

    Returns:
        dict[str, Any]: new dictionary with `prefix` prepended to each key of `d`.
    """
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def get_project_root_dir() -> str:
    """get the absolute path to the project root

    Returns:
        str: absoluate path to project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def interleave_ends(iterable: Sequence[_T]) -> Sequence[_T]:
    """Interleave elements from both ends of a sequence.

    Takes elements alternately from the beginning and end of the input sequence,
    starting with the first element, then the last, then the second, then the
    second-to-last, and so on until all elements are consumed.

    Args:
        iterable (Sequence[_T]): The input sequence to interleave.

    Returns:
        Sequence[_T]: A new sequence with elements interleaved from both ends.

    Examples:
        >>> interleave_ends([1, 2, 3, 4, 5])
        [1, 5, 2, 4, 3]
        >>> interleave_ends(['a', 'b', 'c', 'd'])
        ['a', 'd', 'b', 'c']
        >>> interleave_ends([42])
        [42]
        >>> interleave_ends([])
        []
    """
    dq = deque(iterable)
    outs = list()
    while dq:
        outs.append(dq.popleft())
        if dq:
            outs.append(dq.pop())
    return outs


def clamp(value: _NumberT, minimum: _NumberT, maximum: _NumberT) -> _NumberT:
    """Clamp a value between minimum and maximum bounds.
    
    Args:
        value: The value to clamp.
        minimum: The minimum bound.
        maximum: The maximum bound.
    
    Returns:
        The clamped value, constrained to [minimum, maximum].
    """
    return max(min(value, maximum), minimum)
