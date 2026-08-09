from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    strict: bool = True,
) -> tuple[list[int], list[str], list[Any]]:
    """Match the regex keys of ``data`` against ``list_of_strings``.

    Returns ``(indices, names, values)`` for every target string matched by
    exactly one pattern. Ordered by key-declaration order by default, by
    target order if ``preserve_order`` is set.
    """
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dict of {{regex: value}}, got {type(data)}")
    keys = list(data.keys())
    used = [False] * len(keys)
    matches: list[tuple[int, int, str, Any]] = []  # (key_pos, target_pos, name, value)
    for target_pos, name in enumerate(list_of_strings):
        hits = [k for k, pattern in enumerate(keys) if re.fullmatch(pattern, name)]
        if len(hits) > 1:
            raise ValueError(
                f"'{name}' matches multiple patterns: {[keys[k] for k in hits]} — "
                "patterns must be mutually exclusive."
            )
        if hits:
            used[hits[0]] = True
            matches.append((hits[0], target_pos, name, data[keys[hits[0]]]))
    if strict and not all(used):
        unused = [keys[k] for k, u in enumerate(used) if not u]
        raise ValueError(f"patterns matched no target at all: {unused}")
    matches.sort(key=(lambda m: m[1]) if preserve_order else (lambda m: (m[0], m[1])))
    return [m[1] for m in matches], [m[2] for m in matches], [m[3] for m in matches]


def resolve_param(
    data: dict[str, float],
    names: Sequence[str],
    default: float | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Resolve ``{regex: value}`` onto ``names`` order as a float array.

    Names not covered by any pattern take ``default``; with ``default=None``
    full coverage is required (error otherwise).
    """
    out = np.full(len(names), np.nan, dtype=np.float64)
    indices, _, values = resolve_matching_names_values(data, names)
    out[indices] = values
    missing = np.isnan(out)
    if missing.any():
        if default is None:
            missing_names = [names[i] for i in np.flatnonzero(missing)]
            raise ValueError(f"no value resolved for: {missing_names}")
        out[missing] = default
    return out.astype(dtype)
