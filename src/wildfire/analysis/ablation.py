"""Ablation gate for promoting reintegration candidates (PLAN §C.2).

Pure verdict unit: the notebook fits the base model with and without the
candidate columns under **both** blocked-split protocols (``season``,
``ccaa``) and hands the collected per-fold PR-AUC values here. Promotion
requires the mean ΔPR-AUC to clear ``margin``, a bootstrap CI excluding zero,
and no protocol going negative. ``month`` / ``ccaa`` one-hot negative controls
run through the same function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

#: Mean ΔPR-AUC must exceed this to promote (0 reproduces "mean Δ > 0").
DEFAULT_MARGIN = 0.0

#: Bootstrap resamples for the CI (over folds).
DEFAULT_N_BOOT = 2000

#: Shared seed for the bootstrap (PLAN decision 2026-10-18).
ABLATION_SEED = 42


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = ABLATION_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, seeded and vectorized.

    Returns ``(ci_low, ci_high)`` at the ``1 - alpha`` level.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_mean_ci needs at least one value")
    if not np.isfinite(arr).all():
        raise ValueError("bootstrap_mean_ci needs finite values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[draws].mean(axis=1)
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def _to_protocol_map(metrics: object, name: str) -> dict[str, np.ndarray]:
    """Normalize the accepted metric shapes to ``{protocol: array}``.

    Accepts a per-fold metrics frame (``pr_auc`` column, optional
    ``protocol``), a mapping ``protocol -> sequence``, or a flat sequence
    (single ``"default"`` protocol).
    """
    if isinstance(metrics, pd.DataFrame):
        if "pr_auc" not in metrics.columns:
            raise KeyError(f"{name} frame is missing the 'pr_auc' column")
        if "protocol" in metrics.columns:
            return {
                str(proto): group["pr_auc"].to_numpy(dtype=float)
                for proto, group in metrics.groupby("protocol")
            }
        return {"default": metrics["pr_auc"].to_numpy(dtype=float)}
    if isinstance(metrics, Mapping):
        return {str(k): np.asarray(v, dtype=float) for k, v in metrics.items()}
    arr = np.asarray(metrics, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D per-fold metrics")
    return {"default": arr}


def ablation_verdict(
    with_metrics: object,
    without_metrics: object,
    margin: float = DEFAULT_MARGIN,
    seed: int = ABLATION_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    """Promotion verdict from with/without per-fold PR-AUC metrics.

    Both arguments accept the same shapes as :func:`_to_protocol_map` and must
    expose the same protocols with matching fold counts.

    Returns a dict with ``mean_delta``, bootstrap ``ci_low``/``ci_high``,
    ``per_protocol_mean``, ``per_protocol_min``, ``per_fold_min``,
    ``n_folds``, ``margin`` and the boolean ``promoted``.
    """
    with_map = _to_protocol_map(with_metrics, "with_metrics")
    without_map = _to_protocol_map(without_metrics, "without_metrics")
    if set(with_map) != set(without_map):
        raise ValueError(
            f"Protocols differ: {sorted(with_map)} vs {sorted(without_map)}"
        )

    per_protocol_mean: dict[str, float] = {}
    per_protocol_min: dict[str, float] = {}
    deltas: list[np.ndarray] = []
    for proto in with_map:
        a, b = with_map[proto], without_map[proto]
        if a.shape != b.shape:
            raise ValueError(
                f"Protocol {proto!r}: {a.size} with-metrics vs {b.size} "
                "without-metrics (same folds required)"
            )
        if a.size == 0:
            raise ValueError(f"Protocol {proto!r} has no folds")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"Protocol {proto!r} has non-finite metrics")
        delta = a - b
        per_protocol_mean[proto] = float(delta.mean())
        per_protocol_min[proto] = float(delta.min())
        deltas.append(delta)

    pooled = np.concatenate(deltas)
    mean_delta = float(pooled.mean())
    ci_low, ci_high = bootstrap_mean_ci(pooled, n_boot=n_boot, seed=seed)
    promoted = bool(
        mean_delta > margin
        and ci_low > 0.0
        and all(value >= 0.0 for value in per_protocol_mean.values())
    )
    return {
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_protocol_mean": per_protocol_mean,
        "per_protocol_min": per_protocol_min,
        "per_fold_min": float(pooled.min()),
        "n_folds": int(pooled.size),
        "margin": float(margin),
        "promoted": promoted,
    }
