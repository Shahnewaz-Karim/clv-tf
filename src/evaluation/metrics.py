"""Evaluation metrics — all in raw $ space, not log space."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, MAPE — clipping pred to ≥0 since CLV is non-negative."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # MAPE skips zero targets to avoid division-by-zero.
    nz = y_true > 1.0  # ignore essentially-zero CLV (mostly churned customers)
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]))) if nz.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def revenue_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 1.0) -> float:
    """MAE weighted by max(y_true, floor) — large customers count more."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    w = np.maximum(y_true, 0.0) + floor
    return float(np.sum(np.abs(y_true - y_pred) * w) / np.sum(w))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan")}
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    f1 = float(f1_score(y_true, (y_score >= 0.5).astype(int)))
    return {"auc": auc, "pr_auc": pr_auc, "f1": f1}


def decile_lift(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> dict[str, list]:
    """Sort by predicted CLV descending; report mean actual CLV per decile.

    A working CLV model should put more revenue in the top deciles. The "lift"
    is decile_mean / overall_mean; lift > 1 means the decile is above-average.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    order = np.argsort(-y_pred)  # descending
    y_sorted = y_true[order]
    n = len(y_sorted)
    overall_mean = float(y_sorted.mean()) if n > 0 else 0.0
    sizes = np.array_split(np.arange(n), n_deciles)
    deciles, mean_actual, lift = [], [], []
    for i, idx in enumerate(sizes, start=1):
        deciles.append(int(i))
        m = float(y_sorted[idx].mean()) if len(idx) else 0.0
        mean_actual.append(round(m, 2))
        lift.append(round(m / overall_mean, 3) if overall_mean > 0 else float("nan"))
    return {"decile": deciles, "mean_actual_clv": mean_actual, "lift": lift}


def calibration_by_decile(
    y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10
) -> dict[str, list]:
    """Mean predicted vs. mean actual CLV per predicted-decile (calibration plot data)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    order = np.argsort(y_pred)  # ascending
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    sizes = np.array_split(np.arange(len(y_true_s)), n_deciles)
    deciles = [int(i) for i in range(1, n_deciles + 1)]
    pred_means = [round(float(y_pred_s[idx].mean()), 2) if len(idx) else 0.0 for idx in sizes]
    actual_means = [round(float(y_true_s[idx].mean()), 2) if len(idx) else 0.0 for idx in sizes]
    return {"decile": deciles, "mean_predicted_clv": pred_means, "mean_actual_clv": actual_means}


def segment_failure_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_labels: np.ndarray,
    segment_names: tuple[str, ...] | list[str],
) -> dict[str, list]:
    """Per-segment MAE — surfaces which segments the model fails on."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    segment_labels = np.asarray(segment_labels).reshape(-1)
    out_seg, out_n, out_mae, out_mean_actual = [], [], [], []
    for i, name in enumerate(segment_names):
        mask = segment_labels == i
        if mask.sum() == 0:
            continue
        out_seg.append(name)
        out_n.append(int(mask.sum()))
        out_mae.append(round(float(np.mean(np.abs(y_true[mask] - y_pred[mask]))), 2))
        out_mean_actual.append(round(float(y_true[mask].mean()), 2))
    return {
        "segment": out_seg,
        "n": out_n,
        "mae": out_mae,
        "mean_actual_clv": out_mean_actual,
    }
