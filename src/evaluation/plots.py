"""Plotting helpers — matplotlib only, no seaborn-styled colors locked in."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def decile_lift_plot(
    deciles: list[int], lift: list[float], out_path: Path, title: str = "Decile lift"
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(deciles, lift, color="#2c7fb8", edgecolor="black")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="overall mean")
    ax.set_xlabel("Predicted-CLV decile (1 = highest)")
    ax.set_ylabel("Lift (mean actual CLV / overall mean)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def calibration_plot(
    pred_means: list[float], actual_means: list[float], out_path: Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(min(pred_means), min(actual_means))
    hi = max(max(pred_means), max(actual_means))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.plot(pred_means, actual_means, marker="o", color="#d95f0e", label="model")
    ax.set_xlabel("Mean predicted CLV per decile ($)")
    ax.set_ylabel("Mean actual CLV per decile ($)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def residual_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=2, alpha=0.25, color="#2c7fb8")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("Actual CLV ($)")
    ax.set_ylabel("Predicted CLV ($)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
