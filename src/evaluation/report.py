"""Build the comparison report across deep model + all baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from ..data import pipeline
from ..data.schema import SEGMENTS
from ..models import baselines
from ..models.layers import RecencyWeightedAttention, RevenueWeightedMAE
from . import metrics, plots


@dataclass
class ModelEvaluation:
    name: str
    pred_clv_test: np.ndarray
    pred_churn_test: np.ndarray | None
    metrics: dict[str, float]


def _evaluate_predictions(
    name: str,
    y_clv_true: np.ndarray,
    pred_clv: np.ndarray,
    y_churn_true: np.ndarray | None,
    pred_churn: np.ndarray | None,
) -> dict[str, float]:
    out = {f"clv_{k}": v for k, v in metrics.regression_metrics(y_clv_true, pred_clv).items()}
    out["clv_rev_weighted_mae"] = metrics.revenue_weighted_mae(y_clv_true, pred_clv)
    if pred_churn is not None and y_churn_true is not None:
        out.update({f"churn_{k}": v for k, v in metrics.classification_metrics(y_churn_true, pred_churn).items()})
    return out


def _load_deep_model(path: Path) -> keras.Model:
    # Custom objects are also auto-discovered via @register_keras_serializable, but we
    # pass them explicitly to be robust to renames or future Keras 3 changes.
    return keras.models.load_model(
        path,
        custom_objects={
            "RecencyWeightedAttention": RecencyWeightedAttention,
            "RevenueWeightedMAE": RevenueWeightedMAE,
            "clv_tf>RecencyWeightedAttention": RecencyWeightedAttention,
            "clv_tf>RevenueWeightedMAE": RevenueWeightedMAE,
        },
    )


def _deep_predict(model: keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run model.predict and return (clv $ predictions, churn predictions)."""
    preds = model.predict(ds, verbose=0)
    clv_log = preds["clv"].reshape(-1)
    churn = preds["churn"].reshape(-1)
    clv = np.maximum(np.expm1(clv_log), 0.0).astype(np.float32)
    return clv, churn


def evaluate_all(
    *,
    config: dict[str, Any],
    model_dir: str | Path,
    output_dir: str | Path,
    include_deep: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        config["data"]["splits"]["train"],
        config["data"]["splits"]["val"],
        config["data"]["splits"]["test"],
    )
    datasets = pipeline.load(
        data_dir=config["data"]["output_dir"],
        seed=int(config["seed"]),
        splits=splits,
        batch_size=config["training"]["batch_size"],
        shuffle_buffer=config["training"]["shuffle_buffer"],
        cache=False,
    )
    y_clv_test = datasets.test_arrays.y_clv_raw
    y_churn_test = datasets.test_arrays.y_churn

    rows: list[dict[str, Any]] = []
    evals: dict[str, ModelEvaluation] = {}

    # --- baselines ---
    bl_results = baselines.fit_all(datasets, seed=int(config["seed"]))
    for name, br in bl_results.items():
        m = _evaluate_predictions(name, y_clv_test, br.pred_clv_test, y_churn_test, br.pred_churn_test)
        rows.append({"model": name, **m})
        evals[name] = ModelEvaluation(
            name=name,
            pred_clv_test=br.pred_clv_test,
            pred_churn_test=br.pred_churn_test,
            metrics=m,
        )

    # --- deep model ---
    if include_deep:
        deep_path = Path(model_dir) / "deep_clv.keras"
        if deep_path.exists():
            deep = _load_deep_model(deep_path)
            clv_pred, churn_pred = _deep_predict(deep, datasets.test)
            m = _evaluate_predictions("deep_clv", y_clv_test, clv_pred, y_churn_test, churn_pred)
            rows.append({"model": "deep_clv", **m})
            evals["deep_clv"] = ModelEvaluation(
                name="deep_clv",
                pred_clv_test=clv_pred,
                pred_churn_test=churn_pred,
                metrics=m,
            )

    # --- artefacts ---
    table = pd.DataFrame(rows).set_index("model").round(3)
    table_path = output_dir / "results.csv"
    table.to_csv(table_path)
    (output_dir / "results.md").write_text(table.to_markdown())

    # Plots: deep_clv if present, else best baseline by RMSE.
    primary_name = "deep_clv" if "deep_clv" in evals else min(
        evals, key=lambda n: evals[n].metrics["clv_rmse"]
    )
    primary = evals[primary_name]
    decile = metrics.decile_lift(y_clv_test, primary.pred_clv_test)
    calib = metrics.calibration_by_decile(y_clv_test, primary.pred_clv_test)
    plots.decile_lift_plot(
        decile["decile"], decile["lift"], plots_dir / f"decile_lift_{primary_name}.png",
        title=f"Decile lift — {primary_name}",
    )
    plots.calibration_plot(
        calib["mean_predicted_clv"], calib["mean_actual_clv"],
        plots_dir / f"calibration_{primary_name}.png",
        title=f"Calibration — {primary_name}",
    )
    plots.residual_scatter(
        y_clv_test, primary.pred_clv_test, plots_dir / f"residuals_{primary_name}.png",
        title=f"Predicted vs actual CLV — {primary_name}",
    )

    # Per-segment failure analysis on the primary model.
    seg_idx = datasets.test_arrays.features["segment"]
    segment_breakdown = metrics.segment_failure_analysis(
        y_clv_test, primary.pred_clv_test, seg_idx, SEGMENTS
    )

    summary = {
        "primary_model": primary_name,
        "metrics_table": rows,
        "decile_lift": decile,
        "calibration": calib,
        "segment_failure": segment_breakdown,
        "n_test": int(len(y_clv_test)),
        "test_clv_mean": float(y_clv_test.mean()),
        "test_churn_rate": float(y_churn_test.mean()),
        "paths": {
            "results_csv": str(table_path),
            "plots_dir": str(plots_dir),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def integrated_gradients(
    model: keras.Model,
    sample_features: dict[str, np.ndarray],
    target_head: str = "clv",
    n_steps: int = 32,
) -> dict[str, np.ndarray]:
    """Integrated gradients attribution for a single example w.r.t. each input.

    Numeric inputs are integrated along a linear path from a zero baseline.
    Categorical inputs are attributed via a coarse "actual vs zero category"
    delta on the model output (gradients don't flow into integer indices).

    `sample_features` must already include the leading batch dim of size 1.
    """
    inputs_1 = {k: tf.convert_to_tensor(v) for k, v in sample_features.items()}
    numeric_keys = ("seq_numeric", "tenure")
    cat_keys = tuple(k for k in inputs_1 if k not in numeric_keys)

    alphas = tf.linspace(0.0, 1.0, n_steps)  # (n_steps,)
    attrs: dict[str, np.ndarray] = {}

    for k in numeric_keys:
        baseline = tf.zeros_like(inputs_1[k])
        # alpha shape (n_steps, 1, ..., 1) so it broadcasts against (1, *feat).
        alpha_shape = [n_steps] + [1] * (len(inputs_1[k].shape) - 1)
        a = tf.reshape(alphas, alpha_shape)
        interp = baseline + a * (inputs_1[k] - baseline)  # -> (n_steps, *feat)

        batch = {key: tf.repeat(val, n_steps, axis=0) for key, val in inputs_1.items()}
        batch[k] = interp
        with tf.GradientTape() as tape:
            tape.watch(batch[k])
            out = model(batch, training=False)
            target = out[target_head][:, 0]  # (n_steps,)
        grads = tape.gradient(target, batch[k])
        avg_grads = tf.reduce_mean(grads, axis=0, keepdims=True)
        attr = (inputs_1[k] - baseline) * avg_grads
        attrs[k] = attr.numpy()

    for k in cat_keys:
        baseline_inputs = {key: tf.identity(val) for key, val in inputs_1.items()}
        baseline_inputs[k] = tf.zeros_like(inputs_1[k])
        out_actual = model(inputs_1, training=False)[target_head][:, 0]
        out_baseline = model(baseline_inputs, training=False)[target_head][:, 0]
        attrs[k] = (out_actual - out_baseline).numpy()
    return attrs
