"""Quality invariants caught by the audit.

These tests assert properties that must hold for the project's central claims
to be defensible. They build a brief end-to-end run on the mini dataset to
get a real model + real predictions; they're slower than the unit tests but
they're the ones that catch real bugs (the audit found a confidence-calibration
bug in the sister project's pricing engine via a similar test).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.data import pipeline
from src.evaluation import metrics
from src.evaluation.report import integrated_gradients
from src.models.deep_clv import ModelConfig, build_model, compile_model
from src.training.train import set_seeds


@pytest.fixture(scope="module")
def trained_mini_model(mini_data_dir: Path, mini_seed: int):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    set_seeds(mini_seed)
    ds = pipeline.load(
        mini_data_dir, seed=mini_seed, splits=(0.7, 0.15, 0.15), batch_size=64, cache=True
    )
    # Smaller model + more epochs than the smoke test so it actually learns.
    model = build_model(ModelConfig(lstm_1=24, lstm_2=12, head_dim_1=24, head_dim_2=12))
    compile_model(model, learning_rate=2e-3)
    model.fit(ds.train, validation_data=ds.val, epochs=4, verbose=0)
    return model, ds


def test_ig_completeness_holds(trained_mini_model):
    """IG attributions must approximately sum to f(x) - f(0). This is the
    fundamental completeness property of Integrated Gradients (Sundararajan
    et al. 2017). If it fails, the attribution heatmap in notebook 02 is
    quantitatively meaningless even though it might look plausible visually.

    Tolerance is 20% of |f(x) - f(0)| on this mini-trained model. The
    discretisation error of n_steps=32 IG is ~5-10% on a thinly-trained model;
    a broken implementation would be off by 50%+, so this still catches the
    real failure modes. On the full model the audit measured ~0.75% gap with
    n_steps=64.
    """
    model, ds = trained_mini_model
    y = ds.test_arrays.y_clv_raw
    if len(y) < 5:
        pytest.skip("test split too small for IG completeness check")
    idx = int(np.argsort(-y)[2])
    sample = {k: v[idx : idx + 1] for k, v in ds.test_arrays.features.items()}
    zero_sample = {k: np.zeros_like(v) for k, v in sample.items()}

    f_x = float(model.predict(sample, verbose=0)["clv"][0, 0])
    f_0 = float(model.predict(zero_sample, verbose=0)["clv"][0, 0])
    target_diff = f_x - f_0

    attrs = integrated_gradients(model, sample, target_head="clv", n_steps=32)
    numeric_sum = sum(float(attrs[k].sum()) for k in ("seq_numeric", "tenure"))
    cat_sum = sum(float(attrs[k][0]) for k in ("industry", "region", "segment", "channel"))
    total = numeric_sum + cat_sum

    if abs(target_diff) < 1e-3:
        pytest.skip("f(x) ~= f(0); completeness gap is meaningless here")
    relative_gap = abs(target_diff - total) / abs(target_diff)
    assert relative_gap < 0.20, (
        f"IG completeness violated: f(x)-f(0)={target_diff:.4f}, "
        f"sum(attrs)={total:.4f}, relative gap={relative_gap:.3%}"
    )


def test_top_decile_lift_above_baseline(trained_mini_model):
    """The deep model must produce a ranking lift > 1.05x at the top decile
    of its own predictions on this mini-data smoke test. On the full 50k
    dataset the audit measured top-decile lift = 4.60×; the mini model only
    sees ~350 train customers across 4 epochs so its ranking power is much
    weaker, but lift > 1.05 still catches a model that doesn't rank at all.

    Weak threshold by design — this test is a regression guard, not a quality bar.
    """
    model, ds = trained_mini_model
    y = ds.test_arrays.y_clv_raw
    if len(y) < 50:
        pytest.skip("test split too small for decile-lift test")
    preds_log = model.predict(ds.test, verbose=0)["clv"].reshape(-1)
    preds = np.maximum(np.expm1(preds_log), 0.0)
    decile = metrics.decile_lift(y, preds, n_deciles=10)
    top_lift = float(decile["lift"][0])
    assert top_lift > 1.05, (
        f"top-decile lift={top_lift:.2f} ≤ 1.05 — model is not concentrating "
        f"true revenue at the top of its predictions. Full lift table: {decile['lift']}"
    )


def test_naive_log_lr_baseline_beats_mean(mini_data_dir: Path, mini_seed: int):
    """The single-feature log1p(L12-revenue) Ridge baseline must beat the
    mean predictor on RMSE. If even this trivial baseline can't beat 'predict
    the average', the dataset itself doesn't carry the signal the project
    claims it carries — and every other model's metric is suspect."""
    from src.models.baselines import fit_all

    ds = pipeline.load(mini_data_dir, seed=mini_seed, splits=(0.7, 0.15, 0.15), batch_size=64, cache=False)
    results = fit_all(ds, seed=mini_seed)
    y_te = ds.test_arrays.y_clv_raw
    rmse_mean = float(np.sqrt(((y_te - results["mean"].pred_clv_test) ** 2).mean()))
    rmse_naive = float(np.sqrt(((y_te - results["naive_log_lr"].pred_clv_test) ** 2).mean()))
    assert rmse_naive < rmse_mean, (
        f"naive_log_lr RMSE ({rmse_naive:.0f}) does not beat mean predictor ({rmse_mean:.0f})"
    )
