"""Baselines. The deep model only deserves credit for what it beats here.

Five baselines, in order of increasing expressiveness:

    1. mean             — predicts the train-set mean CLV. Sanity floor.
    2. carry_forward    — predicts last-12-months revenue (no model). The
                          baseline that often embarrasses ML on contractual B2B.
    3. naive_log_lr     — Ridge on a SINGLE feature: log1p(last-12-months revenue).
                          The "is the deep model's complexity earning its keep"
                          ablation — if the 91k-param BiLSTM doesn't beat a
                          1-feature linear regression by a meaningful margin,
                          ship the linear regression.
    4. linear_rfm       — Ridge regression on engineered RFM + categorical features.
    5. lightgbm         — Gradient-boosted trees on the same engineered features.

Each baseline returns a `BaselineResult` with predictions for train/val/test
and (where applicable) churn predictions. CLV predictions are returned in
raw $ space for direct comparison with the deep model after its log1p inversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..data.pipeline import Datasets, SplitArrays, engineered_features


@dataclass
class BaselineResult:
    name: str
    pred_clv_test: np.ndarray
    pred_clv_val: np.ndarray
    pred_churn_test: np.ndarray | None  # None for regression-only baselines
    pred_churn_val: np.ndarray | None


def _last_12_revenue(arr: SplitArrays) -> np.ndarray:
    """Sum revenue over the most recent 12 months in the input window."""
    seq = arr.features["seq_numeric"]  # (n, 24, 6); channel 0 is log1p(revenue)
    rev = np.expm1(seq[..., 0])
    return rev[:, -12:].sum(axis=1).astype(np.float32)


def fit_mean(ds: Datasets) -> BaselineResult:
    mean_clv = float(ds.train_arrays.y_clv_raw.mean())
    base_churn = float(ds.train_arrays.y_churn.mean())
    n_val = len(ds.val_arrays.y_clv_raw)
    n_test = len(ds.test_arrays.y_clv_raw)
    return BaselineResult(
        name="mean",
        pred_clv_test=np.full(n_test, mean_clv, dtype=np.float32),
        pred_clv_val=np.full(n_val, mean_clv, dtype=np.float32),
        pred_churn_test=np.full(n_test, base_churn, dtype=np.float32),
        pred_churn_val=np.full(n_val, base_churn, dtype=np.float32),
    )


def fit_carry_forward(ds: Datasets) -> BaselineResult:
    return BaselineResult(
        name="carry_forward",
        pred_clv_test=_last_12_revenue(ds.test_arrays),
        pred_clv_val=_last_12_revenue(ds.val_arrays),
        pred_churn_test=None,
        pred_churn_val=None,
    )


def fit_linear_rfm(ds: Datasets) -> BaselineResult:
    X_tr, _ = engineered_features(ds.train_arrays)
    X_val, _ = engineered_features(ds.val_arrays)
    X_te, _ = engineered_features(ds.test_arrays)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    # Train on log1p target (much better-conditioned for linear models on heavy-tailed CLV).
    y_tr_log = np.log1p(ds.train_arrays.y_clv_raw)
    reg = Ridge(alpha=1.0).fit(X_tr_s, y_tr_log)
    pred_val_log = reg.predict(X_val_s)
    pred_te_log = reg.predict(X_te_s)
    pred_val = np.maximum(np.expm1(pred_val_log), 0.0).astype(np.float32)
    pred_te = np.maximum(np.expm1(pred_te_log), 0.0).astype(np.float32)

    # Churn classifier on the same features. Guard against degenerate single-class
    # training data (occurs in tiny smoke runs) by falling back to a constant predictor.
    y_tr_churn = ds.train_arrays.y_churn
    if len(np.unique(y_tr_churn)) < 2:
        const = float(y_tr_churn.mean())
        churn_val = np.full(len(X_val_s), const, dtype=np.float32)
        churn_te = np.full(len(X_te_s), const, dtype=np.float32)
    else:
        clf = LogisticRegression(max_iter=300, C=1.0).fit(X_tr_s, y_tr_churn)
        churn_val = clf.predict_proba(X_val_s)[:, 1].astype(np.float32)
        churn_te = clf.predict_proba(X_te_s)[:, 1].astype(np.float32)

    return BaselineResult(
        name="linear_rfm",
        pred_clv_test=pred_te,
        pred_clv_val=pred_val,
        pred_churn_test=churn_te,
        pred_churn_val=churn_val,
    )


def fit_lightgbm(ds: Datasets, seed: int = 42) -> BaselineResult:
    X_tr, _ = engineered_features(ds.train_arrays)
    X_val, _ = engineered_features(ds.val_arrays)
    X_te, _ = engineered_features(ds.test_arrays)

    y_tr_log = np.log1p(ds.train_arrays.y_clv_raw)
    y_val_log = np.log1p(ds.val_arrays.y_clv_raw)

    reg = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        objective="regression",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    reg.fit(
        X_tr, y_tr_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_val = np.maximum(np.expm1(reg.predict(X_val)), 0.0).astype(np.float32)
    pred_te = np.maximum(np.expm1(reg.predict(X_te)), 0.0).astype(np.float32)

    y_tr_churn = ds.train_arrays.y_churn
    if len(np.unique(y_tr_churn)) < 2:
        const = float(y_tr_churn.mean())
        churn_val = np.full(len(X_val), const, dtype=np.float32)
        churn_te = np.full(len(X_te), const, dtype=np.float32)
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=48,
            min_child_samples=40,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_tr, y_tr_churn,
            eval_set=[(X_val, ds.val_arrays.y_churn)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        churn_val = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        churn_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)

    return BaselineResult(
        name="lightgbm",
        pred_clv_test=pred_te,
        pred_clv_val=pred_val,
        pred_churn_test=churn_te,
        pred_churn_val=churn_val,
    )


def fit_naive_log_lr(ds: Datasets) -> BaselineResult:
    """Single-feature ablation: linear regression on log1p(last-12-months revenue).

    Why this matters
    ----------------
    The deep model has 91k params and a custom recency-attention layer. The
    LightGBM baseline has thousands of leaves and full RFM features. The
    *cheapest* possible learned baseline is "fit y = a + b * log1p(L12rev)"
    using ONE input feature. If the deep model does not meaningfully beat
    this one-feature linear regression, every other architectural decision
    in the project is unjustified. This is the analogue of the null-engine
    ablation used in the profitability-pricing repo.
    """
    L12_train = _last_12_revenue(ds.train_arrays).reshape(-1, 1)
    L12_val = _last_12_revenue(ds.val_arrays).reshape(-1, 1)
    L12_test = _last_12_revenue(ds.test_arrays).reshape(-1, 1)
    x_train = np.log1p(L12_train.clip(min=0.0))
    x_val = np.log1p(L12_val.clip(min=0.0))
    x_test = np.log1p(L12_test.clip(min=0.0))
    y_train_log = np.log1p(ds.train_arrays.y_clv_raw.clip(min=0.0))
    reg = Ridge(alpha=1.0).fit(x_train, y_train_log)
    pred_val = np.maximum(np.expm1(reg.predict(x_val)), 0.0).astype(np.float32)
    pred_te = np.maximum(np.expm1(reg.predict(x_test)), 0.0).astype(np.float32)
    return BaselineResult(
        name="naive_log_lr",
        pred_clv_test=pred_te,
        pred_clv_val=pred_val,
        pred_churn_test=None,
        pred_churn_val=None,
    )


def fit_all(ds: Datasets, seed: int = 42) -> dict[str, BaselineResult]:
    return {
        "mean": fit_mean(ds),
        "carry_forward": fit_carry_forward(ds),
        "naive_log_lr": fit_naive_log_lr(ds),
        "linear_rfm": fit_linear_rfm(ds),
        "lightgbm": fit_lightgbm(ds, seed=seed),
    }
