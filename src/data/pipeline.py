"""tf.data pipeline: parquet -> normalised tensors -> batched, prefetched datasets.

The model consumes a dict of tensors (matching `Input` layer names in
`models.deep_clv`). Splits are stratified-by-customer (random with fixed seed).
Targets:
  * clv  — log1p-transformed in the dataset; eval undoes this.
  * churn — float32 in {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .schema import N_SEQ_NUMERIC

_TENURE_DENOM = 60.0  # divide tenure_months_at_t0 by this to get ~[0, 2] range


@dataclass(frozen=True)
class SplitArrays:
    """In-memory arrays for a single split. Held briefly during dataset construction."""

    features: dict[str, np.ndarray]
    y_clv_log: np.ndarray
    y_churn: np.ndarray
    y_clv_raw: np.ndarray  # original $ — used by evaluation, never by training
    customer_ids: np.ndarray


@dataclass(frozen=True)
class Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    train_arrays: SplitArrays
    val_arrays: SplitArrays
    test_arrays: SplitArrays


def _build_seq_numeric(dense: dict[str, np.ndarray]) -> np.ndarray:
    """Stack monthly numerics into (n, T, 6) with log1p on revenue/orders/skus."""
    revenue = np.log1p(dense["revenue"]).astype(np.float32)
    orders = np.log1p(dense["order_count"]).astype(np.float32)
    skus = np.log1p(dense["unique_skus"]).astype(np.float32)
    margin = dense["avg_margin"].astype(np.float32)
    bulk = dense["bulk_flag"].astype(np.float32)
    active = dense["active"].astype(np.float32)
    seq = np.stack([revenue, orders, skus, margin, bulk, active], axis=-1)
    assert seq.shape[-1] == N_SEQ_NUMERIC
    return seq


def _customer_split(n: int, seed: int, ratios: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _slice(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in arrays.items()}


def _to_tf_dataset(
    arr: SplitArrays,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
    cache: bool,
    seed: int,
) -> tf.data.Dataset:
    targets = {"clv": arr.y_clv_log, "churn": arr.y_churn}
    ds = tf.data.Dataset.from_tensor_slices((arr.features, targets))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(arr.y_churn)), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load(
    data_dir: str | Path,
    seed: int,
    splits: tuple[float, float, float] = (0.70, 0.15, 0.15),
    batch_size: int = 256,
    shuffle_buffer: int = 4096,
    cache: bool = True,
) -> Datasets:
    """Load the parquet artifacts produced by `data.generate.generate` and build tf.data."""
    data_dir = Path(data_dir)
    customers = pd.read_parquet(data_dir / "customers.parquet")
    targets = pd.read_parquet(data_dir / "targets.parquet")
    dense_npz = np.load(data_dir / "dense_input.npz")
    dense = {k: dense_npz[k] for k in dense_npz.files}

    # Align: customers and targets both keyed on customer_id; sort to be safe.
    customers = customers.sort_values("customer_id").reset_index(drop=True)
    targets = targets.sort_values("customer_id").reset_index(drop=True)
    assert (customers["customer_id"].to_numpy() == targets["customer_id"].to_numpy()).all()

    seq_numeric = _build_seq_numeric(dense)
    seq_top_category = dense["top_category"].astype(np.int32)
    feats: dict[str, np.ndarray] = {
        "seq_numeric": seq_numeric,
        "seq_top_category": seq_top_category,
        "industry": customers["industry_idx"].to_numpy().astype(np.int32),
        "region": customers["region_idx"].to_numpy().astype(np.int32),
        "segment": customers["segment_idx"].to_numpy().astype(np.int32),
        "channel": customers["channel_idx"].to_numpy().astype(np.int32),
        "tenure": (customers["tenure_months_at_t0"].to_numpy() / _TENURE_DENOM)
        .reshape(-1, 1)
        .astype(np.float32),
    }
    y_clv_raw = targets["clv_next_12m_usd"].to_numpy().astype(np.float32)
    y_clv_log = np.log1p(y_clv_raw).astype(np.float32)
    y_churn = targets["churn_next_12m"].to_numpy().astype(np.float32)
    customer_ids = customers["customer_id"].to_numpy().astype(np.int64)

    train_idx, val_idx, test_idx = _customer_split(len(customers), seed, splits)

    def _split(idx: np.ndarray) -> SplitArrays:
        return SplitArrays(
            features=_slice(feats, idx),
            y_clv_log=y_clv_log[idx],
            y_churn=y_churn[idx],
            y_clv_raw=y_clv_raw[idx],
            customer_ids=customer_ids[idx],
        )

    train_arr = _split(train_idx)
    val_arr = _split(val_idx)
    test_arr = _split(test_idx)

    return Datasets(
        train=_to_tf_dataset(train_arr, batch_size, True, shuffle_buffer, cache, seed),
        val=_to_tf_dataset(val_arr, batch_size, False, shuffle_buffer, cache, seed),
        test=_to_tf_dataset(test_arr, batch_size, False, shuffle_buffer, cache, seed),
        train_arrays=train_arr,
        val_arrays=val_arr,
        test_arrays=test_arr,
    )


def engineered_features(arr: SplitArrays) -> tuple[np.ndarray, list[str]]:
    """Hand-crafted RFM features for the linear / LightGBM baselines.

    Recency: months-since-last-active (within the 24-month window).
    Frequency: count of active months.
    Monetary: total / mean / max revenue; trend (slope of log1p revenue vs month).
    Plus categorical one-hot blocks for industry/region/segment/channel and tenure.
    """
    seq = arr.features["seq_numeric"]  # (n, T, 6) — log1p revenue, log1p orders, log1p skus, margin, bulk, active
    n, T, _ = seq.shape
    log_rev = seq[..., 0]
    active = seq[..., 5].astype(bool)

    months = np.arange(T)[None, :]
    last_active_idx = np.where(
        active.any(axis=1),
        np.argmax(active[:, ::-1], axis=1),  # offset from end
        T,
    )
    recency = last_active_idx.astype(np.float32)  # 0 = active in last month
    frequency = active.sum(axis=1).astype(np.float32)
    monetary_total = np.expm1(log_rev).sum(axis=1).astype(np.float32)
    monetary_mean = np.where(frequency > 0, monetary_total / np.maximum(frequency, 1.0), 0.0)
    monetary_max = np.expm1(log_rev).max(axis=1).astype(np.float32)
    # Trend: slope of log1p(revenue) vs month over active months only.
    # For a vectorised, robust approximation, use the slope on all T months
    # treating zeros as zero — captures decay.
    x_centered = months - months.mean()
    y_centered = log_rev - log_rev.mean(axis=1, keepdims=True)
    denom = (x_centered**2).sum() + 1e-9
    trend = (x_centered * y_centered).sum(axis=1) / denom

    # Recent-window aggregates (last 6 months — strongest signal in B2B).
    log_rev_recent = log_rev[:, -6:]
    recent_active = active[:, -6:].sum(axis=1).astype(np.float32)
    recent_revenue = np.expm1(log_rev_recent).sum(axis=1).astype(np.float32)

    # Static categoricals as one-hots; tenure as scalar.
    def _onehot(col: str, k: int) -> np.ndarray:
        idx = arr.features[col]
        out = np.zeros((n, k), dtype=np.float32)
        out[np.arange(n), idx] = 1.0
        return out

    from .schema import N_CHANNELS, N_INDUSTRIES, N_REGIONS, N_SEGMENTS

    industry_oh = _onehot("industry", N_INDUSTRIES)
    region_oh = _onehot("region", N_REGIONS)
    segment_oh = _onehot("segment", N_SEGMENTS)
    channel_oh = _onehot("channel", N_CHANNELS)
    tenure = arr.features["tenure"].squeeze(-1)

    blocks: list[tuple[str, np.ndarray]] = [
        ("recency", recency[:, None]),
        ("frequency", frequency[:, None]),
        ("monetary_total", monetary_total[:, None]),
        ("monetary_mean", monetary_mean[:, None]),
        ("monetary_max", monetary_max[:, None]),
        ("trend", trend[:, None].astype(np.float32)),
        ("recent_active", recent_active[:, None]),
        ("recent_revenue", recent_revenue[:, None]),
        ("tenure", tenure[:, None]),
        ("industry", industry_oh),
        ("region", region_oh),
        ("segment", segment_oh),
        ("channel", channel_oh),
    ]
    feature_names: list[str] = []
    matrices: list[np.ndarray] = []
    for name, mat in blocks:
        if mat.shape[1] == 1:
            feature_names.append(name)
        else:
            feature_names.extend(f"{name}_{i}" for i in range(mat.shape[1]))
        matrices.append(mat.astype(np.float32))
    X = np.concatenate(matrices, axis=1)
    return X, feature_names
