"""Pipeline tensors must match the schema the model declares — shapes & dtypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import pipeline
from src.data.schema import (
    CATEGORY_VOCAB_SIZE,
    N_CHANNELS,
    N_INDUSTRIES,
    N_REGIONS,
    N_SEGMENTS,
    N_SEQ_NUMERIC,
)


def test_parquets_exist(mini_data_dir: Path):
    for name in ("customers", "products", "transactions_monthly", "targets"):
        assert (mini_data_dir / f"{name}.parquet").exists(), f"missing {name}"
    assert (mini_data_dir / "dense_input.npz").exists()


def test_targets_are_finite_and_well_typed(mini_data_dir: Path, mini_n: int):
    targets = pd.read_parquet(mini_data_dir / "targets.parquet")
    assert len(targets) == mini_n
    assert np.isfinite(targets["clv_next_12m_usd"]).all()
    assert (targets["clv_next_12m_usd"] >= 0).all()
    assert targets["churn_next_12m"].dtype == bool


def test_pipeline_shapes_and_dtypes(mini_data_dir: Path, mini_n: int, mini_seed: int):
    ds = pipeline.load(mini_data_dir, seed=mini_seed, batch_size=32, cache=False)
    train_arr = ds.train_arrays
    n = len(train_arr.y_churn)
    assert n + len(ds.val_arrays.y_churn) + len(ds.test_arrays.y_churn) == mini_n

    feats = train_arr.features
    assert feats["seq_numeric"].shape == (n, 24, N_SEQ_NUMERIC)
    assert feats["seq_top_category"].shape == (n, 24)
    assert feats["industry"].shape == (n,)
    assert feats["tenure"].shape == (n, 1)
    assert feats["seq_numeric"].dtype == np.float32
    assert feats["seq_top_category"].dtype == np.int32
    assert feats["industry"].dtype == np.int32
    assert feats["tenure"].dtype == np.float32

    # Vocab bounds
    assert feats["industry"].max() < N_INDUSTRIES
    assert feats["region"].max() < N_REGIONS
    assert feats["segment"].max() < N_SEGMENTS
    assert feats["channel"].max() < N_CHANNELS
    assert feats["seq_top_category"].max() < CATEGORY_VOCAB_SIZE


def test_tf_dataset_yields_correct_structure(mini_data_dir: Path, mini_seed: int):
    ds = pipeline.load(mini_data_dir, seed=mini_seed, batch_size=16, cache=False)
    x_batch, y_batch = next(iter(ds.train))
    assert set(x_batch.keys()) == {
        "seq_numeric", "seq_top_category", "industry", "region", "segment", "channel", "tenure"
    }
    assert set(y_batch.keys()) == {"clv", "churn"}
    assert tuple(x_batch["seq_numeric"].shape) == (16, 24, N_SEQ_NUMERIC)


def test_engineered_features_shape_and_no_nans(mini_data_dir: Path, mini_seed: int):
    ds = pipeline.load(mini_data_dir, seed=mini_seed, batch_size=32, cache=False)
    X, names = pipeline.engineered_features(ds.train_arrays)
    assert X.shape[0] == len(ds.train_arrays.y_churn)
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all()
