"""Model forward pass yields correct output shapes and is differentiable."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from src.data.schema import N_CHANNELS, N_INDUSTRIES, N_REGIONS, N_SEGMENTS, N_SEQ_NUMERIC
from src.models.deep_clv import ModelConfig, build_model, compile_model


def _make_dummy_batch(batch_size: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "seq_numeric": rng.standard_normal((batch_size, 24, N_SEQ_NUMERIC)).astype(np.float32),
        "seq_top_category": rng.integers(0, 13, size=(batch_size, 24)).astype(np.int32),
        "industry": rng.integers(0, N_INDUSTRIES, size=(batch_size,)).astype(np.int32),
        "region": rng.integers(0, N_REGIONS, size=(batch_size,)).astype(np.int32),
        "segment": rng.integers(0, N_SEGMENTS, size=(batch_size,)).astype(np.int32),
        "channel": rng.integers(0, N_CHANNELS, size=(batch_size,)).astype(np.int32),
        "tenure": rng.standard_normal((batch_size, 1)).astype(np.float32),
    }


def test_forward_pass_shapes():
    model = build_model(ModelConfig())
    x = _make_dummy_batch(batch_size=4)
    out = model(x, training=False)
    assert out["clv"].shape == (4, 1)
    assert out["churn"].shape == (4, 1)
    # softplus / sigmoid both produce non-negative outputs.
    assert tf.reduce_all(out["clv"] >= 0).numpy()
    assert tf.reduce_all(out["churn"] >= 0).numpy() and tf.reduce_all(out["churn"] <= 1).numpy()


def test_compile_and_one_step():
    model = build_model(ModelConfig())
    compile_model(model, learning_rate=1e-3)
    x = _make_dummy_batch(batch_size=8)
    y = {
        "clv": np.random.default_rng(1).standard_normal((8,)).astype(np.float32),
        "churn": np.random.default_rng(2).integers(0, 2, size=(8,)).astype(np.float32),
    }
    history = model.fit(x, y, epochs=1, verbose=0)
    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][0])


def test_param_count_reasonable():
    model = build_model(ModelConfig())
    n = model.count_params()
    # Sanity bounds: not 0, not absurdly huge for a 24-step BiLSTM.
    assert 50_000 < n < 1_000_000, f"unexpected param count: {n}"


def test_save_and_load_roundtrip(tmp_path):
    from src.models.layers import RecencyWeightedAttention, RevenueWeightedMAE

    model = build_model(ModelConfig())
    compile_model(model)
    save_path = tmp_path / "m.keras"
    model.save(save_path)
    loaded = keras.models.load_model(
        save_path,
        custom_objects={
            "RecencyWeightedAttention": RecencyWeightedAttention,
            "RevenueWeightedMAE": RevenueWeightedMAE,
        },
    )
    x = _make_dummy_batch(batch_size=2)
    out_a = model(x, training=False)
    out_b = loaded(x, training=False)
    np.testing.assert_allclose(out_a["clv"].numpy(), out_b["clv"].numpy(), atol=1e-6)
    np.testing.assert_allclose(out_a["churn"].numpy(), out_b["churn"].numpy(), atol=1e-6)
