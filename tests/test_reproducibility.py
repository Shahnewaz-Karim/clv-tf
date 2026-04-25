"""Same seed must produce identical training loss to 6 decimals on a tiny subset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import pipeline
from src.models.deep_clv import ModelConfig, build_model, compile_model
from src.training.train import set_seeds


def _train_one_epoch(mini_data_dir: Path, mini_seed: int) -> float:
    set_seeds(mini_seed)
    ds = pipeline.load(
        mini_data_dir, seed=mini_seed, splits=(0.7, 0.15, 0.15), batch_size=64, cache=True
    )
    model = build_model(ModelConfig(lstm_1=16, lstm_2=8, head_dim_1=16, head_dim_2=8))
    compile_model(model, learning_rate=1e-3)
    history = model.fit(ds.train, validation_data=ds.val, epochs=1, verbose=0)
    return float(history.history["loss"][0])


def test_loss_reproducible_to_6dp(mini_data_dir: Path, mini_seed: int):
    loss_1 = _train_one_epoch(mini_data_dir, mini_seed)
    loss_2 = _train_one_epoch(mini_data_dir, mini_seed)
    # Determinism is best-effort across HW; fail loudly if it slips below 6dp.
    np.testing.assert_almost_equal(loss_1, loss_2, decimal=6,
                                   err_msg=f"non-deterministic loss: {loss_1} vs {loss_2}")
