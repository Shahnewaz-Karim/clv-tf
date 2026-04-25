"""Callbacks bundle for the deep CLV model."""

from __future__ import annotations

from pathlib import Path

import keras


def build(
    *,
    checkpoint_path: Path,
    tensorboard_dir: Path,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_lr: float,
    monitor: str = "val_clv_loss",
) -> list[keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
    ]
