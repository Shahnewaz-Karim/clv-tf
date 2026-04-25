"""Training orchestration: seed -> data -> model -> fit -> save."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ..data import pipeline
from ..models.deep_clv import ModelConfig, build_model, compile_model
from . import callbacks as cb_module


def set_seeds(seed: int) -> None:
    """Set seeds and enable TF op determinism.

    NOTE: TF_ENABLE_ONEDNN_OPTS=0 disables oneDNN's reordering, which is the
    common source of cross-run nondeterminism on x86. Set before TF imports
    elsewhere — we set it here defensively in case this is called late.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TF versions or partial-determinism platforms — best effort.
        pass


def train(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    epochs: int | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train the deep CLV model end-to-end. Returns a result dict with paths and metrics."""
    set_seeds(int(config["seed"]))

    # --- data ---
    ds_cfg = config["data"]
    train_cfg = config["training"]
    splits = (
        ds_cfg["splits"]["train"],
        ds_cfg["splits"]["val"],
        ds_cfg["splits"]["test"],
    )
    datasets = pipeline.load(
        data_dir=ds_cfg["output_dir"],
        seed=int(config["seed"]),
        splits=splits,
        batch_size=train_cfg["batch_size"],
        shuffle_buffer=train_cfg["shuffle_buffer"],
        cache=True,
    )

    # --- model ---
    mcfg = ModelConfig(
        n_months_history=ds_cfg["n_months_history"],
        emb_industry=config["model"]["emb_industry"],
        emb_region=config["model"]["emb_region"],
        emb_segment=config["model"]["emb_segment"],
        emb_channel=config["model"]["emb_channel"],
        emb_category=config["model"]["emb_category"],
        lstm_1=config["model"]["lstm_1"],
        lstm_2=config["model"]["lstm_2"],
        static_dim_1=config["model"]["static_dim_1"],
        static_dim_2=config["model"]["static_dim_2"],
        head_dim_1=config["model"]["head_dim_1"],
        head_dim_2=config["model"]["head_dim_2"],
        dropout_seq=config["model"]["dropout_seq"],
        dropout_static=config["model"]["dropout_static"],
        dropout_head=config["model"]["dropout_head"],
        recency_attn_units=config["model"]["recency_attn_units"],
    )
    model = build_model(mcfg)
    compile_model(
        model,
        learning_rate=train_cfg["learning_rate"],
        clipnorm=train_cfg["clipnorm"],
        loss_weight_clv=train_cfg["loss_weight_clv"],
        loss_weight_churn=train_cfg["loss_weight_churn"],
    )

    # --- callbacks + paths ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best.weights.h5"
    saved_model_dir = output_dir / "saved_model"
    tb_dir = Path(config["paths"]["runs_dir"]) / "tensorboard"
    callbacks = cb_module.build(
        checkpoint_path=ckpt_path,
        tensorboard_dir=tb_dir,
        early_stopping_patience=train_cfg["early_stopping_patience"],
        reduce_lr_patience=train_cfg["reduce_lr_patience"],
        reduce_lr_factor=train_cfg["reduce_lr_factor"],
        min_lr=train_cfg["min_lr"],
    )

    n_epochs = int(epochs if epochs is not None else train_cfg["epochs"])
    t0 = time.perf_counter()
    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - t0

    # Save SavedModel format (Keras 3 uses .keras single-file). Both saved.
    keras_path = output_dir / "deep_clv.keras"
    model.save(keras_path)
    model.export(saved_model_dir)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2))

    summary = {
        "epochs_trained": len(history.history["loss"]),
        "epochs_requested": n_epochs,
        "train_seconds": round(train_seconds, 2),
        "n_train": len(datasets.train_arrays.y_churn),
        "n_val": len(datasets.val_arrays.y_churn),
        "n_test": len(datasets.test_arrays.y_churn),
        "model_params": int(model.count_params()),
        "config": {**config, "model_dataclass": asdict(mcfg)},
        "paths": {
            "weights": str(ckpt_path),
            "saved_model": str(saved_model_dir),
            "keras_file": str(keras_path),
            "history": str(history_path),
            "tensorboard": str(tb_dir),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
