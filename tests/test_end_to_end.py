"""End-to-end smoke: generate -> train (1 epoch) -> evaluate. Must finish fast."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data.generate import GenerateConfig, generate
from src.evaluation.report import evaluate_all
from src.training.train import train


def _smoke_config(tmp_path: Path) -> dict:
    base = yaml.safe_load(Path("configs/default.yaml").read_text())
    base["data"]["n_customers"] = 500
    base["data"]["output_dir"] = str(tmp_path / "synthetic_e2e")
    base["paths"]["models_dir"] = str(tmp_path / "models_e2e")
    base["paths"]["runs_dir"] = str(tmp_path / "runs_e2e")
    base["paths"]["reports_dir"] = str(tmp_path / "reports_e2e")
    base["training"]["epochs"] = 1
    base["training"]["batch_size"] = 64
    base["training"]["shuffle_buffer"] = 256
    return base


def test_smoke_end_to_end(tmp_path: Path):
    cfg = _smoke_config(tmp_path)
    generate(
        GenerateConfig(
            n_customers=cfg["data"]["n_customers"],
            n_months_history=cfg["data"]["n_months_history"],
            n_months_target=cfg["data"]["n_months_target"],
            n_products=cfg["data"]["n_products"],
            seed=cfg["seed"],
            output_dir=cfg["data"]["output_dir"],
        )
    )
    out_dir = Path(cfg["paths"]["models_dir"]) / "deep_clv"
    summary = train(config=cfg, output_dir=out_dir, epochs=1, verbose=0)
    assert summary["epochs_trained"] == 1
    assert (Path(summary["paths"]["keras_file"])).exists()

    report = evaluate_all(
        config=cfg,
        model_dir=out_dir,
        output_dir=Path(cfg["paths"]["reports_dir"]),
        include_deep=True,
    )
    # We expect all four baselines + the deep model to appear.
    names = {row["model"] for row in report["metrics_table"]}
    assert {"mean", "carry_forward", "linear_rfm", "lightgbm", "deep_clv"} <= names
