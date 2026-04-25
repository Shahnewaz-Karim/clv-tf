"""Typer CLI — single entrypoint for every workflow step.

Targets mirror the Makefile so users without `make` can run:

    uv run python -m src.cli generate
    uv run python -m src.cli train --epochs 5
    uv run python -m src.cli evaluate
    uv run python -m src.cli smoke
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="clv-tf — multi-task CLV with TensorFlow."
)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@app.command()
def generate(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    n_customers: Annotated[int | None, typer.Option(help="Override n_customers.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override seed.")] = None,
) -> None:
    """Generate the synthetic dataset (parquet + npz)."""
    from .data.generate import GenerateConfig
    from .data.generate import generate as gen_data

    cfg = _load_config(config)
    n = n_customers if n_customers is not None else cfg["data"]["n_customers"]
    s = seed if seed is not None else cfg["seed"]
    gcfg = GenerateConfig(
        n_customers=n,
        n_months_history=cfg["data"]["n_months_history"],
        n_months_target=cfg["data"]["n_months_target"],
        n_products=cfg["data"]["n_products"],
        seed=int(s),
        output_dir=cfg["data"]["output_dir"],
    )
    paths = gen_data(gcfg)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


@app.command()
def baselines(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output_dir: Annotated[Path, typer.Option()] = Path("runs/reports/baselines"),
) -> None:
    """Fit baselines and write a metrics-only report (no deep model required)."""
    from .evaluation.report import evaluate_all

    cfg = _load_config(config)
    summary = evaluate_all(
        config=cfg, model_dir=cfg["paths"]["models_dir"],
        output_dir=output_dir, include_deep=False,
    )
    typer.echo(json.dumps({"primary_model": summary["primary_model"], "n_test": summary["n_test"]}))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    epochs: Annotated[int | None, typer.Option(help="Override epochs.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override models dir.")] = None,
) -> None:
    """Train the deep CLV model."""
    from .training.train import train as train_fn

    cfg = _load_config(config)
    out = Path(output_dir) if output_dir else Path(cfg["paths"]["models_dir"]) / "deep_clv"
    summary = train_fn(config=cfg, output_dir=out, epochs=epochs)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    model_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path, typer.Option()] = Path("runs/reports/full"),
) -> None:
    """Evaluate baselines + deep model and write the comparison report."""
    from .evaluation.report import evaluate_all

    cfg = _load_config(config)
    md = Path(model_dir) if model_dir else Path(cfg["paths"]["models_dir"]) / "deep_clv"
    summary = evaluate_all(config=cfg, model_dir=md, output_dir=output_dir, include_deep=True)
    table_path = Path(summary["paths"]["results_csv"]).with_suffix(".md")
    typer.echo(table_path.read_text())


@app.command()
def smoke(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    n_customers: Annotated[int, typer.Option()] = 500,
    epochs: Annotated[int, typer.Option()] = 1,
) -> None:
    """End-to-end smoke run on a tiny dataset (default: 500 customers, 1 epoch)."""
    from .data.generate import GenerateConfig
    from .data.generate import generate as gen_data
    from .evaluation.report import evaluate_all
    from .training.train import train as train_fn

    cfg = _load_config(config)
    cfg["data"]["n_customers"] = n_customers
    cfg["data"]["output_dir"] = "data/synthetic_smoke"
    cfg["paths"]["models_dir"] = "models/smoke"

    gen_data(
        GenerateConfig(
            n_customers=n_customers,
            n_months_history=cfg["data"]["n_months_history"],
            n_months_target=cfg["data"]["n_months_target"],
            n_products=cfg["data"]["n_products"],
            seed=cfg["seed"],
            output_dir=cfg["data"]["output_dir"],
        )
    )
    out_dir = Path(cfg["paths"]["models_dir"]) / "deep_clv"
    train_fn(config=cfg, output_dir=out_dir, epochs=epochs, verbose=2)
    summary = evaluate_all(
        config=cfg, model_dir=out_dir, output_dir=Path("runs/reports/smoke"), include_deep=True
    )
    typer.echo(f"smoke OK — primary={summary['primary_model']}, n_test={summary['n_test']}")


if __name__ == "__main__":
    app()
