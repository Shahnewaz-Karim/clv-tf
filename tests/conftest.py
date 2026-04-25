"""Shared fixtures: a tiny generated dataset that every test can re-use."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.generate import GenerateConfig, generate

_MINI_N_CUSTOMERS = 500
_MINI_SEED = 7


@pytest.fixture(scope="session")
def mini_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("clv_mini") / "synthetic"
    generate(
        GenerateConfig(
            n_customers=_MINI_N_CUSTOMERS,
            n_months_history=24,
            n_months_target=12,
            n_products=200,
            seed=_MINI_SEED,
            output_dir=str(out),
        )
    )
    return out


@pytest.fixture(scope="session")
def mini_n() -> int:
    return _MINI_N_CUSTOMERS


@pytest.fixture(scope="session")
def mini_seed() -> int:
    return _MINI_SEED
