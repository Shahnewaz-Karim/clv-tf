"""Synthetic B2B distribution dataset generator.

Generative model
----------------
The goal is realistic structure, not statistical realism. Real B2B distribution data has:

* **Hierarchy.** Industry & segment set the spending scale; region modulates it.
* **Memory.** Customers who spent last month are likely to spend this month (AR(1)).
* **Seasonality.** Most categories have an annual cycle; phase varies by industry.
* **Pareto orders.** A small share of months contains a bulk order ~5-15x typical spend.
* **Churn dynamics.** Hazard rises with declining recency / consecutive inactive months.
* **Cold start.** A minority of customers signed up partway through the window.

Math (per customer i, month t):
    base_i        = scale[industry_i, segment_i] * region_mul[region_i]
    seasonal_i,t  = 1 + 0.15 * sin(2*pi*t/12 + phase[industry_i])
    momentum_i,t  = 0.55 * (revenue_i,t-1 / base_i) + 0.45
    revenue_i,t   = base_i * seasonal_i,t * momentum_i,t * lognormal(0, 0.25)
                  + bulk_order(prob ~ 0.04)
    active_i,t    = 1 with prob = 1 - inactive_hazard_i,t (raises after low-revenue months)

Outputs (under data/synthetic/):
    customers.parquet, products.parquet, transactions_monthly.parquet, targets.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .schema import (
    CATEGORIES,
    CATEGORY_PAD_INDEX,
    CHANNELS,
    INDUSTRIES,
    MARGIN_TIERS,
    N_CATEGORIES,
    N_CHANNELS,
    N_INDUSTRIES,
    N_REGIONS,
    N_SEGMENTS,
    REGIONS,
    SEGMENTS,
)

# Hyper-parameters of the generative process. Treat these as fixed-by-design;
# tweaking them changes the dataset's character (and any committed metrics).
_BASE_SPEND_BY_SEGMENT = {"SMB": 4_500.0, "Mid": 18_000.0, "Enterprise": 65_000.0}
_INDUSTRY_SCALE = {
    "Manufacturing": 1.30,
    "Healthcare": 1.10,
    "Construction": 1.20,
    "Retail": 0.85,
    "Hospitality": 0.75,
    "Logistics": 1.05,
    "Education": 0.65,
    "Government": 0.95,
    "FoodService": 0.70,
    "Automotive": 1.15,
    "Energy": 1.40,
    "ProfessionalServices": 0.55,
}
_REGION_MULT = {
    "NA-East": 1.05,
    "NA-West": 1.10,
    "EU": 1.00,
    "LATAM": 0.75,
    "APAC": 0.95,
    "MEA": 0.85,
}
_CHANNEL_MULT = {"direct": 1.10, "partner": 1.00, "inbound": 0.95, "outbound": 0.90}

_INDUSTRY_TOP_CATEGORIES = {
    "Manufacturing": ["Fasteners", "Lubricants", "PowerTools", "Safety"],
    "Healthcare": ["Cleaning", "Safety", "Packaging", "Storage"],
    "Construction": ["HandTools", "PowerTools", "Hardware", "Safety"],
    "Retail": ["Packaging", "Storage", "Cleaning", "Hardware"],
    "Hospitality": ["Cleaning", "Packaging", "Plumbing", "Hardware"],
    "Logistics": ["Packaging", "Storage", "Safety", "Adhesives"],
    "Education": ["Cleaning", "Storage", "Hardware", "Electrical"],
    "Government": ["Safety", "Hardware", "Cleaning", "Electrical"],
    "FoodService": ["Cleaning", "Packaging", "Storage", "Plumbing"],
    "Automotive": ["Lubricants", "HandTools", "Fasteners", "PowerTools"],
    "Energy": ["Safety", "Electrical", "Lubricants", "PowerTools"],
    "ProfessionalServices": ["Hardware", "Electrical", "Storage", "Cleaning"],
}


class GenerateConfig(NamedTuple):
    n_customers: int = 50_000
    n_months_history: int = 24
    n_months_target: int = 12
    n_products: int = 200
    seed: int = 42
    output_dir: str = "data/synthetic"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    industry_idx = rng.integers(0, N_INDUSTRIES, size=n)
    region_idx = rng.integers(0, N_REGIONS, size=n)
    # Segments are not uniform: SMB > Mid > Enterprise.
    segment_idx = rng.choice(N_SEGMENTS, size=n, p=[0.60, 0.30, 0.10])
    channel_idx = rng.integers(0, N_CHANNELS, size=n)
    # Tenure at t=0 (start of observation window). 8% are cold-start (< 6 months tenure).
    is_cold_start = rng.random(n) < 0.08
    tenure = np.where(
        is_cold_start,
        rng.integers(0, 6, size=n),
        rng.integers(6, 121, size=n),
    ).astype(np.int16)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n, dtype=np.int64),
            "industry_idx": industry_idx.astype(np.int8),
            "industry": [INDUSTRIES[i] for i in industry_idx],
            "region_idx": region_idx.astype(np.int8),
            "region": [REGIONS[i] for i in region_idx],
            "segment_idx": segment_idx.astype(np.int8),
            "segment": [SEGMENTS[i] for i in segment_idx],
            "channel_idx": channel_idx.astype(np.int8),
            "channel": [CHANNELS[i] for i in channel_idx],
            "tenure_months_at_t0": tenure,
        }
    )


def _build_products(rng: np.random.Generator, n: int) -> pd.DataFrame:
    category_idx = rng.integers(0, N_CATEGORIES, size=n)
    margin_idx = rng.choice(len(MARGIN_TIERS), size=n, p=[0.5, 0.35, 0.15])
    unit_price = np.exp(rng.normal(3.0, 0.9, size=n)).astype(np.float32)  # ~$5-$200
    return pd.DataFrame(
        {
            "sku_id": np.arange(n, dtype=np.int32),
            "category_idx": category_idx.astype(np.int8),
            "category": [CATEGORIES[i] for i in category_idx],
            "margin_tier": [MARGIN_TIERS[i] for i in margin_idx],
            "unit_price_usd": unit_price,
        }
    )


def _generate_revenue_series(
    rng: np.random.Generator, customers: pd.DataFrame, n_total_months: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised AR(1) + seasonality revenue path with bulk shocks and inactivity.

    Returns a tuple of (revenue, order_count, unique_skus, avg_margin, bulk_flag) arrays
    of shape (n_customers, n_total_months). Inactive months have all features = 0.
    """
    n = len(customers)
    T = n_total_months

    seg_scale = np.array([_BASE_SPEND_BY_SEGMENT[s] for s in customers["segment"]])
    ind_scale = np.array([_INDUSTRY_SCALE[s] for s in customers["industry"]])
    reg_scale = np.array([_REGION_MULT[s] for s in customers["region"]])
    ch_scale = np.array([_CHANNEL_MULT[s] for s in customers["channel"]])
    base = seg_scale * ind_scale * reg_scale * ch_scale  # (n,)

    # Phase per industry, fixed across customers in that industry.
    phase_lookup = rng.uniform(0, 2 * np.pi, size=N_INDUSTRIES)
    industry_idx = customers["industry_idx"].to_numpy()
    phase = phase_lookup[industry_idx]  # (n,)

    months = np.arange(T)[None, :]  # (1, T)
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * months / 12.0 + phase[:, None])

    # Cold-start cutoff: customers with tenure < 6 only become active after (6 - tenure) months.
    tenure = customers["tenure_months_at_t0"].to_numpy()
    cold_start_offset = np.maximum(0, 6 - tenure)  # (n,)
    not_yet_active = months < cold_start_offset[:, None]

    # AR(1) momentum, applied multiplicatively. Initialised at 1.0.
    momentum = np.ones((n, T), dtype=np.float32)
    noise = rng.lognormal(0.0, 0.20, size=(n, T)).astype(np.float32)

    # Inactivity hazard with an absorbing "decline" state. Once a customer hits
    # 3+ consecutive inactive months their hazard ratchets up and doesn't reset
    # — this is the realistic B2B pattern where lost accounts rarely return.
    inactive_streak = np.zeros(n, dtype=np.int16)
    in_decline = np.zeros(n, dtype=bool)
    revenue = np.zeros((n, T), dtype=np.float32)
    active = np.zeros((n, T), dtype=bool)

    # Bulk shocks: ~3% of months get a 2.5–5x multiplier (compressed from earlier
    # 5–15x range, which produced an implausible long tail).
    bulk_mask = rng.random((n, T)) < 0.03
    bulk_mult = np.where(bulk_mask, rng.uniform(2.5, 5.0, size=(n, T)), 1.0).astype(np.float32)

    last_rev_ratio = np.ones(n, dtype=np.float32)  # rev_{t-1} / base
    for t in range(T):
        not_active_yet_t = not_yet_active[:, t]
        # Base hazard 0.08; +0.06 per inactive-streak month; +0.20 if last revenue
        # collapsed; declined customers have a much higher floor.
        hazard = 0.08 + 0.06 * inactive_streak + 0.20 * (last_rev_ratio < 0.4)
        hazard = np.where(in_decline, np.maximum(hazard, 0.55), hazard)
        hazard = np.minimum(hazard, 0.92)
        is_active_t = (rng.random(n) > hazard) & (~not_active_yet_t)

        momentum_t = 0.55 * last_rev_ratio + 0.45
        rev_t = base * seasonal[:, t] * momentum_t * noise[:, t] * bulk_mult[:, t]
        rev_t = np.where(is_active_t, rev_t, 0.0).astype(np.float32)
        revenue[:, t] = rev_t
        active[:, t] = is_active_t

        last_rev_ratio = np.where(is_active_t, rev_t / np.maximum(base, 1.0), last_rev_ratio * 0.6)
        inactive_streak = np.where(is_active_t & ~in_decline, 0, inactive_streak + 1)
        in_decline = in_decline | (inactive_streak >= 3)
        momentum[:, t] = momentum_t

    # Order count, SKU count, margin tier per active month.
    order_count = np.zeros_like(revenue, dtype=np.int16)
    unique_skus = np.zeros_like(revenue, dtype=np.int16)
    avg_margin = np.zeros_like(revenue, dtype=np.float32)
    seg_idx = customers["segment_idx"].to_numpy()
    seg_avg_orders = np.array([2.0, 5.0, 12.0])  # SMB / Mid / Enterprise
    avg_orders_per_cust = seg_avg_orders[seg_idx]
    # Where active, sample order_count ~ Poisson(seg_avg) clipped at 1.
    sampled_orders = np.maximum(
        1, rng.poisson(avg_orders_per_cust[:, None] * np.ones((1, T)))
    ).astype(np.int16)
    sampled_skus = np.maximum(
        1, (sampled_orders * rng.uniform(0.6, 1.4, size=(n, T))).astype(np.int16)
    )
    # Margin tier: mostly low+med, biased by industry (high-margin industries skew up).
    high_margin_industries = {"Healthcare", "Energy", "ProfessionalServices"}
    is_hi_margin_ind = np.array(
        [c in high_margin_industries for c in customers["industry"]]
    )
    base_margin = np.where(is_hi_margin_ind, 0.30, 0.20)
    sampled_margin = base_margin[:, None] + rng.normal(0, 0.05, size=(n, T))
    sampled_margin = np.clip(sampled_margin, 0.05, 0.55).astype(np.float32)

    order_count = np.where(active, sampled_orders, 0).astype(np.int16)
    unique_skus = np.where(active, sampled_skus, 0).astype(np.int16)
    avg_margin = np.where(active, sampled_margin, 0.0).astype(np.float32)
    bulk_flag = (active & bulk_mask).astype(np.bool_)

    return revenue, order_count, unique_skus, avg_margin, bulk_flag


def _build_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    n_total_months: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the long-form monthly transaction table + the dense top-category index array.

    The top_category for an active month is sampled from the customer's industry-affine mix.
    Inactive months get CATEGORY_PAD_INDEX (=0). The dense array (n, T) is returned
    alongside so callers can build sequence tensors without a join.
    """
    revenue, order_count, unique_skus, avg_margin, bulk_flag = _generate_revenue_series(
        rng, customers, n_total_months
    )
    n, T = revenue.shape

    # Sample top_category each month from the industry's preferred mix (60%) or any (40%).
    pref_lookup = np.zeros((N_INDUSTRIES, 4), dtype=np.int8)
    for ind_name, prefs in _INDUSTRY_TOP_CATEGORIES.items():
        pref_lookup[INDUSTRIES.index(ind_name)] = [CATEGORIES.index(p) for p in prefs]
    industry_idx = customers["industry_idx"].to_numpy()
    pref_choices = pref_lookup[industry_idx]  # (n, 4)

    use_pref = rng.random((n, T)) < 0.60
    pref_idx = rng.integers(0, 4, size=(n, T))
    pref_cat = pref_choices[np.arange(n)[:, None], pref_idx]
    any_cat = rng.integers(0, N_CATEGORIES, size=(n, T))
    top_category_0idx = np.where(use_pref, pref_cat, any_cat).astype(np.int8)
    # Shift to 1-indexed so 0 is the inactive sentinel; zero out inactive months.
    active = revenue > 0
    top_category = np.where(active, top_category_0idx + 1, CATEGORY_PAD_INDEX).astype(np.int8)

    # Long-form: keep only active months to avoid 36*50k = 1.8M rows of zeros.
    cust_ids = customers["customer_id"].to_numpy()
    rows = []
    active_idx = np.argwhere(active)
    for c, t in active_idx:
        rows.append(
            (
                int(cust_ids[c]),
                int(t),
                float(revenue[c, t]),
                int(order_count[c, t]),
                int(unique_skus[c, t]),
                int(top_category[c, t]),
                float(avg_margin[c, t]),
                bool(bulk_flag[c, t]),
            )
        )
    transactions = pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "month_idx",
            "revenue_usd",
            "order_count",
            "unique_skus",
            "top_category_id",
            "avg_margin_tier_score",
            "bulk_flag",
        ],
    )
    transactions = transactions.astype(
        {
            "customer_id": "int64",
            "month_idx": "int8",
            "revenue_usd": "float32",
            "order_count": "int16",
            "unique_skus": "int16",
            "top_category_id": "int8",
            "avg_margin_tier_score": "float32",
            "bulk_flag": "bool",
        }
    )
    # Pack the dense arrays the model will consume into a single payload.
    dense = {
        "revenue": revenue,
        "order_count": order_count.astype(np.float32),
        "unique_skus": unique_skus.astype(np.float32),
        "top_category": top_category,
        "avg_margin": avg_margin,
        "bulk_flag": bulk_flag.astype(np.float32),
        "active": active.astype(np.float32),
    }
    return transactions, dense


def _build_targets(
    customers: pd.DataFrame,
    dense: dict,
    history_months: int,
    target_months: int,
) -> tuple[pd.DataFrame, dict]:
    """Compute 12-month forward CLV and churn flag from the post-history window.

    CLV  = sum(revenue[history:history+target])
    Churn = (sum(active[history:history+target]) <= 1) — at most one active month
    """
    rev_target_window = dense["revenue"][:, history_months : history_months + target_months]
    act_target_window = dense["active"][:, history_months : history_months + target_months]
    clv = rev_target_window.sum(axis=1).astype(np.float32)
    # Functional churn: active in fewer than 4 of the next 12 months. This is a
    # softer threshold than "permanently inactive" — captures customers who have
    # collapsed to incidental orders, which is the operational definition that
    # matters in B2B distribution.
    churn = (act_target_window.sum(axis=1) < 4.0)
    targets = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "clv_next_12m_usd": clv,
            "churn_next_12m": churn,
        }
    )
    # Slice dense arrays down to the input window only (the model never sees target months).
    dense_input = {k: v[:, :history_months] for k, v in dense.items()}
    return targets, dense_input


def generate(config: GenerateConfig | None = None) -> dict[str, Path]:
    """Generate the full dataset and write parquets. Returns a dict of artifact paths."""
    cfg = config or GenerateConfig()
    rng = _rng(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = _build_customers(rng, cfg.n_customers)
    products = _build_products(rng, cfg.n_products)

    n_total = cfg.n_months_history + cfg.n_months_target
    transactions, dense = _build_transactions(rng, customers, n_total)
    targets, dense_input = _build_targets(
        customers, dense, cfg.n_months_history, cfg.n_months_target
    )

    paths = {
        "customers": out_dir / "customers.parquet",
        "products": out_dir / "products.parquet",
        "transactions": out_dir / "transactions_monthly.parquet",
        "targets": out_dir / "targets.parquet",
        "dense_input": out_dir / "dense_input.npz",
    }
    customers.to_parquet(paths["customers"], index=False)
    products.to_parquet(paths["products"], index=False)
    transactions.to_parquet(paths["transactions"], index=False)
    targets.to_parquet(paths["targets"], index=False)
    np.savez_compressed(paths["dense_input"], **dense_input)
    return paths
