"""Single source of truth for vocabularies, feature lists, and tensor shapes.

Importing from this module is preferred over hardcoding strings or dimensions
elsewhere — the model, pipeline, generator, and tests all consume these constants.
"""

from __future__ import annotations

from dataclasses import dataclass

INDUSTRIES: tuple[str, ...] = (
    "Manufacturing",
    "Healthcare",
    "Construction",
    "Retail",
    "Hospitality",
    "Logistics",
    "Education",
    "Government",
    "FoodService",
    "Automotive",
    "Energy",
    "ProfessionalServices",
)
REGIONS: tuple[str, ...] = ("NA-East", "NA-West", "EU", "LATAM", "APAC", "MEA")
SEGMENTS: tuple[str, ...] = ("SMB", "Mid", "Enterprise")
CHANNELS: tuple[str, ...] = ("direct", "partner", "inbound", "outbound")
CATEGORIES: tuple[str, ...] = (
    "Adhesives",
    "Cleaning",
    "Electrical",
    "Fasteners",
    "HandTools",
    "Hardware",
    "Lubricants",
    "Packaging",
    "Plumbing",
    "PowerTools",
    "Safety",
    "Storage",
)
MARGIN_TIERS: tuple[str, ...] = ("low", "med", "high")
MARGIN_TIER_SCORES: dict[str, float] = {"low": 0.10, "med": 0.25, "high": 0.45}

# +1 for the "no-activity" / mask token at index 0 in the sequential category embedding.
N_INDUSTRIES = len(INDUSTRIES)
N_REGIONS = len(REGIONS)
N_SEGMENTS = len(SEGMENTS)
N_CHANNELS = len(CHANNELS)
N_CATEGORIES = len(CATEGORIES)
CATEGORY_PAD_INDEX = 0  # index reserved for inactive months
CATEGORY_VOCAB_SIZE = N_CATEGORIES + 1  # category id 1..N maps to CATEGORIES[i-1]

SEQ_NUMERIC_COLS: tuple[str, ...] = (
    "revenue",
    "order_count",
    "unique_skus",
    "avg_margin_tier_score",
    "bulk_flag",
    "active",
)
N_SEQ_NUMERIC = len(SEQ_NUMERIC_COLS)

STATIC_NUMERIC_COLS: tuple[str, ...] = ("tenure_months_at_t0",)
N_STATIC_NUMERIC = len(STATIC_NUMERIC_COLS)

STATIC_CATEGORICAL_COLS: tuple[str, ...] = ("industry", "region", "segment", "channel")

TARGET_CLV_COL = "clv_next_12m_usd"
TARGET_CHURN_COL = "churn_next_12m"


@dataclass(frozen=True)
class TensorShapes:
    """Reference shapes (without batch dim) consumed by the model + tests."""

    n_months_history: int = 24
    n_seq_numeric: int = N_SEQ_NUMERIC

    @property
    def seq_numeric(self) -> tuple[int, int]:
        return (self.n_months_history, self.n_seq_numeric)

    @property
    def seq_category(self) -> tuple[int]:
        return (self.n_months_history,)


def industry_to_id(name: str) -> int:
    return INDUSTRIES.index(name)


def region_to_id(name: str) -> int:
    return REGIONS.index(name)


def segment_to_id(name: str) -> int:
    return SEGMENTS.index(name)


def channel_to_id(name: str) -> int:
    return CHANNELS.index(name)


def category_to_id(name: str) -> int:
    """1-indexed; 0 is reserved for masked / inactive months."""
    return CATEGORIES.index(name) + 1
