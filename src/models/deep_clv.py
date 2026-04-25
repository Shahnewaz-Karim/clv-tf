"""Hybrid deep CLV model — sequence encoder + static tower + multi-task heads.

Architecture (Keras Functional API):

    seq_numeric (24, 6) ─┐
                         ├─ Concat ─ Masking ─ RecencyAttn ─ BiLSTM(64) ─ LN ─ BiLSTM(32) ─ Drop
    seq_top_cat (24,) ──[Emb,mask=0]┘                                                        │
                                                                                             │
    industry/region/segment/channel ─[Embeddings]─┐                                          │
    tenure (1,) ──────────────────────────────────┴─ Concat ─ Dense(64) ─ Drop ─ Dense(32) ──┤
                                                                                             │
                                                          ┌─ Dense(1, softplus) → clv  (log1p $)
              Concat ─ Dense(64) ─ Drop ─ Dense(32) ──────┤
                                                          └─ Dense(1, sigmoid)  → churn (0/1)

Why BiLSTM (not 1D-CNN or Transformer):
    * 24 timesteps is short — self-attention has no length advantage and
      its O(T^2) overhead is wasted.
    * Masking inactive months matters more than long-range dependencies.
      LSTM with `mask_zero` from the category embedding handles this cleanly.
    * Recency dominates the signal; a learned recency prior (the custom
      RecencyWeightedAttention layer) plus an LSTM is a strong inductive bias.
    * BiLSTM lets the encoder use both early-history (cold-start) and
      late-window (recent activity) signals symmetrically.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers as L

from ..data.schema import (
    CATEGORY_VOCAB_SIZE,
    N_CHANNELS,
    N_INDUSTRIES,
    N_REGIONS,
    N_SEGMENTS,
    N_SEQ_NUMERIC,
)
from .layers import RecencyWeightedAttention, RevenueWeightedMAE


@dataclass(frozen=True)
class ModelConfig:
    n_months_history: int = 24
    emb_industry: int = 4
    emb_region: int = 3
    emb_segment: int = 2
    emb_channel: int = 2
    emb_category: int = 4
    lstm_1: int = 64
    lstm_2: int = 32
    static_dim_1: int = 64
    static_dim_2: int = 32
    head_dim_1: int = 64
    head_dim_2: int = 32
    dropout_seq: float = 0.2
    dropout_static: float = 0.1
    dropout_head: float = 0.2
    recency_attn_units: int = 16


def build_model(cfg: ModelConfig | None = None) -> keras.Model:
    cfg = cfg or ModelConfig()

    # --- inputs ---------------------------------------------------------------
    seq_numeric = keras.Input(
        shape=(cfg.n_months_history, N_SEQ_NUMERIC), name="seq_numeric", dtype="float32"
    )
    seq_top_category = keras.Input(
        shape=(cfg.n_months_history,), name="seq_top_category", dtype="int32"
    )
    industry = keras.Input(shape=(), name="industry", dtype="int32")
    region = keras.Input(shape=(), name="region", dtype="int32")
    segment = keras.Input(shape=(), name="segment", dtype="int32")
    channel = keras.Input(shape=(), name="channel", dtype="int32")
    tenure = keras.Input(shape=(1,), name="tenure", dtype="float32")

    # --- sequence branch ------------------------------------------------------
    cat_emb = L.Embedding(
        CATEGORY_VOCAB_SIZE, cfg.emb_category, mask_zero=True, name="emb_category"
    )(seq_top_category)
    seq = L.Concatenate(axis=-1, name="seq_concat")([seq_numeric, cat_emb])
    seq = RecencyWeightedAttention(units=cfg.recency_attn_units, name="recency_attn")(seq)
    seq = L.Bidirectional(L.LSTM(cfg.lstm_1, return_sequences=True), name="bilstm_1")(seq)
    seq = L.LayerNormalization(name="seq_ln")(seq)
    seq = L.Bidirectional(L.LSTM(cfg.lstm_2), name="bilstm_2")(seq)
    seq = L.Dropout(cfg.dropout_seq, name="seq_drop")(seq)

    # --- static branch --------------------------------------------------------
    ind_emb = L.Embedding(N_INDUSTRIES, cfg.emb_industry, name="emb_industry")(industry)
    reg_emb = L.Embedding(N_REGIONS, cfg.emb_region, name="emb_region")(region)
    seg_emb = L.Embedding(N_SEGMENTS, cfg.emb_segment, name="emb_segment")(segment)
    chan_emb = L.Embedding(N_CHANNELS, cfg.emb_channel, name="emb_channel")(channel)
    static = L.Concatenate(name="static_concat")([ind_emb, reg_emb, seg_emb, chan_emb, tenure])
    static = L.Dense(cfg.static_dim_1, activation="gelu", name="static_dense_1")(static)
    static = L.Dropout(cfg.dropout_static, name="static_drop")(static)
    static = L.Dense(cfg.static_dim_2, activation="gelu", name="static_dense_2")(static)

    # --- shared head ----------------------------------------------------------
    h = L.Concatenate(name="head_concat")([seq, static])
    h = L.Dense(cfg.head_dim_1, activation="gelu", name="head_dense_1")(h)
    h = L.Dropout(cfg.dropout_head, name="head_drop")(h)
    h = L.Dense(cfg.head_dim_2, activation="gelu", name="head_dense_2")(h)

    # --- multi-task outputs ---------------------------------------------------
    # softplus on the CLV head ensures non-negative predictions in log1p-$ space.
    clv_out = L.Dense(1, activation="softplus", name="clv")(h)
    churn_out = L.Dense(1, activation="sigmoid", name="churn")(h)

    return keras.Model(
        inputs={
            "seq_numeric": seq_numeric,
            "seq_top_category": seq_top_category,
            "industry": industry,
            "region": region,
            "segment": segment,
            "channel": channel,
            "tenure": tenure,
        },
        outputs={"clv": clv_out, "churn": churn_out},
        name="clv_deep",
    )


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
    loss_weight_clv: float = 1.0,
    loss_weight_churn: float = 0.5,
) -> keras.Model:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss={
            "clv": keras.losses.Huber(delta=1.0),
            "churn": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"clv": loss_weight_clv, "churn": loss_weight_churn},
        metrics={
            "clv": [keras.metrics.MeanAbsoluteError(name="mae"), RevenueWeightedMAE()],
            "churn": [
                keras.metrics.AUC(name="auc"),
                keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            ],
        },
    )
    return model
