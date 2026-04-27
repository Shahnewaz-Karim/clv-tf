"""Custom Keras layers and metrics for the deep CLV model.

`RecencyWeightedAttention` is a soft-attention pooling that biases attention
toward recent timesteps via a learned exponential decay. It returns the
sequence reweighted in place (shape preserved) so downstream LSTMs receive
a recency-emphasised signal without losing the per-timestep representation.

`RevenueWeightedMAE` weights the absolute error of each example by its
target magnitude. In B2B distribution, large customers dominate revenue
and are disproportionately costly to mispredict — flat MAE understates this.
"""

from __future__ import annotations

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="clv_tf")
class RecencyWeightedAttention(keras.layers.Layer):
    """Soft attention with a learned exponential recency prior.

    Weights for timestep t (1-indexed from the end, so the most recent has t=0):
        w_t = softmax_t( score_t - softplus(decay) * t )
    where `score_t = tanh(W * h_t + b) . v` is a standard additive attention
    score over the input features. The output is `inputs * w[:, :, None]`, so
    inactive (masked) months receive ~0 weight and the sequence shape is preserved.
    """

    def __init__(self, units: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.supports_masking = True

    def build(self, input_shape: tuple[int, ...]) -> None:
        feat_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="W", shape=(feat_dim, self.units), initializer="glorot_uniform"
        )
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(
            name="v", shape=(self.units,), initializer="glorot_uniform"
        )
        self.decay = self.add_weight(name="decay", shape=(), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        # additive attention scores
        proj = tf.tanh(tf.tensordot(inputs, self.W, axes=[[-1], [0]]) + self.b)  # (B, T, U)
        scores = tf.tensordot(proj, self.v, axes=[[-1], [0]])  # (B, T)

        # recency offset — last timestep is t=0, first is t=T-1
        T = tf.shape(inputs)[1]
        rev_pos = tf.cast(T - 1 - tf.range(T), tf.float32)
        scores = scores - tf.nn.softplus(self.decay) * rev_pos[None, :]

        if mask is not None:
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * tf.constant(-1e9, scores.dtype)

        attn = tf.nn.softmax(scores, axis=-1)  # (B, T)
        return inputs * attn[:, :, None]

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor | None:
        return mask

    def get_config(self) -> dict[str, object]:
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


@keras.saving.register_keras_serializable(package="clv_tf")
class RevenueWeightedMAE(keras.metrics.Metric):
    """Mean absolute error weighted by max(y_true, weight_floor) + 1.

    NOTE: training operates in log1p space; this metric is registered on the
    CLV head and therefore reports MAE in log1p space (interpretable as
    "log-dollar error, weighted by log-revenue magnitude"). For raw $ metrics
    see `evaluation.metrics.revenue_weighted_mae`.
    """

    def __init__(self, name: str = "rev_weighted_mae", weight_floor: float = 1.0, **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self.weight_floor = float(weight_floor)
        self.weighted_error = self.add_weight(name="we", initializer="zeros")
        self.total_weight = self.add_weight(name="tw", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None
    ) -> None:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        err = tf.abs(y_true - y_pred)
        w = tf.maximum(y_true, 0.0) + self.weight_floor
        if sample_weight is not None:
            w = w * tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        self.weighted_error.assign_add(tf.reduce_sum(err * w))
        self.total_weight.assign_add(tf.reduce_sum(w))

    def result(self) -> tf.Tensor:
        return self.weighted_error / (self.total_weight + 1e-9)

    def reset_state(self) -> None:
        self.weighted_error.assign(0.0)
        self.total_weight.assign(0.0)

    def get_config(self) -> dict[str, object]:
        cfg = super().get_config()
        cfg.update({"weight_floor": self.weight_floor})
        return cfg
