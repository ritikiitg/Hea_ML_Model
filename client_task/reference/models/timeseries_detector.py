"""
TimeSeriesAnomalyDetector — LSTM + statistical baseline model for detecting
changes in sleep, mood, activity, and stress signals over time.

Architecture:
  Input metrics → LSTM encoder → anomaly scoring head → change-point detection

Purpose: Detect behavioral changes like sleep drops, mood declines, energy loss,
and stress spikes that may indicate emerging health risks.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class TimeSeriesLSTMModel(nn.Module):
        """
        LSTM-based anomaly detector for health metrics time-series.

        Input: (batch, seq_len, num_features) — daily metrics over time
        Output: {
            anomaly_scores: (batch, num_features) — per-metric anomaly scores
            change_points: (batch, seq_len) — change-point probability per timestep
            ts_embedding: (batch, 64) — temporal embedding for fusion
        }
        """

        NUM_FEATURES = 5  # sleep_hours, mood_score, energy_level, stress_level, steps_count

        def __init__(self, input_size: int = 5, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )

            # Anomaly scoring head
            self.anomaly_scorer = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, input_size),
                nn.Sigmoid(),
            )

            # Change-point detection head
            self.changepoint_detector = nn.Sequential(
                nn.Linear(hidden_size * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            # Temporal embedding for fusion
            self.ts_encoder = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
            )

        def forward(self, x):
            # x: (batch, seq_len, num_features)
            lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden*2)

            # Use last timestep for anomaly scoring
            last_hidden = lstm_out[:, -1, :]  # (batch, hidden*2)

            # Anomaly scores per feature
            anomaly_scores = self.anomaly_scorer(last_hidden)  # (batch, num_features)

            # Change point probabilities per timestep
            change_points = self.changepoint_detector(lstm_out).squeeze(-1)  # (batch, seq_len)

            # Temporal embedding
            ts_embedding = self.ts_encoder(last_hidden)  # (batch, 64)

            return {
                "anomaly_scores": anomaly_scores,
                "change_points": change_points,
                "ts_embedding": ts_embedding,
            }

        def get_feature_names(self):
            return ["sleep_hours", "mood_score", "energy_level", "stress_level", "steps_count"]


class StatisticalAnomalyDetector:
    """
    Statistical baseline anomaly detector using Z-scores and IQR.
    Works without PyTorch — production-ready for the prototype.
    """

    # Population-level normal ranges
    NORMAL_RANGES = {
        "sleep_hours": {"mean": 7.5, "std": 1.5, "low_threshold": 4, "high_threshold": 12},
        "mood_score": {"mean": 6.0, "std": 2.0, "low_threshold": 3, "high_threshold": None},
        "energy_level": {"mean": 6.0, "std": 2.0, "low_threshold": 3, "high_threshold": None},
        "stress_level": {"mean": 4.0, "std": 2.0, "low_threshold": None, "high_threshold": 8},
        "steps_count": {"mean": 8000, "std": 3000, "low_threshold": 2000, "high_threshold": None},
    }

    def detect_anomalies(self, current_metrics: dict, historical_metrics: list[dict] = None) -> dict:
        """
        Detect anomalies using Z-score analysis and trend detection.

        Returns:
            dict with anomaly_scores, change_points, ts_embedding, and signal details
        """
        anomaly_scores = {}
        signals = []

        for metric, ranges in self.NORMAL_RANGES.items():
            value = current_metrics.get(metric)
            if value is None:
                anomaly_scores[metric] = 0.0
                continue

            # Z-score relative to population norms
            z_score = abs((value - ranges["mean"]) / ranges["std"]) if ranges["std"] else 0
            anomaly_score = min(1.0, z_score / 3.0)  # Normalize to [0, 1]

            # Threshold-based checks
            if ranges.get("low_threshold") and value < ranges["low_threshold"]:
                anomaly_score = max(anomaly_score, 0.7)
                signals.append({
                    "metric": metric,
                    "type": "below_threshold",
                    "value": value,
                    "threshold": ranges["low_threshold"],
                    "score": anomaly_score,
                })

            if ranges.get("high_threshold") and value > ranges["high_threshold"]:
                anomaly_score = max(anomaly_score, 0.7)
                signals.append({
                    "metric": metric,
                    "type": "above_threshold",
                    "value": value,
                    "threshold": ranges["high_threshold"],
                    "score": anomaly_score,
                })

            anomaly_scores[metric] = round(anomaly_score, 3)

        # Historical trend analysis
        change_points = []
        if historical_metrics and len(historical_metrics) >= 3:
            for metric in self.NORMAL_RANGES:
                values = [h.get(metric) for h in historical_metrics if h.get(metric) is not None]
                if len(values) >= 3:
                    # Simple change-point: significant day-over-day change
                    for i in range(1, len(values)):
                        daily_change = abs(values[i] - values[i - 1])
                        std = self.NORMAL_RANGES[metric]["std"]
                        if daily_change > std * 1.5:
                            change_points.append({
                                "metric": metric,
                                "day_index": i,
                                "change": round(values[i] - values[i - 1], 2),
                                "significance": round(daily_change / std, 2),
                            })

        # Overall anomaly score
        scores = [s for s in anomaly_scores.values() if s > 0]
        overall_score = round(max(scores) if scores else 0.0, 3)

        # Pseudo temporal embedding
        ts_embedding = np.zeros(64)
        for i, (metric, score) in enumerate(anomaly_scores.items()):
            ts_embedding[i * 12:(i + 1) * 12] = score

        return {
            "anomaly_scores": anomaly_scores,
            "change_points": change_points,
            "ts_embedding": ts_embedding.tolist(),
            "overall_score": overall_score,
            "signals": signals,
        }


# Factory function
def create_timeseries_detector(use_lstm: bool = True):
    """Create the appropriate detector based on available dependencies."""
    if HAS_TORCH and use_lstm:
        try:
            model = TimeSeriesLSTMModel()
            logger.info("Created TimeSeriesLSTMModel")
            return model
        except Exception as e:
            logger.warning(f"Could not create LSTM model: {e}")

    logger.info("Created StatisticalAnomalyDetector (fallback)")
    return StatisticalAnomalyDetector()
