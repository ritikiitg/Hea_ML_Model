"""
Training Script for TimeSeriesAnomalyDetector (LSTM)
Trains the LSTM model on daily metrics time-series data.
"""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


METRIC_KEYS = ["sleep_hours", "mood_score", "energy_level", "stress_level", "steps_count"]
RISK_LABELS = {"LOW": 0, "WEAK": 1, "MODERATE": 2, "HIGH": 3}

# Normalization constants
NORM = {
    "sleep_hours": {"mean": 7.5, "std": 1.5},
    "mood_score": {"mean": 6.0, "std": 2.0},
    "energy_level": {"mean": 6.0, "std": 2.0},
    "stress_level": {"mean": 4.0, "std": 2.0},
    "steps_count": {"mean": 8000, "std": 3000},
}


class TimeSeriesDataset(Dataset):
    """Dataset for training the LSTM anomaly detector."""

    def __init__(self, sequences: list[dict], seq_length: int = 7):
        self.sequences = sequences
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        metrics = seq["metrics"]  # list of daily metric dicts
        label = RISK_LABELS.get(seq["label"], 0)

        # Normalize and pad
        normalized = []
        for day_metrics in metrics[-self.seq_length:]:
            day_vec = []
            for key in METRIC_KEYS:
                val = day_metrics.get(key)
                if val is not None:
                    norm_val = (val - NORM[key]["mean"]) / NORM[key]["std"]
                else:
                    norm_val = 0.0
                day_vec.append(norm_val)
            normalized.append(day_vec)

        # Pad if shorter than seq_length
        while len(normalized) < self.seq_length:
            normalized.insert(0, [0.0] * len(METRIC_KEYS))

        # Create anomaly labels (1 if day's risk was high)
        anomaly_target = []
        for day_metrics in metrics[-self.seq_length:]:
            score = 0.0
            for key in METRIC_KEYS:
                val = day_metrics.get(key)
                if val is not None:
                    z = abs((val - NORM[key]["mean"]) / NORM[key]["std"])
                    score = max(score, min(1.0, z / 3.0))
            anomaly_target.append(score)
        while len(anomaly_target) < self.seq_length:
            anomaly_target.insert(0, 0.0)

        return {
            "sequence": torch.tensor(normalized, dtype=torch.float),
            "anomaly_target": torch.tensor(anomaly_target, dtype=torch.float),
            "risk_label": torch.tensor(label, dtype=torch.long),
        }


def prepare_sequences(data: list[dict], seq_length: int = 7) -> list[dict]:
    """Group data by user and create overlapping sequences."""
    from collections import defaultdict

    user_data = defaultdict(list)
    for entry in data:
        user_data[entry["user_id"]].append(entry)

    sequences = []
    for user_id, entries in user_data.items():
        entries.sort(key=lambda x: x["date"])
        for i in range(len(entries) - seq_length + 1):
            window = entries[i:i + seq_length]
            sequences.append({
                "user_id": user_id,
                "metrics": [e["daily_metrics"] for e in window],
                "label": window[-1]["risk_level"],
            })

    return sequences


def train_timeseries_model(
    data_path: str = "ml/training/data/synthetic_training_data.json",
    model_save_path: str = "ml/trained_models/timeseries_detector/",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seq_length: int = 7,
):
    """Train the TimeSeriesAnomalyDetector LSTM model."""
    if not HAS_TORCH:
        logger.error("PyTorch not available. Cannot train time-series model.")
        return

    # Load data
    data_file = Path(data_path)
    if not data_file.exists():
        logger.info("Training data not found. Generating synthetic data...")
        from ml.training.generate_synthetic_data import generate_training_dataset
        generate_training_dataset(num_users=50, days_per_user=30, output_path="ml/training/data/")

    with open(data_file) as f:
        data = json.load(f)

    # Prepare sequences
    sequences = prepare_sequences(data, seq_length)
    logger.info(f"Created {len(sequences)} sequences from {len(data)} data points")

    dataset = TimeSeriesDataset(sequences, seq_length)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    from ml.models.timeseries_detector import TimeSeriesLSTMModel
    model = TimeSeriesLSTMModel(input_size=len(METRIC_KEYS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss & optimizer
    anomaly_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            sequence = batch["sequence"].to(device)
            anomaly_target = batch["anomaly_target"].to(device)

            outputs = model(sequence)
            anomaly_scores = outputs["anomaly_scores"]

            # Loss: predict anomaly score for the last timestep
            target_score = anomaly_target[:, -1].unsqueeze(1).expand_as(anomaly_scores)
            loss = anomaly_criterion(anomaly_scores, target_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sequence = batch["sequence"].to(device)
                anomaly_target = batch["anomaly_target"].to(device)

                outputs = model(sequence)
                target_score = anomaly_target[:, -1].unsqueeze(1).expand_as(outputs["anomaly_scores"])
                loss = anomaly_criterion(outputs["anomaly_scores"], target_score)
                val_loss += loss.item()

        val_avg = val_loss / len(val_loader) if val_loader else 0
        logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_loss:.4f}, Val Loss: {val_avg:.4f}")

    # Save model
    save_path = Path(model_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    logger.info(f"Time-series model saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_timeseries_model()
