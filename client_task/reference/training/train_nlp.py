"""
Training Script for WeakSignalDetector_NLP (DistilBERT)
Trains the NLP model on synthetic health data to detect weak symptom patterns.
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
    from transformers import AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


RISK_TO_LABEL = {"LOW": 0, "WEAK": 1, "MODERATE": 2, "HIGH": 3}
SIGNAL_CATEGORIES = {
    "fatigue": ["tired", "exhausted", "fatigue", "no energy", "drained"],
    "pain": ["pain", "ache", "sore", "cramp", "sharp"],
    "mood": ["sad", "depressed", "anxious", "worry", "irritable"],
    "sleep": ["insomnia", "can't sleep", "waking up", "restless"],
    "digestive": ["nausea", "vomiting", "stomach", "bloating"],
    "respiratory": ["cough", "breathing", "wheeze", "congestion"],
    "cardiovascular": ["heart", "palpitations", "chest pain", "blood pressure"],
    "neurological": ["headache", "migraine", "numbness", "vision", "confusion"],
}


class HealthTextDataset(Dataset):
    """Dataset for training the NLP weak signal detector."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("symptom_text", "")

        # Tokenize
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        # Create signal labels (multi-label)
        signal_labels = self._extract_signal_labels(text)

        # Risk label
        risk_label = RISK_TO_LABEL.get(item.get("risk_level", "LOW"), 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "signal_labels": torch.tensor(signal_labels, dtype=torch.float),
            "risk_label": torch.tensor(risk_label, dtype=torch.long),
        }

    def _extract_signal_labels(self, text: str) -> list[float]:
        """Extract binary signal labels from text content."""
        text_lower = text.lower()
        labels = []
        for category, keywords in SIGNAL_CATEGORIES.items():
            present = any(kw in text_lower for kw in keywords)
            labels.append(1.0 if present else 0.0)
        return labels


def train_nlp_model(
    data_path: str = "ml/training/data/synthetic_training_data.json",
    model_save_path: str = "ml/trained_models/weak_signal_nlp/",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """Train the WeakSignalDetector_NLP model."""
    if not HAS_TORCH:
        logger.error("PyTorch not available. Cannot train NLP model.")
        return

    # Load data
    data_file = Path(data_path)
    if not data_file.exists():
        logger.info("Training data not found. Generating synthetic data...")
        from ml.training.generate_synthetic_data import generate_training_dataset
        generate_training_dataset(num_users=50, days_per_user=30, output_path="ml/training/data/")

    with open(data_file) as f:
        data = json.load(f)

    # Filter entries with text
    data = [d for d in data if d.get("symptom_text")]
    logger.info(f"Training on {len(data)} text samples")

    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = HealthTextDataset(data, tokenizer)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    from ml.models.weak_signal_nlp import WeakSignalNLPModel
    model = WeakSignalNLPModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss & optimizer
    signal_criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            signal_labels = batch["signal_labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = signal_criterion(outputs["signal_scores"], signal_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                signal_labels = batch["signal_labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = signal_criterion(outputs["signal_scores"], signal_labels)
                val_loss += loss.item()

        val_avg = val_loss / len(val_loader) if val_loader else 0
        logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_loss:.4f}, Val Loss: {val_avg:.4f}")

    # Save model
    save_path = Path(model_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"NLP model saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_nlp_model()
