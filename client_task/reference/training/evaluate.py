"""
Evaluation Script for Hea ML Models
Computes signal precision, false alarm rate, and feedback alignment metrics.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """
    Evaluate model predictions against ground truth.

    Metrics:
    - Signal Precision: % of detected signals that were true positives (target ≥ 0.7)
    - False Alarm Rate: % of predictions that were false positives (target ≤ 0.3)
    - Feedback Alignment: correlation between model output and user feedback
    - Risk Level Accuracy: correct risk level classification rate
    """
    if not predictions or not ground_truth:
        return {"error": "No data to evaluate"}

    # Build lookup
    gt_map = {g["id"]: g for g in ground_truth}

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    correct_risk = 0
    total = 0
    risk_confusion = defaultdict(lambda: defaultdict(int))

    for pred in predictions:
        gt = gt_map.get(pred.get("id"))
        if not gt:
            continue

        total += 1
        pred_risk = pred.get("risk_level", "LOW")
        actual_risk = gt.get("risk_level", "LOW")

        risk_confusion[actual_risk][pred_risk] += 1

        if pred_risk == actual_risk:
            correct_risk += 1

        # Binary classification: HIGH/MODERATE = positive, LOW/WEAK = negative
        pred_positive = pred_risk in ("HIGH", "MODERATE")
        actual_positive = actual_risk in ("HIGH", "MODERATE")

        if pred_positive and actual_positive:
            true_positives += 1
        elif pred_positive and not actual_positive:
            false_positives += 1
        elif not pred_positive and not actual_positive:
            true_negatives += 1
        else:
            false_negatives += 1

    # Calculate metrics
    signal_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (signal_precision * recall) / (signal_precision + recall) if (signal_precision + recall) > 0 else 0

    results = {
        "total_samples": total,
        "signal_precision": round(signal_precision, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "risk_accuracy": round(correct_risk / total, 4) if total else 0,
        "confusion_matrix": dict(risk_confusion),
        "targets": {
            "signal_precision_target": 0.7,
            "signal_precision_met": signal_precision >= 0.7,
            "false_alarm_rate_target": 0.3,
            "false_alarm_rate_met": false_alarm_rate <= 0.3,
        },
    }

    return results


def evaluate_feedback_alignment(feedback_data: list[dict]) -> dict:
    """
    Evaluate how well model predictions align with user feedback.
    
    Computes feedback alignment score:
    - 1.0 = perfect alignment (users always confirm)
    - 0.0 = no alignment (users always reject)
    """
    if not feedback_data:
        return {"alignment_score": 0.0, "total_feedback": 0}

    confirms = sum(1 for f in feedback_data if f.get("feedback_type") == "confirm")
    rejects = sum(1 for f in feedback_data if f.get("feedback_type") == "reject")
    adjusts = sum(1 for f in feedback_data if f.get("feedback_type") == "adjust")
    total = len(feedback_data)

    alignment = (confirms + 0.5 * adjusts) / total if total else 0
    relevance_scores = [f["relevance_score"] for f in feedback_data if f.get("relevance_score")]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    return {
        "alignment_score": round(alignment, 4),
        "total_feedback": total,
        "confirm_rate": round(confirms / total, 4) if total else 0,
        "reject_rate": round(rejects / total, 4) if total else 0,
        "adjust_rate": round(adjusts / total, 4) if total else 0,
        "avg_relevance_score": round(avg_relevance, 2),
    }


def generate_evaluation_report(
    predictions: list[dict],
    ground_truth: list[dict],
    feedback_data: list[dict] = None,
    output_path: str = "ml/evaluation_report.json",
) -> dict:
    """Generate a complete evaluation report."""
    report = {
        "prediction_metrics": evaluate_predictions(predictions, ground_truth),
        "feedback_alignment": evaluate_feedback_alignment(feedback_data or []),
    }

    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_file}")
    logger.info(f"Signal Precision: {report['prediction_metrics'].get('signal_precision', 'N/A')}")
    logger.info(f"False Alarm Rate: {report['prediction_metrics'].get('false_alarm_rate', 'N/A')}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo evaluation with synthetic data
    from ml.training.generate_synthetic_data import generate_user_data

    data = generate_user_data(num_days=100, risk_profile="mixed")
    # Simulate predictions (using ground truth with noise)
    import random
    predictions = []
    for d in data:
        pred = dict(d)
        if random.random() < 0.15:  # 15% error rate
            levels = ["LOW", "WEAK", "MODERATE", "HIGH"]
            pred["risk_level"] = random.choice(levels)
        predictions.append(pred)

    report = generate_evaluation_report(predictions, data)
    print(json.dumps(report, indent=2))
