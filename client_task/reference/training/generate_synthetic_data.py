"""
Synthetic Data Generator for Hea ML Pipelines
Generates realistic training data mimicking user health inputs.
"""

import random
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Symptom text templates
SYMPTOM_TEMPLATES = {
    "low_risk": [
        "Feeling pretty good today. Slept well and had a good morning walk.",
        "Normal day, nothing unusual. Had a bit of a snack craving but otherwise fine.",
        "Good energy levels today. Went for a short jog.",
        "Feeling rested and productive. No complaints.",
        "Had a relaxed day. Sleep was fine, mood was okay.",
        "Pretty standard day. Ate well and felt good overall.",
    ],
    "weak_risk": [
        "Woke up feeling a bit tired today. Not sure why, went to bed early.",
        "Slight headache this afternoon. Probably need more water.",
        "Feeling a bit more stressed than usual about work deadlines.",
        "Didn't sleep great last night. Kept waking up around 3am.",
        "A bit of an upset stomach after lunch. Nothing major.",
        "Feeling somewhat low on energy. Maybe I need to exercise more.",
        "Had some trouble concentrating at work today.",
    ],
    "moderate_risk": [
        "Been having recurring headaches for the past few days. Getting worse each day.",
        "Can't sleep properly. Insomnia has been going on for about a week now.",
        "Feeling really anxious lately. Heart races sometimes for no reason.",
        "Persistent fatigue that won't go away. Always tired no matter how much I sleep.",
        "My mood has been consistently low. Feeling depressed and unmotivated.",
        "Stomach issues keep coming back. Nausea almost every morning.",
        "Back pain getting worse. Hard to sit at my desk for long periods.",
        "Dizziness when I stand up quickly. Happened multiple times this week.",
    ],
    "high_risk": [
        "Experiencing chest pain that comes and goes. Feeling short of breath too.",
        "Severe headache with vision changes. Never had anything like this before.",
        "Heart palpitations are frequent now. Also feeling numbness in my left arm.",
        "Can't breathe properly. Shortness of breath even when resting. Very worried.",
        "Fainting episodes twice this week. Feeling confused and disoriented after.",
        "Blood pressure feels high. Constant headache, chest tightness, and sweating.",
        "Sharp abdominal pain that won't stop. Been vomiting and can't keep food down.",
    ],
}

CHECKBOX_OPTIONS = [
    "headache", "fatigue", "nausea", "dizziness", "insomnia",
    "anxiety", "joint_pain", "muscle_ache", "shortness_of_breath",
    "chest_pain", "stomach_pain", "back_pain", "fever", "cough",
    "sore_throat", "congestion", "appetite_change", "weight_change",
    "skin_changes", "vision_changes", "mood_changes", "concentration",
    "memory", "digestive_issues", "heart_palpitations", "sweating",
]

EMOJI_INPUTS = {
    "positive": ["😊", "💪", "🏃", "😴", "🥗", "☀️"],
    "neutral": ["😐", "🤔", "😶"],
    "negative": ["😫", "🤒", "🤧", "😰", "😢", "😵", "🤢", "😓", "💤"],
}


def generate_daily_metrics(risk_level: str) -> dict:
    """Generate realistic daily metrics based on risk level."""
    ranges = {
        "low_risk": {"sleep": (6.5, 9), "mood": (6, 10), "energy": (6, 10), "stress": (1, 4), "steps": (6000, 15000)},
        "weak_risk": {"sleep": (5, 7.5), "mood": (4, 7), "energy": (4, 7), "stress": (4, 6), "steps": (4000, 10000)},
        "moderate_risk": {"sleep": (3.5, 6), "mood": (2, 5), "energy": (2, 5), "stress": (6, 8), "steps": (2000, 7000)},
        "high_risk": {"sleep": (2, 5), "mood": (1, 3), "energy": (1, 3), "stress": (7, 10), "steps": (500, 4000)},
    }

    r = ranges[risk_level]
    return {
        "sleep_hours": round(random.uniform(*r["sleep"]), 1),
        "mood_score": random.randint(*r["mood"]),
        "energy_level": random.randint(*r["energy"]),
        "stress_level": random.randint(*r["stress"]),
        "steps_count": random.randint(*r["steps"]),
        "water_intake_ml": random.randint(800, 3000),
    }


def generate_checkbox_selections(risk_level: str) -> list:
    """Generate checkbox selections based on risk level."""
    selection_counts = {"low_risk": (0, 1), "weak_risk": (1, 2), "moderate_risk": (2, 4), "high_risk": (3, 6)}
    high_concern = ["chest_pain", "shortness_of_breath", "heart_palpitations"]
    moderate_concern = ["headache", "fatigue", "insomnia", "anxiety", "dizziness", "nausea"]
    low_concern = ["muscle_ache", "cough", "congestion", "appetite_change"]

    count = random.randint(*selection_counts[risk_level])

    if risk_level == "high_risk":
        selections = random.sample(high_concern, min(count, len(high_concern)))
        remaining = count - len(selections)
        if remaining > 0:
            selections += random.sample(moderate_concern, min(remaining, len(moderate_concern)))
    elif risk_level == "moderate_risk":
        selections = random.sample(moderate_concern, min(count, len(moderate_concern)))
    elif risk_level == "weak_risk":
        selections = random.sample(low_concern + moderate_concern[:3], min(count, 5))
    else:
        selections = random.sample(low_concern, min(count, len(low_concern)))

    return selections


def generate_emoji_inputs(risk_level: str) -> list:
    """Generate emoji inputs based on risk level."""
    emoji_map = {
        "low_risk": EMOJI_INPUTS["positive"],
        "weak_risk": EMOJI_INPUTS["neutral"] + EMOJI_INPUTS["negative"][:2],
        "moderate_risk": EMOJI_INPUTS["negative"][:5],
        "high_risk": EMOJI_INPUTS["negative"],
    }
    count = random.randint(0, 3)
    return random.sample(emoji_map[risk_level], min(count, len(emoji_map[risk_level])))


def generate_user_data(num_days: int = 30, risk_profile: str = "mixed") -> list[dict]:
    """
    Generate a full user dataset over multiple days.

    Args:
        num_days: Number of days to simulate
        risk_profile: 'low', 'mixed', 'declining', or 'high'
    """
    data = []
    user_id = str(uuid.uuid4())
    start_date = datetime.utcnow() - timedelta(days=num_days)

    for day in range(num_days):
        date = start_date + timedelta(days=day)

        # Determine risk level for this day
        if risk_profile == "low":
            risk_level = random.choices(["low_risk", "weak_risk"], weights=[0.8, 0.2])[0]
        elif risk_profile == "high":
            risk_level = random.choices(["moderate_risk", "high_risk"], weights=[0.6, 0.4])[0]
        elif risk_profile == "declining":
            # Gradually increasing risk over time
            progress = day / num_days
            if progress < 0.3:
                risk_level = "low_risk"
            elif progress < 0.6:
                risk_level = random.choices(["weak_risk", "moderate_risk"], weights=[0.7, 0.3])[0]
            else:
                risk_level = random.choices(["moderate_risk", "high_risk"], weights=[0.6, 0.4])[0]
        else:  # mixed
            risk_level = random.choices(
                ["low_risk", "weak_risk", "moderate_risk", "high_risk"],
                weights=[0.4, 0.3, 0.2, 0.1],
            )[0]

        entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "date": date.isoformat(),
            "risk_level": risk_level.replace("_risk", "").upper(),
            "symptom_text": random.choice(SYMPTOM_TEMPLATES[risk_level]),
            "emoji_inputs": generate_emoji_inputs(risk_level),
            "checkbox_selections": generate_checkbox_selections(risk_level),
            "daily_metrics": generate_daily_metrics(risk_level),
        }
        data.append(entry)

    return data


def generate_training_dataset(
    num_users: int = 100,
    days_per_user: int = 30,
    output_path: str = "ml/training/data/",
) -> str:
    """Generate a full training dataset with multiple users."""
    all_data = []
    profiles = ["low", "mixed", "declining", "high"]

    for i in range(num_users):
        profile = profiles[i % len(profiles)]
        user_data = generate_user_data(num_days=days_per_user, risk_profile=profile)
        all_data.extend(user_data)

    # Save to file
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_training_data.json"

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    print(f"Generated {len(all_data)} training samples for {num_users} users")
    print(f"Saved to: {output_file}")

    # Print distribution
    from collections import Counter
    risk_dist = Counter(d["risk_level"] for d in all_data)
    print(f"Risk distribution: {dict(risk_dist)}")

    return str(output_file)


if __name__ == "__main__":
    generate_training_dataset(num_users=100, days_per_user=30)
