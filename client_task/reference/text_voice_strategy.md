# Text/Voice Feature Extraction Strategy

## Overview

Hea's early health risk detection can be significantly enhanced by extracting structured features from unstructured text and voice inputs. This strategy enables the longitudinal prediction model to incorporate real-time, self-reported symptom data alongside the structured survey responses.

## Text Feature Extraction Pipeline

### Stage 1: Symptom Entity Recognition
Use a lightweight NER model (spaCy + BioBERT fine-tuned) to extract health-related entities from free-text symptom descriptions:
- **Symptom mentions**: "headache", "fatigue", "chest pain", "difficulty breathing"
- **Severity indicators**: "severe", "mild", "getting worse", "constant"
- **Temporal markers**: "for 3 days", "since last week", "every morning"
- **Body locations**: "lower back", "left arm", "stomach"

### Stage 2: Sentiment & Distress Scoring
Apply a distress classifier on the text to produce a continuous distress score (0-1):
- Fine-tune DistilBERT on health-related sentiment data
- Output features: `text_distress_score`, `text_urgency_score`, `text_symptom_count`
- These continuous features integrate directly into the tabular model

### Stage 3: Longitudinal Text Features
Track text-derived features over time to detect trajectories:
- `symptom_count_7d_avg` — rolling average of daily symptom mentions
- `distress_trend_slope` — is distress increasing or stable?
- `new_symptom_flag` — did the user mention a symptom never mentioned before?
- `vocabulary_complexity_change` — cognitive decline may reduce language complexity

## Voice Feature Extraction Pipeline

### Acoustic Biomarkers
Extract clinically-validated acoustic features from voice recordings:
- **Fundamental frequency (F0)**: Depression and fatigue lower vocal pitch
- **Speech rate**: Cognitive decline and medication effects slow speech
- **Jitter/shimmer**: Voice quality degrades with respiratory conditions
- **Pause patterns**: Longer pauses correlate with cognitive load and depression

### Implementation
Use `openSMILE` or `librosa` to extract a standardized feature set (eGeMAPS — 88 features), then:
1. Compute per-user baselines from first 2 weeks of voice data
2. Generate delta features: `f0_change_from_baseline`, `speech_rate_delta`
3. Feed these as additional columns into the trained ensemble model

## Integration with Current Model

The trained LightGBM/XGBoost ensemble accepts tabular features. Text/voice features are converted to numeric columns and appended:

```
Existing features (waves 1-13):  ~400 columns
+ Text features per session:       ~10 columns  
+ Voice features per session:      ~15 columns
= Total input to model:           ~425 columns
```

The model retrains periodically (weekly) to incorporate new feature columns. For real-time inference, the same `prediction_service.py` handles prediction with SHAP explanations covering all feature types.

## Privacy Considerations

- Text is processed locally; only extracted features (numeric scores) are stored
- Voice recordings are deleted after feature extraction — only acoustic features persist
- All processing uses privacy-preserving, on-device NLP where possible
- Users can opt out of text/voice analysis while keeping structured input tracking
