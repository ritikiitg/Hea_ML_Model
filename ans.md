# Hea — Early Health Risk Prediction Project
> **Status**: Prototype / MVP Complete  
> **Goal**: Detect early weak signals of chronic disease using longitudinal self-reported data.

## 1. Project Overview
This project builds a Machine Learning model to predict **future health risks** (like diabetes, heart disease, or general health decline) before they are clinically diagnosed. 

Instead of relying on expensive medical tests (blood work, MRI), we use **longitudinal self-reported data** (e.g., "How do you feel?", "Do you have trouble walking?", "How is your sleep?"). This makes the solution scalable and suitable for proactive health monitoring.

## 2. Technical Approach
The core engine is an **Ensemble ML Pipeline** trained on the **RAND HRS (Health and Retirement Study)** dataset.

### A. Data Pipeline & Leakage Prevention
*   **Source**: RAND HRS (Longitudinal data from 1992–2018).
*   **Leakage Prevention**: We strictly exclude all data from future waves and any direct disease indicators from the target period. We only use *past* behavior/states to predict *future* onset.

### B. Feature Engineering
We create "weak signals" from raw data:
*   **Trajectories**: `slope_bmi`, `delta_mobility` (Health decline often shows in the *rate of change*).
*   **Volatility**: `std_sleep` (Erratic behavior is a risk signal).
*   **Cross-Domain Interactions**: `Depression × Poor Sleep` (Risk multipliers).

### C. Models
We use a **Stacking Ensemble** of **LightGBM** and **XGBoost**, combined with a **Logistic Regression Meta-Learner**.

---

## 3. Key Questions & Answers (Aligned with Evaluation Criteria)

### Q1: "Why did you choose F2-Score as your primary metric?"
**Ans**: We prioritize **Recall** over Precision. In early health screening, **missing a sick person (False Negative) is much worse than a false alarm (False Positive)**. A false alarm just leads to a suggestion for a check-up or lifestyle change, whereas missing a signal means losing the opportunity for early intervention. F2-score weights recall higher than precision, aligning with this clinical reality.

### Q2: "How do you handle 'Data Leakage'? Are you cheating?"
**Ans**: We perform a strict audit of our features.
1.  **Temporal Separation**: We strictly use data from Waves 1–13 to predict outcomes in Waves 14–16.
2.  **Concept Seperation**: We remove features that are direct proxies for the target (e.g., "taking diabetes medication" is removed when predicting diabetes).
3.  **Automated Checks**: We verify that no target variables appear in the feature set.

### Q3: "Real-world user data is messy and unstructured. How does your model handle that?"
**Ans**:
*   **Robustness**: Our model uses tree-based ensembles (XGBoost/LightGBM) which handle missing values natively and are robust to outliers.
*   **Unstructured Input Strategy**: While we train on structured data, in production, we would use an **LLM-based parsing layer** (e.g., Llama 3 or GPT-4o mini) to extract structured features (like "mobility_level", "sleep_quality") from user chat logs or voice notes before feeding them into our risk model.

### Q4: "Is this solution cost-efficient for millions of users?"
**Ans**: Yes.
*   **Lightweight Inference**: The core risk model is a gradient boosting ensemble, which requires minimal CPU and no GPU for inference.
*   **Low Data Cost**: It relies on self-reported data (free/cheap) rather than clinical biomarkers (expensive).
*   **Open Source**: We use 100% open-source libraries (`scikit-learn`, `lightgbm`, `xgboost`, `shap`), avoiding expensive proprietary licenses.

### Q5: "Can you explain WHY the model flagged me as high risk?"
**Ans**: Yes. We use **SHAP (SHapley Additive exPlanations)** to generate individual explanations.
*   **User View**: "We noticed your walking speed has decreased by 15% over the last year, and your sleep has become irregular."
*   **Doctor View**: "High risk due to: `delta_mobility_score` (-0.15) and `volatility_sleep` (+2.0)."

### Q6: "How do you ensure the model doesn't discriminate (Fairness)?"
**Ans**: We explicitly audit the model using `Fairlearn`.
*   **Metrics**: We check **Equalized Odds** and **Demographic Parity** across Age Groups, Gender, and Race.
*   **Mitigation**: If bias is detected, we re-weight the training data (sample weighting) to penalize errors on underrepresented groups more heavily, ensuring the model performs equally well for everyone.

### Q7: "What is the 'Early Weak Signal' you found?"
**Ans**: We found that **trajectory volatility** (e.g., fluctuating weight or sleep patterns) is often a stronger predictor of future disease than the static value itself. A person whose weight is stable at a slightly higher BMI might be lower risk than someone whose weight is fluctuating wildly.
