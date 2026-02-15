Task Third is Judging Criteria

Your submission will be evaluated on two main areas: model performance and practical considerations.

1. Primary Metrics (60% of score)

We will measure how well your model predicts who will get sick using three metrics:

F2-Score measures how well you catch people who will develop a disease. We prioritize recall over precision as missing a sick person is worse than a false alarm.

PR-AUC (Precision-Recall Area Under Curve) shows how your model performs with imbalanced data. Most people in the dataset are healthy, and your model must handle that well.

ROC-AUC is the industry standard metric that allows us to compare your solution with published benchmarks.

2. Additional Criteria (40% of score)

No Data Leakage: Your model must not use features that already reveal the disease. For example, if someone takes medication for diabetes, they already have diabetes (that's cheating 😛). We will audit your feature set.

Real World Usability: Your model will receive self reported data from regular people, not clinical records. It must handle noisy, incomplete, and inconsistent inputs gracefully.

Cost Efficiency: Simple beats expensive. A lightweight model that runs fast is better than an overengineered solution with costly API calls.

Open Source Only: All tools, libraries, and data sources must be open and reproducible. No proprietary black boxes.

Explainability: Can you explain why your model flagged someone as high risk? Both users and doctors need to understand the reasoning.

Fairness: Your model should not discriminate by age, gender, or ethnicity. We will check for bias.

3. Bonus Points

We will award extra points for novel feature engineering, discovery of non-obvious correlations, and production-ready code quality.

