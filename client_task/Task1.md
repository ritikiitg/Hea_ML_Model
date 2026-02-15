First Thing

Your task is to build a model that predicts early health risk using longitudinal, self-reported data. You will work with structured panel datasets (such as RAND HRS, NLSY97, or PSID-SHELF) and train a model that can identify people who are likely to develop a disease or experience health decline in the future. The key goal is to detect weak early signals before clinical diagnosis, using only information that would realistically be available in everyday self-reports.

Your model will be evaluated primarily on F2-score (recall matters most because missing a sick person is worse than a false alarm), PR-AUC (handling imbalanced data), and ROC-AUC. You must strictly avoid data leakage. No features that directly reveal the disease or rely on future information. The solution should be explainable, fair, cost-efficient, and built only with open-source tools. It should also be robust to noisy, incomplete, real-world self-reported inputs.

You will have to work with structured tabular data. However, in real-world deployment, user data is not always available in a clean table format. It may come from conversational text or voice input. You are not required to build a full NLP or speech processing pipeline. Instead, describe your strategy: how you would extract structured features from unstructured text or voice streams, what open-source frameworks or models you would use, what transformations you would apply, and how meaningful health signals would be derived.

We will also evaluate the originality of your approach and its scientific novelty. Extra credit will be given for non-obvious feature engineering, discovery of unexpected correlations, and creative modeling strategies that reveal meaningful early weak signals beyond standard baseline methods.

Think beyond a one-time prediction. The final solution should be realistic for integration into Hea’s core product, a proactive system that interacts with users regularly over time. Your model should be production minded: lightweight, interpretable, and suitable for continuous monitoring and early signal detection within an ongoing user communication flow.
