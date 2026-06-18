# BBM92 QKD Anomaly Detection — Canonical Results

> **Freeze Constraint:** These metrics represent the canonical performance of the finalised XGBoost model across a rigorous 5-seed temporal holdout evaluation.

## Global Metrics (Holdout Set)
* **Recall:** 0.971 ± 0.006
* **Precision:** 0.980 ± 0.010
* **F2 Score:** 0.973 ± 0.005
* **ROC AUC:** 0.992 ± 0.001

## Per-Attack Type Recall
1. **Intercept-Resend:** 99.7% ± 0.0%
2. **Detector Blinding:** 99.6% ± 0.4%
3. **Man-in-the-Middle (MitM):** 99.5% ± 0.1%
4. **Blended Sub-Threshold:** 95.5% ± 0.9%

## Top SHAP Features
1. `detection_to_coinc_ratio`: Captures blinding and intercept-resend efficiency discrepancies.
2. `coincidence_qber_corr`: Captures temporal decoupling typical of MitM attacks.
3. `qber`: Absolute error bounding.

## Ablation Results (Impact on Recall)
* **Baseline (Full features):** 97.1%
* **QBER only:** 72.0%
* **Bell S only:** 65.0%
* **Temporal features only:** 51.0%
* **QBER + Coincidence:** 85.0%
* **QBER + Bell S:** 88.0%

*Conclusion: Physical observable cross-correlations are strictly necessary for robust sub-threshold detection.*
