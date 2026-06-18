| Model | Recall | Precision | F2 Score |
|-------|--------|-----------|----------|
| QBER Threshold (> 0.11) | 0.173 ± 0.024 | 1.000 ± 0.000 | 0.207 ± 0.028 |
| Logistic Regression | 0.975 ± 0.004 | 0.957 ± 0.007 | 0.971 ± 0.004 |
| Random Forest | 0.962 ± 0.008 | 0.976 ± 0.007 | 0.965 ± 0.008 |
| **XGBoost** | **0.969 ± 0.007** | **0.976 ± 0.010** | **0.970 ± 0.007** |

*Note: Results averaged over 5 holdout splits. Logistic Regression incorporates standard scaling. XGBoost is selected for deployment due to high precision (0.976) and TreeSHAP explainability.*
