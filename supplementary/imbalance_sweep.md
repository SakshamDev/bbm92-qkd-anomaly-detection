| Target Ratio | Actual Attack Ratio | F2 Score | Recall |
|---|---|---|---|
| 0.05 | 3.79% | 0.926 | 0.917 |
| 0.10 | 8.32% | 0.960 | 0.958 |
| 0.15 | 12.57% | 0.966 | 0.963 |
| 0.20 | 16.63% | 0.963 | 0.961 |
| 0.30 | 23.79% | 0.957 | 0.953 |

*Conclusion:* Model is highly robust to severe class imbalances typical in cybersecurity. Even when attack traffic represents only 3.79% of the test set, the XGBoost `scale_pos_weight` adjustment ensures an F2 score of >0.92.
