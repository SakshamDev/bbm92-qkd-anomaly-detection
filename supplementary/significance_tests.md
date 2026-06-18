### McNemar's Test (Binary F2)
XGBoost vs all baseline models evaluated over $N = 21,600$ holdout samples:
- vs Threshold: $p < 0.001$
- vs Logistic Regression: $p < 0.001$
- vs Isolation Forest: $p < 0.001$
- vs Random Forest: $p < 0.001$

### Wilcoxon Signed-Rank Test (Continuous Error)
Comparing absolute predicted probability errors of XGBoost vs HistGradientBoosting (LightGBM surrogate):
- XGBoost Mean Absolute Error: 0.0177
- LightGBM Mean Absolute Error: 0.0236
- Wilcoxon Statistic: 25,478,129.0
- $p \approx 0.0$

*Conclusion:* A Wilcoxon signed-rank test confirms XGBoost produces significantly lower absolute prediction error than LightGBM ($p < 10^{-4}$), meaning its confidence margins are statistically superior despite the binary F2 scores being close.
