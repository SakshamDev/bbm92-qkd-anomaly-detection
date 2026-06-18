| Rank | Intercept-Resend | Detector Blinding | MitM | Blended Sub-Threshold |
|------|-------------------|-------------------|------|------------------------|
| 1 | `det_coinc_rat` (3.90) | `det_coinc_rat` (3.27) | `det_coinc_rat` (3.17) | `coinc_qber_r` (2.08) |
| 2 | `coinc_qber_r` (1.29) | `coinc_qber_r` (1.40) | `coinc_qber_r` (1.32) | `qber_delta` (1.11) |
| 3 | `qber_delta` (0.52) | `qber_delta` (0.66) | `qber_delta` (0.67) | `temp_asym` (0.97) |
| 4 | `qber_std` (0.28) | `temp_asym` (0.31) | `qber_std` (0.40) | `det_coinc_rat` (0.80) |
| 5 | `temp_asym` (0.27) | `qber_std` (0.20) | `temp_asym` (0.36) | `qber_std` (0.52) |

*Note: SHAP values represent mean absolute attribution, averaged over 5 random seeds. For continuous attacks (IR, DB, MitM), the model relies overwhelmingly on the detection-to-coincidence ratio. For Blended Sub-Threshold attacks, the model pivots entirely to temporal features, primarily the coincidence-QBER cross-correlation.*
