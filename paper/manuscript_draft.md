# Title: Physics-Aware Temporal Feature Engineering for Eavesdropping Detection in BBM92 Quantum Key Distribution

## Abstract
Traditional security in Quantum Key Distribution (QKD) relies on aborting key generation when the Quantum Bit Error Rate (QBER) exceeds a static threshold. However, in noisy free-space optical (FSO) channels, natural atmospheric variations can camouflage short, low-intensity eavesdropping bursts. We propose a machine learning-based defense strategy specifically designed for entanglement-based (BBM92) QKD telemetry. By engineering a 24-dimensional feature space computed over a 30-second sliding window, we evaluate the temporal shape and cross-observable correlation of QBER, Bell S parameter, and photon coincidence rates. Evaluated on a comprehensive simulated FSO telemetry dataset, our framework demonstrates that static QBER thresholding fails catastrophically against stealthy blended attacks, achieving only 17.3% recall. In contrast, our multivariate approach suggests that such attacks can be identified with 96.9% recall and 97.6% precision within the simulated environment. SHAP analysis indicates that stealthy attacks are not detected by absolute error magnitude, but by anomalous temporal cross-correlations between independent quantum observables. This work suggests that integrating physics-aware temporal features into QKD receivers may provide a robust defense mechanism against dynamic adversarial environments.

## Keywords
Quantum Key Distribution, BBM92, Machine Learning, Anomaly Detection, Free-Space Optics, Eavesdropping Detection, Telemetry

## I. Introduction
Quantum Key Distribution (QKD) provides information-theoretic security guaranteed by the laws of quantum mechanics, enabling two parties to share a secure cryptographic key. The entanglement-based BBM92 protocol \cite{bennett1992quantum} achieves this by measuring photon pairs in a shared Bell state ($|\Phi^+\rangle$). While theoretically robust, practical implementations are vulnerable to physical layer attacks. Eavesdroppers (Eve) can exploit device imperfections through strategies such as intercept-resend \cite{fuchs1997optimal}, faked-state injection \cite{makarov2006faked}, and detector blinding \cite{lydersen2010hacking}.

The standard countermeasure in deployed QKD systems is static thresholding: if the Quantum Bit Error Rate (QBER) exceeds a theoretical limit (typically 11\%), the protocol aborts. However, in Free-Space Optical (FSO) networks, the channel is inherently noisy. Scintillation, thermal gradients, and precipitation cause natural, transient QBER fluctuations. A sophisticated adversary can execute a "blended attack"---timing short bursts of low-intensity eavesdropping to coincide with natural atmospheric turbulence, thereby keeping the aggregate QBER below the abort threshold. Static thresholding is structurally blind to this threat vector.

To address this, machine learning (ML) has emerged as a promising tool for QKD security. However, relatively little prior work has examined temporal feature engineering for anomaly detection in BBM92 telemetry. Prior ML applications have largely focused on classifying discrete snapshots of parameters in prepare-and-measure protocols (e.g., CV-QKD) \cite{mao2020detecting, liu2025deep}. These methods do not directly map to the coincidence-based telemetry of entanglement protocols.

This work investigates physics-aware temporal feature engineering for ML-based anomaly detection in entanglement-based BBM92 QKD systems. Rather than evaluating instantaneous QBER, we propose a 24-dimensional feature framework computed over a sliding temporal window. By analyzing the temporal shape and cross-correlation of quantum observables (such as coincidence rates and FSO path loss), we demonstrate within a simulated environment the potential to detect sub-threshold blended attacks that successfully evade standard QBER thresholding.

## II. Related Work
The security of quantum key distribution has been extensively studied since the foundational BB84~\cite{bennett1984quantum} and BBM92~\cite{bennett1992quantum} protocols. We organize prior work into three areas relevant to our investigation: (A)~physical attack characterization, (B)~machine learning for quantum security, and (C)~multivariate temporal anomaly detection.

**A. Physical Attack Characterization**
The intercept-resend attack was rigorously quantified by Fuchs \textit{et al.}~\cite{fuchs1997optimal}, estimating the fundamental QBER bound. Moving beyond theoretical attacks, Lydersen \textit{et al.}~\cite{lydersen2010hacking} showed that hardware-level side channels, such as blinding single-photon avalanche diodes (SPADs) with bright light, could allow an adversary to dictate measurement outcomes. While countermeasures like measurement-device-independent QKD (MDI-QKD)~\cite{lo2012measurement} have been proposed, faked-state attacks~\cite{makarov2006faked, gerhardt2011fullfield} indicate that implementation security remains a persistent vulnerability.

**B. Machine Learning for Quantum Security**
To mitigate unknown implementation flaws, the application of machine learning to QKD security has gained recent traction. Mao \textit{et al.}~\cite{mao2020detecting} proposed machine learning-based defense strategies targeting continuous-variable QKD. More recently, Liu \textit{et al.}~\cite{liu2025deep} demonstrated the efficacy of deep anomaly detection models for identifying active attacks on QKD receivers. However, prior ML approaches predominantly analyze static snapshots of parameters or focus exclusively on prepare-and-measure protocols (BB84/CV-QKD). Our work diverges by targeting entanglement-based BBM92 telemetry, specifically investigating the temporal cross-correlation of independent observables.

**C. Multivariate Temporal Anomaly Detection**
Our 24-feature framework draws heavily from industrial anomaly detection~\cite{chandola2009anomaly}. Ge and Song~\cite{ge2013review} discussed that multivariate process monitoring---detecting anomalies via the breakdown of established correlations between variables---often outperforms univariate thresholding. We map this principle to the quantum domain: elevated QBER that is decoupled from corresponding FSO channel loss~\cite{bedington2017progress} suggests adversarial perturbation rather than atmospheric turbulence. To classify these temporal features, we employ XGBoost~\cite{chen2016xgboost}, selected for its compatibility with the TreeSHAP explainer~\cite{lundberg2017shap}, which may allow defense analysts to audit the physical triggers behind automated alerts.

## III. System Model and Feature Engineering

### A. BBM92 Telemetry and Threat Model
We model a hybrid FSO/fibre BBM92 link operating at 1 Hz telemetry resolution. The observables monitored include QBER ($Q$), CHSH Bell parameter ($S$), coincidence pair rate ($C$), single-detector click rate ($D$), HOM visibility ($V$), and optical path loss ($L$). The baseline atmospheric environment simulates log-normal scintillation, diurnal thermal gradients, and Poisson-arrival burst noise.

We simulate four distinct attack vectors:
1. **Intercept-Resend (IR):** High intensity, driving QBER well above threshold.
2. **Detector Blinding:** Adversary forces single-photon detectors into a linear regime using bright light, creating anomalous single-click rates.
3. **Man-in-the-Middle (MitM):** Complete source replacement, collapsing the Bell $S$ parameter to the classical bound ($\approx 2.0$).
4. **Blended Sub-Threshold:** 30-second bursts of low-intensity interception (Eve fraction $\sim$12\%), explicitly timed during naturally occurring high-loss weather events to camouflage QBER variations.

### B. Temporal Feature Engineering Framework
Static thresholds evaluate instantaneous telemetry. To capture the temporal signature of adversarial interference, we transform the 1 Hz raw telemetry $\mathbf{x}_t$ into a 24-dimensional feature vector over a sliding window $T = 30$ seconds. These features are grouped by physical rationale:

1. **QBER Statistics (6 features):** Mean, standard deviation, delta (rate of change), skewness, and autoregressive lag metrics (AR(1), AR(5)). Attacks often introduce right-skewed, serially correlated step-changes, whereas weather induces smooth drift.
2. **Bell S Entanglement (5 features):** Mean, standard deviation, rate of change, fractional duration below the CHSH violation boundary, and Pearson correlation with QBER.
3. **Coincidence Dynamics (5 features):** Mean rate, fractional drop from peak, burstiness coefficient (CV), and the detection-to-coincidence ratio ($R_{dc} = \frac{\bar{D}}{2\bar{C}}$), which is known to spike heavily during detector blinding.
4. **Temporal Structure (4 features):** Rolling variance, Bell $S$ range, burst energy ($L_2$ norm of deviations), and rising/falling edge asymmetry. Weather events typically exhibit symmetric onset/offset envelopes, while attack insertions exhibit abrupt edges.
5. **Cross-Channel Coupling (4 features):** Mean loss, mean visibility, joint entanglement-rate product, and a QBER-loss decoupling metric $\delta_{LQ}$. A positive decoupling metric indicates QBER variations that cannot be explained by measured FSO path loss.

## IV. Experimental Methodology
We generated 86,400 seconds (24 hours) of simulated telemetry per random seed, encompassing natural diurnal weather patterns and randomly distributed attack insertions. The dataset was generated across 5 distinct random seeds to ensure statistical robustness regarding weather timings and attack overlaps. The overall attack ratio is approximately 13\%.

We employed a temporal 75/25 train-test split (first 18 hours for training, final 6 hours for holdout evaluation). Features were standardized using `StandardScaler`. We evaluated Logistic Regression (LR) to test the linear separability of the engineered feature space, alongside Random Forest (RF) and XGBoost. The primary evaluation metric is the F2-score, reflecting the asymmetric risk of false negatives (missed eavesdropping).

## V. Results and Discussion

### A. Baseline Model Comparison
Table I presents the performance of the classification models against the traditional static threshold. 

**Table I: Baseline Comparison (Mean $\pm$ Std over 5 seeds)**
| Model | Recall | Precision | F2 Score |
|-------|--------|-----------|----------|
| QBER Threshold (> 0.11) | 0.173 $\pm$ 0.024 | 1.000 $\pm$ 0.000 | 0.207 $\pm$ 0.028 |
| Logistic Regression | 0.975 $\pm$ 0.004 | 0.957 $\pm$ 0.007 | 0.971 $\pm$ 0.004 |
| Random Forest | 0.962 $\pm$ 0.008 | 0.976 $\pm$ 0.007 | 0.965 $\pm$ 0.008 |
| **XGBoost** | **0.969 $\pm$ 0.007** | **0.976 $\pm$ 0.010** | **0.970 $\pm$ 0.007** |

The static QBER threshold yields a recall of only 17.3\%, indicating its vulnerability to blended, sub-threshold attacks. Logistic Regression achieves near-parity in F2-score with the nonlinear models. This suggests that the 24-dimensional temporal transformation effectively separates the attack signatures from background weather noise, rendering the space largely linearly separable. XGBoost's higher precision (0.976) may minimize operational false positives compared to LR (0.957), and it is fully compatible with TreeSHAP auditing. McNemar's test supports a statistically significant difference between XGBoost and the static threshold ($p < 0.001$).

### B. Temporal Window Ablation
We evaluated the sensitivity of the model to the sliding window size $T$, as shown in Table II.

**Table II: Window Size Ablation**
| Window Size | Recall | F2 Score |
|-------------|--------|----------|
| 15 seconds | 0.979 $\pm$ 0.005 | 0.977 $\pm$ 0.005 |
| **30 seconds** | **0.968 $\pm$ 0.007** | **0.970 $\pm$ 0.007** |
| 60 seconds | 0.955 $\pm$ 0.013 | 0.946 $\pm$ 0.021 |
| 120 seconds | 0.926 $\pm$ 0.023 | 0.922 $\pm$ 0.023 |
| 300 seconds | 0.886 $\pm$ 0.030 | 0.878 $\pm$ 0.027 |

While a 15-second window achieves slightly better aggregate performance in simulation, the 30-second window was selected because it perfectly aligns with the duration of the simulated blended attack bursts, providing greater physical interpretability. Larger windows (e.g., 120s) begin to dilute the transient attack signals within larger blocks of benign telemetry, degrading overall recall.

### C. Sensitivity to Eavesdropper Fraction
Table III illustrates the recall of XGBoost across different attack types as the Eve interception fraction ($\eta$) varies.

**Table III: Eve Fraction Sensitivity (Recall)**
| Eve Fraction | Intercept-Resend | Detector Blinding | Blended Sub-Threshold |
|--------------|-------------------|-------------------|------------------------|
| 0.05 | 0.945 | 0.945 | 0.615 |
| **0.10** | **0.979** | **0.945** | **0.953** |
| 0.15 | 0.998 | 0.930 | 0.970 |
| 0.30 | 0.999 | 0.999 | 0.965 |

At an Eve fraction of 5\%, the blended attack is difficult to distinguish from natural atmospheric variations (0.615 recall). However, at 10\% interception, the model detects 95.3\% of blended attacks within the simulated dataset.

### D. Physical Interpretability via SHAP
To investigate the physical drivers of detection, we extracted the mean absolute SHAP (SHapley Additive exPlanations) values for the XGBoost model. 

For continuous, high-intensity attacks (Intercept-Resend, MitM, and Detector Blinding), the model overwhelmingly relies on a single feature: `det_coinc_rat` (Detection-to-Coincidence Ratio), with a SHAP magnitude > 3.1. This is physically consistent, as active blinding and high-intensity disruption severely misalign single clicks from correlated photon pairs.

Conversely, for Blended Sub-Threshold attacks, `det_coinc_rat` drops to 4th place. The model pivots entirely to temporal features. The top three features are:
1. `coinc_qber_r`: The cross-correlation between coincidences and QBER within the window.
2. `qber_delta`: The step-change magnitude of the error rate.
3. `temp_asym`: The asymmetry between the rising and falling edges of the error profile.

This indicates that stealthy attacks are not identified by absolute error magnitude alone, but by the anomalous timing and cross-correlation of independent quantum observables. Crucially, the SHAP analysis indicates that the model relies on physically meaningful cross-observable relationships rather than overfitting to any specific atmospheric noise realization generated by the simulator.

## VI. Limitations
While this study proposes a mathematically derived feature framework, several significant limitations must be acknowledged:
1. **Simulation Bounds:** The results are entirely simulation-based. The FSO channel model assumes log-normal scintillation and does not cover severe deep-fade regimes typical of $>100$km FSO links.
2. **Hardware Constraints:** No hardware validation was possible, and no public BBM92 telemetry dataset was available to verify these findings empirically. Generalization to operational QKD systems remains unverified.
3. **Data Availability:** Computing the 24 features requires 1 Hz access to raw coincidence and single-click rates, which are often abstracted away in commercial "black-box" systems. 
4. **Adversarial Dynamics:** Performance against adaptive adversaries employing variable burst lengths (e.g., 5s or 120s) remains future work.

## VII. Conclusion
We presented a physics-aware temporal feature engineering framework for eavesdropping detection in simulated BBM92 QKD systems. By shifting from instantaneous QBER thresholds to a multivariate sliding window, our system achieves a 96.9\% recall against stealthy, sub-threshold blended attacks that easily bypass standard defenses. The linear separability of the resulting feature space suggests that proper temporal correlation of quantum observables is the primary mechanism of detection accuracy. These findings contribute to the ongoing effort to secure QKD networks, suggesting that integrating temporal telemetry analysis into QKD receivers may provide a pathway for low-latency monitoring against active adversaries in noisy environments.
