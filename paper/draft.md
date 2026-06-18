# Entanglement-Based QKD Network Security Monitor using XGBoost Anomaly Detection

**Author:** Saksham Gupta, B.E. EEE, BITS Pilani Hyderabad Campus
**Organisation:** DRDO · Solid State Physics Laboratory (SSPL), Delhi
**Mentor:** Dr. Lalit Kumar, Scientist F
**Date:** June 2026

---

## 1. Executive Summary

Quantum Key Distribution (QKD) promises unconditional communication security underpinned by the laws of quantum mechanics. However, practical implementations of QKD protocols, such as BBM92, are susceptible to hardware imperfections and sophisticated physical-layer attacks. This project successfully designs, evaluates, and versions an anomaly detection prototype for BBM92 QKD networks, built upon a robust XGBoost classifier. 

Through the simulation of an 86,400-second (24-hour) atmospheric channel incorporating realistic quantum-optical physics, four distinct attack vectors were modelled: Intercept-Resend, Detector Blinding, Man-in-the-Middle (MitM), and Blended Sub-Threshold attacks. By engineering a 24-dimensional feature space from rolling 30-second temporal windows, the model effectively captured the cross-correlated physical observables required to detect eavesdropping.

Rigorous 5-seed temporal holdout evaluations demonstrate that the proposed XGBoost model achieves a mean Recall of **0.971 ± 0.006** and a Precision of **0.980 ± 0.010**. Crucially, the system successfully identified **95.5% ± 0.9%** of Blended Sub-Threshold attacks—a regime where traditional Quantum Bit Error Rate (QBER) thresholds fundamentally fail. Model explainability is ensured via SHAP analysis, providing operational interpretability for defence analysts. The entire repository has been standardized, audited, and version-controlled to support statistical reproduction and structural readiness for academic publication.

---

## 2. Introduction

The advent of quantum computing threatens classical cryptographic primitives such as RSA and ECC, motivating the rapid development of quantum-safe architectures. Quantum Key Distribution (QKD) offers an information-theoretic secure method for cryptographic key exchange. While the theoretical security of QKD is well-established, practical deployments face immense challenges stemming from real-world device constraints, detector non-linearities, and environmental atmospheric noise. 

Adversaries, historically referred to as "Eve," exploit these practical loopholes. Defending against these attacks conventionally relies on static thresholding, primarily terminating key generation if the Quantum Bit Error Rate (QBER) or the Bell parameter ($S$) breaches a theoretical bound. However, sophisticated sub-threshold attacks can manipulate the quantum channel whilst remaining undetected by static metrics.

This project introduces a machine-learning-driven security paradigm. By treating QKD telemetry as a complex, multi-variate time-series, we leverage gradient-boosted decision trees (XGBoost) to capture subtle temporal anomalies and cross-channel correlations indicative of eavesdropping. This report documents the theoretical underpinnings, threat models, methodology, and final empirical results of the DRDO-SSPL QKD Anomaly Detection System.

---

## 3. Background

### 3.1 Quantum Key Distribution
Quantum Key Distribution (QKD) leverages quantum properties—specifically superposition and entanglement—to securely share cryptographic keys between two parties, typically designated Alice and Bob. Any attempt by a third party (Eve) to measure the quantum states inevitably disturbs them (due to the No-Cloning Theorem and wave-function collapse), introducing detectable errors into the communication channel.

### 3.2 BB84 Protocol
Proposed by Charles Bennett and Gilles Brassard in 1984, BB84 was the first QKD protocol. Alice prepares single photons in one of four polarization states spanning two mutually unbiased bases (e.g., rectilinear and diagonal). Bob measures the photons using randomly chosen bases. Through a classical communication channel, they perform basis reconciliation (sifting), error correction, and privacy amplification to distill a secure key. 

### 3.3 BBM92 Protocol
The BBM92 protocol, proposed by Bennett, Brassard, and Mermin in 1992, is an entanglement-based variant of BB84. Instead of Alice preparing specific states, a quantum source generates entangled photon pairs. One photon is sent to Alice and the other to Bob. Both measure their respective photons in randomly selected bases. When their bases match, their measurement outcomes are perfectly anti-correlated (or correlated, depending on the specific entangled state). BBM92 inherently protects against certain hardware trojans, as the source itself does not need to be trusted; security is verified continuously via Bell's Inequality tests (CHSH inequality).

---

## 4. Problem Statement

Current defence mechanisms for QKD rely heavily on evaluating the Quantum Bit Error Rate (QBER) against a strict threshold (typically 11% for BB84/BBM92). If the QBER is below 11%, the channel is deemed secure, and privacy amplification is expected to distill a secure key. 

However, modern physical attacks (such as sophisticated detector blinding or blended sub-threshold attacks) exploit hardware imperfections to control detector outputs without elevating the aggregate QBER above the 11% abort threshold. Consequently, static thresholding is insufficient for detecting intelligent adversaries. There is a critical need for a dynamic, multi-dimensional security monitor capable of identifying the subtle temporal and cross-correlated signatures of eavesdropping that evade simple QBER bounds.

---

## 5. Project Objectives

1. **Simulation:** Develop a physically grounded BBM92 atmospheric channel simulator capable of generating high-fidelity telemetry, inclusive of atmospheric scintillation, thermal gradients, and loss.
2. **Attack Modeling:** Implement four distinct, mathematically accurate quantum physical layer attacks.
3. **Detection:** Design a machine-learning anomaly detection pipeline capable of identifying attacks with >95% recall.
4. **Sub-Threshold Resilience:** Evaluate the ML model's ability to detect eavesdropping attempts designed to keep the QBER below 11%.
5. **Interpretability:** Implement SHAP (SHapley Additive exPlanations) to ensure the ML model's decisions are physically interpretable for defence analysts.
6. **Reproducibility:** Freeze the methodology to enable statistical reproduction over any simulated random seed.

---

## 6. System Architecture

The anomaly detection framework is composed of the following modules:

1. **Physical Simulation Core (`core/`)**: 
   - Generates the entangled photon pairs and models atmospheric attenuation and noise.
   - Evaluates the CHSH Bell parameter ($S$) and calculates secure key rates based on Toeplitz hashing constraints.
2. **Attack Engine (`core/attacks.py`)**: 
   - Dynamically injects physical layer attacks into the telemetry stream, altering observables like coincidence rates and visibility.
3. **Feature Engineering (`ml/features.py`)**: 
   - Applies rolling statistical windows (30 seconds) across raw telemetry (QBER, Bell S, Detection Rates) to generate a 24-dimensional feature vector, computing moving averages, standard deviations, and cross-correlations.
4. **Machine Learning Pipeline (`ml/train.py`, `ml/inference.py`)**: 
   - Utilizes XGBoost configured for imbalanced learning. The model is tuned for high recall (minimizing false negatives) using 5-fold expanding-window cross-validation.
5. **Real-time Dashboard (`app.py`)**: 
   - A Streamlit-based monitoring UI simulating real-time inference at 1 Hz, providing operators with actionable alerts and SHAP attributions.

---

## 7. Threat Model

We assume an adversary, Eve, possessing technological capabilities restricted only by the laws of quantum mechanics. Eve has:
- Full access to the quantum channel (fibre or free-space optical link) between Alice and Bob.
- The ability to intercept, measure, store (quantum memory), and resend photons.
- The ability to inject high-power classical light into the quantum channel to manipulate Single Photon Avalanche Diodes (SPADs).
- Full access to the classical public channel (read-only, authenticated).

The overarching goal of Eve is to maximize the intercepted secure key while maintaining stealth, effectively forcing the QBER and Bell S parameters to remain within conventionally "safe" bounds.

---

## 8. Attack Simulations

### 8.1 Intercept-Resend (IR)
Eve intercepts the entangled photon destined for Bob, measures it in a randomly chosen basis, and prepares a new photon based on her measurement outcome to send to Bob. 
**Physics Validation:** This attack fundamentally breaks entanglement. Sifted bits measured in mismatched bases yield a baseline error of 25%, causing the overall QBER to spike towards 25% and the Bell parameter ($S$) to drop to the classical limit of $\leq 2.0$.

### 8.2 Detector Blinding
Eve exploits the hardware vulnerabilities of Bob's SPADs. By shining a bright continuous-wave (CW) laser, she forces the SPADs out of Geiger mode into linear mode. The detectors are "blinded" to single photons. Eve then intercepts the legitimate photons, measures them, and sends bright classical pulses to precisely trigger Bob's detectors.
**Physics Validation:** Because Eve perfectly controls Bob's detectors, she forces the QBER strictly to 0.0%. However, this alters the fundamental ratio between raw detection rates and coincidence rates, creating a physical anomaly detectable by ML.

### 8.3 Man-in-the-Middle (MitM)
Eve acts as an active intermediary, effectively running two separate QKD sessions: one with Alice and one with Bob. 
**Physics Validation:** Due to the processing delays of interception, measurement, and classical negotiation, MitM introduces a temporal decoupling. While static QBER might stabilize, the cross-correlation between coincidence rates and QBER (`coincidence_qber_corr`) drops significantly.

### 8.4 Blended Sub-Threshold
Eve dynamically toggles between passive observation, partial Intercept-Resend, and Blinding in rapid, short bursts. She monitors the QBER in real-time, withdrawing her attack immediately when the QBER approaches the 11% threshold.
**Physics Validation:** This attack is mathematically designed to defeat simple thresholds. The QBER successfully stays below 11%, resulting in a false-negative for static systems. However, the rapid toggling induces high temporal variance in the Bell parameter and detection rates. 

It is important to emphasize that the blended sub-threshold attack is a synthetic adversarial scenario designed to stress-test the model. It represents a theoretical worst-case robustness test engineered specifically to evade the 11% QBER threshold, rather than a physically demonstrated attack currently documented in experimental quantum hacking literature.

---

## 9. Dataset Generation Pipeline

The canonical dataset (`telemetry_86400.parquet`) models a continuous 24-hour operation (86,400 seconds) of the BBM92 system at 1 Hz resolution. 

- **Normal Baseline:** Simulated with natural drift in visibility, channel loss (dB), and ambient noise. Mean QBER hovers at ~3.5%, and Bell $S$ at ~2.62.
- **Attack Injection:** Attacks are injected at randomized intervals and randomized intensities (Eve fractions ranging from 0.12 to 0.65).
- **Data Integrity:** The pipeline uses strict random seeding to ensure that the 5 distinct evaluation datasets represent physically unique 24-hour periods while maintaining identical macro-statistical distributions. 

*Dataset Distribution (Canonical Seed 42):*
- Total Rows: 86,400
- Normal Data: 86.3%
- Attack Data: 13.7%

---

## 10. Feature Engineering

Raw telemetry is insufficient for detecting sub-threshold blended attacks. A 30-second rolling window is applied to the raw stream to compute a 24-dimensional feature space.

**Key Features Include:**
- `qber_mean_30s`, `qber_std_30s`: Moving average and standard deviation of QBER.
- `bell_S_mean_30s`, `bell_S_std_30s`: Vital for detecting loss of entanglement.
- `detection_to_coinc_ratio`: Specifically engineered to flag Detector Blinding.
- `coincidence_qber_corr`: The Pearson correlation coefficient over 30s between coincidence rates and QBER, effectively highlighting temporal decoupling seen in MitM.
- `visibility_drop_rate`: First derivative of optical visibility, catching rapid hardware trojan insertions.

These features map the isolated variables into an interaction space where anomalous physical correlations are exposed.

---

## 11. Machine Learning Architecture

The final frozen system utilizes an **eXtreme Gradient Boosting (XGBoost)** classifier. 

**Hyperparameters:**
- `n_estimators`: 300
- `max_depth`: 6
- `learning_rate`: 0.05
- `tree_method`: `hist` (optimized for CPU processing constraints)
- `scale_pos_weight`: Dynamically calculated to handle the 86:14 class imbalance.

**Rationale for XGBoost over Deep Learning:**
1. Tabular data: Boosted trees inherently outperform neural networks on 24-dimensional structured telemetry.
2. Computational Efficiency: Training occurs in ~5 seconds on a standard CPU, negating the need for GPU hardware (critical for secure, air-gapped defence deployments).
3. Exact Interpretability: SHAP values can be computed exactly (TreeSHAP) without the approximations required for deep neural networks.

---

## 12. Evaluation Methodology

To reduce the risk of reported metrics being artificially inflated due to temporal data leakage, the repository enforces a rigorous **5-Seed Temporal Holdout Methodology**.

1. Five unique 24-hour datasets are generated using seeds `[42, 1337, 2024, 777, 101010]`.
2. For each dataset, the first 18 hours (64,800s) are used strictly for training and cross-validation (5-fold block CV).
3. The final 6 hours are strictly partitioned as a holdout test set.
4. Model performance is evaluated *only* on the 6-hour holdout set.
5. Metrics are averaged across the 5 seeds to provide a mean and non-zero standard deviation, supporting rigorous evaluation.

---

## 13. Results

The 5-seed holdout evaluation yielded the following canonical performance metrics compared against a traditional static physics threshold (QBER > 11%).

### Table 1: Baseline Comparison
| Method | Recall | Precision | F2 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| **QBER > 11% (Static)** | 0.180 ± 0.045 | 1.000 ± 0.000 | 0.215 ± 0.051 | N/A |
| **Proposed XGBoost** | **0.971 ± 0.006** | **0.980 ± 0.010** | **0.973 ± 0.005** | **0.992 ± 0.001** |

> *Figure 1 Placeholder: ROC Curve (`paper/figures/roc_curve.png`)*  
> *Figure 2 Placeholder: Precision-Recall Curve (`paper/figures/precision_recall_curve.png`)*

### Table 2: Per-Attack Type Recall
| Attack Type | Recall | Avg Count per Holdout |
| :--- | :--- | :--- |
| Intercept-Resend | 99.7% ± 0.0% | 356 |
| Detector Blinding | 99.6% ± 0.4% | 678 |
| MitM | 99.5% ± 0.1% | 232 |
| **Blended Sub-Threshold** | **95.5% ± 0.9%** | 2052 |

> *Figure 3 Placeholder: Confusion Matrix (`paper/figures/confusion_matrix.png`)*  
> *Figure 4 Placeholder: Recall by Attack Type Bar Chart (`paper/figures/per_attack_recall.png`)*

**Discussion:** While the static QBER threshold successfully catches brute-force Intercept-Resend attacks (resulting in perfect precision), it completely fails to catch Detector Blinding and Blended Sub-Threshold attacks, yielding a low overall recall of 18.0%. The XGBoost model successfully correlates secondary variables to achieve 97.1% recall. Most importantly, it catches 95.5% of the blended attacks explicitly configured to evade standard detection.

---

## 14. SHAP Explainability Analysis

To transition the ML model from a "black box" to an operational defence tool, TreeSHAP was integrated to calculate the marginal contribution of every feature to the final alert probability.

**Global SHAP Summary:**
The beeswarm summary plot reveals the overarching strategy learned by the XGBoost model:
1. `detection_to_coinc_ratio`: High values (red) dramatically increase the probability of an attack, acting as the primary flag for Detector Blinding.
2. `coincidence_qber_corr`: Low values (blue) push the prediction toward an attack state, signifying the temporal decoupling introduced by MitM processing delays.
3. `qber`: Acts as a heavy boundary feature; even moderate increases strongly push the probability toward an attack.

> *Figure 5 Placeholder: SHAP Beeswarm Summary Plot (`paper/figures/shap_summary.png`)*

The system leverages `ml/explain.py` to calculate these SHAP attributions in under 200ms per alert, allowing live rendering in the Streamlit dashboard.

---

## 15. Feature Ablation Study

To evaluate the relative importance of cross-correlated physical observables, a feature ablation study was conducted. The baseline recall (97.1%) was compared against restricted feature sets.

### Ablation Results (Impact on Recall)
- **Baseline (All 24 features):** 97.1%
- **QBER only (mean, std, rate):** 72.0%
- **Bell S only:** 65.0%
- **Temporal features only (drifts, delays):** 51.0%
- **QBER + Coincidence:** 85.0%
- **QBER + Bell S:** 88.0%

> *Figure 6 Placeholder: Feature Group Ablation Impact Chart (`paper/figures/ablation_chart.png`)*

**Conclusion:** No single isolated metric is sufficient. Relying solely on QBER results in a 25.1% drop in recall. The full combinatorial feature space significantly improves the model's ability to identify the statistical footprints of blended quantum attacks.

---

## 16. Physics Validation

A thorough physics audit was conducted to ensure the training data mathematically respects quantum mechanical limits:
- **No-Cloning & CHSH Bound:** IR attacks strictly limit Bell $S \leq 2.0$. This is consistently reflected in the generated datasets.
- **Geiger-Mode Saturation:** The Detector Blinding simulation correctly manipulates the detection rates to match the injected classical CW laser power, preserving the physical constraints of an APD detector.
- **Privacy Amplification:** Secure Key Rate calculations in `core/privacy_amplification.py` strictly enforce the Hoeffding bound and theoretical Toeplitz hashing limits based on the instantaneous QBER.

---

## 17. Limitations of the Simulation Environment

The findings in this report must be interpreted strictly within the scope of the simulation environment. All attacks and normal telemetry were generated programmatically; no physical hardware validation has been performed. The APD (Avalanche Photodiode) physics and atmospheric channels are simplified models. For example, the detector blinding simulation models a scalar multiplier on detection rates but does not capture the complex temporal jitter or pulse-shape deformations inherent to true physical APD blinding. Consequently, while the machine learning model excels at recognizing these simulated mathematical anomalies, its zero-shot transferability to real-world hardware trojans remains unproven.

Furthermore, offline feature engineering via a 30-second rolling window requires caching state in memory. In a high-speed FPGA deployment, maintaining 24-dimensional sliding windows at MHz telemetry rates requires optimized HDL programming not covered by this software demonstration.

---

## 18. Future Work

1. **Hardware-in-the-Loop Validation:** Deploy the trained `models/xgb_model.json` to an edge device connected to a physical DRDO-SSPL free-space optical (FSO) testing rig.
2. **Deep Autoencoders:** Investigate unsupervised deep autoencoders for zero-day attack detection. While XGBoost excels at identifying known attack distributions, unsupervised learning may flag entirely novel attack physics without prior training data.
3. **Continuous Learning:** Implement a drifting-window retraining pipeline to handle long-term seasonal atmospheric variations in FSO deployments.

---

## 19. Conclusion

The DRDO-SSPL BBM92 QKD Anomaly Detection System proposes a machine-learning-driven countermeasure to advanced quantum eavesdropping. By moving beyond static thresholding and analyzing 24 dimensions of temporal physical telemetry, the XGBoost ensemble achieves a validated recall of 97.1% ± 0.6% on strictly held-out data. 

Critically, the prototype demonstrates capability in identifying simulated mathematically blended sub-threshold attacks that evade traditional cryptographic abort protocols. With SHAP interpretability and a version-controlled codebase, this project provides a foundational software framework for further research in securing quantum networks.

---

## 20. References

1. Bennett, C. H., Brassard, G., & Mermin, N. D. (1992). *Quantum cryptography without Bell’s theorem*. Physical review letters, 68(5), 557.
2. Bennett, C. H., & Brassard, G. (1984). *Quantum cryptography: Public key distribution and coin tossing*. In Proceedings of IEEE International Conference on Computers, Systems and Signal Processing (pp. 175-179).
3. Lydersen, L., et al. (2010). *Hacking commercial quantum cryptography systems by tailored bright illumination*. Nature photonics, 4(10), 686-689.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).
5. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in neural information processing systems, 30.

---
*End of Report*
