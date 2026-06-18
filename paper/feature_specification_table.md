# 24-Feature Mathematical Specification Table

> For inclusion in IEEE INDISCON 2026 paper, Section IV-A: Feature Engineering Framework.
> All features are computed over a sliding window $W$ of $T = 30$ seconds at 1 Hz resolution.
> Let $\mathbf{x}_t$ denote the raw telemetry vector at second $t$.

---

## LaTeX Table (copy-paste into paper)

```latex
\begin{table*}[!t]
\caption{Complete Mathematical Specification of the 24-Dimensional Physics-Aware Feature Vector}
\label{tab:features}
\centering
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{|c|l|l|l|l|}
\hline
\textbf{\#} & \textbf{Feature Name} & \textbf{Mathematical Definition} & \textbf{Physical Rationale} & \textbf{Group} \\
\hline\hline
% ── QBER Features (6) ──
1  & \texttt{qber\_mean}       & $\bar{Q} = \frac{1}{T}\sum_{i=t-T}^{t-1} Q_i$                                      & Baseline error rate; rises under IR/MitM          & QBER \\
2  & \texttt{qber\_std}        & $\sigma_Q = \sqrt{\frac{1}{T}\sum (Q_i - \bar{Q})^2}$                                & Volatility; attacks cause unstable QBER            & QBER \\
3  & \texttt{qber\_delta}      & $\Delta Q = Q_{t-1} - Q_{t-T}$                                                       & Rate of change; step-change vs. drift              & QBER \\
4  & \texttt{qber\_skewness}   & $\gamma_Q = \frac{\frac{1}{T}\sum(Q_i - \bar{Q})^3}{\sigma_Q^3 + \epsilon}$          & Right-skewed $\Rightarrow$ attack spikes           & QBER \\
5  & \texttt{qber\_ac\_lag1}   & $\rho_Q(1) = \frac{\text{Cov}(Q_i, Q_{i+1})}{\text{Var}(Q)}$                         & AR(1); high for sustained attacks, low for weather & QBER \\
6  & \texttt{qber\_ac\_lag5}   & $\rho_Q(5) = \frac{\text{Cov}(Q_i, Q_{i+5})}{\text{Var}(Q)}$                         & Longer temporal memory of anomaly persistence      & QBER \\
\hline
% ── Bell S Features (5) ──
7  & \texttt{bell\_S\_mean}    & $\bar{S} = \frac{1}{T}\sum S_i$                                                      & Entanglement health; $S < 2$ violates CHSH         & Bell \\
8  & \texttt{bell\_S\_std}     & $\sigma_S = \sqrt{\frac{1}{T}\sum (S_i - \bar{S})^2}$                                & Entanglement stability within window               & Bell \\
9  & \texttt{bell\_S\_delta}   & $\Delta S = S_{t-1} - S_{t-T}$                                                       & Rate of entanglement degradation                   & Bell \\
10 & \texttt{bell\_S\_sub2414} & $f_{<2.414} = \frac{1}{T}\sum \mathbb{1}[S_i < 2\sqrt{2} \times 0.854]$              & Fraction of window below $0.854 \times S_{\max}$   & Bell \\
11 & \texttt{bell\_S\_pears}   & $r(S, Q) = \frac{\sum(S_i - \bar{S})(Q_i - \bar{Q})}{T \cdot \sigma_S \sigma_Q}$    & $S$-$Q$ coupling; negative in normal, flips under attack & Bell \\
\hline
% ── Coincidence / Detection Features (5) ──
12 & \texttt{coinc\_mean}      & $\bar{C} = \frac{1}{T}\sum C_i$                                                      & Mean pair rate; drops under PNS/MitM               & Coinc \\
13 & \texttt{coinc\_drop}      & $d_C = \frac{C_{\max} - \bar{C}}{C_{\max} + \epsilon}$                               & Fractional drop from peak; PNS signature           & Coinc \\
14 & \texttt{coinc\_cv}        & $\text{CV}_C = \frac{\sigma_C}{\bar{C} + \epsilon}$                                  & Burstiness coefficient; high during blended bursts & Coinc \\
15 & \texttt{coinc\_qber\_r}   & $r(C, Q) = \frac{\sum(C_i - \bar{C})(Q_i - \bar{Q})}{T \cdot \sigma_C \sigma_Q}$    & Cross-correlation; anomalous under detector blind  & Coinc \\
16 & \texttt{det\_coinc\_rat}  & $R_{dc} = \frac{\bar{D}}{2\bar{C} + \epsilon}$                                       & Singles/coincidences ratio; $\gg 1$ under blinding & Coinc \\
\hline
% ── Temporal Structure Features (4) ──
17 & \texttt{qber\_roll\_var}  & $\text{Var}(Q) = \frac{1}{T}\sum (Q_i - \bar{Q})^2$                                  & Smooth drift (weather) vs. step-change (attack)    & Temp \\
18 & \texttt{bell\_S\_range}   & $R_S = S_{\max} - S_{\min}$                                                          & Entanglement stability range in window             & Temp \\
19 & \texttt{burst\_energy}    & $E_b = \sqrt{\frac{1}{T}\sum (Q_i - \bar{Q})^2}$                                    & $L_2$ norm of QBER deviations; high during bursts  & Temp \\
20 & \texttt{temp\_asym}       & $A = \frac{1}{T/2}\sum_{i=T/2}^{T-1} Q_i - \frac{1}{T/2}\sum_{i=0}^{T/2-1} Q_i$    & Rising vs. falling edge asymmetry                  & Temp \\
\hline
% ── Cross-Channel Features (4) ──
21 & \texttt{loss\_mean}       & $\bar{L} = \frac{1}{T}\sum L_i \text{ (dB)}$                                         & Mean FSO path loss; high $\Rightarrow$ weather     & Cross \\
22 & \texttt{vis\_mean}        & $\bar{V} = \frac{1}{T}\sum V_i$                                                      & HOM visibility; $V \approx 1 - 2Q$ under normal    & Cross \\
23 & \texttt{loss\_qber\_dec}  & $\delta_{LQ} = \bar{Q} - (0.02 + 0.05(1 - 10^{-\bar{L}/10}))$                       & Positive $\Rightarrow$ QBER unexplained by loss    & Cross \\
24 & \texttt{S\_coinc\_prod}   & $P_{SC} = \frac{\bar{S} \times \bar{C}}{10000}$                                      & Joint entanglement-rate health metric              & Cross \\
\hline
\end{tabular}
\vspace{1mm}
\begin{flushleft}
\small{Window size $T = 30$\,s. $\epsilon = 10^{-9}$ prevents division by zero. $Q$: QBER, $S$: CHSH Bell parameter, $C$: coincidence rate (pairs/s), $D$: single-detector rate, $V$: HOM visibility, $L$: channel loss (dB). All features computed at 1\,Hz sliding resolution. QBER autocorrelation returns 0 for constant signals (zero variance).}
\end{flushleft}
\end{table*}
```

---

## Feature Group Summary

### Group 1: QBER Statistical Features (Features 1–6)

These six features capture the complete statistical profile of the Quantum Bit Error Rate within each 30-second window. The mean and standard deviation provide first-order anomaly detection, while the delta (rate of change) discriminates between the gradual onset of weather-induced degradation and the sharp step-changes characteristic of attack insertion. Skewness captures the asymmetric tail structure of attack-induced QBER spikes, and the two autocorrelation features (lag-1 and lag-5) exploit the key insight that attack-induced QBER elevations exhibit high serial correlation (the attack persists), whereas atmospheric scintillation produces uncorrelated noise.

**Physical basis:** Under an Intercept-Resend attack with Eve fraction $\eta$, the QBER increases by $\Delta Q = 0.25\eta$ (from Eve's 50% basis mismatch causing 50% errors on intercepted photons).

### Group 2: Bell S Parameter Features (Features 7–11)

The CHSH Bell parameter $S$ is the gold-standard test for genuine quantum entanglement. For the $|\Phi^+\rangle$ Bell state used in BBM92, the theoretical maximum is $S = 2\sqrt{2} \approx 2.828$. Any eavesdropper who breaks entanglement (IR, MitM) will cause $S$ to collapse toward the classical bound of 2.0.

**Physical basis:** $S = 2\sqrt{2} \cos(2 \arcsin(\sqrt{Q}))$ for the $|\Phi^+\rangle$ state. The Pearson correlation $r(S, Q)$ is normally negative (higher QBER → lower S), but under detector blinding this relationship can invert because Eve's classical trigger pulses produce anomalous $S$-$Q$ coupling patterns.

### Group 3: Coincidence / Detection Features (Features 12–16)

These features target Photon Number Splitting (PNS) and detector blinding signatures. The detection-to-coincidence ratio (Feature 16) is the unique discriminator for detector blinding: Eve's bright blinding laser causes excess single-detector clicks without corresponding coincidences, pushing $R_{dc} \gg 1$.

**Physical basis:** In a healthy BBM92 channel, the detection rate $D \approx 2C + D_{\text{dark}}$ where $D_{\text{dark}} \sim 200$/s. Under detector blinding, $D$ spikes anomalously while $C$ drops.

### Group 4: Temporal Structure Features (Features 17–20)

These four features capture the *temporal shape* of anomalies within each window, which is the core innovation of this feature framework. Burst energy (Feature 19) measures the $L_2$ norm of QBER deviations—high during the short, intense bursts of a Blended Sub-Threshold attack. Temporal asymmetry (Feature 20) discriminates between the symmetric onset/offset profile of weather events (which ramp up and down smoothly) and the abrupt edges of attack insertion.

**Physical basis:** Weather-induced QBER degradation follows smooth sinusoidal envelopes (precipitation model). Attack insertion produces rectangular step-functions with sharp rising and falling edges, creating asymmetric temporal profiles.

### Group 5: Cross-Channel Features (Features 21–24)

The most physically motivated group. Feature 23 (`loss_qber_decoupling`) implements the key anomaly detection principle: if QBER is elevated but channel loss is normal, the degradation *cannot be explained by atmospheric effects* and must originate from an eavesdropper. This is computed as:

$$\delta_{LQ} = \bar{Q} - \left(0.02 + 0.05 \times (1 - 10^{-\bar{L}/10})\right)$$

where $0.02$ is the baseline QBER from detector dark counts and $0.05 \times (1 - 10^{-\bar{L}/10})$ is the expected QBER contribution from channel loss (Beer-Lambert model). A positive $\delta_{LQ}$ indicates an anomaly that cannot be attributed to the physical channel.

---

## SHAP-Validated Feature Importance (per Attack Type)

From the TreeSHAP analysis on the holdout set (Seed 42):

| Rank | Intercept-Resend | Detector Blinding | MitM | Blended Sub-Threshold |
|------|-------------------|-------------------|------|------------------------|
| 1 | `det_coinc_rat` (3.90) | `det_coinc_rat` (3.27) | `det_coinc_rat` (3.17) | `coinc_qber_r` (2.08) |
| 2 | `coinc_qber_r` (1.29) | `coinc_qber_r` (1.40) | `coinc_qber_r` (1.32) | `qber_delta` (1.11) |
| 3 | `qber_delta` (0.52) | `qber_delta` (0.66) | `qber_delta` (0.67) | `temp_asym` (0.97) |
| 4 | `qber_std` (0.28) | `temp_asym` (0.31) | `qber_std` (0.40) | `det_coinc_rat` (0.80) |
| 5 | `temp_asym` (0.27) | `qber_std` (0.20) | `temp_asym` (0.36) | `qber_std` (0.52) |

**Key insight:** For the three continuous, high-intensity attacks (IR, DB, MitM), the model relies overwhelmingly on `det_coinc_rat` (SHAP > 3.1). However, for the operationally critical Blended Sub-Threshold attack, `det_coinc_rat` drops to 4th place, and the model instead relies on three purely temporal features: the cross-correlation `coinc_qber_r`, the step-change `qber_delta`, and the edge-detection `temp_asym`. This validates the core thesis: static thresholding cannot detect stealthy blended attacks, necessitating a temporal feature engineering framework.
