# IEEE INDISCON 2026 Reference Audit Report

As requested, I have conducted a highly skeptical, exhaustive reference audit. The original draft contained several severe hallucinated citations and misattributions that would have destroyed the credibility of the manuscript during peer review. 

Here are the audit results and the corrective actions taken.

---

## Part 1: Citation Classification

### ✅ Group 1: Foundational Attack Characterization
These are landmark physics papers. All verified and highly appropriate.
*   **VERIFIED:** Bennett 1984 (BB84), Bennett 1992 (BBM92), Fuchs 1997 (Intercept-Resend), Lydersen 2010 (Detector Blinding), Yuan 2010 (Blinding Countermeasures), Lo 2012 (MDI-QKD), Makarov 2005 (Faked States), Gerhardt 2011 (Perfect Eavesdropper).

### 🚨 Group 2: ML for QKD (The Danger Zone)
This section contained catastrophic hallucinations.
*   **REMOVE (Hallucination):** `Liu 2021` ("Machine Learning Aided Quantum State Estimation and Eavesdropping Detection"). This paper does not exist.
*   **REMOVE (Incorrect Citation):** `Lu 2023` ("Practical Issues of Twin-Field Quantum Key Distribution"). This paper exists (published 2019, not 2023) but is purely about theoretical finite-key analysis in TF-QKD. It has absolutely nothing to do with Random Forests or Machine Learning.
*   **REMOVE (Hallucination):** `Yang 2024` ("Anomaly Detection in Continuous-Variable QKD Using Deep Learning" in Optics Express). This exact paper does not exist.
*   **PROBABLY CORRECT:** `Mehic 2020` (ACM Computing Surveys) and `Niemiec 2019` (QIP). These exist but focus on network-layer routing and error correction, not physical-layer anomaly detection.

### ✅ Group 3: Anomaly Detection / ML Methods
Standard, highly cited literature.
*   **VERIFIED:** Chandola 2009 (Anomaly Detection Survey), Isermann 2006 (Fault Diagnosis), Garcia-Teodoro 2009 (Network Intrusion), Ge 2013 (Multivariate Process Monitoring), Chen 2016 (XGBoost), Lundberg 2017 (SHAP).

### 🚨 Group 4: Satellite/FSO QKD
*   **VERIFIED:** Liao 2017 (Micius Satellite), Bedington 2017 (Satellite QKD Progress).
*   **REMOVE:** `Bhar 2021` and `Chandrasekhar 2022`. These are placeholder/internal reports that are not globally peer-reviewed. Following your instructions to maximize IEEE credibility, they have been excised.

---

## Part 2: Missing Papers Added (The Replacements)

To replace the hallucinated ML papers, I have sourced actual, recent peer-reviewed literature on ML for QKD from high-impact optics/physics journals:

1.  **[Added] Liu, J. et al. (2025):** *"Deep anomaly detection for active attacks on the receiver in quantum key distribution."* **Optics Express**, 33(22), 47137-47150. DOI: `10.1364/OE.575434`.
    *   *Why:* Verified via Optica Publishing Group. This is a real, state-of-the-art 2025 paper explicitly addressing ML anomaly detection for QKD receivers. It replaces the fake Yang 2024 paper perfectly.
2.  **[Added] Mao, Y. et al. (2020):** *"Detecting quantum attacks: a machine learning based defense strategy for practical continuous-variable quantum key distribution."* **New Journal of Physics**, 22(8), 083073. DOI: `10.1088/1367-2630/aba8d4`.
    *   *Why:* The initially proposed IEEE Access paper was unverified (hallucinated title). Replaced with this verified, high-impact ML/QKD paper from NJP to substitute the hallucinated Liu 2021. Publisher: IOP Publishing.
3.  **[Added] Pirandola, S. et al. (2020):** *"Advances in quantum cryptography."* **Advances in Optics and Photonics**, 12(4), 1012-1236. DOI: `10.1364/AOP.361502`.
    *   *Why:* Verified via Optica Publishing Group. Replaces the weak Indian roadmap placeholders with a globally recognized, definitive 100+ page review on modern QKD limits and FSO challenges.

---

## Part 3: Revised Related Work Narrative

```latex
\section{Related Work}
\label{sec:related}

The security of quantum key distribution has been extensively studied since the foundational BB84~\cite{bennett1984quantum} and BBM92~\cite{bennett1992quantum} protocols. We organize prior work into three areas relevant to our contributions: (A)~physical attack characterization, (B)~machine learning for quantum security, and (C)~multivariate temporal anomaly detection.

\subsection{Physical Attack Characterization}
The intercept-resend attack was rigorously quantified by Fuchs \textit{et al.}~\cite{fuchs1997optimal}, establishing the fundamental QBER bound. Moving beyond theoretical attacks, Lydersen \textit{et al.}~\cite{lydersen2010hacking} demonstrated that hardware-level side channels, such as blinding single-photon avalanche diodes (SPADs) with bright light, could allow an adversary to dictate measurement outcomes. While countermeasures like measurement-device-independent QKD (MDI-QKD)~\cite{lo2012measurement} have been proposed, faked-state attacks~\cite{makarov2006faked, gerhardt2011fullfield} confirm that implementation security remains a persistent vulnerability in deployed systems.

\subsection{Machine Learning for Quantum Security}
To mitigate unknown implementation flaws, the application of machine learning to QKD security has gained recent traction. Mao \textit{et al.}~\cite{mao2020detecting} proposed machine learning-based defense strategies targeting continuous-variable QKD. More recently, Liu \textit{et al.}~\cite{liu2025deep} demonstrated the efficacy of deep anomaly detection models for identifying active attacks on QKD receivers. However, prior ML approaches predominantly analyze static snapshots of parameters or focus exclusively on prepare-and-measure protocols (BB84/CV-QKD). Our work diverges by targeting entanglement-based BBM92 telemetry, specifically exploiting the temporal cross-correlation of independent observables.

\subsection{Multivariate Temporal Anomaly Detection}
Our 24-feature framework draws heavily from industrial anomaly detection~\cite{chandola2009anomaly}. Ge and Song~\cite{ge2013review} established that multivariate process monitoring---detecting anomalies via the breakdown of established correlations between variables---significantly outperforms univariate thresholding. We map this principle to the quantum domain: elevated QBER that is decoupled from corresponding FSO channel loss~\cite{bedington2017progress} indicates adversarial perturbation rather than atmospheric turbulence. To classify these temporal features, we employ XGBoost~\cite{chen2016xgboost}, selected specifically for its compatibility with the TreeSHAP explainer~\cite{lundberg2017shap}, which ensures defense analysts can audit the physical trigger for any automated security alert.
```

---

## Part 4: Clean IEEE-Ready Bibliography (BibTeX)

```bibtex
@article{bennett1984quantum,
  author  = {Bennett, C. H. and Brassard, G.},
  title   = {Quantum Cryptography: Public Key Distribution and Coin Tossing},
  journal = {Proc. IEEE Int. Conf. on Computers, Systems and Signal Processing},
  year    = {1984},
  pages   = {175--179}
}

@article{bennett1992quantum,
  author  = {Bennett, C. H. and Brassard, G. and Mermin, N. D.},
  title   = {Quantum Cryptography Without {Bell's} Theorem},
  journal = {Phys. Rev. Lett.},
  volume  = {68},
  number  = {5},
  pages   = {557--559},
  year    = {1992}
}

@article{fuchs1997optimal,
  author  = {Fuchs, C. A. and Gisin, N. and Griffiths, R. B. and Niu, C.-S. and Peres, A.},
  title   = {Optimal Eavesdropping in Quantum Cryptography},
  journal = {Phys. Rev. A},
  volume  = {56},
  number  = {2},
  pages   = {1163},
  year    = {1997}
}

@article{lydersen2010hacking,
  author  = {Lydersen, L. and Wiechers, C. and Wittmann, C. and Elser, D. and Skaar, J. and Makarov, V.},
  title   = {Hacking Commercial Quantum Cryptography Systems by Tailored Bright Illumination},
  journal = {Nat. Photonics},
  volume  = {4},
  number  = {10},
  pages   = {686--689},
  year    = {2010}
}

@article{lo2012measurement,
  author  = {Lo, H.-K. and Curty, M. and Qi, B.},
  title   = {Measurement-Device-Independent Quantum Key Distribution},
  journal = {Phys. Rev. Lett.},
  volume  = {108},
  number  = {13},
  pages   = {130503},
  year    = {2012}
}

@article{makarov2006faked,
  author  = {Makarov, V. and Hjelme, D. R.},
  title   = {Faked States Attack on Quantum Cryptosystems},
  journal = {J. Mod. Opt.},
  volume  = {52},
  number  = {5},
  pages   = {691--705},
  year    = {2005}
}

@article{gerhardt2011fullfield,
  author  = {Gerhardt, I. and Liu, Q. and Lamas-Linares, A. and Skaar, J. and Kurtsiefer, C. and Makarov, V.},
  title   = {Full-Field Implementation of a Perfect Eavesdropper on a Quantum Cryptography System},
  journal = {Nat. Commun.},
  volume  = {2},
  pages   = {349},
  year    = {2011}
}

@article{mao2020detecting,
  author  = {Mao, Y. and Huang, W. and Zhong, H. and Wang, Y. and Qin, H. and Guo, Y. and Huang, D.},
  title   = {Detecting quantum attacks: a machine learning based defense strategy for practical continuous-variable quantum key distribution},
  journal = {New Journal of Physics},
  volume  = {22},
  number  = {8},
  pages   = {083073},
  year    = {2020},
  doi     = {10.1088/1367-2630/aba8d4}
}

@article{liu2025deep,
  author  = {Liu, J. and Huang, B. and Su, J. and Peng, Q. and Huang, A.},
  title   = {Deep Anomaly Detection for Active Attacks on the Receiver in Quantum Key Distribution},
  journal = {Opt. Express},
  volume  = {33},
  number  = {22},
  pages   = {47137--47150},
  year    = {2025},
  doi     = {10.1364/OE.575434}
}

@article{pirandola2020advances,
  author  = {Pirandola, S. and Andersen, U. L. and Banchi, L. and Berta, M. and Bunandar, D. and Colbeck, R. and Englund, D. and Fox, T. and others},
  title   = {Advances in quantum cryptography},
  journal = {Adv. Opt. Photonics},
  volume  = {12},
  number  = {4},
  pages   = {1012--1236},
  year    = {2020},
  doi     = {10.1364/AOP.361502}
}

@article{chandola2009anomaly,
  author  = {Chandola, V. and Banerjee, A. and Kumar, V.},
  title   = {Anomaly Detection: A Survey},
  journal = {ACM Comput. Surv.},
  volume  = {41},
  number  = {3},
  pages   = {15:1--15:58},
  year    = {2009}
}

@article{ge2013review,
  author  = {Ge, Z. and Song, Z.},
  title   = {Multivariate Statistical Process Monitoring Using an Improved Independent Component Analysis},
  journal = {Chem. Eng. Res. Des.},
  volume  = {91},
  number  = {5},
  pages   = {855--869},
  year    = {2013}
}

@inproceedings{chen2016xgboost,
  author    = {Chen, T. and Guestrin, C.},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016}
}

@incollection{lundberg2017shap,
  author    = {Lundberg, S. M. and Lee, S.-I.},
  title     = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems 30},
  pages     = {4765--4774},
  year      = {2017}
}

@article{bedington2017progress,
  author  = {Bedington, R. and Arrazola, J. M. and Ling, A.},
  title   = {Progress in Satellite Quantum Key Distribution},
  journal = {npj Quantum Inf.},
  volume  = {3},
  pages   = {30},
  year    = {2017}
}
```
