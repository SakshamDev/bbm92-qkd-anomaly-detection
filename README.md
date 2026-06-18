# Physics-Aware Temporal Feature Engineering for Eavesdropping Detection in BBM92 QKD

> This repository contains the code, experiments, simulator, and paper associated with the research project.

**Author:** Saksham Gupta
**Institution:** BITS Pilani, Hyderabad Campus

*Research internship conducted at:*
Solid State Physics Laboratory (SSPL), DRDO
under the mentorship of Dr. Lalit Kumar

## Abstract
Traditional security in Quantum Key Distribution (QKD) relies on aborting key generation when the Quantum Bit Error Rate (QBER) exceeds a static threshold. However, in noisy free-space optical (FSO) channels, natural atmospheric variations can camouflage short, low-intensity eavesdropping bursts. We propose a machine learning-based defense strategy specifically designed for entanglement-based (BBM92) QKD telemetry. By engineering a 24-dimensional feature space computed over a 30-second sliding window, we evaluate the temporal shape and cross-observable correlation of QBER, Bell S parameter, and photon coincidence rates. Evaluated on a comprehensive simulated FSO telemetry dataset, our framework demonstrates that static QBER thresholding fails catastrophically against stealthy blended attacks, achieving only 17.3% recall. In contrast, our multivariate approach suggests that such attacks can be identified with 96.9% recall and 97.6% precision within the simulated environment. SHAP analysis indicates that stealthy attacks are not detected by absolute error magnitude, but by anomalous temporal cross-correlations between independent quantum observables.

## Research Motivation
A sophisticated adversary can execute a "blended attack"—timing short bursts of low-intensity eavesdropping to coincide with natural atmospheric turbulence, thereby keeping the aggregate QBER below the abort threshold. Static thresholding is structurally blind to this threat vector. This framework uses XGBoost and rolling window telemetry to catch the non-physical cross-correlations introduced by such attacks.

## Repository Structure
```
bbm92-qkd-anomaly-detection/
├── core/                # Physics simulation engine and attack models
├── ml/                  # Machine learning pipeline and feature engineering
├── scripts/             # Experiment scripts to reproduce paper tables/figures
├── tests/               # Pytest validation suite
├── paper/               # Research paper LaTeX source and supplementary material
├── data/                # Generated telemetry datasets
├── models/              # Persisted XGBoost model
├── docs/                # Research documentation
└── app.py               # Streamlit real-time monitoring dashboard
```

## Installation
Requires Python 3.10+.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Paper Results
1. Generate the canonical dataset:
   ```bash
   python scripts/generate_dataset.py
   ```
2. Train the model:
   ```bash
   python ml/train.py
   ```
3. Run the master evaluation script to generate tables and figures:
   ```bash
   python scripts/reproduce_paper_results.py
   ```

## Experiment Scripts

| Script | Paper Reference | Description |
|--------|----------------|-------------|
| `reproduce_paper_results.py` | Tables I-III, Figures 1-6 | Master 5-seed evaluation |
| `evaluate_baselines.py` | Table I | ML baseline comparison with McNemar's test |
| `ablate_window_size.py` | Table II | Window size sensitivity |
| `eve_fraction_sweep.py` | Table III | Eve fraction sensitivity |
| `cross_seed_baselines.py` | Table I (verification) | Cross-seed stability |
| `cross_seed_shap.py` | Section V.D | Cross-seed SHAP stability |
| `shap_analysis.py` | Section V.D | Per-attack SHAP breakdown |
| `analyze_latency.py` | Supplementary | Detection latency analysis |
| `wilcoxon_test.py` | Section V.A | Statistical significance test |
| `imbalance_sweep.py` | Supplementary | Class imbalance robustness |
| `check_lr_blended.py` | Section V.A | LR vs XGBoost per-attack comparison |
| `simulate_live_telemetry.py` | Dashboard | UDP test transmitter for Streamlit UI |

## Streamlit Dashboard
Launch the real-time monitoring interface:
```bash
streamlit run app.py
```

## External Datasets
The external satellite QKD validation dataset (from Zenodo) is not committed to the repository due to size and licensing. If you wish to run `evaluate_on_zenodo.py`:
1. Download the dataset from the relevant Zenodo archive (URL/DOI to be provided).
2. Place the raw files in `data/zenodo/`.
3. Run `python scripts/convert_zenodo_dataset.py` to prepare the data.

## Paper
The paper source is located in `paper/src/`. Compile it with:
```bash
cd paper/src
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Tests
```bash
python -m pytest tests/ -v
```

## Citation
```bibtex
@article{gupta2026bbm92,
  title={Physics-Aware Temporal Feature Engineering for Eavesdropping Detection in BBM92 Quantum Key Distribution},
  author={Gupta, Saksham},
  journal={IACR Cryptology ePrint Archive},
  year={2026}
}
```

## License
MIT License
