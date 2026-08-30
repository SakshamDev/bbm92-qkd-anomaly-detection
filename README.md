# Temporal Cross-Channel Features for Eavesdropping Detection in Free-Space BBM92 QKD under Hierarchical Threat Models

> **Accepted at IEEE INDISCON 2026**
> 
> This repository contains the code, experiments, and simulator associated with the research project. The research paper source is maintained separately.

**Author:** Saksham Gupta
**Institution:** BITS Pilani, Hyderabad Campus

*Research internship conducted at:*
Solid State Physics Laboratory (SSPL), DRDO
under the mentorship of Dr. Lalit Kumar

## Abstract
Static QBER thresholding can miss adaptive eavesdropping in free-space BBM92 QKD because atmospheric turbulence masks short attack bursts, keeping the aggregate error rate below the abort threshold. This paper evaluates a temporal feature engineering framework for machine learning-based detection in simulated BBM92 telemetry. We define a hierarchical attacker capability model at three levels: Naive, AR1 (statistical count-rate spoofing), and Replay (exact marginal-distribution replay). A feature leakage audit shows that raw count-rate separability drops from AUC 1.000 to near chance as attacker capability increases. A 22-dimensional temporal feature vector, computed over 30-second sliding windows, captures cross-channel correlations and higher-order statistics of QBER, the Bell S parameter, and coincidence rates. The cross-channel representation achieves 91.3% recall and 94.2% precision (F2 = 0.914 ± 0.113) on unseen attacks, vastly outperforming static loss and error thresholds. We conclude that temporal dynamics of Bell-inequality violations and coincidences provide robust indicators of malicious intervention even when raw quantum error rates remain below traditional alarm thresholds. These results indicate that temporal cross-channel feature modeling provides a physics-informed detection capability that is robust to progressive adversarial concealment in high-variance atmospheric conditions.

## Research Motivation
A sophisticated adversary can execute a "blended attack"—timing short bursts of low-intensity eavesdropping to coincide with natural atmospheric turbulence, thereby keeping the aggregate QBER below the abort threshold. Static thresholding is structurally blind to this threat vector. This framework uses XGBoost and rolling window telemetry to catch the non-physical cross-correlations introduced by such attacks.

## Repository Structure
```
core/
    Physics simulation and attack models
ml/
    Machine learning pipeline and feature engineering
scripts/
    Dataset generation
    Training
    Evaluation
    SHAP
    Baselines
tests/
    Pytest validation suite
data/
    Generated telemetry datasets and output figures
models/
    Persisted XGBoost model
app.py
    Streamlit real-time monitoring dashboard
```

## Installation
Requires Python 3.10+.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce paper
1. `python scripts/generate_dataset.py`
2. `python scripts/reproduce_paper_results.py`
3. `python scripts/shap_analysis.py`
4. `python scripts/evaluate_baselines.py`

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

## Tests
```bash
python -m pytest tests/ -v
```

## Citation
```bibtex
@inproceedings{gupta2026bbm92,
  title={Temporal Cross-Channel Features for Eavesdropping Detection in Free-Space BBM92 QKD under Hierarchical Threat Models},
  author={Gupta, Saksham},
  booktitle={Proceedings of the IEEE INDISCON 2026},
  year={2026},
  organization={IEEE}
}
```

## License
MIT License
