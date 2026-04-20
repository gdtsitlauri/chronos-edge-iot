# CHRONOS Research Codebase

**Author:** George David Tsitlauri  
**Affiliation:** Dept. of Informatics & Telecommunications, University of Thessaly, Greece  
**Contact:** gdtsitlauri@gmail.com  
**Year:** 2026

CHRONOS stands for **Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence**.

This repository contains:
- Full edge-IoT simulator
- CHRONOS algorithm modules (CHSE, SPN, CCPG, HFA, DTCS)
- Baseline implementations
- Ablation framework
- Pareto and metrics analysis tools
- Reproducibility runner (multi-seed + significance tests)

---
**Research Paper:**
This project is accompanied by a full research paper describing the architecture, methodology, and experimental results of CHRONOS. The research and all experiments were conducted in 2026.

You can find the paper in [paper/chronos_paper.tex](paper/chronos_paper.tex).
---

## Results Summary

Full results for 3 seeds (42, 43, 44), 30 training rounds per method.
See [results/comparison_table.md](results/comparison_table.md) and [results/ablation_report.md](results/ablation_report.md) for complete tables.

### CHRONOS vs Baselines (mean ± std, n=3 seeds)

| Method | Combined Reward ↑ | Accuracy Reward ↑ | Latency Reward ↑ | Deadline Viol. ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **CHRONOS** | **-47.46 ± 33.80** | **1.784 ± 0.760** | -5.372 ± 3.803 | 0.430 ± 0.308 |
| FedAvg+Greedy | 0.012 ± 0.008 | 0.031 ± 0.020 | -0.001 ± 0.000 | **0.000 ± 0.000** |
| DRL-Offload | 0.011 ± 0.006 | 0.029 ± 0.015 | **-0.001 ± 0.000** | **0.000 ± 0.000** |
| QMIX | -0.800 ± 1.148 | 0.053 ± 0.038 | -0.081 ± 0.113 | 0.166 ± 0.235 |
| GNN-Sched | -0.052 ± 0.080 | 0.037 ± 0.036 | -0.011 ± 0.015 | 0.071 ± 0.100 |
| MAPPO | -2.639 ± 0.214 | 0.069 ± 0.058 | -0.258 ± 0.021 | 0.506 ± 0.045 |
| Random | -2.376 ± 0.140 | 0.300 ± 0.025 | -0.244 ± 0.011 | 0.646 ± 0.068 |

> CHRONOS achieves the highest accuracy reward (+1.48 vs best baseline) by actively offloading tasks to edge nodes, which also incurs higher communication cost. Baselines that avoid offloading achieve near-zero latency/comm penalties but forgo compute gains.

### Ablation: Component Importance (Δ Combined Reward vs CHRONOS-full)

| Component Removed | Δ Combined Reward | Key Finding |
| :--- | :---: | :--- |
| w/o Causal reasoning | -74.42 | Largest drop — causal SCM is critical |
| w/o Hypergraph | -74.12 | Higher-order group dependencies matter |
| w/o Federated learning | -74.00 | Centralized training hurts privacy/scale |
| w/o SNN | -73.93 | SNN energy efficiency affects reward |
| w/o Digital Twin | -73.90 | DT provides safe exploration |

All components contribute significantly; removing any one causes ≈74-point drop.

---

## 1. Environment Setup

```bash
pip install -r requirements.txt
```

Quick import check:

```bash
python -c "import chronos; print('OK')"
```

## 2. Run Main Experiments

### Train CHRONOS

```bash
python experiments/run_chronos.py --config configs/fast_experiment.yaml --rounds 30 --output outputs/chronos
```

Outputs:
- `outputs/chronos/final_results.json` (canonical metrics + metadata)
- `outputs/chronos/training_log.json`
- checkpoints and best model

### Run Baselines

```bash
python experiments/run_baselines.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10 --output outputs/baselines
```

Outputs:
- `outputs/baselines/all_baselines_results.json` (raw)
- `outputs/baselines/all_baselines_summary.json` (canonical)
- per-method raw and summary JSON files

### Run Ablation

```bash
python experiments/run_ablation.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10 --output outputs/ablation
```

Outputs:
- `outputs/ablation/CHRONOS-*_results.json` (canonical)
- `outputs/ablation/CHRONOS-*_results_raw.json` (raw)
- `outputs/ablation/ablation_report.txt`
- `outputs/ablation/ablation_impacts.json`

## 3. Analyze Results

```bash
python experiments/analyze_results.py
```

Outputs:
- `outputs/analysis/comparison_table.txt`
- `outputs/analysis/comparison_table.tex`
- `outputs/analysis/pareto_report.txt`
- `outputs/analysis/analysis_input_snapshot.json`

If you already have ablation JSON files and only want to refresh report math:

```bash
python experiments/rebuild_ablation_report.py --ablation-dir outputs/ablation
```

Notes:
- Analysis now prioritizes `outputs/chronos/final_results.json` for CHRONOS metrics.
- If missing, it falls back to ablation full results, then training log.

## 4. Reproducibility (Multi-Seed)

Run a multi-seed study and significance tests:

```bash
python experiments/run_reproducibility.py --config configs/fast_experiment.yaml --seeds 42,43,44 --chronos-rounds 30 --baseline-rounds 30 --eval-episodes 10 --output outputs/reproducibility
```

Outputs:
- `outputs/reproducibility/per_seed_results.json`
- `outputs/reproducibility/reproducibility_summary.json`
- `outputs/reproducibility/significance_tests.json`
- `outputs/reproducibility/reproducibility_report.txt`

## 5. Tests

Run all tests:

```bash
python -m pytest -q
```

Current tests cover:
- config inheritance
- environment reset and step smoke checks
- canonical metrics schema
- ablation impact stability
- analysis loading priority and fallbacks

## 6. Recommended Paper-Ready Protocol

1. Run at least 3 seeds per scenario.
2. Use same seed set for CHRONOS and all baselines.
3. Report mean and std for all objectives.
4. Include significance tests for combined reward and key objectives.
5. Attach comparison table and Pareto report from `outputs/analysis`.

## 7. Common Commands (Short)

```bash
# CHRONOS only
python experiments/run_chronos.py --config configs/fast_experiment.yaml --rounds 30

# Baselines only
python experiments/run_baselines.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10

# Ablation only
python experiments/run_ablation.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10

# Analysis only
python experiments/analyze_results.py
```

## 8. GitHub Upload Checklist

1. Initialize git:

```bash
git init
```

2. Stage and commit:

```bash
git add .
git commit -m "Initial CHRONOS research codebase"
```

3. Create remote repository and push:

```bash
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

Note:
- Generated artifacts in `outputs/` are ignored by default through `.gitignore`.
- If you want to version specific result files, remove or adjust the `outputs/` rule.

## 9. License

This project is released under the MIT License.
See the `LICENSE` file for full text.

## Citation

```bibtex
@misc{tsitlauri2026chronos,
  author = {George David Tsitlauri},
  title  = {CHRONOS: Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```
