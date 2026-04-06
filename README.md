# CHRONOS Research Codebase

CHRONOS stands for **Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence**.

This repository contains:
- Full edge-IoT simulator
- CHRONOS algorithm modules (CHSE, SPN, CCPG, HFA, DTCS)
- Baseline implementations
- Ablation framework
- Pareto and metrics analysis tools
- Reproducibility runner (multi-seed + significance tests)

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
