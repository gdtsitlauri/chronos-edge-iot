# CHRONOS

**Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence**


CHRONOS is a simulation-first research repository for edge/IoT orchestration under conflicting objectives: task accuracy, latency, communication cost, energy use, and deadline compliance. The repository includes a full simulator, CHRONOS-specific decision modules, baseline runners, ablation tooling, and analysis scripts.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Evidence Status

| Item | Current status |
| --- | --- |
| Core simulator and algorithm modules | Present |
| Baseline implementations | Present |
| Multi-seed summaries | Present |
| Ablation artifacts | Present |
| Statistical claims of overall superiority | Not supported by current committed results |
| Real deployment / hardware validation | Not present |

## Research Positioning

The strongest claim supported by the committed artifacts is:

> CHRONOS explores a high-capacity orchestration policy that can increase accuracy-related reward in a synthetic edge simulation, but in its current configuration it pays a heavy communication and deadline cost and does not dominate the simpler baselines on the overall combined objective.

That trade-off is scientifically useful. It is not the same thing as state-of-the-art orchestration performance.

## What Exists

- A causal/hypergraph-oriented simulation environment under `chronos/`
- CHRONOS-specific training and evaluation scripts under `experiments/`
- Baseline comparison and ablation pipelines
- Multi-seed result aggregation under `results/` and `outputs/`
- Tests for schema, fallbacks, and basic analysis workflows

## Current Result Snapshot

Source: `results/summary.json`

| Method | Combined Reward | Accuracy Reward | Communication Reward | Deadline Violation |
| --- | ---: | ---: | ---: | ---: |
| `CHRONOS` | `-47.46 ± 33.80` | `1.784 ± 0.760` | `-435.92 ± 308.25` | `0.430 ± 0.308` |
| `FedAvg+Greedy` | `0.012 ± 0.008` | `0.031 ± 0.020` | `0.00 ± 0.00` | `0.000 ± 0.000` |
| `DRL-Offload` | `0.011 ± 0.006` | `0.029 ± 0.015` | `0.00 ± 0.00` | `0.000 ± 0.000` |
| `GNN-Sched` | `-0.052 ± 0.080` | `0.037 ± 0.036` | `-0.59 ± 0.83` | `0.071 ± 0.100` |

### Interpretation

- CHRONOS achieves the highest accuracy-related reward among the committed methods.
- That gain comes with a very large communication penalty and a much worse deadline-violation profile.
- Under the current scalarized reward, the simpler baselines remain stronger on the aggregate objective.
- The current repo therefore supports a trade-off study, not a headline claim that CHRONOS is already the best scheduler overall.

## Methodology Notes

- Evaluation is entirely simulation-based.
- The reported means come from three seeds (`42`, `43`, `44`) in the committed summary.
- The result files separate raw run outputs from rebuilt analysis artifacts.
- Any future paper-ready claim should report both objective-level metrics and the scalarized reward, because the current result picture changes materially depending on which objective is emphasized.

## Repository Layout

```text
chronos/
  __init__.py
  env.py / agents / policies / metrics
configs/
  fast_experiment.yaml
experiments/
  run_chronos.py
  run_baselines.py
  run_ablation.py
  run_reproducibility.py
  analyze_results.py
outputs/
  chronos/
  baselines/
  ablation/
  reproducibility/
results/
  comparison_table.md
  ablation_report.md
  summary.json
tests/
paper/
  chronos_paper.tex
```

## Reproducibility

Install:

```bash
pip install -r requirements.txt
```

Run the main experiment stack:

```bash
python experiments/run_chronos.py --config configs/fast_experiment.yaml --rounds 30 --output outputs/chronos
python experiments/run_baselines.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10 --output outputs/baselines
python experiments/run_ablation.py --config configs/fast_experiment.yaml --train-rounds 30 --eval-episodes 10 --output outputs/ablation
python experiments/analyze_results.py
```

Multi-seed study:

```bash
python experiments/run_reproducibility.py --config configs/fast_experiment.yaml --seeds 42,43,44 --chronos-rounds 30 --baseline-rounds 30 --eval-episodes 10 --output outputs/reproducibility
```

Tests:

```bash
python -m pytest -q
```

## Limitations

- No real edge cluster or wireless deployment is included.
- The current scalarized reward strongly penalizes the communication-heavy behavior of CHRONOS.
- The repo does not yet show a tuned operating point that preserves CHRONOS's accuracy benefit while recovering aggregate reward competitiveness.

## Future Work

- Rebalance the reward design and scheduling constraints.
- Add communication-aware calibration experiments.
- Evaluate on richer traffic/task arrival regimes and, eventually, hardware-in-the-loop setups.


