# CHRONOS Ablation Study

Each variant removes one component of CHRONOS. Δ is relative to CHRONOS-full.

| Variant | Combined Reward | Deadline Viol. | Latency Reward | Energy Reward | SNN Energy Ratio | Δ Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CHRONOS-full** | 0.277 | 0.000 | -0.017 | -0.001 | 0.020 | — |
| w/o SNN (→ MLP) | -73.649 | 0.467 | -7.330 | -0.036 | 0.000 | -73.926 |
| w/o Hypergraph (→ graph) | -73.847 | 0.480 | -7.494 | -0.321 | 0.020 | -74.124 |
| w/o Causal reasoning | -74.138 | 0.486 | -7.468 | -0.024 | 0.020 | -74.416 |
| w/o Federated learning | -73.727 | 0.473 | -7.364 | -0.035 | 0.020 | -74.004 |
| w/o Digital Twin | -73.623 | 0.465 | -7.283 | -0.090 | 0.020 | -73.901 |

## Component Importance (by impact on Combined Reward)

| Component Removed | Δ Combined Reward | Relative Change |
| :--- | :---: | :---: |
| w/o Causal reasoning | -74.416 | -26853.6% |
| w/o Hypergraph (→ graph) | -74.124 | -26748.4% |
| w/o Federated learning | -74.004 | -26705.2% |
| w/o SNN (→ MLP) | -73.926 | -26676.9% |
| w/o Digital Twin | -73.901 | -26667.8% |

> Larger negative Δ = that component contributes more to performance.
