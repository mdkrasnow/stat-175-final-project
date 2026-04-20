# Project Plan

**Title:** Which *Kinds* of Graph Structure Add Predictive Value Beyond Text? A Sampler-as-Probe, Schema-Generalization Study on PrimeKG

**Team:** Matt Krasnow, Seager Hunt, Cory Wu, Reade Park

**Course:** Stat 175 Final Project

See [`PIVOT_DESIGN.md`](PIVOT_DESIGN.md) for the full experimental design, estimands, hypotheses, and timeline.

## Repository Layout

```
final-project/
├── PIVOT_DESIGN.md          # Full experimental design (source of truth)
├── PROJECT_PLAN.md          # This file
├── TODO.md                  # Active task list
├── README.md
├── requirements.txt
├── src/
│   ├── data/                # PrimeKG loading, schema partitioning
│   ├── samplers/            # Word2Vec, DeepWalk, node2vec, GraphSAGE
│   ├── estimation/          # DML estimator, cross-fitting, bootstrap
│   └── evaluation/          # Metrics, permutation tests, schema transfer
├── experiments/             # Experiment runner scripts
├── results/                 # Numerical results + figures
├── notebooks/               # Exploratory analysis
├── latex/                   # Paper source
├── papers/                  # Reference PDFs
├── references/
└── archive/
    └── old_graphrag_project/  # Previous GraphRAG retrieval project (shelved)
```

## Ownership

| Member | Primary Responsibility |
|---|---|
| Matt Krasnow | DML estimator, cross-fitting, synthetic-graph validation, statistical analysis |
| Seager Hunt | Schema partitioning, sampler training pipeline, drug–gene replication |
| Cory Wu | Schema-transfer experiments, permutation tests, joint-sampler analysis |
| Reade Park | Hop stratification, ablations, write-up lead |

All members collaborate on final analysis and write-up.
