---
name: Project hypothesis and goal
description: Stat 175 final project — benchmarking GraphRAG retrieval strategies on STaRK, hypothesis that path/subgraph beat node-based
type: project
---

**Hypothesis:** Path-based and subgraph-based retrieval outperform node-based retrieval on multi-hop relational reasoning queries in text-rich KGs, and hybrid (structural + textual) retrieval outperforms any single strategy.

**Goal:** Benchmark node-centric, path-centric, subgraph-centric, and hybrid retrieval on STaRK (PrimeKG primary, Amazon secondary) to answer which retrieval strategy best supports multi-hop reasoning in GraphRAG.

**Why:** Three papers (GRAG 2024, MoR 2025, GraphFlow 2025) propose different retrieval approaches but haven't been compared head-to-head on the same benchmark under controlled conditions.

**How to apply:** All implementation decisions should serve this comparison — same frozen LLM, same dataset, same metrics. The retrieval strategy is the only independent variable.
