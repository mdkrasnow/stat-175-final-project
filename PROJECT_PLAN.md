# Phase Transition in Graph Retrieval: When Does Structure Beat Text for Multi-Hop Reasoning?

**Track 2: Theoretical & Exploratory Research — Experimental Proof of Concept**

## Project Members
- Matt Krasnow
- Seager Hunt
- Cory Wu
- Reade Park

## Research Question

**Does a phase transition exist in GraphRAG retrieval performance as query complexity (hop count) increases, where text-only retrieval suffices for simple queries but graph-structural retrieval becomes necessary beyond a critical complexity threshold?**

We hypothesize that:
1. For 1-hop queries, node-centric (text-only) retrieval performs comparably to structural methods — graph structure provides no benefit.
2. There exists a critical hop count *h\** beyond which structural retrieval (path or subgraph) significantly outperforms text-only retrieval — a **phase transition** in retrieval effectiveness.
3. The location of this phase transition *h\** depends on graph properties (density, degree distribution, textual richness).

This is an exhaustive experimental study characterizing this phase transition, inspired by but distinct from the three foundational papers below.

## Foundational Papers

1. **GRAG: Graph Retrieval-Augmented Generation (2024)** — Introduces subgraph-based retrieval for RAG. Demonstrates that graph structure improves generation quality, but does not characterize *when* or *why* structure matters as a function of query complexity. ([arXiv](https://arxiv.org/abs/2405.16506), [GitHub](https://github.com/HuieL/GRAG))

2. **Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases (MoR, 2025)** — Proposes mixing structural and textual signals with a learned weight. Reports "uneven retrieving performance across different query logics" but does not formally characterize this unevenness as a function of hop count. ([arXiv](https://arxiv.org/abs/2502.20317), [GitHub](https://github.com/Yoega/MoR))

3. **GraphFlow: Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need? (2025)** — Critically examines retrieval fidelity in GraphRAG. Shows that existing methods often fail to retrieve what's needed, but focuses on a single flow-based solution rather than characterizing the failure boundary. ([arXiv](https://arxiv.org/abs/2510.16582), [GitHub — code not yet released](https://github.com/Samyu0304/GraphFlow))

**Our contribution:** None of these papers systematically vary query complexity to identify *where* text-only retrieval breaks down and structural retrieval becomes necessary. We fill this gap with a controlled experiment that isolates hop count as the independent variable and characterizes the resulting phase transition.

## Experimental Design

### Independent Variable
- **Query hop count** (1, 2, 3, 4+): The number of relational hops required to answer the query. We stratify STaRK's QA pairs by the shortest path length between query entity and answer entity in the ground-truth KG.

### Conditions (Retrieval Strategies)
| Strategy | Type | Description | Paper Reference |
|---|---|---|---|
| **Node-centric** | Text-only | Top-k nodes by embedding similarity | GRAG baseline |
| **Path-centric** | Structural | Reasoning paths between seed entities | GraphFlow |
| **Subgraph-centric** | Structural | k-hop neighborhood expansion from seeds | GRAG + MoR |
| **Hybrid** | Mixed | Weighted combination of text + structural scores | MoR |

### Dependent Variables
- **Hit@1, Hit@5, Hit@10** — retrieval accuracy
- **MRR** — mean reciprocal rank
- **EM, Token F1** — generation quality (via frozen LLM)

### Controls
- Same frozen LLM (GPT-4o-mini) across all conditions
- Same embedding model (all-MiniLM-L6-v2) for all retrievers
- Same top-k budget across strategies at each comparison point

### Statistical Methodology
- **Paired bootstrap tests** for pairwise strategy comparisons at each hop count
- **Interaction analysis**: strategy x hop-count interaction via logistic regression on Hit@1
- **Crossover point estimation**: fit performance curves per strategy as a function of hop count; estimate *h\** where structural methods first significantly outperform text-only
- **Effect size**: Cohen's d at each hop count to quantify the practical significance of structural retrieval

## Dataset: STaRK Benchmark

| Dataset | Domain | Processing Time | Role |
|---|---|---|---|
| **STaRK-PrimeKG** | Biomedical (drugs, diseases, genes) | ~5 minutes | **Primary** — fast iteration, rich multi-hop structure |
| **STaRK-Amazon** | E-commerce / product search | ~1 hour | **Secondary** — test generalization across domains |
| **STaRK-MAG** | Academic papers | ~1 hour | **Tertiary** — use if time permits |

**Why STaRK:** Used by 2 of 3 papers (MoR, GraphFlow), publicly available via `pip install stark-qa`, text-rich nodes with relational structure, train/val/test splits provided.

### Repo References

| Repo | Status | Our Use |
|---|---|---|
| **[MoR](https://github.com/Yoega/MoR)** | Complete with pretrained checkpoints | Reference for hybrid baseline |
| **[GRAG](https://github.com/HuieL/GRAG)** | Complete but heavy (Llama-2, 4x GPUs) | Reference for subgraph retrieval code |
| **[GraphFlow](https://github.com/Samyu0304/GraphFlow)** | Empty repo, code not released | Read paper only for path-based methodology |

## Implementation Plan

### Phase 1: Data & Infrastructure (Week 1)
- Load STaRK-PrimeKG and characterize its graph properties (degree distribution, diameter, clustering)
- **Stratify QA pairs by hop count** — compute shortest path from query entity to answer entity; bin into 1-hop, 2-hop, 3-hop, 4+ categories
- Verify sufficient sample size per hop-count bin for statistical power
- Set up evaluation harness with retrieval + generation metrics

### Phase 2: Implement Retrieval Strategies (Weeks 2-3)
- Implement and validate all four retrievers (node, path, subgraph, hybrid)
- Each retriever takes a query → returns context string → frozen LLM generates answer
- Sanity check: all strategies should perform non-trivially on 1-hop queries

### Phase 3: Phase Transition Experiments (Week 3-4)
- Run all 4 strategies on each hop-count stratum
- Record per-query results (not just aggregates) for paired statistical tests
- Sweep hyperparameters per strategy (top-k, k_hops, path_length, text_weight) — report best configuration per strategy to ensure fair comparison
- Repeat on STaRK-Amazon to test generalization

### Phase 4: Statistical Analysis (Week 4)
- **Phase transition characterization**:
  - Plot Hit@k and MRR as a function of hop count for each strategy
  - Identify crossover point *h\** via curve fitting
  - Test strategy x hop-count interaction (logistic regression)
- **Pairwise comparisons**: paired bootstrap at each hop count
- **Effect sizes**: Cohen's d for structural vs text-only at each hop count
- **Ablations**: subgraph size, path length, top-k budget
- **Graph property analysis**: correlate performance gaps with local graph properties (degree, clustering coefficient around query entities)

### Phase 5: Write-Up (Week 5)
- Frame as phase transition characterization
- Present crossover curves and statistical evidence for *h\**
- Discuss implications: when should practitioners use structural retrieval?
- Connect findings to MoR's "uneven performance across query logics" and GraphFlow's retrieval fidelity critique
- Discuss limitations and future work

## Team Division

| Member | Primary Responsibility |
|---|---|
| **Matt Krasnow** | Infrastructure, data pipeline, hop-count stratification, statistical analysis |
| **Seager Hunt** | Node-centric retriever + baseline experiments |
| **Cory Wu** | Path-centric retriever (GraphFlow-inspired) + path length ablations |
| **Reade Park** | Subgraph-centric retriever (GRAG-inspired) + hybrid (MoR) + structural ablations |

All members collaborate on phase transition analysis and write-up.
