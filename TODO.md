# Project TODO

**Track 2: Theoretical & Exploratory Research — Experimental Proof of Concept**

**Research Question:** Does a phase transition exist in GraphRAG retrieval performance as query complexity (hop count) increases, where text-only retrieval suffices for simple queries but graph-structural retrieval becomes necessary beyond a critical complexity threshold *h\**?

**Hypotheses:**
1. For 1-hop queries, node-centric (text-only) retrieval matches structural methods.
2. There exists a critical hop count *h\** beyond which structural retrieval significantly outperforms text-only.
3. The phase transition location depends on graph properties (density, degree distribution).

---

## Phase 1: Data & Infrastructure (Week 1)

- [x] Select dataset: STaRK benchmark (PrimeKG primary, Amazon secondary)
- [x] Write STaRK data loader with NetworkX graph wrapper (`src/data/stark_loader.py`)
- [x] Set up evaluation harness with Hit@1, Hit@5, Hit@10, MRR (`src/evaluation/metrics.py`)
- [x] Set up frozen LLM generation wrapper (`src/generation/llm.py`)
- [x] Define base retriever interface (`src/retrievers/base.py`)
- [x] Create experiment runner (`experiments/run_experiment.py`)
- [x] Write `requirements.txt`
- [x] Install dependencies and verify STaRK-PrimeKG loads successfully
- [x] Clone MoR repo as reference (`references/MoR`)
- [x] Clone GRAG repo as reference (`references/GRAG`)
- [ ] Set up API keys (OpenAI or Anthropic) for frozen LLM
- [x] **Characterize graph properties**: 129K nodes, 4M edges, mean degree 62.6, clustering 0.079, diameter ~15
- [x] **Stratify QA pairs by hop count**: 1-hop: 3851, 2-hop: 3699, 3-hop: 2193, 4-hop: 1001, 5-hop: 313, 6+: 147
- [x] **Verify statistical power**: all bins >= 50 samples — good to proceed

## Phase 2: Implement Retrieval Strategies (Weeks 2-3)

- [x] Node-centric retriever — scaffold (`src/retrievers/node_retriever.py`)
- [x] Path-centric retriever — scaffold (`src/retrievers/path_retriever.py`)
- [x] Subgraph-centric retriever — scaffold (`src/retrievers/subgraph_retriever.py`)
- [x] Hybrid retriever — scaffold (`src/retrievers/hybrid_retriever.py`)
- [x] Build shared FAISS index to avoid redundant encoding (`src/retrievers/shared_index.py`)
- [x] Validate node-centric retriever on 50 samples — Hit@10=0.38, MRR=0.241
- [x] Validate path-centric retriever on 50 samples — Hit@10=0.46, MRR=0.272
- [x] Validate subgraph-centric retriever on 50 samples — Hit@10=0.48, MRR=0.260
- [x] Validate hybrid retriever on 50 samples — Hit@10=0.48, MRR=0.259
- [x] Sanity check: all strategies perform non-trivially (Hit@1 >= 0.16)
- [ ] Run end-to-end test: retriever -> LLM -> answer on 50 samples

## Phase 3: Phase Transition Experiments (Weeks 3-4)

- [x] Run all 4 strategies on each hop-count stratum (1-hop through 6+) — n=11,204
- [x] Record per-query results for paired tests (`prime_perquery_*.json`)
- [ ] Hyperparameter sweep per strategy to ensure fair comparison:
  - [ ] Node: top-k values, embedding models
  - [ ] Path: num_seeds, max_path_length
  - [ ] Subgraph: num_seeds, k_hops
  - [ ] Hybrid: text_weight (sweep 0.0 to 1.0)
- [ ] Report best configuration per strategy
- [ ] Repeat full experiment on STaRK-Amazon for generalization

## Phase 4: Statistical Analysis (Week 4)

- [x] Plot Hit@k and MRR as function of hop count for each strategy (`prime_phase_transition.png`)
- [ ] Estimate crossover point *h\** via curve fitting
- [x] Logistic regression: strategy × hop-count interaction on Hit@1 — interaction coef = +2.09 (structural gains as hops increase)
- [x] Paired bootstrap tests at each hop count — path-centric sig. better Hit@1 at 2-5 hop (p<0.05)
- [x] Cohen's d effect sizes — path vs node: d=0.16-0.20 on Hit@1 at multi-hop
- [ ] Ablation: effect of subgraph size (k-hop depth)
- [ ] Ablation: effect of path length
- [ ] Ablation: effect of top-k budget
- [ ] Correlate performance gap with local graph properties (degree, clustering around query entities)
- [ ] Qualitative failure case analysis (10-20 examples per strategy per hop count)
- [x] Generate comparison tables and phase transition figure

## Phase 5: Write-Up (Week 5)

- [ ] Introduction: frame as phase transition characterization in GraphRAG
- [ ] Related work: position relative to GRAG, MoR, GraphFlow
- [ ] Experimental design section: independent/dependent variables, controls, statistical methods
- [ ] Results: crossover curves, significance tests, effect sizes
- [ ] Discussion: when should practitioners use structural retrieval?
- [ ] Connect to MoR's "uneven performance across query logics"
- [ ] Connect to GraphFlow's retrieval fidelity critique
- [ ] Limitations and future work
- [ ] Final paper draft
- [ ] Prepare defense presentation
