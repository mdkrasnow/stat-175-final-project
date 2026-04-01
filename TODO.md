# Project TODO

**Hypothesis:** Path-based and subgraph-based retrieval outperform node-based retrieval on queries requiring multi-hop relational reasoning in text-rich knowledge graphs, and hybrid (structural + textual) retrieval outperforms any single strategy.

**Goal:** Systematically benchmark node-centric, path-centric, subgraph-centric, and hybrid retrieval strategies on STaRK to determine which best supports multi-hop reasoning in GraphRAG.

---

## Phase 1: Data & Infrastructure (Week 1)

- [x] Select dataset: STaRK benchmark (PrimeKG as primary, Amazon as secondary)
- [x] Write STaRK data loader with NetworkX graph wrapper (`src/data/stark_loader.py`)
- [x] Set up evaluation harness with Hit@1, Hit@5, Hit@10, MRR (`src/evaluation/metrics.py`)
- [x] Set up frozen LLM generation wrapper (`src/generation/llm.py`)
- [x] Define base retriever interface (`src/retrievers/base.py`)
- [x] Create experiment runner (`experiments/run_experiment.py`)
- [x] Write `requirements.txt`
- [ ] Install dependencies and verify STaRK-PrimeKG loads successfully
- [ ] Clone MoR repo as reference (`git clone https://github.com/Yoega/MoR.git`)
- [ ] Clone GRAG repo as reference (`git clone https://github.com/HuieL/GRAG.git`)
- [ ] Set up API keys (OpenAI or Anthropic) for frozen LLM

## Phase 2: Implement Three Retrieval Strategies (Weeks 2-3)

- [x] Node-centric retriever — scaffold (`src/retrievers/node_retriever.py`)
- [x] Path-centric retriever — scaffold (`src/retrievers/path_retriever.py`)
- [x] Subgraph-centric retriever — scaffold (`src/retrievers/subgraph_retriever.py`)
- [ ] Validate node-centric retriever on small sample (Seager)
- [ ] Validate path-centric retriever on small sample (Cory)
- [ ] Validate subgraph-centric retriever on small sample (Reade)
- [ ] Tune node retriever: experiment with top-k values, embedding models
- [ ] Tune path retriever: experiment with num_seeds, max_path_length
- [ ] Tune subgraph retriever: experiment with num_seeds, k_hops
- [ ] Run end-to-end test: retriever -> LLM -> answer on 50 samples

## Phase 3: Hybrid / MoR Baseline (Week 3)

- [x] Hybrid retriever — scaffold (`src/retrievers/hybrid_retriever.py`)
- [ ] Validate hybrid retriever on small sample (Reade)
- [ ] Tune text_weight parameter (sweep 0.0 to 1.0)
- [ ] Compare hybrid vs individual strategies on dev set

## Phase 4: Evaluation & Analysis (Week 4)

- [ ] Run full retrieval evaluation on STaRK-PrimeKG (all 4 strategies)
- [ ] Run full retrieval evaluation on STaRK-Amazon (all 4 strategies)
- [ ] Break down results by query hop count (1-hop, 2-hop, 3-hop)
- [ ] Ablation: effect of subgraph size (k-hop depth)
- [ ] Ablation: effect of path length
- [ ] Ablation: effect of number of retrieved nodes (top-k)
- [ ] Qualitative failure case analysis (10-20 examples per strategy)
- [ ] Statistical significance tests (paired bootstrap)
- [ ] Generate comparison tables and figures

## Phase 5: Write-Up (Week 5)

- [ ] Frame results around the three research questions
- [ ] Discuss when graph structure helps vs hurts (connect to GraphFlow findings)
- [ ] Compare our results to MoR and GRAG reported numbers
- [ ] Write recommendations for practitioners
- [ ] Final paper draft
