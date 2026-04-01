# Benchmarking Retrieval Strategies for GraphRAG on Text-Rich Knowledge Graphs

## Project Members
- Matt Krasnow
- Seager Hunt
- Cory Wu
- Reade Park

## Project Goal

Systematically compare **node-centric**, **path-centric**, and **subgraph-centric** retrieval strategies over text-rich knowledge graphs (KGs) to determine which approach best supports multi-hop relational reasoning in retrieval-augmented generation (RAG). The project directly addresses three research questions:

1. Does path-based or subgraph-based retrieval outperform node-based retrieval on queries that require multi-hop relational reasoning in text-rich knowledge graphs?
2. How does retrieval/sampling strategy over a knowledge graph affect answer quality in GraphRAG?
3. Compare node-centric, path-centric, and subgraph-centric retrieval on text-rich KGs.

## Key Papers & Their Roles

1. **GRAG: Graph Retrieval-Augmented Generation (2024)** — Introduces graph retrieval-augmented generation with subgraph-based retrieval. Provides the baseline subgraph retrieval method and the core idea of using graph structure to improve RAG. ([arXiv](https://arxiv.org/abs/2405.16506), [GitHub](https://github.com/HuieL/GRAG))

2. **Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases (MoR, 2025)** — Proposes a *Mixture of Retrieval* approach combining structural (graph topology) and textual (semantic similarity) signals over text-rich KGs. Provides the framework for hybrid retrieval and the text-rich KG setting the research questions target. ([arXiv](https://arxiv.org/abs/2502.20317), [GitHub](https://github.com/Yoega/MoR))

3. **GraphFlow: Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need? (2025)** — Critically examines whether KG-based RAG actually retrieves what's needed, introducing flow-based retrieval analysis. Provides evaluation methodology and the path-based retrieval perspective. ([arXiv](https://arxiv.org/abs/2510.16582), [GitHub — code not yet released](https://github.com/Samyu0304/GraphFlow))

## Dataset Decision: STaRK Benchmark

After investigating all three papers' datasets and repos, we chose the **STaRK benchmark** (Stanford's Semi-structured Retrieval Benchmark on Textual and Relational Knowledge Bases).

### Why STaRK

- **Used by 2 of 3 papers** — both MoR and GraphFlow evaluate on STaRK, making results directly comparable to the literature
- **Easiest setup** — `pip install stark-qa`, data auto-downloads from HuggingFace
- **Text-rich KGs** — nodes have textual descriptions + relational graph structure, exactly matching our research questions
- **Clean Python API** — `load_skb('prime')`, `load_qa('prime')`
- **Train/val/test splits provided**

### STaRK Datasets

| Dataset | Domain | Processing Time | Notes |
|---|---|---|---|
| **STaRK-PrimeKG** | Biomedical (drugs, diseases, genes) | ~5 minutes | **Start here** — fastest to iterate on |
| **STaRK-Amazon** | E-commerce / product search | ~1 hour | Good second domain for generalization |
| **STaRK-MAG** | Academic papers (Microsoft Academic Graph) | ~1 hour | Largest, use if time permits |

### Alternatives Considered

| Dataset | Paper | Why Not |
|---|---|---|
| ExplaGraphs | GRAG | Tiny graphs (~5 nodes) — not real KGs, poor fit for multi-hop reasoning |
| WebQSP | GRAG | Heavy preprocessing (SentenceBERT + PyTorch Geometric + subgraph caching), fragile dependency chain |

### Repo Availability

| Repo | Status | Our Use |
|---|---|---|
| **[MoR](https://github.com/Yoega/MoR)** | Complete with pretrained checkpoints | Clone — reference for hybrid baseline, structural+textual retrieval on STaRK |
| **[GRAG](https://github.com/HuieL/GRAG)** | Complete but heavy (Llama-2, 4x GPUs, PyG) | Clone for reference only — read subgraph retrieval code |
| **[GraphFlow](https://github.com/Samyu0304/GraphFlow)** | Empty repo, code not released | Do not clone — read the paper for path-based methodology |

### LLM Strategy

Use a **frozen API model** (GPT-4o-mini or Claude) rather than local Llama. This:
- Eliminates GPU requirements
- Lets us focus on retrieval quality (the actual research question)
- Ensures identical generation across all retrieval strategies

## Implementation Plan

### Phase 1: Data & Infrastructure (Week 1)
- Install `stark-qa` and verify data loading with STaRK-PrimeKG
- Build a shared KG loading pipeline — load STaRK's semi-structured KB into NetworkX for graph operations + FAISS for text embeddings
- Set up the evaluation harness — use STaRK's QA pairs with standard metrics (Hit@1, Hit@5, MRR)
- Clone MoR and GRAG repos as reference implementations

### Phase 2: Implement Three Retrieval Strategies (Weeks 2-3)
Each team member (or pair) owns one strategy:

| Strategy | Description | Primary Paper Reference |
|---|---|---|
| **Node-centric** | Retrieve top-k nodes by embedding similarity to the query; concatenate their text as context | Baseline from GRAG |
| **Path-centric** | Retrieve reasoning paths (chains of entities/relations) connecting query entities to candidate answers | GraphFlow's flow-based retrieval |
| **Subgraph-centric** | Extract a local subgraph around query entities (e.g., k-hop neighborhood or PPR-based); serialize as context | GRAG's subgraph method + MoR's structural retrieval |

- Each retriever takes a query and returns a context string
- Feed each context into the **same frozen LLM** (GPT-4o-mini or Claude) to isolate retrieval quality from generation quality

### Phase 3: Hybrid / MoR Baseline (Week 3)
- Implement MoR's mixture approach: combine structural retrieval scores with textual similarity scores
- This serves as a fourth condition and directly tests whether mixing modalities outperforms pure strategies

### Phase 4: Evaluation & Analysis (Week 4)
- **Quantitative**: Compare all strategies on multi-hop QA accuracy (Hit@1, Hit@5, MRR), varying hop count (1-hop, 2-hop, 3-hop)
- **Qualitative**: Analyze failure cases — when does each strategy retrieve irrelevant context? (Ties to GraphFlow's critique)
- **Ablations**:
  - Effect of subgraph size (k-hop depth)
  - Effect of path length
  - Effect of number of retrieved nodes
- **Statistical testing**: Significance tests across strategies (paired bootstrap or similar)

### Phase 5: Write-Up (Week 5)
- Frame results around the three research questions
- Discuss when graph structure helps vs. hurts (connecting to GraphFlow's findings)
- Propose recommendations for practitioners choosing a GraphRAG retrieval strategy

## Suggested Team Division

| Member | Primary Responsibility |
|---|---|
| **Matt Krasnow** | Infrastructure, KG pipeline, evaluation harness |
| **Seager Hunt** | Node-centric retrieval + baseline experiments |
| **Cory Wu** | Path-centric retrieval (GraphFlow-inspired) |
| **Reade Park** | Subgraph-centric retrieval (GRAG-inspired) + MoR hybrid |

All members collaborate on evaluation and write-up.
