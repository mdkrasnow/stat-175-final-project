# Experimental Results

## Dataset: STaRK-PrimeKG

- **129,375 nodes**, **4,049,642 edges**, single connected component
- Mean degree: 62.6, Median: 4.0, Max: 17,355 (heavy-tailed)
- Clustering coefficient: 0.079
- Approximate diameter: 15
- **11,204 QA pairs** stratified by hop count: 1-hop (3851), 2-hop (3699), 3-hop (2193), 4-hop (1001), 5-hop (313), 6+ (147)

## Main Experiment: Retrieval Performance by Hop Count (n=11,204)

| Hop | Node Hit@1 | Node Hit@10 | Path Hit@1 | Path Hit@10 | Subgraph Hit@1 | Subgraph Hit@10 | Hybrid Hit@1 | Hybrid Hit@10 |
|-----|-----------|------------|-----------|------------|---------------|----------------|-------------|--------------|
| 1 | 0.237 | 0.434 | 0.183 | 0.514 | 0.237 | **0.611** | 0.237 | 0.522 |
| 2 | 0.000 | 0.176 | **0.020** | 0.178 | 0.000 | 0.160 | 0.000 | **0.181** |
| 3 | 0.000 | 0.112 | **0.013** | 0.115 | 0.000 | **0.117** | 0.000 | 0.113 |
| 4 | 0.000 | 0.121 | **0.018** | 0.109 | 0.000 | 0.112 | 0.000 | **0.129** |
| 5 | 0.000 | 0.137 | **0.013** | **0.144** | 0.000 | 0.134 | 0.000 | 0.137 |
| 6+ | 0.000 | **0.150** | **0.014** | 0.068 | 0.000 | 0.109 | 0.000 | 0.143 |

### Key Finding: The Cliff, Not the Crossover

We hypothesized a gradual phase transition where structural methods overtake text-only as hop count increases. Instead, we found a **performance cliff from 1-hop to 2-hop for all strategies** — Hit@10 drops ~70% (0.43-0.61 down to 0.16-0.18). After 2-hop, everything plateaus near zero.

The phase transition isn't *between strategies* — it's between "solvable" (1-hop) and "nearly unsolvable" (2+ hop) for all current approaches.

### Statistical Evidence

- **Logistic regression interaction term** (hop x structural) = **+2.09** — structural methods gain relative advantage as hops increase
- **Paired bootstrap tests**: path-centric significantly outperforms node-centric on Hit@1 at 2-5 hops (p < 0.05, Cohen's d = 0.16-0.20)
- Path-centric is the **only strategy achieving Hit@1 > 0 at multi-hop** (1-2% — statistically significant but practically tiny)

## Ablation Studies (200 samples/bin)

### Ablation 1: Top-k Budget (Node-centric)

| top-k | 1-hop Hit | 2-hop Hit | 1-hop MRR | 2-hop MRR |
|-------|-----------|-----------|-----------|-----------|
| 1 | 0.30 | 0.00 | 0.30 | 0.00 |
| 10 | 0.46 | 0.26 | 0.35 | 0.08 |
| 50 | 0.59 | 0.44 | 0.36 | 0.09 |

**Takeaway:** Casting a wider net improves hit rate but **MRR barely moves** — the answer doesn't rise in rank, you're just hoping it falls somewhere in a bigger bucket. Brute-force doesn't solve the problem.

### Ablation 2: k-hop Expansion Depth (Subgraph-centric)

| k-hops | 1-hop Hit@10 | 2-hop Hit@10 | 1-hop MRR |
|--------|-------------|-------------|-----------|
| 1 | **0.64** | 0.24 | **0.38** |
| 2 | 0.48 | **0.30** | 0.35 |
| 3 | 0.48 | 0.27 | 0.35 |

**Takeaway:** More expansion hurts 1-hop and barely helps multi-hop. With mean degree 62, a 2-hop expansion floods the subgraph with ~3,600 nodes. The relevance-based pruning recovers some signal, but noise from irrelevant neighbors dominates. **Density kills expansion.**

### Ablation 3: Max Path Length (Path-centric)

| max_length | 1-hop Hit@10 | 2-hop Hit@10 | 2-hop MRR |
|-----------|-------------|-------------|-----------|
| 2 | 0.52 | 0.23 | **0.107** |
| 3 | 0.54 | 0.23 | 0.082 |
| 4 | 0.56 | 0.25 | 0.109 |
| 5 | 0.56 | 0.25 | 0.089 |

**Takeaway:** Longer paths give marginal Hit improvements but **shorter paths rank better** (higher MRR). Short, direct connections are more informative than long chains through a dense graph.

### Ablation 4: Number of Seed Nodes (Path + Subgraph)

Subgraph-centric results:

| seeds | 1-hop Hit@10 | 2-hop Hit@10 | 1-hop MRR |
|-------|-------------|-------------|-----------|
| 1 | **0.77** | 0.00 | **0.45** |
| 3 | 0.64 | 0.24 | 0.38 |
| 5 | 0.58 | 0.30 | 0.37 |
| 10 | 0.46 | 0.26 | 0.35 |

**Takeaway:** This is the clearest result. One seed achieves **Hit@10 = 0.77 and MRR = 0.45 at 1-hop** — the best number in any experiment — but **completely zeros out at 2+ hops**. More seeds spread coverage and help multi-hop but dilute focus at 1-hop. **This is a fundamental focus-vs-coverage tradeoff** that no single configuration resolves.

### Ablation 5: Text Weight (Hybrid)

| text_weight | 1-hop Hit@10 | 2-hop Hit@10 | 1-hop MRR | 4-hop Hit@10 |
|------------|-------------|-------------|-----------|-------------|
| 0.0 (pure structural) | 0.44 | 0.16 | 0.26 | 0.10 |
| 0.2 | **0.57** | 0.23 | **0.37** | 0.16 |
| 0.5 | 0.55 | **0.28** | 0.37 | **0.21** |
| 1.0 (pure text) | 0.46 | 0.26 | 0.35 | 0.18 |

**Takeaway:** Pure structural (0.0) is worst. Pure text (1.0) is mediocre. The **optimal weight shifts by query complexity** — 0.2 (mostly structural) for 1-hop, 0.5-0.6 (balanced) for multi-hop. This validates MoR's mixture intuition but adds nuance: a fixed weight is suboptimal because the ideal balance depends on query complexity.

## Synthesis

The ablations collectively tell one story: **there is a fundamental tension between focus and coverage that no single hyperparameter setting resolves.**

- Focused strategies (1 seed, 1 hop, short paths, low text weight) excel at 1-hop but fail at multi-hop
- Broad strategies (many seeds, deep expansion, long paths, high text weight) help multi-hop but dilute 1-hop performance
- The dense graph (mean degree 62) punishes expansion aggressively — noise overwhelms signal within 2 hops

This suggests that the right approach isn't picking one configuration but **adapting the retrieval strategy to estimated query complexity** — which is essentially what MoR's planning stage tries to do. The practical implication is that GraphRAG systems on dense KGs need query-adaptive retrieval, not a one-size-fits-all strategy.
