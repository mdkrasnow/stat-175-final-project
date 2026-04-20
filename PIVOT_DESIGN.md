# Which *Kinds* of Graph Structure Add Predictive Value Beyond Text?
### A Sampler-as-Probe, Schema-Generalization Study on PrimeKG

**Stat 175 Final Project · Krasnow, Hunt, Wu, Park**

---

## Research Question

Different graph sampling methods implicitly capture different *types* of structural information — proximity, structural roles, community membership, neighborhood aggregation. **After controlling for text, which of these structural signals add residual predictive value for link prediction on PrimeKG, and which of them generalize across schema shifts?**

## Core Idea

Each sampler is a **probe for a distinct structural hypothesis**:

| Sampler | Structural signal it encodes | Hypothesis it probes |
|---|---|---|
| **DeepWalk** (uniform walks) | Proximity / homophily | "Nodes close in the graph share labels" |
| **node2vec, low q** (BFS-biased) | Structural roles | "Nodes with similar local topology behave similarly" |
| **node2vec, high q** (DFS-biased) | Community membership | "Nodes in the same community share labels" |
| **GraphSAGE / mean-pool** | Neighborhood feature aggregation | "A node is the average of its neighbors" |
| **Text-only (Word2Vec on descriptions)** | Semantic content | Baseline — no structure |

Estimating residual predictive value per sampler *after* partialling out text tells us **which structural signals PrimeKG's drug–disease biology actually depends on**. Evaluating that residual value on held-out schemas tells us **which of those signals are generalizable structural primitives vs. artifacts of a specific relation topology**.

---

## Core Estimand (Residual Value of Structure)

For each sampler $s$, with $T$ = text embeddings and $S_s$ = structural embeddings:

$$\tau_s(u,v) \;=\; \mathbb{E}[Y \mid T, S_s] - \mathbb{E}[Y \mid T]$$

Target quantities:

- **Per-sampler residual value:** $\bar{\tau}_s = \mathbb{E}[\tau_s(u,v)]$
- **Sampler contrasts:** $\bar{\tau}_s - \bar{\tau}_{s'}$ with joint CIs
- **Hop-stratified residual value:** $\bar{\tau}_{s,h}$ at geodesic distance $h \in \{2,3,4+\}$
- **Joint model:** $\bar{\tau}_{\text{all}} = \mathbb{E}[Y \mid T, S_1, \dots, S_k] - \mathbb{E}[Y \mid T]$ — are samplers redundant or complementary?

---

## Estimation Procedure

- **Double ML with cross-fitting** (Chernozhukov et al. 2018), 5 folds split *by node* to prevent embedding leakage
- Nuisances $\hat{\eta}_T$, $\hat{\eta}_{T,S_s}$ fit with XGBoost on Hadamard-product endpoint features
- **Cluster-bootstrap (B=1000)** over nodes for CIs; **Holm correction** across samplers
- **Permutation null:** shuffle structural embeddings across nodes per sampler → sharp null of no residual structural signal
- **Sanity check:** synthetic graph with known generative structure (SBM + planted text features), verify estimator recovers true $\bar{\tau}_s$

---

## Schema-Generalization Layer

The original "out-of-sample node prediction" question is only meaningful if the test distribution actually shifts. Held-out *edges* is a weak version of OOD; held-out *schema* is the real thing.

### Setup

PrimeKG has multiple node types (drug, disease, gene, protein, pathway, phenotype) and ~30 relation types. A **schema** = a subgraph induced by a chosen subset of node/relation types.

- **Schema A:** drug–gene–disease triangles (direct + indirect indications)
- **Schema B:** drug–protein–pathway–disease chains (mechanism-of-action subgraph)
- **Schema C:** disease–phenotype–gene (clinical presentation subgraph)
- **Schema D (held-out):** drug–disease direct only

### Protocol

1. Induce $k$ schema-subgraphs $G_1, \dots, G_k$ from PrimeKG; designate one $G_{\text{test}}$ as fully held out.
2. For each sampler $s$, train structural embeddings $S_s^{(i)}$ on each $G_i$ independently. Text embeddings are trained once on the full node-description corpus (text is schema-agnostic).
3. Fit nuisance models $\hat{\eta}_T$ and $\hat{\eta}_{T, S_s}$ on $\bigcup_{i \neq \text{test}} G_i$ with cross-fitting.
4. **Transfer:** apply frozen $\hat{\eta}_{T, S_s}$ to node pairs in $G_{\text{test}}$, using embeddings induced by running sampler $s$ on $G_{\text{test}}$'s topology.
5. Estimate $\bar{\tau}_s^{\text{OOD}}$ = residual value of structure on the held-out schema.

### Added Estimand

- **Schema-transfer gap:** $\Delta_s = \bar{\tau}_s^{\text{OOD}} - \bar{\tau}_s^{\text{in-dist}}$
- **Statistical test:** paired bootstrap of $\hat{\Delta}_s$ over node pairs in $G_{\text{test}}$, clustered by node
- **Leave-one-schema-out CV:** report $\bar{\tau}_s^{\text{OOD}}$ for each held-out schema to avoid cherry-picking

---

## Dataset & Task

- **PrimeKG**, drug–disease indication edges (primary); drug–gene as replication
- Binary link prediction, degree-matched negative sampling
- Held-out disease classes for an additional OOD axis orthogonal to schema

---

## Falsifiable Predictions

### Core (residual value)
- **H1:** $\bar{\tau}_{\text{DeepWalk}} > 0$ — proximity carries residual signal beyond text
- **H2:** $\bar{\tau}_{\text{node2vec-BFS}}$ vs $\bar{\tau}_{\text{node2vec-DFS}}$ differ significantly — drug–disease relations favor one structural signal type
- **H3:** $\bar{\tau}_{s,h}$ grows with hop distance $h$ for proximity-based samplers (phase-transition-style)
- **H4:** $\bar{\tau}_{\text{all}} \approx \max_s \bar{\tau}_s$ (redundancy) OR $\bar{\tau}_{\text{all}} \gg \max_s \bar{\tau}_s$ (complementarity)

### Schema-generalization
- **H5:** Proximity samplers (DeepWalk) transfer poorly — proximity is schema-specific
- **H6:** Structural-role samplers (node2vec, low q) transfer best — roles are schema-invariant abstractions
- **H7:** GraphSAGE transfers intermediate — leans on transferable text features but non-transferable aggregation topology

---

## Ablations

- Walk length, window size, p/q grid for node2vec
- Text encoder swap (Word2Vec → SBERT): does residual value shrink with stronger text?
- Relation-type stratification: dominant structural signal for drug–disease vs drug–gene
- Sample-size curves: does $\bar{\tau}_s$ stabilize before we run out of data?
- Schema choice ablation (leave-one-schema-out)

---

## Deliverables

- Table of $\bar{\tau}_s$ per sampler with CIs and permutation p-values
- Heatmap of $\bar{\tau}_{s,h}$ across sampler × hop distance
- Schema-transfer table: $\bar{\tau}_s^{\text{OOD}}$ and $\hat{\Delta}_s$ per (sampler, held-out schema) pair
- Synthetic-graph validation plot
- Full reproducible pipeline

---

## Timeline (2 weeks)

| Days | Work | Owner |
|---|---|---|
| 1–2 | Schema partitioning of PrimeKG + train Word2Vec (text) + DeepWalk + node2vec (BFS/DFS) + GraphSAGE per schema | Seager, Reade |
| 3–4 | DML estimator + synthetic-graph validation | Matt |
| 5–7 | Per-sampler $\bar{\tau}_s$ estimation + cluster bootstrap + permutation null (in-distribution) | Matt, Cory |
| 8–9 | Schema-transfer estimation: $\bar{\tau}_s^{\text{OOD}}$ + leave-one-schema-out | Cory, Matt |
| 10–11 | Hop stratification, joint model, ablations, drug–gene replication | Reade, Seager |
| 12–14 | Write-up, figures, final polish | All |

---

## Why This Wins

- **Sampler-as-probe reframing** turns the methods comparison into a *scientific* one: we're not asking "which sampler is best," we're asking "what kinds of structural signal does PrimeKG's biology actually encode, and which of them generalize."
- Crisp, semiparametric estimand per sampler — statistically deep, well-posed, CI-bounded.
- Schema-generalization layer addresses the real OOS question head-on: held-out *edges* is a weak test; held-out *schema* is a genuine distribution shift.
- Any outcome is interesting: all-zero $\bar{\tau}_s$ → provocative null; large $\bar{\tau}_s$ → quantifies a field-wide assertion; differential $\bar{\tau}_s$ across samplers → maps PrimeKG's structural geometry; differential $\hat{\Delta}_s$ across samplers → identifies which structural signals are generalizable primitives.
- Clean controls, falsification tests, multiple-comparison correction, synthetic-data validation.
