# TODO

Task-level detail with file paths, reusable archived code, and acceptance criteria. Archive root: `archive/old_graphrag_project/`.

---

## Phase 0 — Environment (Day 1, before anything else)

- [ ] **Update `requirements.txt`** — add missing deps:
  - `gensim>=4.3` (Word2Vec + skip-gram for DeepWalk/node2vec)
  - `torch-geometric>=2.4` (GraphSAGE, optional node2vec impl)
  - `xgboost>=2.0` (DML nuisance learners)
  - `doubleml>=0.7` OR `econml>=0.15` (cross-fit DML — `doubleml` is cleaner for PLR)
  - `statsmodels>=0.14` (Holm correction, permutation utilities)
  - Keep existing: `networkx`, `numpy`, `pandas`, `scikit-learn`, `faiss-cpu`, `sentence-transformers`, `torch`, `tqdm`, `matplotlib`, `seaborn`
  - Remove: `openai`, `anthropic`, `stark-qa` no longer strictly required — but KEEP `stark-qa` since the loader still uses it to pull PrimeKG
- [ ] `pip install -r requirements.txt` in a fresh venv; verify imports
- [ ] Decide on DML backend (recommend `doubleml` for partially-linear-regression interface)

---

## Phase 1 — Data & Samplers (Days 1–2)

### 1.1 PrimeKG loader → `src/data/primekg_loader.py`
- [ ] Lift `load_stark_dataset()` and `StarkGraphWrapper` from `archive/old_graphrag_project/src/data/stark_loader.py`. Strip out anything QA-retrieval specific; keep `_build_networkx_graph()` and `_extract_node_texts()`.
- [ ] Add a caching layer: pickle the NetworkX graph + node-text dict to `data/cache/primekg_graph.pkl` on first load (old loader regenerated on every run — wasteful at 4M edges).
- [ ] Expose: `load_primekg() -> (nx.Graph, dict[node_id -> text], dict[node_id -> node_type], dict[edge -> relation_type])`.
- [ ] Acceptance: `load_primekg()` returns graph with 129,375 nodes, 4,049,642 edges (sanity check matches `archive/.../results/prime_graph_summary.json`).

### 1.2 Schema partitioning → `src/data/schemas.py`
- [ ] Inventory PrimeKG node types (drug, disease, gene/protein, pathway, phenotype, anatomy, …) and edge/relation types — dump counts to `results/schema_inventory.json`.
- [ ] Implement `induce_schema_subgraph(G, allowed_node_types, allowed_relation_types) -> nx.Graph`.
- [ ] Define four schemas as dicts (see `PIVOT_DESIGN.md §Schema-Generalization Layer`):
  - **A:** drug–gene–disease triangles
  - **B:** drug–protein–pathway–disease chains
  - **C:** disease–phenotype–gene
  - **D (held-out):** drug–disease direct only
- [ ] Acceptance: each schema subgraph has ≥10k edges and ≥1 connected component of ≥1k nodes; log sizes to `results/schema_inventory.json`.

### 1.3 Link-prediction splits → `src/data/splits.py`
- [ ] Adapt `split_dataset()` from `archive/old_graphrag_project/src/data/qctr_data.py` for edge-level splits instead of query-level.
- [ ] Implement degree-matched negative sampling: for each positive edge (u,v), sample a non-edge (u,v') where deg(v') ∈ [deg(v)/2, 2·deg(v)].
- [ ] Implement **node-disjoint** 5-fold splits (critical for DML — a node's embedding cannot appear in both train and eval folds).
- [ ] Separate splits: (a) in-distribution per schema, (b) held-out-schema, (c) held-out disease class.
- [ ] Acceptance: no node overlap across folds; pos/neg ratio = 1:1; splits serialized to `data/splits/{schema}_fold{k}.npz`.

### 1.4 Text embeddings → `src/samplers/text_word2vec.py`
- [ ] Train Word2Vec (gensim) on tokenized node descriptions from the full PrimeKG corpus (schema-agnostic, trained once).
- [ ] Output: `data/embeddings/word2vec_200d.npy` (shape `[n_nodes, 200]`), plus `node_id → row_index` mapping.
- [ ] Also compute a SBERT baseline embedding (`all-MiniLM-L6-v2`, 384-d) for the text-encoder-swap ablation; save to `data/embeddings/sbert_384d.npy`.

### 1.5 Structural samplers → `src/samplers/`
- [ ] `deepwalk.py` — uniform random walks + gensim Word2Vec skip-gram. Params: `walk_length=40, num_walks=10, window=5, dim=128`.
- [ ] `node2vec.py` — two configs: BFS-biased (`p=1, q=0.25`) and DFS-biased (`p=1, q=4`). Implement via `gensim` + biased-walk sampler OR `torch-geometric.nn.Node2Vec`.
- [ ] `graphsage.py` — unsupervised GraphSAGE with mean aggregator (PyG `SAGEConv`), trained with link-prediction loss on each schema's edges. Dim=128.
- [ ] For each sampler and each schema: save embedding matrix to `data/embeddings/{sampler}/{schema}.npy`.
- [ ] **Acceptance:** per-sampler link-prediction AUC > 0.7 on in-distribution holdout for at least Schema A (sanity).

---

## Phase 2 — DML Estimator (Days 3–4)

### 2.1 Feature construction → `src/estimation/features.py`
- [ ] `hadamard_features(emb_u, emb_v) -> np.ndarray` — element-wise product of endpoint embeddings (standard for link prediction).
- [ ] `build_pair_features(pairs, text_emb, struct_emb=None) -> (X_T, X_TS)` where `X_T` is text-only Hadamard, `X_TS = concat(X_T, hadamard(struct))`.

### 2.2 Cross-fit DML → `src/estimation/dml.py`
- [ ] Implement `estimate_tau(pairs, labels, text_emb, struct_emb, n_folds=5) -> dict` returning per-pair $\hat\tau$, $\bar\tau$, and nuisance predictions.
- [ ] Use `doubleml.DoubleMLPLR` OR roll a manual cross-fit loop (cleaner if we want per-pair τ). Recommend manual: fit `η_T = XGBoost(Y ~ X_T)` and `η_TS = XGBoost(Y ~ X_TS)` on 4 folds, score on the 5th, rotate. Compute $\hat\tau_i = \hat\eta_{TS}(x_i) - \hat\eta_T(x_i)$ on held-out fold only.
- [ ] **Node-disjoint folds** (not edge-disjoint) — reuse splits from 1.3.
- [ ] Acceptance: running on a trivial case where struct = noise → $\bar\tau \approx 0$ with CI covering 0.

### 2.3 Cluster bootstrap + inference → `src/estimation/inference.py`
- [ ] Lift `paired_bootstrap_test()` and `cohens_d()` from `archive/old_graphrag_project/experiments/phase4_analysis.py` — adapt for cluster resampling.
- [ ] `cluster_bootstrap_ci(tau_per_pair, node_ids_u, node_ids_v, B=1000) -> (lower, upper, se)` — resample nodes (not pairs) with replacement, aggregate τ over pairs whose endpoints are in the resampled set.
- [ ] `permutation_null(Y, X_T, X_TS, B=500) -> p_value` — shuffle the structural block of X_TS across nodes, refit nuisance, recompute τ, compare to observed.
- [ ] `holm_correction(p_values: dict) -> dict` — for multi-sampler family.

### 2.4 Synthetic-graph validation → `experiments/validate_dml_synthetic.py`
- [ ] Generate SBM with 3 blocks, 500 nodes/block, within-block p=0.2, between-block p=0.02.
- [ ] Planted text features: per-block mean + noise.
- [ ] Planted edge labels: logistic(α·same_block + β·text_similarity), choose α, β so true $\bar\tau$ is analytically known (α drives residual structural value).
- [ ] Run DML estimator; verify 95% CI covers true $\bar\tau$ in ≥90% of 20 simulation seeds.
- [ ] Save results to `results/dml_synthetic_validation.json` and plot to `results/dml_synthetic_validation.png`.

---

## Phase 3 — In-Distribution Estimation (Days 5–7)

### 3.1 Per-sampler τ̄_s → `experiments/run_in_dist_tau.py`
- [ ] For each schema ∈ {A, B, C, D} and each sampler s ∈ {DeepWalk, n2v-BFS, n2v-DFS, GraphSAGE}: compute $\hat\tau$ per edge-pair on in-distribution holdout.
- [ ] Report: $\bar\tau_s$, 95% cluster-bootstrap CI, permutation p-value.
- [ ] Save to `results/in_dist_tau.json` with schema-{schema}_{sampler} keys.
- [ ] Apply Holm correction across the sampler family within each schema.

### 3.2 Sampler contrasts → `experiments/run_sampler_contrasts.py`
- [ ] Paired bootstrap on $(\hat\tau_s - \hat\tau_{s'})$ per pair (within-pair differences, same folds).
- [ ] All $\binom{4}{2} = 6$ pairwise contrasts per schema.
- [ ] Save to `results/sampler_contrasts.json`.

### 3.3 Hop stratification → `experiments/run_hop_stratified_tau.py`
- [ ] **Reuse** hop-stratified edge bins: lift `stratify_by_hop_count()` logic from `archive/old_graphrag_project/src/data/graph_analysis.py` — but adapt for edge pairs (not query-answer pairs). For each held-out pair (u,v), compute shortest-path distance in G \ {(u,v)}.
- [ ] Bin into hops ∈ {2, 3, 4+} (1-hop is the observed edge).
- [ ] Report $\bar\tau_{s,h}$ with CIs; make heatmap `results/tau_sampler_hop_heatmap.png`.

### 3.4 Joint-sampler model → `experiments/run_joint_tau.py`
- [ ] Fit $\eta_{T, S_1, \dots, S_4}$ with all samplers concatenated; compute $\bar\tau_{\text{all}}$.
- [ ] Test H4: is $\bar\tau_{\text{all}} \approx \max_s \bar\tau_s$ (redundancy) or $\gg$ (complementarity)?
- [ ] Bootstrap CI on $\bar\tau_{\text{all}} - \max_s \hat\tau_s$.
- [ ] Save to `results/joint_tau.json`.

---

## Phase 4 — Schema Generalization (Days 8–9)

### 4.1 Schema-transfer estimation → `experiments/run_schema_transfer.py`
- [ ] For each held-out schema $G_{\text{test}}$:
  - Train each sampler on $G_{\text{test}}$ topology (sampler is applied to held-out structure — we transfer the *learned nuisance model*, not the embeddings).
  - Fit $\hat\eta_T, \hat\eta_{T,S_s}$ on $\bigcup_{i \neq \text{test}} G_i$ with node-disjoint cross-fit.
  - Apply frozen $\hat\eta_{T,S_s}$ to pairs in $G_{\text{test}}$.
  - Compute $\bar\tau_s^{\text{OOD}}$.
- [ ] Paired bootstrap on $\hat\Delta_s = \hat\tau_s^{\text{OOD}} - \hat\tau_s^{\text{in-dist}}$ per sampler.
- [ ] Save to `results/schema_transfer.json`, plus transfer-gap plot `results/schema_transfer_gap.png`.

### 4.2 Leave-one-schema-out CV → `experiments/run_loso_cv.py`
- [ ] Iterate: hold out each schema once, report $\bar\tau_s^{\text{OOD}}$ — avoids cherry-picking the held-out set.
- [ ] Summary table: rows = samplers, cols = held-out schemas, cells = $\bar\tau_s^{\text{OOD}}$ ± CI.
- [ ] Save to `results/loso_summary.json`, LaTeX-ready table to `results/loso_table.tex`.

---

## Phase 5 — Ablations & Replication (Days 10–11)

### 5.1 Sampler hyperparameter sweeps → `experiments/run_ablations.py`
- [ ] node2vec: grid over walk length ∈ {20, 40, 80}, window ∈ {5, 10}, (p,q) ∈ {(1, 0.25), (1, 1), (1, 4)}.
- [ ] DeepWalk: walk length ∈ {20, 40, 80}, window ∈ {5, 10}.
- [ ] GraphSAGE: layers ∈ {1, 2, 3}, hidden dim ∈ {64, 128, 256}.
- [ ] For each config, re-estimate $\bar\tau_s$ on Schema A only (budget). Save to `results/sampler_ablations.json`.

### 5.2 Text-encoder swap → `experiments/run_text_swap.py`
- [ ] Re-run in-distribution τ̄_s with SBERT (384-d) replacing Word2Vec (200-d) as T.
- [ ] H: residual value should shrink if stronger text absorbs more signal.
- [ ] Save to `results/text_encoder_swap.json`; comparison plot `results/text_swap.png`.

### 5.3 Drug–gene replication → `experiments/run_drug_gene.py`
- [ ] Repeat Phase 3 pipeline with drug–gene indication edges as the outcome.
- [ ] Same four samplers, same schemas (where applicable — drug–gene edges define target, not schema).
- [ ] Save to `results/drug_gene_replication.json`.

### 5.4 Held-out disease class OOD → `experiments/run_disease_class_ood.py`
- [ ] Identify disease classes in PrimeKG (e.g., via MONDO hierarchy) — hold out one top-level class entirely.
- [ ] Re-run Phase 3 with this as an additional OOD axis (orthogonal to schema shift).
- [ ] Save to `results/disease_class_ood.json`.

### 5.5 Sample-size stability → `experiments/run_sample_size.py`
- [ ] Subsample edges at {10%, 25%, 50%, 100%}; report $\bar\tau_s$ stability curve.
- [ ] Plot to `results/sample_size_curves.png`.

---

## Phase 6 — Write-Up (Days 12–14)

- [ ] Initialize `latex/paper.tex` — NeurIPS-style template with abstract, intro, background, methods, experiments, results, discussion.
- [ ] Figures to generate/finalize:
  - Fig 1: Conceptual diagram (samplers-as-probes)
  - Fig 2: Per-sampler $\bar\tau_s$ bar chart with CIs (from 3.1)
  - Fig 3: Hop × sampler heatmap (from 3.3)
  - Fig 4: Schema transfer gap $\Delta_s$ (from 4.1)
  - Fig 5: LOSO CV table (from 4.2)
  - Fig 6: Synthetic validation (from 2.4)
  - Appendix: ablations, text swap, replication
- [ ] Results tables:
  - Table 1: $\bar\tau_s$ per (schema, sampler) with p-values (Holm-corrected)
  - Table 2: LOSO schema-transfer matrix
- [ ] Reproducibility: `make all` target that runs Phase 1–5 end-to-end. Seeds fixed throughout.
- [ ] Final polish: typography, citations (BibTeX from `references/`), page-limit trim.

---

## Nice-to-have (only if we have spare time)

- [ ] Relation-type stratification within Schema A (drug–gene edges vs drug–disease edges within the same schema)
- [ ] Per-node τ̄ analysis — are high-degree hub drugs driving most of the residual structural value?
- [ ] Compare DeepWalk skip-gram objective as implicit matrix factorization (ties into the 1905.01669-style NetMF framing from the email)
