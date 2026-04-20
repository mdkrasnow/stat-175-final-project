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

### 1.1 PrimeKG loader → `src/data/primekg_loader.py` ✅
- [x] Lift loader, add node-type + relation-type capture (old loader didn't have these)
- [x] Pickle cache at `data/cache/primekg_graph.pkl` (first run ~3 min, cached load <5 s)
- [x] Exposes `PrimeKG` dataclass + `load_primekg()`
- [x] Verified: 129,375 nodes, 4,049,642 edges (exact match)

### 1.2 Schema partitioning → `src/data/schemas.py` ✅
- [x] `Schema` dataclass, four schemas defined (A/B/C/D) with real PrimeKG node and relation vocab
- [x] `induce_schema_subgraph()`, `target_edges()`, `schema_inventory()`
- [x] Inventory written to `results/schema_inventory.json`
- [x] All four schemas pass acceptance (≥10k edges, ≥1k-node component):
  - A (drug–gene/protein–disease): 52,708 nodes / 469,337 edges / 9,293 indication targets / largest cc 31,963
  - B (+pathway): 55,224 / 511,983 / 9,293 / 34,554
  - C (disease–phenotype–gene): 60,062 / 556,154 / 83,741 associated-with targets / 36,986
  - D (drug–disease direct, held-out): 25,037 / 42,383 / 9,293 / 4,024

### 1.3 Link-prediction splits → `src/data/splits.py` ✅ (in-distribution per schema)
- [x] Node-disjoint 5-fold split: TEST = edges with BOTH endpoints in fold-k's node group; TRAIN = edges with NEITHER; edges with exactly one are dropped (leakage)
- [x] Degree-matched negative sampling with O(1) lookup (bucket by degree, tolerance 2×, up to 50 tries per positive)
- [x] Serialized to `data/splits/{schema}_fold{k}.npz`; summary in `results/splits_summary.json`
- [x] Verified: zero node leaks across all 20 folds; pos/neg = 1:1
- [ ] TODO (deferred to Phase 4): held-out-schema splits, held-out disease class splits

### 1.4 Text embeddings → `src/samplers/text_word2vec.py` ✅ (Word2Vec done; SBERT deferred)
- [x] Word2Vec skip-gram (dim=200, window=5, epochs=10) trained on 10.3M tokens from 129k node descriptions; 76,997-word vocab
- [x] Per-node embeddings via IDF-weighted mean of word vectors; 100% coverage
- [x] Saved to `data/embeddings/word2vec_200d.npz` (shape [129375, 200])
- [ ] SBERT baseline (`all-MiniLM-L6-v2`, 384-d) — deferred to Phase 5.2 (text-encoder-swap ablation)

### 1.5 Structural samplers → `src/samplers/` 🔄 (training in progress)
- [x] `random_walk.py` — shared DeepWalk + node2vec (BFS p=1,q=0.25 and DFS p=1,q=4) via gensim skip-gram (walk_length=40, num_walks=10, window=5, dim=128)
- [x] `graphsage.py` — 2-layer SAGEConv with unsupervised link-prediction loss; learned structural-only node features (no text leakage)
- [x] Batch training pipeline: `train_per_schema()` dispatches each sampler across all 4 schemas → `data/embeddings/{sampler}/{schema}.npz`
- [ ] Training running: DeepWalk/node2vec ETA ~30 min, GraphSAGE ~2-4 hours
- [ ] Acceptance: per-sampler link-prediction AUC > 0.7 on in-distribution holdout (measured during Phase 3.1)

---

## Phase 2 — DML Estimator (Days 3–4)

### 2.1 Feature construction → `src/estimation/features.py` ✅
- [x] `hadamard_features()`, `build_pair_features()`, `stack_pos_neg()`, `load_embedding()`

### 2.2 Cross-fit DML → `src/estimation/dml.py` ✅
- [x] Manual cross-fit loop (cleaner per-pair τ̂ than doubleml); XGBoost nuisances
- [x] Node-disjoint folds reused from Phase 1.3 via `_pool_folds()` in Phase 3.1 runner
- [x] Returns `DMLResult` with per-pair τ̂, per-fold means, AUC of η̂_T and η̂_TS

### 2.3 Cluster bootstrap + inference → `src/estimation/inference.py` ✅
- [x] `cluster_bootstrap_ci()`, `permutation_null()`, `holm_correction()`

### 2.4 Synthetic-graph validation → `experiments/validate_dml_synthetic.py` ✅ (with caveat)
- [x] Three-regime test + acceptance suite
- [x] Regime 1 (struct=noise): τ̄≈0, CI covers 0, perm p=0.45 ✅
- [x] Regime 3 (text dominant): τ̄=0 exactly ✅
- [⚠️] Regime 2 (struct informative, text noisy, N=645): AUC_T=0.62 → AUC_TS=0.72 clearly shows structure helps ranking, perm p=0.01 correctly detects the effect; but τ̄=-0.018 has the wrong sign due to finite-sample calibration drift between two independently-fit XGBoost classifiers
- [x] **Known limitation**: τ̄ point estimate is sensitive to classifier calibration at small N; report permutation p-value as primary significance test and AUC gap as complementary magnitude metric. At PrimeKG scale (N = 5k-80k per schema) this should be less of an issue.

---

## Phase 3 — In-Distribution Estimation (Days 5–7)

### 3.1 Per-sampler τ̄_s → `experiments/run_in_dist_tau.py` ✅ (code ready, awaiting embeddings)
- [x] Pools test pairs across folds, builds features, runs cross_fit_dml per sampler
- [x] Cluster-bootstrap CI (B=500), permutation null (B=100), Holm correction per-schema
- [x] Output: `results/in_dist_tau.json` with schema × sampler × (τ̄, CI, AUC, p_raw, p_holm)
- [ ] Execution blocked on Phase 1.5 embeddings completing

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
