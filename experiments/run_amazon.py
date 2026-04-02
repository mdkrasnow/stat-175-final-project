"""STaRK-Amazon replication — memory-optimized.

Problem: SKB + NetworkX graph + embeddings exceeds laptop RAM.
Solution: Extract what we need from SKB, delete it, then proceed.
"""

import sys
import os
import time
import json
import random
import gc
import pickle

import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATASET = "amazon"
INDEX_SIZE = 100000
MAX_PER_BIN = 200
CACHE_DIR = os.path.join(RESULTS_DIR, "amazon_cache")


def step1_extract():
    """Extract edges, QA, and candidate texts from SKB, then free it."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    edge_path = os.path.join(CACHE_DIR, "edges.npy")
    qa_path = os.path.join(CACHE_DIR, "qa.pkl")

    if os.path.exists(edge_path) and os.path.exists(qa_path):
        print("Cache found, skipping extraction.")
        return

    from stark_qa import load_skb, load_qa

    print("Loading SKB...")
    t0 = time.time()
    skb = load_skb(DATASET, download_processed=True)
    num_nodes = skb.num_nodes()
    print(f"SKB: {num_nodes} nodes ({time.time()-t0:.0f}s)")

    # Extract edges as numpy array
    print("Extracting edges...")
    edges = skb.edge_index.numpy().copy()
    np.save(edge_path, edges)
    print(f"Edges saved: {edges.shape}")

    # Extract candidate IDs
    candidate_ids = skb.candidate_ids
    if not isinstance(candidate_ids, list):
        candidate_ids = candidate_ids.tolist()

    # Save graph summary
    summary = {"dataset": DATASET, "num_nodes": num_nodes, "num_edges": edges.shape[1]}

    # Load QA
    print("Loading QA...")
    qa = load_qa(DATASET)
    qa_data = []
    all_gold = set()
    for i in range(len(qa)):
        q, qid, gids, meta = qa[i]
        qa_data.append({"query": q, "query_id": qid, "gold_ids": gids})
        all_gold.update(gids)

    # Sample 100K candidates (include all gold for evaluated queries)
    # We'll include gold nodes later per-bin, for now just save candidates
    random.seed(42)
    sample = random.sample(candidate_ids, min(INDEX_SIZE, len(candidate_ids)))

    # Extract texts for sampled candidates
    print(f"Extracting texts for {len(sample)} sampled candidates...")
    sample_texts = {}
    for i, nid in enumerate(sample):
        try:
            sample_texts[nid] = skb.get_doc_info(nid, add_rel=False)[:200]
        except Exception:
            sample_texts[nid] = ""
        if (i+1) % 25000 == 0:
            print(f"  ...{i+1}/{len(sample)}")

    # Also extract texts for all gold IDs (might need them for index)
    print(f"Extracting texts for {len(all_gold)} gold nodes...")
    for nid in all_gold:
        if nid not in sample_texts:
            try:
                sample_texts[nid] = skb.get_doc_info(nid, add_rel=False)[:200]
            except Exception:
                sample_texts[nid] = ""

    with open(qa_path, "wb") as f:
        pickle.dump({
            "qa_data": qa_data,
            "candidate_ids": candidate_ids,
            "sample_texts": sample_texts,
            "num_nodes": num_nodes,
            "summary": summary,
        }, f)
    print(f"Cached {len(sample_texts)} node texts + {len(qa_data)} QA pairs")

    # Free SKB
    del skb, qa, edges
    gc.collect()
    print("SKB freed from memory.")


def step2_experiment():
    """Build graph from cached edges, encode, stratify, and run experiment."""
    import networkx as nx
    import faiss
    from sentence_transformers import SentenceTransformer
    from src.evaluation.metrics import hit_at_k, mrr

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load cache
    print("\nLoading cached data...")
    edges = np.load(os.path.join(CACHE_DIR, "edges.npy"))
    with open(os.path.join(CACHE_DIR, "qa.pkl"), "rb") as f:
        cache = pickle.load(f)

    qa_data = cache["qa_data"]
    sample_texts = cache["sample_texts"]
    num_nodes = cache["num_nodes"]
    summary = cache["summary"]

    # Build graph
    print(f"Building graph ({num_nodes} nodes)...")
    t0 = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edges[0].tolist(), edges[1].tolist()))
    del edges; gc.collect()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time()-t0:.0f}s)")

    # Graph properties
    degrees = [d for _, d in G.degree()]
    summary.update({
        "density": nx.density(G),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(max(degrees)),
    })
    print(f"Mean degree: {summary['mean_degree']:.2f}, Median: {summary['median_degree']:.1f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{DATASET}_graph_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    del degrees; gc.collect()

    # Build FAISS index from cached texts
    idx_nodes = sorted(sample_texts.keys())
    idx_texts = [sample_texts[nid] for nid in idx_nodes]
    nid_to_pos = {nid: j for j, nid in enumerate(idx_nodes)}
    del sample_texts; gc.collect()

    print(f"\nEncoding {len(idx_texts)} node texts...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.time()
    idx_emb = encoder.encode(idx_texts, show_progress_bar=True,
                              normalize_embeddings=True, batch_size=64).astype(np.float32)
    del idx_texts; gc.collect()
    print(f"Encoded in {time.time()-t0:.0f}s")

    idx_index = faiss.IndexFlatIP(idx_emb.shape[1])
    idx_index.add(idx_emb)
    print(f"FAISS index: {idx_index.ntotal} vectors")

    # Encode queries
    print(f"Encoding {len(qa_data)} queries...")
    queries = [item["query"] for item in qa_data]
    query_embs = encoder.encode(queries, show_progress_bar=True,
                                 normalize_embeddings=True, batch_size=64).astype(np.float32)

    # Stratify
    print("\n--- Stratification ---")
    max_hops = 6
    bins = {f"{h}-hop": [] for h in range(1, max_hops)}
    bins[f"{max_hops}+-hop"] = []
    skipped = 0

    for i, item in enumerate(qa_data):
        q_emb = query_embs[i:i+1]
        _, top_idx = idx_index.search(q_emb, 1)
        query_node = idx_nodes[top_idx[0][0]]

        min_dist = float("inf")
        for gid in item["gold_ids"]:
            if gid >= num_nodes: continue
            try:
                dist = nx.shortest_path_length(G, query_node, gid)
                min_dist = min(min_dist, dist)
            except (nx.NetworkXNoPath, nx.NodeNotFound): continue

        if min_dist == float("inf"):
            skipped += 1; continue

        hop = max(1, int(min_dist))
        key = f"{hop}-hop" if hop < max_hops else f"{max_hops}+-hop"
        bins[key].append({"index": i, "query_id": item["query_id"],
                          "gold_ids": [int(g) for g in item["gold_ids"]],
                          "hop_count": hop, "query_node": query_node})
        if (i+1) % 2000 == 0:
            print(f"  ...{i+1}/{len(qa_data)}")

    print(f"\nHop distribution (skipped {skipped}):")
    for k, v in bins.items():
        print(f"  {k}: {len(v)}")

    with open(os.path.join(RESULTS_DIR, f"{DATASET}_hop_stratification.json"), "w") as f:
        json.dump(bins, f, indent=2)

    # Run experiment
    print("\n--- Retrieval Experiment (200/bin) ---")
    ks = [1, 5, 10]
    all_results = {}

    for bin_name, items in bins.items():
        if not items: continue
        sub = items[:MAX_PER_BIN]
        print(f"\n{'='*60}\n  {bin_name} (n={len(sub)})\n{'='*60}")
        br = {}

        # Node-centric
        hs = {k: [] for k in ks}; ms = []
        for item in sub:
            q = query_embs[item["index"]:item["index"]+1]
            _, ii = idx_index.search(q, max(ks))
            pred = [idx_nodes[x] for x in ii[0]]
            for k in ks: hs[k].append(hit_at_k(pred, item["gold_ids"], k))
            ms.append(mrr(pred, item["gold_ids"]))
        br["node_centric"] = {f"Hit@{k}": float(np.mean(hs[k])) for k in ks}
        br["node_centric"]["MRR"] = float(np.mean(ms))

        # Path-centric
        hs = {k: [] for k in ks}; ms = []
        for item in sub:
            q = query_embs[item["index"]:item["index"]+1]
            _, ii = idx_index.search(q, 5)
            seeds = [idx_nodes[x] for x in ii[0]]
            pn = set(seeds)
            for si in range(len(seeds)):
                for sj in range(si+1, len(seeds)):
                    try:
                        p = nx.shortest_path(G, seeds[si], seeds[sj])
                        if len(p) <= 5: pn.update(p)
                    except: pass
            scored = [(n, float(idx_emb[nid_to_pos[n]] @ q.T)) for n in pn if n in nid_to_pos]
            scored.sort(key=lambda x: x[1], reverse=True)
            pred = [n for n,_ in scored[:max(ks)]]
            for s in seeds:
                if s not in set(pred) and len(pred)<max(ks): pred.append(s)
            for k in ks: hs[k].append(hit_at_k(pred, item["gold_ids"], k))
            ms.append(mrr(pred, item["gold_ids"]))
        br["path_centric"] = {f"Hit@{k}": float(np.mean(hs[k])) for k in ks}
        br["path_centric"]["MRR"] = float(np.mean(ms))

        # Subgraph-centric
        hs = {k: [] for k in ks}; ms = []
        for item in sub:
            q = query_embs[item["index"]:item["index"]+1]
            _, ii = idx_index.search(q, 3)
            seeds = [idx_nodes[x] for x in ii[0]]
            exp = set(seeds)
            for s in seeds:
                for nb in G.neighbors(s):
                    exp.add(nb)
                    if len(exp)>200: break
                if len(exp)>200: break
            scored = [(n, float(idx_emb[nid_to_pos[n]] @ q.T)) for n in exp if n in nid_to_pos]
            scored.sort(key=lambda x: x[1], reverse=True)
            pred = [n for n,_ in scored[:max(ks)]]
            for k in ks: hs[k].append(hit_at_k(pred, item["gold_ids"], k))
            ms.append(mrr(pred, item["gold_ids"]))
        br["subgraph_centric"] = {f"Hit@{k}": float(np.mean(hs[k])) for k in ks}
        br["subgraph_centric"]["MRR"] = float(np.mean(ms))

        # Hybrid
        hs = {k: [] for k in ks}; ms = []
        for item in sub:
            q = query_embs[item["index"]:item["index"]+1]
            ts, ti = idx_index.search(q, 50)
            tm = {idx_nodes[x]: float(s) for s,x in zip(ts[0],ti[0])}
            _, si = idx_index.search(q, 3)
            seeds = [idx_nodes[x] for x in si[0]]
            exp = set(seeds)
            for s in seeds:
                for nb in G.neighbors(s):
                    exp.add(nb)
                    if len(exp)>200: break
                if len(exp)>200: break
            sm = {n:(1.0 if n in set(seeds) else 0.5 if any(G.has_edge(n,s) for s in seeds) else 0.25) for n in exp}
            comb = [(n, 0.5*tm.get(n,0)+0.5*sm.get(n,0)) for n in set(tm)|exp]
            comb.sort(key=lambda x: x[1], reverse=True)
            pred = [n for n,_ in comb[:max(ks)]]
            for k in ks: hs[k].append(hit_at_k(pred, item["gold_ids"], k))
            ms.append(mrr(pred, item["gold_ids"]))
        br["hybrid_mor"] = {f"Hit@{k}": float(np.mean(hs[k])) for k in ks}
        br["hybrid_mor"]["MRR"] = float(np.mean(ms))

        all_results[bin_name] = br
        print(f"{'Strategy':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
        print("-" * 52)
        for rn, r in br.items():
            print(f"{rn:<20} {r['Hit@1']:>8.4f} {r['Hit@5']:>8.4f} {r['Hit@10']:>8.4f} {r['MRR']:>8.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{DATASET}_experiment_{ts}.json")
    with open(path, "w") as f:
        json.dump({"dataset": DATASET, "timestamp": ts, "results": all_results}, f, indent=2)
    print(f"\nSaved to {path}")
    print(f"\n{'='*60}\n  Complete!\n{'='*60}")


def main():
    step1_extract()
    gc.collect()
    step2_experiment()


if __name__ == "__main__":
    main()
