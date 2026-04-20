"""Data pipeline for Query-Conditioned Transition Retrieval (QCTR).

Builds training data from STaRK-PrimeKG shortest-path trajectories:
positive transitions along gold paths, plus random and hard negatives.
"""

import json
import os
import random

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# 1. Edge type lookup
# ---------------------------------------------------------------------------

def build_edge_type_lookup(skb) -> dict:
    """Build a dict mapping (src, dst) -> edge_type_id from the SKB.

    Since the graph is undirected, both (src, dst) and (dst, src) are stored.
    """
    edge_index = skb.edge_index  # [2, E] tensor
    edge_types = skb.edge_types  # [E] tensor

    sources = edge_index[0].numpy()
    targets = edge_index[1].numpy()
    types = edge_types.numpy()

    lookup = {}
    for i in range(len(types)):
        s, t, etype = int(sources[i]), int(targets[i]), int(types[i])
        lookup[(s, t)] = etype
        lookup[(t, s)] = etype

    print(f"[build_edge_type_lookup] Built lookup with {len(lookup)} entries "
          f"from {len(types)} edges")
    return lookup


# ---------------------------------------------------------------------------
# 2. Trajectory extraction
# ---------------------------------------------------------------------------

def extract_trajectories(graph, stratification, edge_type_lookup,
                         max_paths_per_query=5) -> list[dict]:
    """Extract step-level transitions from shortest paths between query and gold nodes.

    Args:
        graph: StarkGraphWrapper instance.
        stratification: dict from hop_stratification JSON (keys like "1-hop").
        edge_type_lookup: dict from build_edge_type_lookup.
        max_paths_per_query: max total paths collected per QA item.

    Returns:
        List of transition dicts with keys: query_idx, query_node,
        current_node, next_node, edge_type, hop_count, step_position, path_id.
    """
    transitions = []
    path_id_counter = 0
    items_processed = 0

    for hop_key, items in stratification.items():
        hop_count = int(hop_key.split("-")[0].rstrip("+"))

        for item in items:
            query_idx = item["query_id"]
            query_node = item["query_node"]
            gold_ids = item["gold_ids"]

            # Skip 0-hop (query_node is the answer)
            paths_collected = 0

            for gold_id in gold_ids:
                if gold_id == query_node:
                    continue
                if paths_collected >= max_paths_per_query:
                    break

                try:
                    path = nx.shortest_path(
                        graph.graph, query_node, gold_id
                    )

                    for step in range(len(path) - 1):
                        v_t = path[step]
                        v_next = path[step + 1]
                        etype = edge_type_lookup.get((v_t, v_next), -1)

                        transitions.append({
                            "query_idx": query_idx,
                            "query_node": query_node,
                            "current_node": v_t,
                            "next_node": v_next,
                            "edge_type": etype,
                            "hop_count": hop_count,
                            "step_position": step,
                            "path_id": path_id_counter,
                        })

                    path_id_counter += 1
                    paths_collected += 1

                except nx.NetworkXNoPath:
                    continue

            items_processed += 1
            if items_processed % 1000 == 0:
                print(f"[extract_trajectories] Processed {items_processed} QA items, "
                      f"{len(transitions)} transitions so far")

    print(f"[extract_trajectories] Done: {items_processed} QA items, "
          f"{len(transitions)} transitions, {path_id_counter} paths")
    return transitions


# ---------------------------------------------------------------------------
# 3. Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(graph, trajectories, shared_index,
                     edge_type_lookup=None,
                     num_random=5, num_hard=3) -> list[dict]:
    """Sample random and hard negative transitions for each positive.

    Args:
        graph: StarkGraphWrapper instance.
        trajectories: list of positive transition dicts from extract_trajectories.
        shared_index: SharedIndex instance with precomputed embeddings.
        edge_type_lookup: dict mapping (src, dst) -> edge_type_id. If provided,
            negatives get their real edge type instead of -1.
        num_random: number of random neighbor negatives per positive.
        num_hard: number of hard (high cosine to query) negatives per positive.

    Returns:
        Combined list of positive (label=1) and negative (label=0) samples.
    """
    all_samples = []
    query_emb_cache = {}

    for i, pos in enumerate(trajectories):
        # Add positive with label
        pos_sample = dict(pos)
        pos_sample["label"] = 1
        pos_sample["neg_type"] = None
        all_samples.append(pos_sample)

        current_node = pos["current_node"]
        next_node = pos["next_node"]
        query_idx = pos["query_idx"]

        # Get neighbors of current node, excluding the positive next_node
        neighbors = list(graph.graph.neighbors(current_node))
        neg_candidates = [n for n in neighbors if n != next_node]

        if not neg_candidates:
            continue

        # --- Random negatives ---
        if len(neg_candidates) >= num_random:
            rand_negs = random.sample(neg_candidates, num_random)
        else:
            rand_negs = random.choices(neg_candidates, k=num_random)

        for neg_node in rand_negs:
            neg_sample = dict(pos)
            neg_sample["next_node"] = neg_node
            neg_sample["edge_type"] = (
                edge_type_lookup.get((current_node, neg_node), -1)
                if edge_type_lookup else -1
            )
            neg_sample["label"] = 0
            neg_sample["neg_type"] = "random"
            all_samples.append(neg_sample)

        # --- Hard negatives (highest cosine to query) ---
        if query_idx not in query_emb_cache:
            # We'll cache the query node embedding as proxy; the actual query
            # text encoding happens in build_feature_vectors. Here use node emb.
            try:
                query_emb_cache[query_idx] = shared_index.get_node_embedding(
                    pos["query_node"]
                )
            except KeyError:
                query_emb_cache[query_idx] = None

        q_emb = query_emb_cache[query_idx]
        if q_emb is not None and len(neg_candidates) > 0:
            # Score all negative candidates by cosine similarity to query
            try:
                cand_embs = shared_index.get_node_embeddings(neg_candidates)
                scores = cand_embs @ q_emb  # dot product (normalized embeddings)
                # Get top-k indices
                k = min(num_hard, len(neg_candidates))
                top_indices = np.argpartition(scores, -k)[-k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                hard_neg_nodes = [neg_candidates[idx] for idx in top_indices]
            except (KeyError, IndexError):
                hard_neg_nodes = neg_candidates[:num_hard]

            for neg_node in hard_neg_nodes:
                neg_sample = dict(pos)
                neg_sample["next_node"] = neg_node
                neg_sample["edge_type"] = (
                    edge_type_lookup.get((current_node, neg_node), -1)
                    if edge_type_lookup else -1
                )
                neg_sample["label"] = 0
                neg_sample["neg_type"] = "hard"
                all_samples.append(neg_sample)

        if (i + 1) % 5000 == 0:
            print(f"[sample_negatives] Processed {i + 1}/{len(trajectories)} "
                  f"transitions, {len(all_samples)} total samples")

    print(f"[sample_negatives] Done: {len(trajectories)} positives -> "
          f"{len(all_samples)} total samples")
    return all_samples


# ---------------------------------------------------------------------------
# 4. Feature vector construction
# ---------------------------------------------------------------------------

def build_feature_vectors(samples, shared_index, qa_dataset) -> dict:
    """Convert sample dicts into numpy arrays for training.

    Feature vector per sample: [query_emb(384), current_emb(384),
    candidate_emb(384), cos_q_curr, cos_q_cand, cos_curr_cand] = 1155 dims.

    Args:
        samples: list of sample dicts (from sample_negatives).
        shared_index: SharedIndex instance.
        qa_dataset: STaRK QA dataset (indexable, qa_dataset[i][0] is query text).

    Returns:
        dict with keys: features, edge_types, labels, metadata.
    """
    n = len(samples)
    dim = shared_index.embeddings.shape[1]  # 384
    feature_dim = dim * 3 + 3

    features = np.zeros((n, feature_dim), dtype=np.float32)
    edge_types = np.zeros(n, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int32)
    metadata = []

    # Cache query embeddings by query_idx to avoid re-encoding
    query_emb_cache = {}

    for i, sample in enumerate(samples):
        qidx = sample["query_idx"]

        # Get or compute query embedding
        if qidx not in query_emb_cache:
            query_text = qa_dataset[qidx][0]
            query_emb_cache[qidx] = shared_index.encode_query(query_text).flatten()
        q_emb = query_emb_cache[qidx]

        curr_emb = shared_index.get_node_embedding(sample["current_node"])
        cand_emb = shared_index.get_node_embedding(sample["next_node"])

        # Cosine similarities (embeddings are L2-normalized, so dot product)
        cos_q_curr = float(np.dot(q_emb, curr_emb))
        cos_q_cand = float(np.dot(q_emb, cand_emb))
        cos_curr_cand = float(np.dot(curr_emb, cand_emb))

        features[i] = np.concatenate([
            q_emb, curr_emb, cand_emb,
            [cos_q_curr, cos_q_cand, cos_curr_cand]
        ])
        edge_types[i] = sample["edge_type"]
        labels[i] = sample["label"]
        metadata.append({
            "query_idx": qidx,
            "hop_count": sample["hop_count"],
            "step_position": sample["step_position"],
            "neg_type": sample.get("neg_type"),
        })

        if (i + 1) % 10000 == 0:
            print(f"[build_feature_vectors] Processed {i + 1}/{n} samples")

    print(f"[build_feature_vectors] Done: {n} samples, "
          f"feature dim {feature_dim}, "
          f"{len(query_emb_cache)} unique queries cached")

    return {
        "features": features,
        "edge_types": edge_types,
        "labels": labels,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# 5. Train/val/test split (by query, not by sample)
# ---------------------------------------------------------------------------

def split_dataset(features_dict, train_frac=0.7, val_frac=0.15,
                  seed=42) -> dict:
    """Split dataset by query_idx so no query leaks across splits.

    Args:
        features_dict: dict from build_feature_vectors.
        train_frac: fraction of queries for training.
        val_frac: fraction of queries for validation.
        seed: random seed for reproducibility.

    Returns:
        dict with keys train, val, test, each containing
        features, edge_types, labels, metadata.
    """
    metadata = features_dict["metadata"]

    # Gather unique query indices
    query_indices = sorted(set(m["query_idx"] for m in metadata))
    rng = random.Random(seed)
    rng.shuffle(query_indices)

    n_train = int(len(query_indices) * train_frac)
    n_val = int(len(query_indices) * val_frac)

    train_queries = set(query_indices[:n_train])
    val_queries = set(query_indices[n_train:n_train + n_val])
    test_queries = set(query_indices[n_train + n_val:])

    splits = {}
    for split_name, split_qset in [("train", train_queries),
                                    ("val", val_queries),
                                    ("test", test_queries)]:
        mask = np.array([m["query_idx"] in split_qset for m in metadata])
        indices = np.where(mask)[0]

        split_meta = [metadata[j] for j in indices]
        splits[split_name] = {
            "features": features_dict["features"][indices],
            "edge_types": features_dict["edge_types"][indices],
            "labels": features_dict["labels"][indices],
            "metadata": split_meta,
        }

    # Print split summary
    for split_name in ("train", "val", "test"):
        s = splits[split_name]
        n_pos = int(s["labels"].sum())
        n_neg = len(s["labels"]) - n_pos
        total = len(s["labels"])
        print(f"[split_dataset] {split_name}: {total} samples "
              f"({n_pos} pos, {n_neg} neg, "
              f"balance {n_pos / total:.2%} pos)" if total > 0
              else f"[split_dataset] {split_name}: 0 samples")

        # Per-hop breakdown
        hop_counts = {}
        for m in s["metadata"]:
            h = m["hop_count"]
            hop_counts[h] = hop_counts.get(h, 0) + 1
        for h in sorted(hop_counts):
            print(f"  hop {h}: {hop_counts[h]} samples")

    return splits


# ---------------------------------------------------------------------------
# 6. Save to disk
# ---------------------------------------------------------------------------

def save_dataset(dataset_dict, output_dir) -> None:
    """Save each split as .npz + metadata JSON.

    Files created per split:
        {split}_features.npz  — features, edge_types, labels
        {split}_metadata.json — list of metadata dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in dataset_dict.items():
        npz_path = os.path.join(output_dir, f"{split_name}_features.npz")
        np.savez(
            npz_path,
            features=split_data["features"],
            edge_types=split_data["edge_types"],
            labels=split_data["labels"],
        )

        meta_path = os.path.join(output_dir, f"{split_name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(split_data["metadata"], f)

        print(f"[save_dataset] Saved {split_name} -> {npz_path} "
              f"({split_data['features'].shape[0]} samples)")

    print(f"[save_dataset] All splits saved to {output_dir}")
