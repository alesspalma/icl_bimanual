"""
Recompute max_distance.json from the actual RICL demo pool.

The original assets/max_distance.json was computed over the DROID dataset
(real-robot third-person RGB).  This script computes it over your own
RLBench simulation demo pool so that the lambda interpolation is calibrated
to the actual embedding distances in your data.

Method
------
For each task/arm pool we load all top_image_embeddings, then compute pairwise
L2 distances across all frames across all demos.  We report:
  - max   : the absolute largest pairwise distance (hard ceiling)
  - p99   : 99th percentile (recommended — avoids extreme outliers)
  - p95   : 95th percentile (more conservative normalisation)

The script writes the chosen value (p99 by default) back to
  ricl_openpi/assets/max_distance.json

Usage (openpi venv):
    source ricl_openpi/.venv/bin/activate
    python compute_max_dist.py [--stat max|p99|p95] [--tasks task1 task2 ...]
"""

import argparse
import json
import os
import random

import numpy as np

DEMOS_ROOT = "ricl_openpi/preprocessing/collected_demos"
ASSETS_PATH = "ricl_openpi/assets/max_distance.json"


def load_all_embeddings(demos_root: str, tasks: list[str] | None = None,
                        max_frames_per_demo: int = 50) -> np.ndarray:
    """Return (N, EMBED_DIM) array of top_image_embeddings from the whole pool."""
    all_embs = []
    for folder in sorted(os.listdir(demos_root)):
        folder_path = os.path.join(demos_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if tasks is not None:
            if not any(t in folder for t in tasks):
                continue
        for demo in sorted(os.listdir(folder_path))[:30]:
            npz_path = os.path.join(folder_path, demo, "processed_demo.npz")
            if not os.path.isfile(npz_path):
                continue
            data = np.load(npz_path)
            emb = data["top_image_embeddings"]  # (T, EMBED_DIM)
            # Subsample to avoid memory explosion on long demos
            if emb.shape[0] > max_frames_per_demo:
                idx = np.linspace(0, emb.shape[0] - 1, max_frames_per_demo, dtype=int)
                emb = emb[idx]
            all_embs.append(emb)
            print(f"  loaded {folder}/{demo}: {emb.shape}")
    if not all_embs:
        raise RuntimeError(f"No processed_demo.npz found under {demos_root}")
    return np.concatenate(all_embs, axis=0)  # (N, EMBED_DIM)


def compute_pairwise_max(embs: np.ndarray, sample_size: int = 4000,
                         stat: str = "p99") -> dict:
    """
    Estimate the distribution of pairwise L2 distances.

    Computing all N^2 pairs is O(N^2) — prohibitively slow for large N.
    We instead sample a random subset and compute all pairwise distances
    within that subset, which gives a tight estimate.
    """
    N = embs.shape[0]
    if N > sample_size:
        print(f"  Sampling {sample_size} / {N} frames for pairwise computation...")
        idx = np.random.choice(N, sample_size, replace=False)
        embs = embs[idx]

    print(f"  Computing pairwise distances for {embs.shape[0]} frames...")
    # Vectorised ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq_norms = np.sum(embs ** 2, axis=1, keepdims=True)  # (N, 1)
    dot = embs @ embs.T                                    # (N, N)
    sq_dists = np.clip(sq_norms + sq_norms.T - 2 * dot, 0, None)
    # Only upper triangle (exclude diagonal / duplicates)
    triu = np.triu_indices(embs.shape[0], k=1)
    dists = np.sqrt(sq_dists[triu])

    result = {
        "max":  float(np.max(dists)),
        "p99":  float(np.percentile(dists, 99)),
        "p95":  float(np.percentile(dists, 95)),
        "p90":  float(np.percentile(dists, 90)),
        "mean": float(np.mean(dists)),
        "std":  float(np.std(dists)),
        "n_pairs": int(dists.shape[0]),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", choices=["max", "p99", "p95", "p90"],
                        default="p99",
                        help="Which statistic to write as the normalisation constant (default: p99)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Restrict to specific tasks (substring match). Default: all tasks.")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Max frames to draw for pairwise computation (default: 4000)")
    parser.add_argument("--max_frames_per_demo", type=int, default=50,
                        help="Max frames to keep per demo (default: 50, subsampled uniformly)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print result but do not overwrite max_distance.json")
    args = parser.parse_args()

    # Change to project root so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Loading embeddings from: {DEMOS_ROOT}")
    if args.tasks:
        print(f"  Restricting to tasks: {args.tasks}")

    np.random.seed(42)
    embs = load_all_embeddings(DEMOS_ROOT, tasks=args.tasks,
                               max_frames_per_demo=args.max_frames_per_demo)
    print(f"\nTotal frames loaded: {embs.shape[0]}, embed dim: {embs.shape[1]}")

    stats = compute_pairwise_max(embs, sample_size=args.sample_size, stat=args.stat)
    print("\nPairwise L2 distance statistics:")
    for k, v in stats.items():
        print(f"  {k:8s}: {v:.4f}")

    chosen = stats[args.stat]
    print(f"\nChosen statistic ({args.stat}): {chosen:.4f}")

    # Read current value
    with open(ASSETS_PATH) as f:
        current = json.load(f)
    print(f"Current max_distance.json value: {current['distances']['max']:.4f}")

    if not args.dry_run:
        current["distances"]["max"] = chosen
        current["distances"]["stats"] = stats
        current["distances"]["stat_used"] = args.stat
        with open(ASSETS_PATH, "w") as f:
            json.dump(current, f, indent=2)
        print(f"\nWritten to {ASSETS_PATH}: max = {chosen:.4f}")
    else:
        print("\n[dry-run] Not writing.")


if __name__ == "__main__":
    main()
