# Utility functions for model evaluation
import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
import re
import torch

# Strip _partial to match partial_mesh_path and ground_truth_path pairs
def strip_partial_mesh_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name.endswith("_partial"):
        name = name[:-8]
    return name

# Build partial - ground truth mesh pairs
def build_partial_gt_mesh_pairs(partial_mesh_summary, mesh_names, split_key):
    pairs = []
    skipped = 0
    for m in partial_mesh_summary["meshes"]:
        base_name = m["base_name"]
        if base_name in mesh_names:
            pairs.append((m["partial"], m["ground_truth"]))
        else:
            skipped += 1
    print(f"Built {len(pairs)} (partial, ground_truth) pairs")
    print(f"Skipped {skipped} meshes not in {split_key}")
    return pairs

# Accuracy Metrics
def uniform_surface_sample(poly, n):
    # Triangulate the mesh
    poly = poly.triangulate().extract_surface(algorithm=None)
    verts = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]  # (T,3)
    # Calculate areas of each triangle
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    # Select triangles to sample from
    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(faces), size=n, p=probs)
    a = v0[tri_idx]; b = v1[tri_idx]; c = v2[tri_idx]
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    # Barycentric sampling to find random points inside each triangle
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return pts

# Calculate chamfer distance on partial-completed mesh vs original-ground truth mesh
def chamfer_distance(pm, gt, n_samples=20000):
    # Sample points across surface
    sp = uniform_surface_sample(pm, n_samples)
    sg = uniform_surface_sample(gt, n_samples)
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0].mean()
    d2 = t2.query(sp, k=1)[0].mean()
    return float(0.5*(d1 + d2)) # Return average distance (symmetric penalty)

# Load in best_cfg from shape_completion_grid_search.py (or manually enter)
def load_best_cfg_from_csv(csv_path, device):
    df = pd.read_csv(csv_path)
    best_row = df.loc[df["mean_cd"].idxmin()]
    # Parse latent_std whether stored as "tensor(0.4401)" or plain float
    raw_std = best_row["latent_std"]
    if isinstance(raw_std, str):
        raw_std = re.search(r"[-+eE0-9\.]+", raw_std).group()
    latent_std = torch.tensor(float(raw_std), device=device)
    # clamp2 may be None/NaN
    clamp2 = None if pd.isna(best_row["clamp2"]) else best_row["clamp2"]

    best_cfg = {"top_k":       int(best_row["top_k"]),
                "iters1":      int(best_row["iters1"]),
                "iters2":      int(best_row["iters2"]),
                "lr1":         float(best_row["lr1"]),
                "lr2":         float(best_row["lr2"]),
                "lambda1":     float(best_row["lambda1"]),
                "lambda2":     float(best_row["lambda2"]),
                "clamp1":      best_row["clamp1"],
                "clamp2":      best_row["clamp2"],
                "latent_std":  best_row["latent_std"],
                "sched_step":  int(best_row["sched_step"]),
                "sched_gamma1": float(best_row["sched_gamma1"]),
                "sched_gamma2": float(best_row["sched_gamma2"]),
                "batch_infer": int(best_row["batch_infer"]),
                "gridN":       int(best_row["gridN"])}

    print(f"\nLoaded best_cfg from CSV (min mean_cd = {best_row['mean_cd']:.4f}):")
    return best_cfg