# classify_vertebrae_v5.py
# ========================
# Version 5: Fixed distance-based classification and full ablation framework.
#
# Fixes from v4:
#   - Distance-based species matching now uses consistent label parsing
#     (v4 compared parse_taxonomy_from_filename species vs extract_species_prefix,
#      which never matched due to different formats)
#   - Added genus/family/position top-1 and top-5 for distance methods
#   - Per-specimen predictions include full top-5 taxonomy breakdown
#
# Behavior modeled after classify_vertebrae_new.py with v4 enhancements:
#   - Ablation configs (A/B/C/D) for incremental comparison
#   - Supervised classifiers (KNN, SVM, RF, MLP, LR)
#   - NCA metric learning
#   - CLI argument parsing (--run-dir, --checkpoint)
#   - Confusion matrices, comparison charts, impact analysis
#
# Configurations:
#   A) Baseline           — raw latent codes, distance-based retrieval only
#   B) + Metric Learning  — NCA-transformed latents, distance-based retrieval
#   C) + Classifiers      — raw latents, supervised classifiers (no metric learning)
#   D) + ML + Classifiers — NCA-transformed latents + supervised classifiers (full pipeline)
#
# Output structure:
#   results/ablation_YYYYMMDD_HHMMSS/
#   ├── README.md
#   ├── summary_statistics.md
#   ├── comparison_metrics.csv
#   ├── comparison_charts/
#   ├── A_baseline/
#   │   ├── README.md
#   │   ├── predictions.csv
#   │   ├── detailed_metrics.csv
#   │   ├── distance_metrics.csv
#   │   └── confusion_matrices/
#   ├── B_metric_learning/
#   ├── C_classifiers/
#   └── D_full_pipeline/
# ========================

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
import argparse
import glob as glob_module
import torch
import numpy as np
import pandas as pd
import json
import datetime
import time as time_module
import pyvista as pv
import pymskt.mesh.meshes as meshes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs, create_mesh
import vtk
import re
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

from NSM.helper_funcs import (
    NumpyTransform, load_config, load_model_and_latents,
    convert_ply_to_vtk, get_sdfs, fixed_point_coords,
    safe_load_mesh_scalars,
)
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, find_similar, find_similar_cos
from .classifiers import (
    train_classifiers, predict_classifiers,
    train_position_regressors, predict_position_regressors,
)
from .metric_learning import LatentMetricLearner
from .taxonomy import parse_taxonomy_from_filename
from .evaluation import (
    calculate_metrics,
    metrics_to_dataframe,
    generate_hierarchical_confusion_matrices,
    generate_position_confusion_matrix,
    calculate_regression_metrics,
)
from .spine_position import SpinePositionMapper
from .interpretation import run_lr_interpretation
from .run_utils import create_run_directory, write_run_manifest

# Monkey-patch pymskt
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)


# ======================================================================
# CLI Arguments
# ======================================================================
def _detect_latest_checkpoint(run_dir: str) -> str:
    """Find the highest-numbered checkpoint in run_dir/model/."""
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No model directory found at {model_dir}")
    pth_files = glob_module.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")
    epochs = []
    for p in pth_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit():
            epochs.append(int(stem))
    if not epochs:
        raise ValueError(f"No numerically-named checkpoints in {model_dir}")
    return str(max(epochs))


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for vertebrae classification using NSM latent codes."
    )
    parser.add_argument(
        "--run-dir", type=str, default="run_v57",
        help="Training run directory to load model/latent codes from (default: run_v57)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint epoch to use, e.g. '2000'. If omitted, uses the latest available checkpoint.",
    )
    args = parser.parse_args()


    # ======================================================================
    # Ablation Configuration Definitions
    # ======================================================================

    ABLATION_CONFIGS = OrderedDict([
        ("A_baseline", {
            "label": "Baseline (Distance Only)",
            "description": "Raw latent codes from NSM, distance-based retrieval only (cosine + euclidean). No metric learning, no supervised classifiers.",
            "use_metric_learning": False,
            "use_classifiers": False,
        }),
        ("B_metric_learning", {
            "label": "Baseline + Metric Learning",
            "description": "NCA-transformed latent codes, distance-based retrieval only. Metric learning reshapes the latent space to cluster by species before distance matching.",
            "use_metric_learning": True,
            "use_classifiers": False,
        }),
        ("C_classifiers", {
            "label": "Baseline + Classifiers",
            "description": "Raw latent codes with supervised classifiers (KNN, SVM, RF, MLP, LR) trained on top. No metric learning transformation.",
            "use_metric_learning": False,
            "use_classifiers": True,
        }),
        ("D_full_pipeline", {
            "label": "Baseline + ML + Classifiers",
            "description": "Full pipeline: NCA metric learning followed by supervised classifiers. This is equivalent to the v3 approach.",
            "use_metric_learning": True,
            "use_classifiers": True,
        }),
    ])


    # ======================================================================
    # Configuration
    # ======================================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SPINE_CSV_PATH = os.path.join(SCRIPT_DIR, "vertebrae_counts.csv")
    spine_mapper = SpinePositionMapper(SPINE_CSV_PATH) if os.path.isfile(SPINE_CSV_PATH) else None
    if spine_mapper:
        print(f"SpinePositionMapper loaded: {len(spine_mapper.specimens)} specimens")
    else:
        print("Warning: vertebrae_counts.csv not found, continuous position disabled")

    TRAIN_DIR = os.path.join(SCRIPT_DIR, args.run_dir)
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Run directory not found: {TRAIN_DIR}")

    os.chdir(TRAIN_DIR)

    CKPT = args.checkpoint if args.checkpoint else _detect_latest_checkpoint(".")
    LC_PATH = f'latent_codes/{CKPT}.pth'
    MODEL_PATH = f'model/{CKPT}.pth'

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint {CKPT} not found at {MODEL_PATH}")
    if not os.path.isfile(LC_PATH):
        raise FileNotFoundError(f"Latent codes not found at {LC_PATH}")

    RUN_NAME = os.path.basename(args.run_dir)
    print(f"Run directory: {RUN_NAME}")
    print(f"Checkpoint:    {CKPT}")

    config = load_config(config_path='model_params_config.json')

    # Determine device
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS (M1/M2/M3 GPU)")
    else:
        device = "cpu"
        print("Using CPU")

    config['device'] = device

    train_paths = config['list_mesh_paths']
    all_vtk_files = [os.path.basename(f) for f in train_paths]

    # ======================================================================
    # Pre-parse taxonomy for ALL training files (used by distance matching)
    # ======================================================================
    train_taxonomy = []  # One entry per training file, same indexing as all_vtk_files
    for f in all_vtk_files:
        parsed = parse_taxonomy_from_filename(f)
        train_taxonomy.append(parsed)  # May be None for unparseable files

    # ======================================================================
    # Locate mesh directory
    # ======================================================================
    potential_dirs = ["vertebrae_meshes", "../vertebrae_meshes",
                      "../../vertebrae_meshes", "data/vertebrae_meshes"]
    mesh_dir = None
    for d in potential_dirs:
        if os.path.isdir(d):
            mesh_dir = d
            break
    if mesh_dir is None:
        print("Warning: Could not locate 'vertebrae_meshes' directory.")
        mesh_dir = "."

    test_paths = config['test_paths']
    mesh_list_raw = test_paths
    mesh_list = [os.path.join(mesh_dir, os.path.basename(p)) for p in mesh_list_raw]

    # ======================================================================
    # Create top-level ablation run directory
    # ======================================================================
    run_dir = create_run_directory(base_dir="results", prefix=f"ablation_{RUN_NAME}_ckpt{CKPT}")
    print(f"\n{'='*60}")
    print(f"Ablation run directory: {run_dir}")
    print(f"{'='*60}\n")


    # ======================================================================
    # Helper: optimize latent vector for a novel mesh
    # ======================================================================
    def optimize_latent(decoder, points, sdf_vals, latent_size, mean_latent,
                        latent_codes, top_k_reg, device, iters=1000, lr=1e-3):
        init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
        latent = init_latent_torch.clone().detach().float().to(device).requires_grad_()
        optimizer = torch.optim.Adam([latent], lr=lr)
        sdf_vals = sdf_vals.float().to(device)
        decoder = decoder.to(device)
        points = points.float().to(device)
        for i in range(iters):
            optimizer.zero_grad()
            pred_sdf = get_sdfs(decoder, points, latent, device=device)
            loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals)
            loss.backward()
            optimizer.step()
            if i % 200 == 0 or i == iters - 1:
                print(f"[{i}/{iters}] Loss: {loss.item():.6f}")
        return latent.detach().to(device)


    # ======================================================================
    # Helper: extract top-k taxonomy predictions from distance indices
    # ======================================================================
    def _get_topk_taxonomy(indices: np.ndarray) -> List[Optional[Dict[str, str]]]:
        """Return parsed taxonomy dicts for each index into the training set."""
        return [train_taxonomy[idx] for idx in indices]


    def _check_match(gt_value: Optional[str], pred_values: List[Optional[str]], k: int = 1) -> str:
        """Check if ground truth appears in the first k predictions. Returns 'yes'/'no'."""
        if gt_value is None:
            return "no"
        for pred in pred_values[:k]:
            if pred is not None and pred == gt_value:
                return "yes"
        return "no"


    # ======================================================================
    # Load model and latent codes
    # ======================================================================
    model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)


    # ======================================================================
    # Prepare training data & labels
    # ======================================================================
    X_train_raw = latent_codes.cpu().numpy()

    y_train_species = []
    y_train_genera = []
    y_train_positions = []
    y_train_continuous_position = []
    valid_indices = []

    for f_idx, f in enumerate(all_vtk_files):
        parsed = parse_taxonomy_from_filename(f)
        if parsed and parsed.get('species') and parsed.get('position'):
            y_train_species.append(parsed['species'])
            y_train_genera.append(parsed['genus'])
            y_train_positions.append(parsed['position'])
            cont_pos = spine_mapper.get_normalized_position(f) if spine_mapper else None
            y_train_continuous_position.append(cont_pos if cont_pos is not None else float('nan'))
            valid_indices.append(f_idx)
        else:
            print(f"Skipping training file with invalid label format: {f}")

    X_train_valid = X_train_raw[valid_indices]
    y_train = np.column_stack((y_train_species, y_train_genera, y_train_positions))
    y_train_cont_pos = np.array(y_train_continuous_position, dtype=float)
    cont_pos_valid_mask = np.isfinite(y_train_cont_pos)
    n_cont_pos = int(cont_pos_valid_mask.sum())
    print(f"Continuous position labels available for {n_cont_pos}/{len(y_train_cont_pos)} training samples")


    # ======================================================================
    # Pre-compute metric learning transformation (used by configs B and D)
    # ======================================================================
    unique_species = np.unique(y_train_species)
    metric_learner = None
    X_train_ml = None

    if len(y_train_species) > 0 and len(unique_species) > 1:
        print("\nPre-computing NCA Metric Learning transformation...")
        ml_start = time_module.time()
        metric_learner = LatentMetricLearner(method='NCA', max_iter=100)
        X_train_ml = metric_learner.fit_transform(X_train_valid, np.array(y_train_species))
        ml_time = time_module.time() - ml_start
        print(f"Metric learning fitted in {ml_time:.3f}s")
    else:
        ml_time = 0.0
        print("Warning: Not enough species for metric learning.")


    # ======================================================================
    # Pre-train classifiers for each applicable config
    # ======================================================================
    print("\nTraining classifiers for each configuration...")

    classifiers_raw = None
    classifiers_raw_times = {}
    classifiers_ml = None
    classifiers_ml_times = {}

    if len(y_train) >= 2:
        # Config C: raw latent classifiers
        print("  Training classifiers on raw latent codes (Config C)...")
        classifiers_raw, classifiers_raw_times = train_classifiers(X_train_valid, y_train)
        for name, t in classifiers_raw_times.items():
            print(f"    {name}: {t:.3f}s")

        # Config D: metric-learning transformed classifiers
        if X_train_ml is not None:
            print("  Training classifiers on NCA-transformed latent codes (Config D)...")
            classifiers_ml, classifiers_ml_times = train_classifiers(X_train_ml, y_train)
            for name, t in classifiers_ml_times.items():
                print(f"    {name}: {t:.3f}s")
    else:
        raise ValueError("Not enough valid training labels!")


    # ======================================================================
    # Pre-train position regressors (continuous spine position)
    # ======================================================================
    pos_regressors_raw = None
    pos_regressors_raw_times = {}
    pos_regressors_ml = None
    pos_regressors_ml_times = {}

    if n_cont_pos >= 5:
        print("\nTraining position regressors for continuous spine position...")
        X_train_pos = X_train_valid[cont_pos_valid_mask]
        y_train_pos = y_train_cont_pos[cont_pos_valid_mask]

        print("  Training regressors on raw latent codes...")
        pos_regressors_raw, pos_regressors_raw_times = train_position_regressors(X_train_pos, y_train_pos)
        for name, t in pos_regressors_raw_times.items():
            print(f"    {name}: {t:.3f}s")

        if X_train_ml is not None:
            X_train_pos_ml = X_train_ml[cont_pos_valid_mask[valid_indices.index(valid_indices[0]):]]
            # Recompute: X_train_ml corresponds to X_train_valid rows
            X_train_pos_ml = X_train_ml[cont_pos_valid_mask]
            print("  Training regressors on NCA-transformed latent codes...")
            pos_regressors_ml, pos_regressors_ml_times = train_position_regressors(X_train_pos_ml, y_train_pos)
            for name, t in pos_regressors_ml_times.items():
                print(f"    {name}: {t:.3f}s")
    else:
        print(f"\nSkipping position regressors: only {n_cont_pos} continuous position labels available")


    # ======================================================================
    # Pre-compute training tensors on device for distance computation
    # ======================================================================
    print("\nPre-computing training tensors on device for distance computation...")
    train_tensor_raw = torch.tensor(X_train_raw, dtype=torch.float32).to(device)
    if X_train_ml is not None:
        X_train_ml_all = metric_learner.transform(X_train_raw)
        train_tensor_ml = torch.tensor(X_train_ml_all, dtype=torch.float32).to(device)
    else:
        X_train_ml_all = None
        train_tensor_ml = None


    # ======================================================================
    # Process each test mesh — optimize latent once, then evaluate
    # across all ablation configurations
    # ======================================================================
    print(f"\nProcessing {len(mesh_list)} test meshes...")

    # Per-config accumulators
    config_logs = {cfg_name: [] for cfg_name in ABLATION_CONFIGS}

    for mesh_idx, vert_fname in enumerate(mesh_list):
        mesh_basename = os.path.basename(vert_fname)
        mesh_stem = os.path.splitext(mesh_basename)[0]
        print(f"\033[32m\n=== Processing {mesh_basename} ({mesh_idx + 1}/{len(mesh_list)}) ===\033[0m")

        # --- Parse ground truth using parse_taxonomy_from_filename ---
        parsed_truth = parse_taxonomy_from_filename(mesh_basename)
        gt_species = parsed_truth.get('species') if parsed_truth else None
        gt_genus = parsed_truth.get('genus') if parsed_truth else None
        gt_family = parsed_truth.get('family') if parsed_truth else None
        gt_position = parsed_truth.get('position') if parsed_truth else None
        gt_cont_pos = spine_mapper.get_normalized_position(mesh_basename) if spine_mapper else None
        gt_derived_region = spine_mapper.get_derived_region(mesh_basename) if spine_mapper else None

        # --- Set up inference dataset ---
        actual_fname = vert_fname
        if '.ply' in vert_fname:
            _, actual_fname = convert_ply_to_vtk(vert_fname)

        sdf_dataset = SDFSamples(
            list_mesh_paths=[actual_fname],
            multiprocessing=False,
            subsample=config["samples_per_object_per_batch"],
            print_filename=True,
            n_pts=config["n_pts_per_object"],
            p_near_surface=config['percent_near_surface'],
            p_further_from_surface=config['percent_further_from_surface'],
            sigma_near=config['sigma_near'],
            sigma_far=config['sigma_far'],
            rand_function=config['random_function'],
            center_pts=config['center_pts'],
            norm_pts=config['normalize_pts'],
            reference_mesh=None,
            verbose=config['verbose'],
            save_cache=config['cache'],
            equal_pos_neg=config['equal_pos_neg'],
            fix_mesh=config['fix_mesh'],
        )

        print("Setting up dataset")
        sdf_sample = sdf_dataset[0]
        sample_dict, _ = sdf_sample
        points = sample_dict['xyz'].float().to(device)
        sdf_vals = sample_dict['gt_sdf'].float()

        # --- Optimize latent (done once per mesh, shared across all configs) ---
        print("Optimizing latent vector...")
        latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'],
                                       mean_latent, latent_codes, top_k_reg, device=device)
        novel_vec_raw = latent_novel.cpu().detach().float().numpy()
        print("Latent optimization complete.")

        # Pre-compute NCA-transformed novel vector
        novel_vec_ml = metric_learner.transform(novel_vec_raw) if metric_learner is not None else None

        # --- Evaluate each ablation config ---
        for cfg_name, cfg in ABLATION_CONFIGS.items():
            entry = {
                "mesh": mesh_basename,
                "ground_truth_species": gt_species,
                "ground_truth_genus": gt_genus,
                "ground_truth_family": gt_family,
                "ground_truth_position": gt_position,
                "ground_truth_continuous_position": gt_cont_pos,
                "ground_truth_derived_region": gt_derived_region,
            }

            # Choose pre-computed training tensor and novel vector for distance
            if cfg["use_metric_learning"] and train_tensor_ml is not None:
                cur_train_tensor = train_tensor_ml
                novel_for_distance = novel_vec_ml
            else:
                cur_train_tensor = train_tensor_raw
                novel_for_distance = novel_vec_raw

            # --- Distance-based classification (all configs) ---
            with torch.no_grad():
                novel_tensor = torch.tensor(novel_for_distance, dtype=torch.float32).to(device)

                # Cosine similarity top-5
                cos_sims = F.cosine_similarity(novel_tensor, cur_train_tensor)
                cos_top5_indices = torch.argsort(cos_sims, descending=True)[:5].cpu().numpy()
                cos_top5_distances = (1 - cos_sims[cos_top5_indices]).cpu().numpy()

                # Euclidean distance top-5
                euc_dists = torch.norm(cur_train_tensor - novel_tensor, dim=1)
                euc_top5_indices = torch.argsort(euc_dists)[:5].cpu().numpy()
                euc_top5_distances = euc_dists[euc_top5_indices].cpu().numpy()

            # --- Parse taxonomy from top-5 matches using SAME parser as ground truth ---
            cos_top5_parsed = _get_topk_taxonomy(cos_top5_indices)
            euc_top5_parsed = _get_topk_taxonomy(euc_top5_indices)

            # Extract species/genus/family/position lists from parsed dicts
            cos_top5_species = [p.get('species') if p else None for p in cos_top5_parsed]
            cos_top5_genera = [p.get('genus') if p else None for p in cos_top5_parsed]
            cos_top5_families = [p.get('family') if p else None for p in cos_top5_parsed]
            cos_top5_positions = [p.get('position') if p else None for p in cos_top5_parsed]

            euc_top5_species = [p.get('species') if p else None for p in euc_top5_parsed]
            euc_top5_genera = [p.get('genus') if p else None for p in euc_top5_parsed]
            euc_top5_families = [p.get('family') if p else None for p in euc_top5_parsed]
            euc_top5_positions = [p.get('position') if p else None for p in euc_top5_parsed]

            # --- Cosine similarity predictions ---
            entry["cos_top1_species"] = cos_top5_species[0]
            entry["cos_top1_genus"] = cos_top5_genera[0]
            entry["cos_top1_family"] = cos_top5_families[0]
            entry["cos_top1_position"] = cos_top5_positions[0]
            # Top-1 matches
            entry["cos_top1_species_match"] = _check_match(gt_species, cos_top5_species, k=1)
            entry["cos_top1_genus_match"] = _check_match(gt_genus, cos_top5_genera, k=1)
            entry["cos_top1_family_match"] = _check_match(gt_family, cos_top5_families, k=1)
            entry["cos_top1_position_match"] = _check_match(gt_position, cos_top5_positions, k=1)
            # Top-5 matches
            entry["cos_top5_species_match"] = _check_match(gt_species, cos_top5_species, k=5)
            entry["cos_top5_genus_match"] = _check_match(gt_genus, cos_top5_genera, k=5)
            entry["cos_top5_family_match"] = _check_match(gt_family, cos_top5_families, k=5)
            entry["cos_top5_position_match"] = _check_match(gt_position, cos_top5_positions, k=5)

            # --- Euclidean distance predictions ---
            entry["euc_top1_species"] = euc_top5_species[0]
            entry["euc_top1_genus"] = euc_top5_genera[0]
            entry["euc_top1_family"] = euc_top5_families[0]
            entry["euc_top1_position"] = euc_top5_positions[0]
            # Top-1 matches
            entry["euc_top1_species_match"] = _check_match(gt_species, euc_top5_species, k=1)
            entry["euc_top1_genus_match"] = _check_match(gt_genus, euc_top5_genera, k=1)
            entry["euc_top1_family_match"] = _check_match(gt_family, euc_top5_families, k=1)
            entry["euc_top1_position_match"] = _check_match(gt_position, euc_top5_positions, k=1)
            # Top-5 matches
            entry["euc_top5_species_match"] = _check_match(gt_species, euc_top5_species, k=5)
            entry["euc_top5_genus_match"] = _check_match(gt_genus, euc_top5_genera, k=5)
            entry["euc_top5_family_match"] = _check_match(gt_family, euc_top5_families, k=5)
            entry["euc_top5_position_match"] = _check_match(gt_position, euc_top5_positions, k=5)

            # --- Per-rank details (top-5 names, distances, parsed taxonomy) ---
            for rank in range(5):
                # Cosine
                entry[f"cos_similar_{rank+1}_name"] = all_vtk_files[cos_top5_indices[rank]]
                entry[f"cos_similar_{rank+1}_distance"] = float(cos_top5_distances[rank])
                entry[f"cos_similar_{rank+1}_species"] = cos_top5_species[rank]
                entry[f"cos_similar_{rank+1}_genus"] = cos_top5_genera[rank]
                entry[f"cos_similar_{rank+1}_position"] = cos_top5_positions[rank]
                # Euclidean
                entry[f"euc_similar_{rank+1}_name"] = all_vtk_files[euc_top5_indices[rank]]
                entry[f"euc_similar_{rank+1}_distance"] = float(euc_top5_distances[rank])
                entry[f"euc_similar_{rank+1}_species"] = euc_top5_species[rank]
                entry[f"euc_similar_{rank+1}_genus"] = euc_top5_genera[rank]
                entry[f"euc_similar_{rank+1}_position"] = euc_top5_positions[rank]

            # --- Distance-based continuous position predictions ---
            if spine_mapper:
                for method, top5_indices, top5_distances in [
                    ("cos", cos_top5_indices, cos_top5_distances),
                    ("euc", euc_top5_indices, euc_top5_distances),
                ]:
                    # Top-1 continuous position
                    top1_fname = all_vtk_files[top5_indices[0]]
                    top1_cont_pos = spine_mapper.get_normalized_position(top1_fname)
                    entry[f"{method}_top1_continuous_position"] = top1_cont_pos

                    # Inverse-distance-weighted average of top-5 continuous positions
                    top5_cont_positions = []
                    top5_weights = []
                    for rank in range(5):
                        fname = all_vtk_files[top5_indices[rank]]
                        cp = spine_mapper.get_normalized_position(fname)
                        if cp is not None:
                            top5_cont_positions.append(cp)
                            dist = float(top5_distances[rank])
                            top5_weights.append(1.0 / (dist + 1e-8))
                    if top5_cont_positions:
                        weights = np.array(top5_weights)
                        weights /= weights.sum()
                        entry[f"{method}_top5_weighted_continuous_position"] = float(
                            np.dot(weights, top5_cont_positions)
                        )
                    else:
                        entry[f"{method}_top5_weighted_continuous_position"] = None

            # --- Supervised classifier predictions (configs C and D only) ---
            if cfg["use_classifiers"]:
                if cfg["use_metric_learning"] and classifiers_ml is not None:
                    trained = classifiers_ml
                    test_vec = novel_vec_ml
                else:
                    trained = classifiers_raw
                    test_vec = novel_vec_raw

                predictions, probabilities, inference_times = predict_classifiers(trained, test_vec)

                for clf_name in trained.keys():
                    preds = predictions[clf_name][0]
                    entry[f"{clf_name}_predicted_species"] = preds[0]
                    entry[f"{clf_name}_predicted_genus"] = preds[1]
                    entry[f"{clf_name}_predicted_position"] = preds[2]
                    entry[f"{clf_name}_match_species"] = "yes" if gt_species and gt_species == preds[0] else "no"
                    entry[f"{clf_name}_match_genus"] = "yes" if gt_genus and gt_genus == preds[1] else "no"
                    entry[f"{clf_name}_match_position"] = "yes" if gt_position and str(gt_position) == str(preds[2]) else "no"
                    entry[f"{clf_name}_inf_time"] = inference_times[clf_name]

                    if probabilities[clf_name] is not None:
                        entry[f"{clf_name}_species_confidence"] = f"{max(probabilities[clf_name][0][0]):.2%}"
                        entry[f"{clf_name}_genus_confidence"] = f"{max(probabilities[clf_name][1][0]):.2%}"
                        entry[f"{clf_name}_position_confidence"] = f"{max(probabilities[clf_name][2][0]):.2%}"

            # --- Position regressor predictions ---
            if pos_regressors_raw is not None or pos_regressors_ml is not None:
                if cfg["use_metric_learning"] and pos_regressors_ml is not None:
                    pos_reg_models = pos_regressors_ml
                    pos_test_vec = novel_vec_ml
                elif pos_regressors_raw is not None:
                    pos_reg_models = pos_regressors_raw
                    pos_test_vec = novel_vec_raw
                else:
                    pos_reg_models = None
                    pos_test_vec = None

                if pos_reg_models is not None and pos_test_vec is not None:
                    pos_preds, pos_inf_times = predict_position_regressors(pos_reg_models, pos_test_vec)
                    for reg_name, pred_val in pos_preds.items():
                        entry[f"{reg_name}_continuous_position"] = float(pred_val[0])
                        entry[f"{reg_name}_inf_time"] = pos_inf_times[reg_name]

            config_logs[cfg_name].append(entry)


    # ======================================================================
    # Compute metrics and save per-config outputs
    # ======================================================================
    print(f"\n{'='*60}")
    print("Computing metrics for each ablation configuration...")
    print(f"{'='*60}")

    CLASSIFIER_NAMES = list(classifiers_raw.keys()) if classifiers_raw else []

    all_config_summaries = []

    for cfg_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n--- {cfg['label']} ({cfg_name}) ---")

        cfg_dir = os.path.join(run_dir, cfg_name)
        os.makedirs(cfg_dir, exist_ok=True)
        cm_dir = os.path.join(cfg_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        df = pd.DataFrame(config_logs[cfg_name])
        df.to_csv(os.path.join(cfg_dir, "predictions.csv"), index=False)

        # --- Distance-based metrics (computed for all configs) ---
        distance_rows = []
        for method in ["cos", "euc"]:
            method_label = "Cosine Similarity" if method == "cos" else "Euclidean Distance"

            # Species
            for k_val, k_label in [(1, "top_1"), (5, "top_5")]:
                col = f"{method}_top{k_val}_species_match"
                if col in df.columns:
                    acc = (df[col] == "yes").mean()
                    distance_rows.append({"method": method_label, "metric": f"species_{k_label}_accuracy", "value": acc})

            # Genus
            for k_val, k_label in [(1, "top_1"), (5, "top_5")]:
                col = f"{method}_top{k_val}_genus_match"
                if col in df.columns:
                    acc = (df[col] == "yes").mean()
                    distance_rows.append({"method": method_label, "metric": f"genus_{k_label}_accuracy", "value": acc})

            # Family
            for k_val, k_label in [(1, "top_1"), (5, "top_5")]:
                col = f"{method}_top{k_val}_family_match"
                if col in df.columns:
                    acc = (df[col] == "yes").mean()
                    distance_rows.append({"method": method_label, "metric": f"family_{k_label}_accuracy", "value": acc})

            # Position
            for k_val, k_label in [(1, "top_1"), (5, "top_5")]:
                col = f"{method}_top{k_val}_position_match"
                if col in df.columns:
                    acc = (df[col] == "yes").mean()
                    distance_rows.append({"method": method_label, "metric": f"position_{k_label}_accuracy", "value": acc})

        distance_df = pd.DataFrame(distance_rows)
        if not distance_df.empty:
            distance_df.to_csv(os.path.join(cfg_dir, "distance_metrics.csv"), index=False)
            for _, row in distance_df.iterrows():
                print(f"  {row['method']} — {row['metric']}: {row['value']:.4f}")

        # Build summary entry for this config
        config_summary = {"config": cfg_name, "label": cfg["label"]}
        for _, row in distance_df.iterrows():
            method_prefix = row['method'].split()[0].lower()
            key = f"{method_prefix}_{row['metric']}"
            config_summary[key] = row['value']

        # --- Continuous position regression metrics ---
        regression_rows = []
        gt_col = "ground_truth_continuous_position"
        if gt_col in df.columns and df[gt_col].notna().any():
            gt_vals = df[gt_col].values.astype(float)

            # Distance-based continuous position predictions
            for method in ["cos", "euc"]:
                method_label = "Cosine" if method == "cos" else "Euclidean"
                for pred_col_suffix, pred_label in [
                    ("top1_continuous_position", "Top-1"),
                    ("top5_weighted_continuous_position", "Top-5 Weighted"),
                ]:
                    pred_col = f"{method}_{pred_col_suffix}"
                    if pred_col in df.columns and df[pred_col].notna().any():
                        pred_vals = df[pred_col].values.astype(float)
                        reg_m = calculate_regression_metrics(gt_vals, pred_vals)
                        if reg_m:
                            reg_m["method"] = f"{method_label} {pred_label}"
                            regression_rows.append(reg_m)
                            config_summary[f"{method}_{pred_col_suffix}_mae"] = reg_m["mae"]
                            config_summary[f"{method}_{pred_col_suffix}_r2"] = reg_m["r2"]
                            print(f"  {method_label} {pred_label} Position: MAE={reg_m['mae']:.4f}, R²={reg_m['r2']:.4f}, ρ={reg_m['spearman_rho']:.4f}")

            # Regressor predictions
            regressor_names = list(pos_regressors_raw.keys()) if pos_regressors_raw else []
            for reg_name in regressor_names:
                pred_col = f"{reg_name}_continuous_position"
                if pred_col in df.columns and df[pred_col].notna().any():
                    pred_vals = df[pred_col].values.astype(float)
                    reg_m = calculate_regression_metrics(gt_vals, pred_vals)
                    if reg_m:
                        reg_m["method"] = reg_name
                        regression_rows.append(reg_m)
                        config_summary[f"{reg_name}_mae"] = reg_m["mae"]
                        config_summary[f"{reg_name}_r2"] = reg_m["r2"]
                        print(f"  {reg_name}: MAE={reg_m['mae']:.4f}, R²={reg_m['r2']:.4f}, ρ={reg_m['spearman_rho']:.4f}")

        if regression_rows:
            reg_df = pd.DataFrame(regression_rows)
            reg_df.to_csv(os.path.join(cfg_dir, "position_regression_metrics.csv"), index=False)

        # --- Derived region accuracy comparison ---
        if "ground_truth_derived_region" in df.columns and "ground_truth_position" in df.columns:
            mask = df["ground_truth_derived_region"].notna() & df["ground_truth_position"].notna()
            if mask.sum() > 0:
                derived_acc = (df.loc[mask, "ground_truth_derived_region"] == df.loc[mask, "ground_truth_position"]).mean()
                config_summary["derived_region_accuracy"] = derived_acc
                print(f"  Derived region matches ground truth position: {derived_acc:.4f}")

        # --- Supervised classifier metrics (configs C and D only) ---
        all_metrics_rows = []
        if cfg["use_classifiers"]:
            for clf_name in CLASSIFIER_NAMES:
                for target, clf_gt_col, pred_col in [
                    ("species", "ground_truth_species", f"{clf_name}_predicted_species"),
                    ("genus", "ground_truth_genus", f"{clf_name}_predicted_genus"),
                    ("position", "ground_truth_position", f"{clf_name}_predicted_position"),
                ]:
                    if pred_col not in df.columns or clf_gt_col not in df.columns:
                        continue
                    mask = df[clf_gt_col].notna() & df[pred_col].notna()
                    y_true = df.loc[mask, clf_gt_col].tolist()
                    y_pred = df.loc[mask, pred_col].tolist()
                    if not y_true:
                        continue
                    m = calculate_metrics(y_true, y_pred)
                    m_df = metrics_to_dataframe(m, classifier_name=clf_name)
                    m_df["target"] = target
                    all_metrics_rows.append(m_df)

                    if target == "species":
                        config_summary[f"{clf_name}_species_accuracy"] = m["instance_accuracy"]
                        config_summary[f"{clf_name}_species_f1_macro"] = m["f1_macro"]
                        config_summary[f"{clf_name}_species_precision_macro"] = m["precision_macro"]
                        config_summary[f"{clf_name}_species_recall_macro"] = m["recall_macro"]
                    elif target == "genus":
                        config_summary[f"{clf_name}_genus_accuracy"] = m["instance_accuracy"]
                        config_summary[f"{clf_name}_genus_f1_macro"] = m["f1_macro"]
                    elif target == "position":
                        config_summary[f"{clf_name}_position_accuracy"] = m["instance_accuracy"]
                        config_summary[f"{clf_name}_position_f1_macro"] = m["f1_macro"]
                        config_summary[f"{clf_name}_position_precision_macro"] = m["precision_macro"]
                        config_summary[f"{clf_name}_position_recall_macro"] = m["recall_macro"]

        if all_metrics_rows:
            metrics_df = pd.concat(all_metrics_rows, ignore_index=True)
            metrics_df.to_csv(os.path.join(cfg_dir, "detailed_metrics.csv"), index=False)

            summary_rows = metrics_df[metrics_df["level"] == "summary"]
            for _, row in summary_rows.iterrows():
                target = row.get("target", "?")
                clf = row.get("classifier", "?")
                acc = row.get("instance_accuracy", "N/A")
                f1 = row.get("f1_macro", "N/A")
                if isinstance(acc, float):
                    print(f"  [{clf}] {target}: Acc={acc:.4f}, F1={f1:.4f}")

        all_config_summaries.append(config_summary)

        # --- Confusion matrices ---
        if len(df) >= 2:
            best_approach = None
            best_acc = 0

            # Check distance-based species accuracy
            for method in ["cos", "euc"]:
                col = f"{method}_top1_species_match"
                if col in df.columns:
                    acc = (df[col] == "yes").mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_approach = ("distance", method)

            # Check classifier accuracy
            if cfg["use_classifiers"]:
                for clf_name in CLASSIFIER_NAMES:
                    col = f"{clf_name}_predicted_species"
                    cm_gt_col = "ground_truth_species"
                    if col in df.columns and cm_gt_col in df.columns:
                        mask = df[cm_gt_col].notna() & df[col].notna()
                        acc = (df.loc[mask, cm_gt_col] == df.loc[mask, col]).mean()
                        if acc > best_acc:
                            best_acc = acc
                            best_approach = ("classifier", clf_name)

            # Generate CMs using the best approach
            if best_approach is not None:
                approach_type, approach_name = best_approach

                if approach_type == "classifier":
                    y_pred_species = df[f"{approach_name}_predicted_species"].tolist()
                    cm_title_suffix = approach_name
                else:
                    y_pred_species = df[f"{approach_name}_top1_species"].tolist()
                    cm_title_suffix = f"{'Cosine' if approach_name == 'cos' else 'Euclidean'} Top-1"

                print(f"  Generating confusion matrices using: {cm_title_suffix} ({best_acc:.1%})")

                # Hierarchical CMs
                true_dicts, pred_dicts = [], []
                species_taxonomy_map = {}
                for filename in df["mesh"]:
                    parsed = parse_taxonomy_from_filename(filename)
                    if parsed:
                        true_dicts.append(parsed)
                        species_taxonomy_map[parsed["species"]] = parsed
                    else:
                        true_dicts.append({"family": "unknown", "genus": "unknown", "species": "unknown"})

                for pred_sp in y_pred_species:
                    if pred_sp in species_taxonomy_map:
                        pred_dicts.append(species_taxonomy_map[pred_sp])
                    else:
                        pred_dicts.append({"family": "unknown", "genus": "unknown", "species": pred_sp or "unknown"})

                cm_results = generate_hierarchical_confusion_matrices(true_dicts, pred_dicts)
                for level, data in cm_results.items():
                    matrix, labels = data["matrix"], data["labels"]
                    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.7)))
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f"CM — {level.capitalize()} [{cfg['label']}] ({cm_title_suffix})")
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(cm_dir, f"confusion_matrix_{level}.png"), dpi=300)
                    plt.close()

                # Position CM
                if approach_type == "classifier":
                    pos_pred_col = f"{approach_name}_predicted_position"
                else:
                    pos_pred_col = f"{approach_name}_top1_position"

                if pos_pred_col in df.columns and "ground_truth_position" in df.columns:
                    mask = df["ground_truth_position"].notna() & df[pos_pred_col].notna()
                    y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
                    y_pred_pos = df.loc[mask, pos_pred_col].tolist()
                    if y_true_pos:
                        pos_cm = generate_position_confusion_matrix(y_true_pos, y_pred_pos)
                        labels = pos_cm["labels"]
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(pos_cm["matrix"], annot=True, fmt="d", cmap="Oranges",
                                    xticklabels=labels, yticklabels=labels)
                        plt.title(f"CM — Position [{cfg['label']}] ({cm_title_suffix})")
                        plt.xlabel("Predicted Position")
                        plt.ylabel("True Position")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cm_dir, "confusion_matrix_position.png"), dpi=300)
                        plt.close()

        # --- Logistic Regression interpretation ---
        if cfg["use_classifiers"]:
            lr_interp_dir = os.path.join(cfg_dir, "lr_interpretation")
            if cfg["use_metric_learning"] and classifiers_ml is not None:
                lr_model = classifiers_ml.get("LogisticRegression")
                lr_X = X_train_ml
            else:
                lr_model = classifiers_raw.get("LogisticRegression") if classifiers_raw else None
                lr_X = X_train_valid

            if lr_model is not None:
                print(f"  Running LR interpretation for position...")
                try:
                    lr_results_pos = run_lr_interpretation(
                        lr_model, lr_X, y_train,
                        target_index=2, target_name="position",
                        output_dir=lr_interp_dir, top_n=20,
                    )
                    n_sig = sum(lr_results_pos["sig_counts"].values())
                    print(f"    Position: {n_sig} significant latent dims (p<0.05) across all classes")
                except Exception as e:
                    print(f"    Warning: LR position interpretation failed: {e}")

                print(f"  Running LR interpretation for species...")
                try:
                    lr_results_sp = run_lr_interpretation(
                        lr_model, lr_X, y_train,
                        target_index=0, target_name="species",
                        output_dir=lr_interp_dir, top_n=20,
                    )
                except Exception as e:
                    print(f"    Warning: LR species interpretation failed: {e}")

        # --- Continuous position plots ---
        if regression_rows and gt_col in df.columns and df[gt_col].notna().any():
            pos_plots_dir = os.path.join(cfg_dir, "position_plots")
            os.makedirs(pos_plots_dir, exist_ok=True)
            gt_vals = df[gt_col].values.astype(float)

            # Collect all prediction methods for plotting
            pred_methods = {}
            for method in ["cos", "euc"]:
                for suffix in ["top1_continuous_position", "top5_weighted_continuous_position"]:
                    col = f"{method}_{suffix}"
                    if col in df.columns and df[col].notna().any():
                        label = f"{'Cosine' if method == 'cos' else 'Euclidean'} {'Top-1' if 'top1' in suffix else 'Top-5 Wt'}"
                        pred_methods[label] = df[col].values.astype(float)
            regressor_names = list(pos_regressors_raw.keys()) if pos_regressors_raw else []
            for reg_name in regressor_names:
                col = f"{reg_name}_continuous_position"
                if col in df.columns and df[col].notna().any():
                    pred_methods[reg_name] = df[col].values.astype(float)

            # Scatter plot: true vs predicted for each method
            n_methods = len(pred_methods)
            if n_methods > 0:
                cols = min(3, n_methods)
                rows_needed = (n_methods + cols - 1) // cols
                fig, axes = plt.subplots(rows_needed, cols, figsize=(6 * cols, 5 * rows_needed), squeeze=False)
                for ax_idx, (label, pred_vals) in enumerate(pred_methods.items()):
                    ax = axes[ax_idx // cols][ax_idx % cols]
                    mask = np.isfinite(gt_vals) & np.isfinite(pred_vals)
                    ax.scatter(gt_vals[mask], pred_vals[mask], alpha=0.6, s=30)
                    ax.plot([0, 1], [0, 1], 'r--', lw=1, label="Perfect")
                    ax.set_xlabel("True Position")
                    ax.set_ylabel("Predicted Position")
                    ax.set_title(label)
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_aspect('equal')
                    ax.legend(fontsize=8)
                # Hide unused axes
                for ax_idx in range(n_methods, rows_needed * cols):
                    axes[ax_idx // cols][ax_idx % cols].set_visible(False)
                fig.suptitle(f"Continuous Position: True vs Predicted [{cfg['label']}]")
                fig.tight_layout()
                fig.savefig(os.path.join(pos_plots_dir, "scatter_true_vs_pred.png"), dpi=300)
                plt.close(fig)

                # Residual histogram (best method by MAE)
                best_method = min(regression_rows, key=lambda r: r["mae"])["method"]
                if best_method in pred_methods:
                    best_pred = pred_methods[best_method]
                    mask = np.isfinite(gt_vals) & np.isfinite(best_pred)
                    residuals = best_pred[mask] - gt_vals[mask]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                    ax.axvline(0, color='red', linestyle='--', lw=1)
                    ax.set_xlabel("Residual (Predicted - True)")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Position Residuals — {best_method} [{cfg['label']}]")
                    fig.tight_layout()
                    fig.savefig(os.path.join(pos_plots_dir, "residual_histogram.png"), dpi=300)
                    plt.close(fig)

                # Error-by-region box plot (best method)
                if best_method in pred_methods and "ground_truth_position" in df.columns:
                    best_pred = pred_methods[best_method]
                    mask = np.isfinite(gt_vals) & np.isfinite(best_pred) & df["ground_truth_position"].notna()
                    abs_errors = np.abs(best_pred[mask] - gt_vals[mask])
                    regions = df.loc[mask, "ground_truth_position"].values
                    err_df = pd.DataFrame({"Absolute Error": abs_errors, "Region": regions})
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=err_df, x="Region", y="Absolute Error", ax=ax,
                                order=["Cervical", "Thoracic", "Lumbar"])
                    ax.set_title(f"Position Error by Region — {best_method} [{cfg['label']}]")
                    fig.tight_layout()
                    fig.savefig(os.path.join(pos_plots_dir, "error_by_region.png"), dpi=300)
                    plt.close(fig)

        # --- Per-config README ---
        cfg_files = {
            "predictions.csv": "Per-mesh predictions with full top-5 taxonomy breakdown and ground truth",
            "distance_metrics.csv": "Top-1/Top-5 accuracy for distance-based retrieval (species, genus, family, position)",
        }
        if regression_rows:
            cfg_files["position_regression_metrics.csv"] = "MAE, RMSE, R², Spearman for continuous position prediction"
            cfg_files["position_plots/"] = "Scatter, residual, and error-by-region plots for continuous position"
        if cfg["use_classifiers"]:
            cfg_files["detailed_metrics.csv"] = "Full per-classifier, per-class metrics"
            cfg_files["lr_interpretation/"] = "Logistic Regression coefficient analysis, Wald-test p-values, and feature importance plots"
        cfg_files["confusion_matrices/"] = "Confusion matrices at family/genus/species/position levels"

        write_run_manifest(
            cfg_dir,
            description=cfg["description"],
            approach=cfg["label"],
            script_path=os.path.abspath(__file__),
            test_data_paths=[os.path.basename(p) for p in mesh_list],
            checkpoint=CKPT,
            metric_learning_method="NCA" if cfg["use_metric_learning"] else None,
            classifier_names=CLASSIFIER_NAMES if cfg["use_classifiers"] else None,
            extra_files=cfg_files,
        )


    # ======================================================================
    # Cross-configuration comparison
    # ======================================================================
    print(f"\n{'='*60}")
    print("Generating cross-configuration comparison...")
    print(f"{'='*60}")

    comparison_df = pd.DataFrame(all_config_summaries)
    comparison_csv_path = os.path.join(run_dir, "comparison_metrics.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Comparison CSV saved to: {comparison_csv_path}")

    # --- Comparison charts ---
    charts_dir = os.path.join(run_dir, "comparison_charts")
    os.makedirs(charts_dir, exist_ok=True)


    # ---- Helper: build a unified "all approaches" dataframe for a target ----
    def _build_unified_rows(comparison_df, target="species"):
        """
        Builds a flat list of dicts with columns:
            Configuration, Approach, Accuracy, F1, Precision, Recall
        covering distance methods AND classifiers for every config.
        """
        rows = []
        for _, row in comparison_df.iterrows():
            label = row["label"]
            # Distance-based approaches
            for method, pretty in [("cosine", "Cosine Top-1"), ("euclidean", "Euclidean Top-1")]:
                acc_key = f"{method}_{target}_top_1_accuracy"
                val = row.get(acc_key)
                if val is not None and pd.notna(val):
                    rows.append({"Configuration": label, "Approach": pretty,
                                 "Accuracy": val, "F1": None, "Precision": None, "Recall": None})
            # Top-5 for species/genus/position
            for method, pretty in [("cosine", "Cosine Top-5"), ("euclidean", "Euclidean Top-5")]:
                val = row.get(f"{method}_{target}_top_5_accuracy")
                if val is not None and pd.notna(val):
                    rows.append({"Configuration": label, "Approach": pretty,
                                 "Accuracy": val, "F1": None, "Precision": None, "Recall": None})
            # Supervised classifiers
            for clf_name in CLASSIFIER_NAMES:
                acc_key = f"{clf_name}_{target}_accuracy"
                f1_key = f"{clf_name}_{target}_f1_macro"
                prec_key = f"{clf_name}_{target}_precision_macro"
                rec_key = f"{clf_name}_{target}_recall_macro"
                acc_val = row.get(acc_key)
                if acc_val is not None and pd.notna(acc_val):
                    rows.append({
                        "Configuration": label,
                        "Approach": clf_name,
                        "Accuracy": acc_val,
                        "F1": row.get(f1_key) if pd.notna(row.get(f1_key, None)) else None,
                        "Precision": row.get(prec_key) if pd.notna(row.get(prec_key, None)) else None,
                        "Recall": row.get(rec_key) if pd.notna(row.get(rec_key, None)) else None,
                    })
        return rows


    # ======================================================================
    # SPECIES charts
    # ======================================================================
    species_rows = _build_unified_rows(comparison_df, target="species")

    if species_rows:
        sp_df = pd.DataFrame(species_rows)

        # Chart S1: Unified accuracy — all approaches x all configs
        plt.figure(figsize=(18, 8))
        sns.barplot(data=sp_df, x="Approach", y="Accuracy", hue="Configuration", palette="Set2")
        plt.title("Species — Instance Accuracy: All Approaches Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("Instance Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "species_accuracy_all_approaches.png"), dpi=300)
        plt.close()

        # Chart S2: Unified F1 — classifiers only
        f1_rows = [r for r in species_rows if r["F1"] is not None]
        if f1_rows:
            f1_df = pd.DataFrame(f1_rows)
            plt.figure(figsize=(14, 7))
            sns.barplot(data=f1_df, x="Approach", y="F1", hue="Configuration", palette="Set1")
            plt.title("Species — F1 (Macro): Classifiers Across Configurations")
            plt.ylim(0, 1.05)
            plt.ylabel("F1 Score (macro)")
            plt.xticks(rotation=30, ha="right")
            plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "species_f1_all_approaches.png"), dpi=300)
            plt.close()

        # Chart S3: Precision & Recall side-by-side for classifiers
        prec_rec_rows = [r for r in species_rows if r["Precision"] is not None]
        if prec_rec_rows:
            pr_df = pd.DataFrame(prec_rec_rows)
            pr_melted = pd.melt(pr_df, id_vars=["Configuration", "Approach"],
                                value_vars=["Precision", "Recall"],
                                var_name="Metric", value_name="Score")
            plt.figure(figsize=(16, 7))
            sns.barplot(data=pr_melted, x="Approach", y="Score", hue="Metric", palette="coolwarm")
            plt.title("Species — Precision & Recall (Macro): Classifiers")
            plt.ylim(0, 1.05)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "species_precision_recall.png"), dpi=300)
            plt.close()

        # Chart S4: Distance-based retrieval comparison across configs
        dist_only = [r for r in species_rows if "Top-" in r["Approach"]]
        if dist_only:
            dist_df = pd.DataFrame(dist_only)
            plt.figure(figsize=(14, 7))
            sns.barplot(data=dist_df, x="Configuration", y="Accuracy", hue="Approach", palette="viridis")
            plt.title("Species — Distance-Based Retrieval Accuracy Across Configurations")
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy")
            plt.xticks(rotation=15, ha="right")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "species_distance_comparison.png"), dpi=300)
            plt.close()


    # ======================================================================
    # GENUS charts
    # ======================================================================
    genus_rows = _build_unified_rows(comparison_df, target="genus")

    if genus_rows:
        gen_df = pd.DataFrame(genus_rows)

        plt.figure(figsize=(18, 8))
        sns.barplot(data=gen_df, x="Approach", y="Accuracy", hue="Configuration", palette="Set2")
        plt.title("Genus — Instance Accuracy: All Approaches Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("Instance Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "genus_accuracy_all_approaches.png"), dpi=300)
        plt.close()

        dist_only_gen = [r for r in genus_rows if "Top-" in r["Approach"]]
        if dist_only_gen:
            dist_df_gen = pd.DataFrame(dist_only_gen)
            plt.figure(figsize=(14, 7))
            sns.barplot(data=dist_df_gen, x="Configuration", y="Accuracy", hue="Approach", palette="viridis")
            plt.title("Genus — Distance-Based Retrieval Accuracy Across Configurations")
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy")
            plt.xticks(rotation=15, ha="right")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "genus_distance_comparison.png"), dpi=300)
            plt.close()


    # ======================================================================
    # POSITION (Spinal Region) charts
    # ======================================================================
    position_rows = _build_unified_rows(comparison_df, target="position")

    if position_rows:
        pos_df = pd.DataFrame(position_rows)

        # Chart P1: Unified accuracy — all approaches x all configs
        plt.figure(figsize=(18, 8))
        sns.barplot(data=pos_df, x="Approach", y="Accuracy", hue="Configuration", palette="Set2")
        plt.title("Spinal Position — Instance Accuracy: All Approaches Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("Instance Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "position_accuracy_all_approaches.png"), dpi=300)
        plt.close()

        # Chart P2: Unified F1 — classifiers only
        f1_rows_pos = [r for r in position_rows if r["F1"] is not None]
        if f1_rows_pos:
            f1_df_pos = pd.DataFrame(f1_rows_pos)
            plt.figure(figsize=(14, 7))
            sns.barplot(data=f1_df_pos, x="Approach", y="F1", hue="Configuration", palette="Set1")
            plt.title("Spinal Position — F1 (Macro): Classifiers Across Configurations")
            plt.ylim(0, 1.05)
            plt.ylabel("F1 Score (macro)")
            plt.xticks(rotation=30, ha="right")
            plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "position_f1_all_approaches.png"), dpi=300)
            plt.close()

        # Chart P3: Precision & Recall side-by-side
        prec_rec_pos = [r for r in position_rows if r["Precision"] is not None]
        if prec_rec_pos:
            pr_df_pos = pd.DataFrame(prec_rec_pos)
            pr_melted_pos = pd.melt(pr_df_pos, id_vars=["Configuration", "Approach"],
                                    value_vars=["Precision", "Recall"],
                                    var_name="Metric", value_name="Score")
            plt.figure(figsize=(16, 7))
            sns.barplot(data=pr_melted_pos, x="Approach", y="Score", hue="Metric", palette="coolwarm")
            plt.title("Spinal Position — Precision & Recall (Macro): Classifiers")
            plt.ylim(0, 1.05)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "position_precision_recall.png"), dpi=300)
            plt.close()

        # Chart P4: Distance-based position retrieval
        dist_only_pos = [r for r in position_rows if "Top-" in r["Approach"]]
        if dist_only_pos:
            dist_df_pos = pd.DataFrame(dist_only_pos)
            plt.figure(figsize=(14, 7))
            sns.barplot(data=dist_df_pos, x="Configuration", y="Accuracy", hue="Approach", palette="viridis")
            plt.title("Spinal Position — Distance-Based Retrieval Accuracy Across Configurations")
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy")
            plt.xticks(rotation=15, ha="right")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "position_distance_comparison.png"), dpi=300)
            plt.close()


    # ======================================================================
    # Combined best-of-each-approach chart (species + genus + position)
    # ======================================================================
    combined_chart_data = []
    for _, row in comparison_df.iterrows():
        label = row["label"]
        for target_name, target_key in [
            ("Species", "species"),
            ("Genus", "genus"),
            ("Position", "position"),
        ]:
            # Best distance top-1 accuracy
            best_dist = max(
                row.get(f"cosine_{target_key}_top_1_accuracy", 0) or 0,
                row.get(f"euclidean_{target_key}_top_1_accuracy", 0) or 0,
            )
            if best_dist > 0:
                combined_chart_data.append({"Configuration": label, "Target": target_name,
                                            "Approach": "Best Distance (Top-1)", "Accuracy": best_dist})
            # Best distance top-5 accuracy
            best_dist5 = max(
                row.get(f"cosine_{target_key}_top_5_accuracy", 0) or 0,
                row.get(f"euclidean_{target_key}_top_5_accuracy", 0) or 0,
            )
            if best_dist5 > 0:
                combined_chart_data.append({"Configuration": label, "Target": target_name,
                                            "Approach": "Best Distance (Top-5)", "Accuracy": best_dist5})
            # Best classifier accuracy
            best_clf_acc = 0
            for clf_name in CLASSIFIER_NAMES:
                val = row.get(f"{clf_name}_{target_key}_accuracy", 0) or 0
                if val > best_clf_acc:
                    best_clf_acc = val
            if best_clf_acc > 0:
                combined_chart_data.append({"Configuration": label, "Target": target_name,
                                            "Approach": "Best Classifier", "Accuracy": best_clf_acc})

    if combined_chart_data:
        combined_df = pd.DataFrame(combined_chart_data)
        g = sns.catplot(data=combined_df, x="Configuration", y="Accuracy", hue="Approach",
                        col="Target", kind="bar", palette="Dark2", height=6, aspect=1.3)
        g.set_titles("{col_name} Prediction")
        g.set_xticklabels(rotation=20, ha="right")
        for ax in g.axes.flat:
            ax.set_ylim(0, 1.05)
        g.fig.suptitle("Best Approach per Configuration — Species vs Genus vs Position", y=1.02)
        g.tight_layout()
        g.savefig(os.path.join(charts_dir, "best_approach_all_targets.png"), dpi=300)
        plt.close()


    # ======================================================================
    # Summary statistics markdown
    # ======================================================================
    summary_md_path = os.path.join(run_dir, "summary_statistics.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Summary Statistics\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Run Directory:** {RUN_NAME}  \n")
        f.write(f"**Checkpoint:** {CKPT}  \n")
        f.write(f"**Test Meshes:** {len(mesh_list)}  \n")
        f.write(f"**Training Samples:** {len(X_train_valid)} ({len(X_train_raw) - len(X_train_valid)} skipped)  \n\n")

        f.write("## Configurations\n\n")
        f.write("| ID | Configuration | Metric Learning | Classifiers |\n")
        f.write("|----|---------------|-----------------|-------------|\n")
        for cfg_name, cfg in ABLATION_CONFIGS.items():
            ml = "NCA" if cfg["use_metric_learning"] else "None"
            clf = "KNN, SVM, RF, MLP, LR" if cfg["use_classifiers"] else "None"
            f.write(f"| {cfg_name} | {cfg['label']} | {ml} | {clf} |\n")
        f.write("\n")

        def _fmt(val):
            return f"{val:.4f}" if isinstance(val, float) else str(val if val is not None else "N/A")

        # ================================================================
        # SPECIES PREDICTION
        # ================================================================
        f.write("---\n\n# Species Prediction\n\n")

        f.write("## Distance-Based Species Retrieval\n\n")
        f.write("| Configuration | Cosine Top-1 | Cosine Top-5 | Euclidean Top-1 | Euclidean Top-5 |\n")
        f.write("|---------------|-------------|-------------|----------------|----------------|\n")
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['label']} "
                    f"| {_fmt(row.get('cosine_species_top_1_accuracy'))} "
                    f"| {_fmt(row.get('cosine_species_top_5_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_species_top_1_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_species_top_5_accuracy'))} |\n")
        f.write("\n")

        # Classifier species table
        clf_configs = [cfg_name for cfg_name, cfg in ABLATION_CONFIGS.items() if cfg["use_classifiers"]]
        if clf_configs and CLASSIFIER_NAMES:
            f.write("## Supervised Classifier — Species\n\n")
            header = "| Classifier |"
            sep = "|------------|"
            for cfg_name in clf_configs:
                lbl = ABLATION_CONFIGS[cfg_name]['label']
                header += f" {lbl} Acc | {lbl} F1 | {lbl} Prec | {lbl} Rec |"
                sep += "------|------|------|------|"
            f.write(header + "\n" + sep + "\n")

            for clf_name in CLASSIFIER_NAMES:
                row_str = f"| {clf_name} |"
                for cfg_name in clf_configs:
                    cfg_row = comparison_df[comparison_df["config"] == cfg_name].iloc[0]
                    row_str += (f" {_fmt(cfg_row.get(f'{clf_name}_species_accuracy'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_species_f1_macro'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_species_precision_macro'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_species_recall_macro'))} |")
                f.write(row_str + "\n")
            f.write("\n")

        # ================================================================
        # GENUS PREDICTION
        # ================================================================
        f.write("---\n\n# Genus Prediction\n\n")

        f.write("## Distance-Based Genus Retrieval\n\n")
        f.write("| Configuration | Cosine Top-1 | Cosine Top-5 | Euclidean Top-1 | Euclidean Top-5 |\n")
        f.write("|---------------|-------------|-------------|----------------|----------------|\n")
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['label']} "
                    f"| {_fmt(row.get('cosine_genus_top_1_accuracy'))} "
                    f"| {_fmt(row.get('cosine_genus_top_5_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_genus_top_1_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_genus_top_5_accuracy'))} |\n")
        f.write("\n")

        if clf_configs and CLASSIFIER_NAMES:
            f.write("## Supervised Classifier — Genus\n\n")
            header = "| Classifier |"
            sep = "|------------|"
            for cfg_name in clf_configs:
                lbl = ABLATION_CONFIGS[cfg_name]['label']
                header += f" {lbl} Acc | {lbl} F1 |"
                sep += "------|------|"
            f.write(header + "\n" + sep + "\n")

            for clf_name in CLASSIFIER_NAMES:
                row_str = f"| {clf_name} |"
                for cfg_name in clf_configs:
                    cfg_row = comparison_df[comparison_df["config"] == cfg_name].iloc[0]
                    row_str += (f" {_fmt(cfg_row.get(f'{clf_name}_genus_accuracy'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_genus_f1_macro'))} |")
                f.write(row_str + "\n")
            f.write("\n")

        # ================================================================
        # SPINAL POSITION PREDICTION
        # ================================================================
        f.write("---\n\n# Spinal Position Prediction (Cervical / Thoracic / Lumbar)\n\n")

        f.write("## Distance-Based Position Retrieval\n\n")
        f.write("| Configuration | Cosine Top-1 | Cosine Top-5 | Euclidean Top-1 | Euclidean Top-5 |\n")
        f.write("|---------------|-------------|-------------|----------------|----------------|\n")
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['label']} "
                    f"| {_fmt(row.get('cosine_position_top_1_accuracy'))} "
                    f"| {_fmt(row.get('cosine_position_top_5_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_position_top_1_accuracy'))} "
                    f"| {_fmt(row.get('euclidean_position_top_5_accuracy'))} |\n")
        f.write("\n")

        if clf_configs and CLASSIFIER_NAMES:
            f.write("## Supervised Classifier — Position\n\n")
            header = "| Classifier |"
            sep = "|------------|"
            for cfg_name in clf_configs:
                lbl = ABLATION_CONFIGS[cfg_name]['label']
                header += f" {lbl} Acc | {lbl} F1 | {lbl} Prec | {lbl} Rec |"
                sep += "------|------|------|------|"
            f.write(header + "\n" + sep + "\n")

            for clf_name in CLASSIFIER_NAMES:
                row_str = f"| {clf_name} |"
                for cfg_name in clf_configs:
                    cfg_row = comparison_df[comparison_df["config"] == cfg_name].iloc[0]
                    row_str += (f" {_fmt(cfg_row.get(f'{clf_name}_position_accuracy'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_position_f1_macro'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_position_precision_macro'))} |"
                                f" {_fmt(cfg_row.get(f'{clf_name}_position_recall_macro'))} |")
                f.write(row_str + "\n")
            f.write("\n")

        # ================================================================
        # CONTINUOUS POSITION PREDICTION
        # ================================================================
        f.write("---\n\n# Continuous Spine Position Prediction\n\n")
        f.write("Continuous position is a 0-1 normalized value: `(ordinal - 0.5) / total`.\n")
        f.write("First vertebra ≈ 0.02, last ≈ 0.98.\n\n")

        # Distance-based continuous position
        f.write("## Distance-Based Continuous Position\n\n")
        f.write("| Configuration | Method | MAE | R² |\n")
        f.write("|---------------|--------|-----|----|\n")
        for _, row in comparison_df.iterrows():
            for method in ["cos", "euc"]:
                method_label = "Cosine" if method == "cos" else "Euclidean"
                for suffix, pred_label in [
                    ("top1_continuous_position", "Top-1"),
                    ("top5_weighted_continuous_position", "Top-5 Weighted"),
                ]:
                    mae_key = f"{method}_{suffix}_mae"
                    r2_key = f"{method}_{suffix}_r2"
                    mae_val = row.get(mae_key)
                    r2_val = row.get(r2_key)
                    if isinstance(mae_val, float):
                        f.write(f"| {row['label']} | {method_label} {pred_label} | {mae_val:.4f} | {_fmt(r2_val)} |\n")
        f.write("\n")

        # Regressor continuous position
        regressor_names_for_summary = list(pos_regressors_raw.keys()) if pos_regressors_raw else []
        if regressor_names_for_summary:
            f.write("## Supervised Regressors — Continuous Position\n\n")
            f.write("| Configuration | Regressor | MAE | R² |\n")
            f.write("|---------------|-----------|-----|----|\n")
            for _, row in comparison_df.iterrows():
                for reg_name in regressor_names_for_summary:
                    mae_key = f"{reg_name}_mae"
                    r2_key = f"{reg_name}_r2"
                    mae_val = row.get(mae_key)
                    r2_val = row.get(r2_key)
                    if isinstance(mae_val, float):
                        f.write(f"| {row['label']} | {reg_name} | {mae_val:.4f} | {_fmt(r2_val)} |\n")
            f.write("\n")

        # Derived region accuracy
        f.write("## Derived Region Accuracy\n\n")
        f.write("Accuracy of deriving categorical region (C/T/L) from continuous position ")
        f.write("using specimen-specific boundary counts from vertebrae_counts.csv.\n\n")
        f.write("| Configuration | Derived Region Accuracy |\n")
        f.write("|---------------|------------------------|\n")
        for _, row in comparison_df.iterrows():
            val = row.get("derived_region_accuracy")
            f.write(f"| {row['label']} | {_fmt(val)} |\n")
        f.write("\n")

        # ================================================================
        # IMPACT ANALYSIS
        # ================================================================
        f.write("---\n\n# Impact Analysis\n\n")

        # --- Impact of Metric Learning ---
        f.write("## Impact of Metric Learning\n\n")
        f.write("Comparing A vs B (distance only) and C vs D (with classifiers) "
                "isolates the effect of NCA metric learning.\n\n")

        for pair_label, without_cfg, with_cfg in [
            ("Distance-Based", "A_baseline", "B_metric_learning"),
            ("With Classifiers", "C_classifiers", "D_full_pipeline"),
        ]:
            row_without = comparison_df[comparison_df["config"] == without_cfg]
            row_with = comparison_df[comparison_df["config"] == with_cfg]
            if row_without.empty or row_with.empty:
                continue
            row_without = row_without.iloc[0]
            row_with = row_with.iloc[0]

            f.write(f"### {pair_label}\n\n")

            for target_name, target_key in [("Species", "species"), ("Genus", "genus"), ("Position", "position")]:
                f.write(f"**{target_name}:**\n")
                for metric_label, key in [
                    ("Cosine Top-1", f"cosine_{target_key}_top_1_accuracy"),
                    ("Cosine Top-5", f"cosine_{target_key}_top_5_accuracy"),
                    ("Euclidean Top-1", f"euclidean_{target_key}_top_1_accuracy"),
                    ("Euclidean Top-5", f"euclidean_{target_key}_top_5_accuracy"),
                ]:
                    before = row_without.get(key)
                    after = row_with.get(key)
                    if isinstance(before, float) and isinstance(after, float):
                        delta = after - before
                        sign = "+" if delta >= 0 else ""
                        f.write(f"- {metric_label}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")
                f.write("\n")

            # Classifier deltas (C vs D)
            if pair_label == "With Classifiers" and CLASSIFIER_NAMES:
                for target_name, target_key in [("Species", "species"), ("Genus", "genus"), ("Position", "position")]:
                    f.write(f"**Classifier {target_name} Accuracy (without ML -> with ML):**\n")
                    for clf_name in CLASSIFIER_NAMES:
                        before = row_without.get(f"{clf_name}_{target_key}_accuracy")
                        after = row_with.get(f"{clf_name}_{target_key}_accuracy")
                        if isinstance(before, float) and isinstance(after, float):
                            delta = after - before
                            sign = "+" if delta >= 0 else ""
                            f.write(f"- {clf_name}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")
                    f.write("\n")

            f.write("\n")

        # --- Impact of Classifiers ---
        f.write("## Impact of Supervised Classifiers\n\n")
        f.write("Comparing A vs C (without ML) and B vs D (with ML) "
                "isolates the effect of adding classifiers.\n\n")

        for pair_label, without_cfg, with_cfg in [
            ("Without Metric Learning", "A_baseline", "C_classifiers"),
            ("With Metric Learning", "B_metric_learning", "D_full_pipeline"),
        ]:
            row_without = comparison_df[comparison_df["config"] == without_cfg]
            row_with = comparison_df[comparison_df["config"] == with_cfg]
            if row_without.empty or row_with.empty:
                continue
            row_without = row_without.iloc[0]
            row_with = row_with.iloc[0]

            f.write(f"### {pair_label}\n\n")

            for target_name, target_key in [("Species", "species"), ("Genus", "genus"), ("Position", "position")]:
                cos_val = row_without.get(f"cosine_{target_key}_top_1_accuracy", 0) or 0
                euc_val = row_without.get(f"euclidean_{target_key}_top_1_accuracy", 0) or 0
                best_dist = max(cos_val, euc_val)

                best_clf = 0
                best_clf_name = "N/A"
                for clf_name in CLASSIFIER_NAMES:
                    val = row_with.get(f"{clf_name}_{target_key}_accuracy", 0) or 0
                    if val > best_clf:
                        best_clf = val
                        best_clf_name = clf_name

                if best_dist > 0 or best_clf > 0:
                    f.write(f"**{target_name}:**\n")
                    f.write(f"- Best Distance Top-1: {best_dist:.4f}\n")
                    if best_clf > 0:
                        delta = best_clf - best_dist
                        sign = "+" if delta >= 0 else ""
                        f.write(f"- Best Classifier ({best_clf_name}): {best_clf:.4f} ({sign}{delta:.4f})\n")
                    f.write("\n")

            f.write("\n")

        # Training times
        f.write("## Training Times\n\n")
        f.write("| Component | Time (s) |\n")
        f.write("|-----------|----------|\n")
        f.write(f"| NCA Metric Learning | {ml_time:.4f} |\n")
        for name, t in classifiers_raw_times.items():
            f.write(f"| {name} (raw latents) | {t:.4f} |\n")
        if classifiers_ml_times:
            for name, t in classifiers_ml_times.items():
                f.write(f"| {name} (NCA-transformed) | {t:.4f} |\n")
        if pos_regressors_raw_times:
            for name, t in pos_regressors_raw_times.items():
                f.write(f"| {name} (raw latents) | {t:.4f} |\n")
        if pos_regressors_ml_times:
            for name, t in pos_regressors_ml_times.items():
                f.write(f"| {name} (NCA-transformed) | {t:.4f} |\n")
        f.write("\n")

    print(f"Summary statistics written to: {summary_md_path}")


    # ======================================================================
    # Save training times
    # ======================================================================
    all_training_times = {
        "metric_learning_NCA": ml_time,
        "classifiers_raw": classifiers_raw_times,
        "classifiers_ml": classifiers_ml_times,
        "position_regressors_raw": pos_regressors_raw_times,
        "position_regressors_ml": pos_regressors_ml_times,
    }
    with open(os.path.join(run_dir, "training_times.json"), "w", encoding="utf-8") as f:
        json.dump(all_training_times, f, indent=2)


    # ======================================================================
    # Write top-level run manifest
    # ======================================================================
    produced_files = {
        "README.md": "Top-level run manifest for this ablation study",
        "summary_statistics.md": "Human-readable summary with comparison tables and impact analysis",
        "comparison_metrics.csv": "All configurations side-by-side in a single CSV",
        "training_times.json": "Training times for all components",
        "comparison_charts/": "Cross-configuration comparison visualizations",
        "comparison_charts/species_accuracy_all_approaches.png": "All approaches x all configs — species accuracy",
        "comparison_charts/species_f1_all_approaches.png": "Classifier F1 across configs — species",
        "comparison_charts/species_precision_recall.png": "Precision & recall across configs — species",
        "comparison_charts/species_distance_comparison.png": "Distance retrieval across configs — species",
        "comparison_charts/genus_accuracy_all_approaches.png": "All approaches x all configs — genus accuracy",
        "comparison_charts/genus_distance_comparison.png": "Distance retrieval across configs — genus",
        "comparison_charts/position_accuracy_all_approaches.png": "All approaches x all configs — position accuracy",
        "comparison_charts/position_f1_all_approaches.png": "Classifier F1 across configs — position",
        "comparison_charts/position_precision_recall.png": "Precision & recall across configs — position",
        "comparison_charts/position_distance_comparison.png": "Distance retrieval across configs — position",
        "comparison_charts/best_approach_all_targets.png": "Best approach per config, species vs genus vs position faceted",
    }
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        produced_files[f"{cfg_name}/"] = cfg["description"]

    write_run_manifest(
        run_dir,
        description=(
            f"Ablation study comparing {len(ABLATION_CONFIGS)} classification configurations "
            f"on {len(mesh_list)} test meshes from {RUN_NAME} checkpoint {CKPT}. "
            f"Isolates the contribution of NCA metric learning and supervised classifiers."
        ),
        approach=f"Ablation: Baseline / +Metric Learning / +Classifiers / +Both (model: {RUN_NAME})",
        script_path=os.path.abspath(__file__),
        train_data_paths=train_paths,
        test_data_paths=[os.path.basename(p) for p in mesh_list],
        config=config,
        classifier_names=CLASSIFIER_NAMES,
        metric_learning_method="NCA (toggled per config)",
        checkpoint=CKPT,
        notes=(
            f"Model Run: {RUN_NAME} (checkpoint {CKPT})\n\n"
            "Ablation Configurations:\n"
            "  A) Baseline — raw latent codes, distance-based retrieval only\n"
            "  B) + Metric Learning — NCA-transformed latents, distance-based retrieval\n"
            "  C) + Classifiers — raw latents, supervised classifiers (KNN/SVM/RF/MLP/LR)\n"
            "  D) + ML + Classifiers — NCA-transformed latents + classifiers (full v3 pipeline)\n\n"
            "v5 fixes from v4:\n"
            "- Distance-based species matching now uses parse_taxonomy_from_filename consistently\n"
            "  for BOTH ground truth AND predictions (v4 used different parsers, causing 0% accuracy)\n"
            "- Added genus and family top-1/top-5 accuracy for distance methods\n"
            "- Added position top-5 accuracy for distance methods\n"
            "- Per-specimen predictions include full top-5 taxonomy breakdown\n\n"
            "Metrics computed per configuration:\n"
            "- Instance Accuracy, Average Class Accuracy\n"
            "- Precision (macro, weighted, per-class)\n"
            "- Recall (macro, weighted, per-class)\n"
            "- F1 (macro, weighted, per-class)\n"
            "- Top-1 / Top-5 Accuracy (distance methods) at species/genus/family/position levels\n"
            "- Continuous spine position: MAE, RMSE, R², Spearman correlation\n"
            "- Position regressors (KNN, SVR, RF, MLP) on continuous 0-1 position\n"
            "- Derived region accuracy: categorical region from continuous position\n"
            "- Confusion Matrices (species, genus, family, spinal position)\n"
            "- Scatter/residual/error-by-region plots for continuous position\n"
            "- Training and inference times\n\n"
            "Cross-configuration outputs:\n"
            "- Side-by-side comparison CSV and markdown tables\n"
            "- Impact analysis: isolated effect of metric learning and classifiers\n"
            "- Comparison bar charts for species, genus, and position"
        ),
        extra_files=produced_files,
    )

    print(f"\n{'='*60}")
    print(f"Ablation study complete! All results saved to:")
    print(f"  {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
