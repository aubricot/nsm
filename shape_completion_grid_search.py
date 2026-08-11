# Fine-tune parameters for shape completion of partial vertebrae
# Use partial vertebrae dataset made with create_partial_meshes.py for validation against ground truth meshes

import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.reconstruct import reconstruct_latent
import torch.nn.functional as F
import json
import sys
import pyvista as pv
import pymskt.mesh.meshes as meshes
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from NSM.mesh import create_mesh
import vtk
import re
import random
import open3d as o3d
from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, optimize_latent_partial, normalize_mesh, get_norm_params, encode_latent, reconstruct_mesh_from_latent, build_sdf_dataset
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)
import time

# Define training directory
TRAIN_DIR = "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
CKPT = '2500' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH =  TRAIN_DIR + '/latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = TRAIN_DIR +  '/model' + '/' + CKPT + '.pth'
val_sum_fn = TRAIN_DIR + "/shape_completion/meshes/" + "partial_meshing_summary.json" # TO DO: Choose path to partial_meshing_summary.json from (generated using create_partial_meshes.py)
N_TRIALS = 15   # TO DO: Choose the number of trials for the grid search
N_TRIAL_INF = 30   # TO DO: Choose the number of meshes to use for inference in each grid search trial
N_FINAL_INF = 50  # TO DO: Choose how many meshes to use for final inference with best config reconstruction parameters from grid search

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

#  Load partial meshing summary
with open(val_sum_fn, "r") as f:
    partial_mesh_summary = json.load(f)

# Build partial_mesh_path and ground_truth_path pairs
def strip_mesh_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name.endswith("_partial"):
        name = name[:-8]
    return name

# Extract test mesh names from config to use as ground truths
test_mesh_names = {strip_mesh_name(p) for p in config["test_paths"]}
print(f"Found {len(test_mesh_names)} test meshes in config")

# Find corresponding partial meshes to use for shape completion against ground truth meshes
pairs = []
skipped = 0
for m in partial_mesh_summary["meshes"]:
    base_name = m["base_name"]
    if base_name in test_mesh_names:
        pairs.append((m["partial"], m["ground_truth"])) # Add paths to partial and ground_truth meshes
    else:
        skipped += 1

print(f"Built {len(pairs)} (partial, ground_truth) pairs")
print(f"Skipped {skipped} meshes not in test_paths")

# Accuracy Metrics
def _uniform_surface_sample(poly, n):
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
def chamfer_distance(pred_path, gt_path, n_samples=20000):
    # Read in completed and ground truth meshes
    mp = pv.read(pred_path).triangulate().extract_surface(algorithm=None)
    gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
    # Sample points across surface
    sp = _uniform_surface_sample(mp, n_samples)
    sg = _uniform_surface_sample(gt, n_samples)
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0].mean()
    d2 = t2.query(sp, k=1)[0].mean()
    return float(0.5*(d1 + d2)) # Return average distance (symmetric penalty)

# Random search on a small validation subset to pick best cfg
def grid_search(pairs, model, config, mean_latent, latent_codes, device, out_dir, n_trials=15, valN=30, log_path_csv=None, log_path_json=None):
    # Set up directory for fine-tuning experiemnts
    os.makedirs(out_dir, exist_ok=True)
    val_subset = pairs[:valN]
    best = {'score': float('inf'), 'cfg': None}
    rows = []
    # Define how many PCs describe X% of variance
    _, k95 = get_top_k_pcs(latent_codes, threshold=0.95)
    _, k90 = get_top_k_pcs(latent_codes, threshold=0.90)
    _, k99 = get_top_k_pcs(latent_codes, threshold=0.99)
    latent_std = latent_codes.std().mean()

    # Randomly pick optimization parameters from provided values
    for t in range(n_trials):
        trial_cfg = {
            'top_k': random.choice([k95, k90, k99]),
            'iters1': random.choice([3000, 5000, 7000]),
            'iters2': random.choice([6000, 8000, 10000]),
            'lr1': random.choice([1e-5, 1e-4, 1e-3]),
            'lr2': random.choice([1e-6, 1e-5, 1e-4]),
            'lambda1': random.choice([1e-6, 1e-4, 1e-2]),
            'lambda2': random.choice([1e-7, 1e-5, 1e-3]),
            'clamp1': random.choice([None, 1, 2]),
            'clamp2': random.choice([None, 1, 2]),
            'latent_std': latent_std,
            'sched_step': random.choice([500, 800, 1000]),
            'sched_gamma1': random.choice([0.5, 0.7, 0.9]),
            'sched_gamma2': random.choice([0.5, 0.7, 0.9]),
            'batch_infer': random.choice([16384, 32768]),
            'gridN': random.choice([256, 320, 384]),}
        scores = []
        times = []
        # Set up directory for each trial
        trial_dir = os.path.join(out_dir, f"trial_{t:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        # Run trial on randomly chosen config params and log chamfer score
        for i, (pred_path, gt_path) in enumerate(val_subset):
            start = time.time()
            if '.ply' in pred_path:
                _, pred_path = convert_ply_to_vtk(pred_path, save=True)
            # Build SDF dataset
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(pred_path, config, n_samples=240)
            # Encode latent via 2 stage optimization (auto-decoder framework)
            latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                               mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=trial_cfg['top_k'], latent_std=trial_cfg['latent_std'],
                               iters1=trial_cfg['iters1'], iters2=trial_cfg['iters2'], lr1=trial_cfg['lr1'], lr2=trial_cfg['lr2'], 
                               lambda_reg1=trial_cfg['lambda1'], lambda_reg2=trial_cfg['lambda2'], clamp_val1=trial_cfg['clamp1'], clamp_val2=trial_cfg['clamp2'], 
                               scheduler_step1=trial_cfg['sched_step'], scheduler_step2=trial_cfg['sched_step'], scheduler_gamma1=trial_cfg['sched_gamma1'],
                               scheduler_gamma2=trial_cfg['sched_gamma2'], batch_inference_size=trial_cfg['batch_infer']) 
            # Reconstruction mesh from latent
            mesh_out = reconstruct_mesh_from_latent(pred_path, model, latent_opt, trial_cfg)
            
            # Normalize and scale output using model config training params
            center, max_radius = get_norm_params(sdf_dataset, sample_dict, pred_path)
            mesh_pv = normalize_mesh(mesh_out, pred_path, config, center, max_radius)
            
            # Save to file
            mesh_pv = mesh_pv.clean().triangulate()
            for arr in ['RegionId', 'vtkOriginalCellIds']:
                if arr in mesh_pv.cell_data.keys():
                    mesh_pv.cell_data.remove(arr)
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            new_filename = f"{base_name}_completed.vtk"
            completed_path = os.path.join(trial_dir, new_filename)
            mesh_pv.save(completed_path)

            # Calculate chamfer distance between predicted mesh and ground truth
            cd = chamfer_distance(completed_path, gt_path)
            mesh_time = time.time() - start
            scores.append(cd)
            times.append(mesh_time)

        # Get mean chamfer for all meshes from trial
        mean_cd = float(np.mean(scores))
        if mean_cd < best['score']:
            best = {'score': mean_cd, 'cfg': trial_cfg}
        # Append the current trial's results to the list
        mean_time = float(np.mean(times))
        rows.append({'trial': t, 'mean_cd': mean_cd, 'mean_time': mean_time, **trial_cfg})
        # Save results to csv
        if log_path_csv is not None:
            pd.DataFrame(rows).to_csv(log_path_csv, index=False)
    print(f"Best cfg: {best['cfg']} (mean Chamfer={best['score']:.4f})")
    # Save logs
    if log_path_csv:
        print(f"Finished. Final trial log with all trials saved to {log_path_csv}")
    return best['cfg'], rows

# Find the best hyperparameters for shape completion using random grid search
best_cfg, trial_rows = grid_search(pairs, model, config, mean_latent, latent_codes, device,
                                    out_dir= TRAIN_DIR + "/shape_completion/fine_tuning",
                                    n_trials=N_TRIALS, valN=N_TRIAL_INF,
                                    log_path_csv= TRAIN_DIR + "/shape_completion/fine_tuning/trial_scores.csv")

# Loop through meshes using best parameters from grid search
best_summary_log = []
inf_subset = random.sample(pairs, N_FINAL_INF) # Get the inference subset
for pm_path, gt_path in inf_subset:    
    try:
        print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
        # Make a new dir to save predictions
        vert_fname = pm_path
        outfpath = TRAIN_DIR + '/shape_completion/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
        print("Making a new directory to save model predictions and outputs at: ", outfpath)
        os.makedirs(outfpath, exist_ok=True)

        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

        # Setup your dataset with just one mesh
        points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=240)

        # Encode latent via 2 stage optimization (auto-decoder framework)
        latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                                mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=best_cfg['top_k'], latent_std=best_cfg['latent_std'],
                                iters1=best_cfg['iters1'], iters2=best_cfg['iters2'], lr1=best_cfg['lr1'], lr2=best_cfg['lr2'], 
                                lambda_reg1=best_cfg['lambda1'], lambda_reg2=best_cfg['lambda2'], clamp_val1=best_cfg['clamp'], clamp_val2=None, 
                                scheduler_step1=best_cfg['sched_step'], scheduler_step2=best_cfg['sched_step'], scheduler_gamma1=best_cfg['sched_gamma'],
                                scheduler_gamma2=best_cfg['sched_gamma'], batch_inference_size=best_cfg['batch_infer']) 

        # Reconstruction mesh from latent
        mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, best_cfg)

        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        
        # Save to file
        mesh_pv = mesh_pv.clean().triangulate()
        for arr in ['RegionId', 'vtkOriginalCellIds']:
            if arr in mesh_pv.cell_data.keys():
                mesh_pv.cell_data.remove(arr)
        base_name = os.path.splitext(os.path.basename(vert_fname))[0]
        outfname = f"{base_name}_completed.vtk"
        compl_path = os.path.join(outfpath, outfname)
        mesh_pv.save(compl_path)

        # Calculate chamfer distance between shape completed and original-ground truth mesh
        cd = chamfer_distance(compl_path, gt_path)
        print(f"\n{os.path.basename(pm_path)} Chamfer={cd:.4f} → {compl_path}\n") 
        best_summary_log.append({'mesh': os.path.basename(pm_path),
                            'chamfer': cd,
                            'completed_path': compl_path,
                            'gt_path': gt_path})

    except Exception as e:
        print(f"\033[31mERROR processing {os.path.basename(pm_path)}: {e}\033[0m")
        best_summary_log.append({'mesh': os.path.basename(pm_path),
                                 'chamfer': None,
                                 'completed_path': None,
                                 'gt_path': gt_path})
        continue

summary_df = pd.DataFrame(best_summary_log)
summary_df_fpath = TRAIN_DIR + '/shape_completion/predictions/inference_summary.csv'
summary_df.to_csv(summary_df_fpath, index=False)
print(f"\nMean Chamfer across {len(summary_df)} meshes: {summary_df['chamfer'].mean():.4f}")
print("Outputs logged to: ", summary_df_fpath)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)