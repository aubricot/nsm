# Fine-tune parameters for shape completion of partial vertebrae
# Use partial vertebrae dataset made with create_partial_meshes.py for validation against ground truth meshes

import os
import torch
import numpy as np
import pandas as pd
import json
import sys
import pyvista as pv
import pymskt.mesh.meshes as meshes
import vtk
import random
from NSM.helper_funcs import load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import normalize_mesh, get_norm_params, encode_latent, reconstruct_mesh_from_latent, build_sdf_dataset
from NSM.evaluation import strip_partial_mesh_name, build_partial_gt_mesh_pairs, uniform_surface_sample, chamfer_distance, load_best_cfg_from_csv
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
N_INF = 5  # TO DO: Choose how many meshes to use for inference (or use "all" to run for all)
BEST_CFG_CSV = TRAIN_DIR + "/shape_completion/fine_tuning/trial_scores.csv"  # TO DO: set to your CSV path
LOAD_BEST_CFG_FROM_CSV = False
split = "val"  # TO DO: Which dataset split to use - "train", "val", or "test"

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Load best optimization config from shape_completion_grid_search.py (or manually enter)
if LOAD_BEST_CFG_FROM_CSV:
    best_cfg = load_best_cfg_from_csv(BEST_CFG_CSV, device)

else:           # TO DO: Manually enter chosen vals
    best_cfg = {"top_k": 466,
                "iters1": 5000,
                "iters2": 8000,
                "lr1": 1e-4,
                "lr2": 1e-4,
                "lambda1": 1e-2,
                "lambda2": 1e-7,
                "clamp1": 1,
                "clamp2": None,
                "latent_std": torch.tensor(0.4401),
                "sched_step": 800,
                "sched_gamma1": 0.7,
                "sched_gamma2": 0.9,
                "batch_infer": 32768,
                "gridN": 256}

# Build validation ground truth dataset
ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
split_key = ds_split_keys[split]
mesh_names = {strip_partial_mesh_name(p) for p in config[split_key]}
print(f"Found {len(config[split_key])} meshes in config['{split_key}'] (split='{split}')")

# Load partial meshing summary
with open(val_sum_fn, "r") as f:
    partial_mesh_summary = json.load(f)

# Find corresponding partial meshes to use for shape completion against ground truth meshes
pairs = build_partial_gt_mesh_pairs(partial_mesh_summary, mesh_names, split_key)

# Loop through meshes using best parameters from grid search
summary_log = []
inf_subset = random.sample(pairs, N_INF) if N_INF != "all" else pairs
print(f"Using {len(inf_subset)} (partial, ground_truth) pairs for inference")

for pm_path, gt_path in inf_subset:    
    try:
        print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
        # Make a new dir to save predictions
        vert_fname = pm_path

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
                                lambda_reg1=best_cfg['lambda1'], lambda_reg2=best_cfg['lambda2'], clamp_val1=best_cfg['clamp1'], clamp_val2=best_cfg['clamp2'], 
                                scheduler_step1=best_cfg['sched_step'], scheduler_step2=best_cfg['sched_step'], scheduler_gamma1=best_cfg['sched_gamma1'],
                                scheduler_gamma2=best_cfg['sched_gamma2'], batch_inference_size=best_cfg['batch_infer']) 

        # Reconstruction mesh from latent
        mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, best_cfg)

        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        
        # Calculate chamfer distance between shape completed and original-ground truth mesh
        pred = mesh_pv.clean().triangulate().extract_surface(algorithm=None)
        gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
        cd = chamfer_distance(pred, gt, n_samples=20000)
        print(f"\n{os.path.basename(pm_path)} Chamfer={cd:.4f}\n") 
        summary_log.append({'mesh': os.path.basename(pm_path),
                                 'chamfer': cd,
                                 'gt_path': gt_path})

    except Exception as e:
        print(f"\033[31mERROR processing {os.path.basename(pm_path)}: {e}\033[0m")
        summary_log.append({'mesh': os.path.basename(pm_path),
                            'chamfer': None,
                            'gt_path': gt_path})
        continue

# Save results
summary_df = pd.DataFrame(summary_log)
outfpath = TRAIN_DIR + '/shape_completion/evaluation/'
os.makedirs(outfpath, exist_ok=True)

summary_df_fpath = outfpath + f"{split_key}_chamfer.csv"
summary_df.to_csv(summary_df_fpath, index=False)

print(f"\nMean Chamfer across {len(summary_df)} meshes: {summary_df['chamfer'].mean():.4f}")
print("Outputs logged to: ", summary_df_fpath)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)