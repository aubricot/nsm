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
import re
import random
from NSM.helper_funcs import load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import normalize_mesh, get_norm_params, encode_latent, encode_latent_pointnet, reconstruct_mesh_from_latent, build_sdf_dataset
from NSM.evaluation import strip_partial_mesh_name, build_partial_gt_mesh_pairs, uniform_surface_sample, chamfer_distance, load_best_cfg_from_csv, grid_search, grid_search_pointnet
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
N_FINAL_INF = 50   # TO DO: Choose how many meshes to use for final inference with best config reconstruction parameters from grid search
split = "val"  # TO DO: Which dataset split to use - "train", "val", or "test"
fast_mode = True  # TO DO: Use fast mode to encode via PointNet or using 2-phase latent optimization (slow)
if fast_mode == True:
    OUTDIR = TRAIN_DIR + "/shape_completion/fine_tuning_encoder"
    encoder_path = TRAIN_DIR + "/encoder/checkpoints/encoder.pt" # TO DO: Point to encoder ckpt
    encoder_ckpt = os.path.abspath(encoder_path)
else: 
    OUTDIR = TRAIN_DIR + "/shape_completion/fine_tuning"

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

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

# Find the best hyperparameters for shape completion using random grid search
if fast_mode: 
    # Fast mode: one-shot encoder instead of full latent optimization 
    best_cfg, trial_rows = grid_search_pointnet(pairs, model, config, mean_latent, latent_codes, device, encoder_ckpt,
                                                out_dir=OUTDIR,
                                                n_trials=N_TRIALS, valN=N_TRIAL_INF,
                                                log_path_csv=OUTDIR + "/trial_scores.csv")

else:   
    # Encode latent via 2 stage optimization (auto-decoder framework)
    best_cfg, trial_rows = grid_search(pairs, model, config, mean_latent, latent_codes, device, 
                                        out_dir=OUTDIR,
                                        n_trials=N_TRIALS, valN=N_TRIAL_INF,
                                        log_path_csv=OUTDIR + "/trial_scores.csv")

# Loop through meshes using best parameters from grid search
best_summary_log = []
inf_subset = random.sample(pairs, N_FINAL_INF)
for i, (pm_path, gt_path) in enumerate(inf_subset, start=1):    
    try:
        print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
        print(f"\033[32m\n=== {i} of {len(inf_subset)} ===\033[0m")
        # Make a new dir to save predictions
        vert_fname = pm_path
        outfpath = OUTDIR + '/best_cfg_predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
        print("Making a new directory to save model predictions and outputs at: ", outfpath)
        os.makedirs(outfpath, exist_ok=True)

        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

        # Latent encoding
        if fast_mode:
            # Build the SDF dataset using all surface samples
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=None) # Use all points instead of downsampling by n_samples
           
            # Fast mode: one-shot encoder instead of full latent optimization 
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_opt = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, device)
            if best_cfg['iters'] > 0:
                latent_opt, _ = optimize_latent_partial(decoder=model, partial_pts=points.squeeze(), sdfs=sdf_vals, latent_dim=latent_codes.shape[1], latent_init=latent_opt, 
                                                        top_k=best_cfg['top_k'], iters=best_cfg['iters'], lr=best_cfg['lr'], lambda_reg=best_cfg['lambda_reg'], 
                                                        clamp_val=best_cfg['clamp'], latent_std=best_cfg['latent_std'], scheduler_step=best_cfg['sched_step'], 
                                                        scheduler_gamma=best_cfg['sched_gamma'], batch_inference_size=best_cfg['batch_infer'], multi_stage=True, device=device)        

        else:
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
        compl = pv.read(compl_path).triangulate().extract_surface(algorithm=None)
        gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
        cd = chamfer_distance(compl, gt, n_samples=20000)
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
summary_df_fpath = OUTDIR + '/best_cfg_predictions/inference_summary.csv'
summary_df.to_csv(summary_df_fpath, index=False)
print(f"\nMean Chamfer across {len(summary_df)} meshes: {summary_df['chamfer'].mean():.4f}")
print("Outputs logged to: ", summary_df_fpath)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)