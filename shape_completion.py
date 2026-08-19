# Shape completion for partial vertebrae

import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.reconstruct import reconstruct_latent
import torch.nn.functional as F
import json
import pyvista as pv
import pymskt.mesh.meshes as meshes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from NSM.mesh import create_mesh
import vtk
import re
import random
import open3d as o3d
from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars, find_shape_completion_files 
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, optimize_latent_partial, normalize_mesh, get_norm_params, encode_latent, reconstruct_mesh_from_latent, build_sdf_dataset, encode_latent_pointnet
from NSM.evaluation import load_best_cfg_from_csv
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Define training directory
TRAIN_DIR = "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '2500' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
N_INF = 5  # TO DO: Choose how many meshes to use for inference (or use "all" to run for all)
BEST_CFG_CSV = TRAIN_DIR + "/shape_completion/fine_tuning/trial_scores.csv"  # TO DO: set to your CSV path; produced by shape_completion_grid_search.py
LOAD_BEST_CFG_FROM_CSV = False
mesh_dir = "fossils/models_smooth_hollow/aligned" # TO DO: Define your mesh directory
fast_mode = True  # TO DO: Use fast mode to encode via PointNet or using 2-phase latent optimization (slow)
if fast_mode == True:
    encoder_path = "encoder/checkpoints/encoder.pt" # TO DO: Point to encoder ckpt
    encoder_ckpt = os.path.abspath(encoder_path)
    refine_iters = 0   # TO DO: Refine pointnet encoding with additional iterations (Ex: 300-500)
    refine_lr = 1e-5 # TO DO: Define learning rate for refine_iters  (ex: 1e-3 - 1e-5)
    refine_lambda_reg = 1e-5 # TO DO: Define lambda reg for refine_iters  (ex: 1e-4 - 1e-7)


# Load model config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")

# Select matching paths of partial meshes for shape completion
mesh_list = os.listdir(mesh_dir)
mesh_list = [os.path.join(mesh_dir, f) for f in mesh_list]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
latent_std = latent_codes.std().mean()
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

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

# Loop through meshes
summary_log = []
inf_subset = random.sample(mesh_list, N_INF) if N_INF != "all" else mesh_list
for i, vert_fname in enumerate(mesh_list):    
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    print(f"\033[32m\n=== Mesh {i+1} / {len(mesh_list)} ===\033[0m")
    # Make a new dir to save predictions
    outfpath = 'shape_completion/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
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
        if refine_iters > 0:
            lr = refine_lr if refine_lr is not None else phase2_lr
            lam = refine_lambda_reg if refine_lambda_reg is not None else phase2_lambda_reg
            print(f"Refining encoder latent for {refine_iters} iters (lr={lr}, lambda={lam})...")
            latent_opt, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_opt, top_k=top_k_reg,
                                                    iters=refine_iters, lr=lr, lambda_reg=lam, clamp_val=None, latent_std=latent_std, scheduler_step=800, 
                                                    scheduler_gamma=0.7, batch_inference_size=32768, multi_stage=True, device=device)        

    else:
        # Build the SDF dataset using downsampled surface samples
        points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=240)
    
        # Encode latent via 2 stage optimization (auto-decoder framework)
        latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                                    mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=top_k_reg, latent_std=best_cfg['latent_std'],
                                    iters1=best_cfg['iters1'], iters2=best_cfg['iters2'], lr1=best_cfg['lr1'], lr2=best_cfg['lr2'], 
                                    lambda_reg1=best_cfg['lambda1'], lambda_reg2=best_cfg['lambda2'], clamp_val1=best_cfg['clamp1'], clamp_val2=best_cfg['clamp2'], 
                                    scheduler_step1=best_cfg['sched_step'], scheduler_step2=best_cfg['sched_step'], scheduler_gamma1=best_cfg['sched_gamma1'],
                                    scheduler_gamma2=best_cfg['sched_gamma2'], batch_inference_size=best_cfg['batch_infer']) 

    # Reconstruction mesh from latent
    mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, config)

    # Normalize and scale output using model config training params
    center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
    mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)

    # Save mesh
    mesh_pv = mesh_pv.clean().triangulate().extract_surface(algorithm=None)
    for arr in ['RegionId', 'vtkOriginalCellIds']:
        if arr in mesh_pv.cell_data.keys():
            mesh_pv.cell_data.remove(arr)
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_shape_completion.vtk"
    # Set color: RGB in range 0–255 or 0–1
    color = np.array([112, 215, 222], dtype=np.uint8)  
    # Broadcast color to all points
    rgb = np.tile(color, (mesh_pv.n_points, 1))
    mesh_pv.point_data.clear()
    mesh_pv.point_data['Colors'] = rgb
    mesh_pv.save(output_path)
    print(f"Completed mesh from partial pointcloud saved to: {output_path}")