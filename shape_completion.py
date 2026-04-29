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
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, optimize_latent_partial, normalize_mesh, get_norm_params
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Define training directory
TRAIN_DIR = "run_v57" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Load model config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")

# Select matching paths of partial meshes for shape completion
mesh_dir = "fossils/models_smooth_hollow/aligned"
mesh_list = os.listdir(mesh_dir)
mesh_list = [os.path.join(mesh_dir, f) for f in random.sample(mesh_list, 5)]
#mesh_list = [os.path.join(mesh_dir, f) for f in mesh_list]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
latent_std = latent_codes.std().mean()
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

# Loop through meshes
summary_log = []
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

    # Setup your dataset with just one mesh
    sdf_dataset = SDFSamples(
        list_mesh_paths=[vert_fname],
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
        scale_method=config['scale_method'],
        scale_jointly=config['scale_jointly'],
        reference_mesh=None,
        verbose=config['verbose'],
        save_cache=config['cache'],
        equal_pos_neg=config['equal_pos_neg'],
        fix_mesh=config['fix_mesh'])

    # Get the point/SDF data
    print("\n-----Setting up dataset-----\n")
    sdf_sample = sdf_dataset[0]  # returns a dict
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf'].to(device)  # shape: [N, 1]

    # Use a subset of the points for optizimation/reconstruction
    n_samples = 240 # TO DO: Define how many points to sample
    indices = torch.randperm(points.size(0))[:n_samples] # Generate n_samples random indices
    # Downsample the points and SDF values
    points = points[indices]
    points = points.squeeze()
    sdf_vals = sdf_vals[indices]
    sdf_vals = sdf_vals.reshape(-1, 1)

    # Optimize latents
    print("\n-----Optimizing latents----\n")
    # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg, 
                                                       iters=3000, lr=1e-4, lambda_reg=1e-3, clamp_val=1.0, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.9, 
                                                       batch_inference_size=32768, multi_stage=False, device=device)
    # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg, 
                                                        iters=8000, lr=1e-5, lambda_reg=1e-5, clamp_val=None, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7, 
                                                        batch_inference_size=32768, multi_stage=True, device=device) # True because second stage using already initialized latent
    print("\nTranslated novel mesh into latent space!\n")
    
    # Reconstruction parameters
    recon_grid_origin = 1.0
    n_pts_per_axis = 256 # TO DO: Adjust resolution
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # Reconstruct the novel mesh
    with torch.no_grad():
        mesh_out = create_mesh(decoder=model, latent_vector=latent_partial, n_pts_per_axis=n_pts_per_axis,
                                voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=vert_fname,
                                offset=offset, scale=scale, icp_transform=icp_transform, objects=objects,
                                verbose=True, device=device, scale_to_original_mesh=False) #, smooth=1.0)
        
    # Normalize and scale output using model config training params
    center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
    mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)

    # Save mesh
    mesh_pv = mesh_pv.clean()
    mesh_pv = mesh_pv.triangulate()
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_shape_completion.vtk"
    # Set color: RGB in range 0–255 or 0–1
    color = np.array([112, 215, 222], dtype=np.uint8)  
    # Broadcast color to all points
    rgb = np.tile(color, (mesh_pv.n_points, 1))
    mesh_pv.point_data.clear()
    mesh_pv.point_data['Colors'] = rgb
    mesh_pv.save(output_path)
    print(f"Completed mesh from partial pointcloud saved to: {output_path}")