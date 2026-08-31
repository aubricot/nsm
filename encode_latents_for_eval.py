"""
Bulk encode train, validation, and test sets into latent space for classification evaluation.
Uses the same SDF sampling and optimization logic as the classification baseline.
"""
import argparse
import os
import sys
import numpy as np

import torch
import pymskt.mesh.meshes as meshes
import pymskt.mesh.meshTools as meshTools

from NSM.helper_funcs import fixed_point_coords, load_config, load_model_and_latents, safe_load_mesh_scalars, convert_ply_to_vtk
from NSM.optimization import get_top_k_pcs, optimize_latent, build_sdf_dataset

# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

def resolve_model_root(root_dir):
    config = os.path.join(root_dir, "model_params_config.json")
    model_dir = os.path.join(root_dir, "model")
    latent_dir = os.path.join(root_dir, "latent_codes")

    missing = []
    if not os.path.isfile(config):
        missing.append("model_params_config.json")
    if not os.path.isdir(model_dir):
        missing.append("model/")
    if not os.path.isdir(latent_dir):
        missing.append("latent_codes/")
    if missing:
        raise ValueError("Invalid model package. Missing: {}".format(missing))

    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith(".pth")])

    if not model_files:
        raise ValueError("No .pth file found in model/")
    if not latent_files:
        raise ValueError("No .pth file found in latent_codes/")

    model_path = os.path.join(model_dir, model_files[-1])
    latent_path = os.path.join(latent_dir, latent_files[-1])

    return config, model_path, latent_path

def _load_model_bundle(args, device):
    config_path, model_path, latent_path = resolve_model_root(args.model_root)
    config = load_config(config_path)
    print("Classification model root: {}".format(args.model_root))
    print("Classification config: {}".format(config_path))
    print("Classification model: {}".format(model_path))
    print("Classification latent codes: {}".format(latent_path))
    print("Classification device: {}".format(device))
    print("Classification iterations: {}".format(args.iterations))
    
    model, _, latent_codes = load_model_and_latents(model_path, latent_path, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
    
    return config, model, latent_codes, mean_latent, top_k_reg

def _encode_split(split_name, mesh_paths, config, model, latent_codes, mean_latent, top_k_reg, device, args):
    if not mesh_paths:
        print("No meshes found for split: {}".format(split_name))
        return

    print("\n--- Encoding {} split ({} meshes) ---".format(split_name, len(mesh_paths)))
    optimized_latents = []

    for i, mesh_path in enumerate(mesh_paths, start=1):
        base = os.path.splitext(os.path.basename(mesh_path))[0]
        print(f"\033[32m\n=== Optimizing {base} ===\033[0m")
        print(f"\033[32m\n=== {i} of {len(mesh_paths)} ===\033[0m")
        
        # 1. Prepare Mesh
        _, prepared_path = convert_ply_to_vtk(mesh_path)
        
        # 2. Build Dataset (exactly as done in the baseline)
        points, sdf_vals, _, _ = build_sdf_dataset(prepared_path, config, config["n_pts_per_object"])
        
        # 3. Optimize Latent
        latent = optimize_latent(model, points.squeeze(), sdf_vals, config["latent_size"], 
                                 top_k_reg, mean_latent, latent_codes, 
                                 iters=args.iterations, lr=args.learning_rate, device=device)  
        optimized_latents.append(latent.detach().cpu())

    # Stack into [N, latent_size]
    stacked_latents = torch.stack(optimized_latents)
    if stacked_latents.dim() == 3 and stacked_latents.shape[1] == 1:
        stacked_latents = stacked_latents.squeeze(1)

    out_file = os.path.join(args.model_root, args.output_dir, "latent_codes_{}.pth".format(split_name))
    torch.save(stacked_latents, out_file)
    print("Saved {} latents to: {}".format(split_name, out_file))

def encode_datasets(args):
    repository_root = os.path.dirname(os.path.abspath(__file__))
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    
    os.makedirs(os.path.join(args.model_root, args.output_dir), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, model, latent_codes, mean_latent, top_k_reg = _load_model_bundle(args, device)

    # Encode files fromd dataset split (train, val, test)
    ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
    split_key = ds_split_keys[args.dataset_split]
    ds_paths = config[split_key]
    _encode_split(split_key, ds_paths, config, model, latent_codes, mean_latent, top_k_reg, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode train, val, and test sets into latents for evaluation.")
    parser.add_argument("--model_root", required=True, help="Path to the model directory (e.g. run_vXX)")
    parser.add_argument("--output_dir", required=True, help="Directory to save latent_codes_{train,val,test}.pth (e.g. classification/evaluation/encoded_latents)")
    parser.add_argument("--dataset_split", choices=["train", "val", "test"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    
    encode_datasets(args)
