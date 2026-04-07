"""
Classify novel vertebrae meshes using Neural Shape Models (NSM).

This script takes novel VTK mesh files and projects them into the learned latent
space to find similar vertebrae from the training set.

PREREQUISITES:
    Before using --init family/genus/species (latent prior initialization), you must first run:
        python create_latents_per_group.py

    This generates the average latent vectors (NPZ files) for each taxonomic
    group in {TRAIN_DIR}/latent_codes_average/{family,genus,species}/

USAGE:
    # Default: no taxonomic initialization, uses PCA-based init (original behavior):
    python classify_vertebrae_normpos.py

    # Initialize latents using family-level averages:
    python classify_vertebrae_normpos.py --init family

    # Initialize latents using genus-level averages:
    python classify_vertebrae_normpos.py --init genus

    # Initialize latents using species-level averages:
    python classify_vertebrae_normpos.py --init species

    # Custom suffix for output files:
    python classify_vertebrae_normpos.py --suffix my_run_001

    # Add Gaussian noise to vtk sampled points:
    python classify_vertebrae_normpos.py --sample_noise_std 0.03

    # Combine options:
    python classify_vertebrae_normpos.py --init genus --suffix test_run --sample_noise_std 0.05

ARGUMENTS:
    --init {none,family,genus,species}
        Initialization method for latent vectors. Default: none
        - none: Use PCA-based initialization near mean (original behavior)
        - family: Use average latent of the mesh's taxonomic family
        - genus: Use average latent of the mesh's taxonomic genus
        - species: Use average latent of the mesh's taxonomic species

    --suffix TEXT
        Suffix for output files (CSV summary, VTK predictions). Default: timestamp

    --latent_noise_std FLOAT
        Standard deviation of Gaussian noise to add to initial latent. Default: 0.0

    --sample_noise_std FLOAT
        Standard deviation of Gaussian noise to add to sampled x,y,z point coordinates. Default: 0.0

    --run_name TEXT
        Training directory name containing the model checkpoint, latent codes,
        and model_params_config.json. Default: run_v57a

    --save_vtk {true,false}
        Whether to reconstruct and save the decoded VTK mesh for each novel
        vertebra. Set to false to skip mesh reconstruction and speed up runs.
        Default: true

OUTPUT:
    - {TRAIN_DIR}/classify_vertebrae/results/summary_matches_{suffix}.csv - Summary of all classifications
    - {TRAIN_DIR}/classify_vertebrae/results/init_success_log_{suffix}.txt - Log of successful latent initializations
    - {TRAIN_DIR}/classify_vertebrae/results/init_failure_log_{suffix}.txt - Log of failed latent initializations
    - {TRAIN_DIR}/classify_vertebrae/predictions_{suffix}/ - Per-mesh outputs (VTK, plots, etc.)
"""

# Identify novel meshes from latent space
import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs
import torch.nn.functional as F
import json
import pyvista as pv
import pymskt.mesh.meshes as meshes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from NSM.mesh import create_mesh
import vtk
import re
import random
import gc
import argparse
from datetime import datetime

from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, get_sdfs, fixed_point_coords, safe_load_mesh_scalars, parse_labels_from_filepaths
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, find_similar, find_similar_cos
from vtk_parsing_logic import parse_vtk_filename

# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Important monkey patch for pymskt.mesh.meshTools.pcu.signed_distance_to_mesh to ensure double precision inputs
# Monkey patch for data types ----
from NSM.helper_funcs import safe_load_mesh_scalars, fixed_point_coords
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

import pymskt.mesh.meshTools as meshTools
_original_signed_distance_to_mesh = meshTools.pcu.signed_distance_to_mesh
def _signed_distance_to_mesh_patch(pts, points, faces):
    pts = np.asarray(pts, dtype=np.float64)     # force double precision
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)   # ensure integer type for faces
    return _original_signed_distance_to_mesh(pts, points, faces)
meshTools.pcu.signed_distance_to_mesh = _signed_distance_to_mesh_patch
# End monkey patch ----


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify novel vertebrae meshes using Neural Shape Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--init',
        type=str,
        choices=['none', 'family', 'genus', 'species'],
        default='none',
        help='Initialization method for latent vectors (default: none = PCA-based init)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d_%H%M%S'),
        help='Suffix for output files (default: timestamp)'
    )
    parser.add_argument(
        '--latent_noise_std',
        type=float,
        default=0.0,
        help='Standard deviation of Gaussian noise to add to initial latent (default: 0.0)'
    )
    parser.add_argument(
        '--sample_noise_std',
        type=float,
        default=0.0,
        help='Standard deviation of Gaussian noise to add to sampled x,y,z point coordinates (default: 0.0)'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='run_v57a',
        help='Training directory name containing model checkpoint and latent codes (default: run_v57a)'
    )
    parser.add_argument(
        '--save_vtk',
        type=lambda x: x.lower() != 'false',
        default=True,
        help='Save reconstructed VTK mesh to disk (default: true). Set to false to skip saving.'
    )
    return parser.parse_args()


def load_taxonomic_latents(train_dir, level):
    """
    Load average latent vectors for a taxonomic level from NPZ file.

    Args:
        train_dir: str, training directory path
        level: str, one of 'family', 'genus', 'species'

    Returns:
        dict mapping taxonomic name -> latent vector (numpy array)
        Returns None if file not found
    """
    npz_path = os.path.join(train_dir, 'latent_codes_average', level, f'mean_latents_by_{level}.npz')

    if not os.path.exists(npz_path):
        print(f"WARNING: NPZ file not found: {npz_path}")
        print(f"Please run create_latents_per_group.py first to generate average latents.")
        return None

    data = np.load(npz_path)
    names = data[level]  # array of taxonomic names
    latents = data['latents']  # array of latent vectors

    # Create lookup dictionary
    latent_dict = {name: latent for name, latent in zip(names, latents)}
    print(f"Loaded {len(latent_dict)} {level}-level average latents from {npz_path}")

    return latent_dict


def get_init_latent_from_taxonomy(vtk_filename, init_method, taxonomic_latents,
                                   mean_latent, latent_codes, top_k_reg, device):
    """
    Get initial latent vector based on initialization method and parsed VTK filename.

    Args:
        vtk_filename: str, path to VTK file
        init_method: str, one of 'none', 'family', 'genus', 'species'
        taxonomic_latents: dict, mapping taxonomic names to latent vectors (or None for 'none')
        mean_latent: torch.Tensor, mean latent for PCA initialization
        latent_codes: torch.Tensor, all training latent codes for PCA initialization
        top_k_reg: int, number of top PCs for regularization
        device: torch device

    Returns:
        tuple: (init_latent as torch.Tensor, status_dict)
        status_dict contains: success (bool), method (str), matched_name (str or None), error (str or None)
    """
    status = {
        'success': True,
        'method': init_method,
        'matched_name': None,
        'error': None
    }

    if init_method == 'none':
        # Original PCA-based initialization (no taxonomic info used)
        init_latent = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
        status['matched_name'] = 'pca_mean'
        return init_latent, status

    # Parse VTK filename to get taxonomic info
    parsed = parse_vtk_filename(vtk_filename)

    if taxonomic_latents is None:
        status['success'] = False
        status['error'] = f"Taxonomic latents not loaded for level '{init_method}'"
        # Fall back to PCA
        init_latent = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
        return init_latent, status

    # Get the taxonomic name for the requested level
    taxonomic_name = parsed.get(init_method)

    if taxonomic_name is None:
        status['success'] = False
        status['error'] = f"Could not parse {init_method} from filename: {vtk_filename}"
        # Fall back to PCA
        init_latent = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
        return init_latent, status

    # Look up the latent vector
    if taxonomic_name in taxonomic_latents:
        latent_np = taxonomic_latents[taxonomic_name]
        init_latent = torch.from_numpy(latent_np).float().unsqueeze(0).to(device)
        status['matched_name'] = taxonomic_name
        return init_latent, status
    else:
        status['success'] = False
        status['error'] = f"{init_method.capitalize()} '{taxonomic_name}' not found in training data"
        # Fall back to PCA
        init_latent = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
        return init_latent, status


def add_sample_noise(points, noise_std):
    """
    Add Gaussian noise to sampled x,y,z point coordinates.

    Args:
        points: torch.Tensor of shape [N, 3], the sampled point coordinates
        noise_std: float, standard deviation of Gaussian noise

    Returns:
        torch.Tensor of shape [N, 3] with noise added
    """
    if noise_std > 0:
        noise = torch.randn_like(points) * noise_std
        points = points + noise
        print(f"  Added Gaussian noise to sample points (std={noise_std})")
    return points


def optimize_latent(decoder, points, sdf_vals, latent_size, init_latent,
                    iters=1000, lr=1e-3, noise_std=0.0, device='cuda:0', log_interval=200):
    """
    Optimize latent vector for inference.

    Since DeepSDF has no encoder, this is how you run novel data through for inference.

    Args:
        decoder: the decoder model
        points: point coordinates
        sdf_vals: SDF values at points
        latent_size: size of latent vector
        init_latent: initial latent vector (torch.Tensor)
        iters: number of optimization iterations
        lr: learning rate
        noise_std: standard deviation of Gaussian noise to add to initial latent
        device: torch device
        log_interval: interval at which to log losses (default: 200)

    Returns:
        tuple: (optimized latent vector, losses tensor on GPU, log_indices)
    """
    print("Using device: ", device)
    init_latent_torch = init_latent.clone().detach()
    if noise_std > 0:
        init_latent_torch = init_latent_torch + torch.randn_like(init_latent_torch) * noise_std
    latent = init_latent_torch.clone().detach().to(device).requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)
    sdf_vals = sdf_vals.to(device)
    decoder = decoder.to(device)
    points = points.to(device)

    # Pre-allocate GPU tensor for losses at log intervals
    # Indices: 0, 200, 400, ..., and final iteration (iters-1)
    log_indices = list(range(0, iters, log_interval))
    if (iters - 1) not in log_indices:
        log_indices.append(iters - 1)
    num_logs = len(log_indices)
    losses_gpu = torch.zeros(num_logs, device=device)
    log_idx = 0

    for i in range(iters):
        optimizer.zero_grad()
        pred_sdf = get_sdfs(decoder, points, latent)
        l1_loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals)

        loss = l1_loss
        loss.backward()
        optimizer.step()
        if i % 200 == 0 or i == iters - 1:
            print(f"[{i}/{iters}] Loss: {loss.item():.6f}")
        # Store loss on GPU without CPU transfer (store total loss)
        if i in log_indices:
            losses_gpu[log_idx] = loss.detach()
            log_idx += 1

    return latent.detach().to(device), losses_gpu, log_indices


def main():
    # Parse command line arguments
    args = parse_args()

    # Define PC index and model checkpoint to use for analysis of novel meshes
    TRAIN_DIR = args.run_name

    # Use suffix from command line
    suffix = args.suffix
    # Insert init method after the date portion (YYYY-MM-DD) in suffix
    # e.g., "2026-02-23_noise00" -> "2026-02-23_family_noise00"
    date_part = args.suffix[:10]  # "2026-02-23"
    rest_part = args.suffix[10:]  # "_noise00"
    suffix = f"{date_part}_{args.init}{rest_part}"
    # Add sample noise std to suffix if set
    if args.sample_noise_std > 0:
        suffix = f"{suffix}_sn{args.sample_noise_std}"

    final_csv_name = f'./{TRAIN_DIR}/classify_vertebrae/results/summary_matches_{suffix}.csv'

    CKPT = '3000'  # TO DO: Choose the ckpt value you want to analyze results for
    LC_PATH = f'{TRAIN_DIR}/latent_codes/{CKPT}.pth'
    MODEL_PATH = f'{TRAIN_DIR}/model/{CKPT}.pth'

    # Load config - after every model run, the config will save in TRAIN_DIR/model_params_config.json
    config = load_config(config_path=f'{TRAIN_DIR}/model_params_config.json')
    device = config.get("device", "cuda:0")
    train_paths = config['list_mesh_paths']
    all_vtk_files = [os.path.basename(f) for f in train_paths]

    # Load normalized position mapping
    norm_pos_df = pd.read_csv('./vtk_name_to_mapping_v2.csv')
    vtk_to_normpos = dict(zip(norm_pos_df['vtk_name'], norm_pos_df['normalized_position']))

    # List of meshes to be classified
    mesh_list = config['test_paths']

    # Load model and latent codes
    model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
    
    # Optional: Compile model once, reduce python to cuda memory ping pong overhead
    # model = torch.compile(model, mode="default")
    
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

    # Load taxonomic latents if using taxonomic initialization
    taxonomic_latents = None
    if args.init in ['family', 'genus', 'species']:
        taxonomic_latents = load_taxonomic_latents(TRAIN_DIR, args.init)
        if taxonomic_latents is None:
            print(f"ERROR: Could not load {args.init}-level latents. Falling back to PCA initialization.")
            print("Please run: python create_latents_per_group.py")

    # Create results directory
    results_dir = f'./{TRAIN_DIR}/classify_vertebrae/results'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize logging files
    success_log_path = os.path.join(results_dir, f'init_success_log_{suffix}.txt')
    failure_log_path = os.path.join(results_dir, f'init_failure_log_{suffix}.txt')
    loss_log_path = os.path.join(results_dir, f'optimization_loss_log_{suffix}.csv')

    # Write log headers
    with open(success_log_path, 'w') as f:
        f.write(f"Initialization Success Log\n")
        f.write(f"Init method: {args.init}\n")
        f.write(f"Noise std: {args.latent_noise_std}\n")
        f.write(f"Suffix: {suffix}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Index':<8}{'VTK Filename':<60}{'Matched Name':<30}\n")
        f.write("-" * 80 + "\n")

    with open(failure_log_path, 'w') as f:
        f.write(f"Initialization Failure Log\n")
        f.write(f"Init method: {args.init}\n")
        f.write(f"Noise std: {args.latent_noise_std}\n")
        f.write(f"Suffix: {suffix}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Index':<8}{'VTK Filename':<60}{'Error':<50}\n")
        f.write("-" * 80 + "\n")

    print(f"\n{'='*60}")
    print(f"Classification Settings:")
    print(f"  Init method: {args.init}")
    print(f"  Noise std: {args.latent_noise_std}")
    print(f"  Suffix: {suffix}")
    print(f"  Number of meshes: {len(mesh_list)}")
    print(f"{'='*60}\n")

    # Loop through meshes
    summary_log = []
    success_count = 0
    failure_count = 0

    for idx, vert_fname in enumerate(mesh_list):
        print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
        print(f"\033[32m\n=== Mesh {idx} / {len(mesh_list)} ===\033[0m")

        # Parse VTK filename
        parsed_info = parse_vtk_filename(vert_fname)
        print(f"Parsed info: family={parsed_info['family']}, genus={parsed_info['genus']}, species={parsed_info['species']}")

        # Get initial latent based on initialization method
        init_latent, init_status = get_init_latent_from_taxonomy(
            vert_fname, args.init, taxonomic_latents,
            mean_latent, latent_codes, top_k_reg, device
        )

        # Log success/failure
        if init_status['success']:
            success_count += 1
            with open(success_log_path, 'a') as f:
                f.write(f"{idx:<8}{os.path.basename(vert_fname):<60}{init_status['matched_name']:<30}\n")
            print(f"  Init: SUCCESS - Using {args.init} latent for '{init_status['matched_name']}'")
        else:
            failure_count += 1
            with open(failure_log_path, 'a') as f:
                f.write(f"{idx:<8}{os.path.basename(vert_fname):<60}{init_status['error']:<50}\n")
            print(f"  Init: FAILED - {init_status['error']} (falling back to PCA)")

        # Make a new dir to save predictions
        outfpath = f'./{TRAIN_DIR}/classify_vertebrae/predictions_{suffix}/' + os.path.splitext(os.path.basename(vert_fname))[0]
        print("Making a new directory to save model predictions and outputs at: ", outfpath)
        os.makedirs(outfpath, exist_ok=True)

        # --- Set up inference dataset ---

        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            _, vert_fname = convert_ply_to_vtk(ply_fname)

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
            reference_mesh=None,
            verbose=config['verbose'],
            save_cache=config['cache'],
            equal_pos_neg=config['equal_pos_neg'],
            fix_mesh=config['fix_mesh']
            )

        # Get the point/SDF data
        print("Setting up dataset")
        sdf_sample = sdf_dataset[0]  # returns a dict
        sample_dict, _ = sdf_sample
        points = sample_dict['xyz'].to(device)  # shape: [N, 3]
        # Points go from -1 to 1
        # Points stats (x): min=-0.9990, mean=-0.0050, max=0.9990
        # Points stats (y): min=-0.9996, mean=0.0036, max=0.9995
        # Points stats (z): min=-0.9995, mean=-0.0077, max=0.9999
        # print(f"  Points stats (x): min={points[:,0].min():.4f}, mean={points[:,0].mean():.4f}, max={points[:,0].max():.4f}")
        # print(f"  Points stats (y): min={points[:,1].min():.4f}, mean={points[:,1].mean():.4f}, max={points[:,1].max():.4f}")
        # print(f"  Points stats (z): min={points[:,2].min():.4f}, mean={points[:,2].mean():.4f}, max={points[:,2].max():.4f}")
        points = add_sample_noise(points, args.sample_noise_std)
        sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]

        # Optimize latents (DeepSDF has no encoder, so must use optimization to encode novel data)
        print("Optimizing latents")
        start_time01 = datetime.now()

        latent_novel, losses_gpu, log_indices = optimize_latent(
            model, points, sdf_vals, config['latent_size'],
            init_latent=init_latent,
            noise_std=args.latent_noise_std,
            device=device,
        )
        end_time01 = datetime.now()

        # Transfer losses to CPU once at the end and print
        losses_cpu = losses_gpu.cpu().numpy()
        print(f"Optimization time: {end_time01 - start_time01}")
        print("Translated novel mesh into latent space!")

        # Log losses to CSV (write header on first mesh, append data for all)
        mesh_name = os.path.basename(vert_fname)
        opt_time_seconds = (end_time01 - start_time01).total_seconds()
        if idx == 0:
            # Write header: name, time_seconds, loss_0, loss_200, loss_400, ...
            header_cols = ["name", "time_seconds"] + [f"loss_{i}" for i in log_indices]
            with open(loss_log_path, 'w') as f:
                f.write(",".join(header_cols) + "\n")
        # Append this mesh's losses
        loss_values = [f"{v:.6f}" for v in losses_cpu]
        with open(loss_log_path, 'a') as f:
            f.write(mesh_name + "," + f"{opt_time_seconds:.3f}" + "," + ",".join(loss_values) + "\n")

        # --- Classify vertebra ---

        # Find most similar latents (Compare to existing latents)
        similar_ids, distances = find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device=device)

        # Write most similar meshes to txt file
        sim_mesh_fpath = outfpath + '/' + 'similar_meshes_pca_regularized_95pct_cos.txt'
        with open(sim_mesh_fpath, "w") as f:
            print(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}\n")
            f.write(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}:\n")
            f.write(f"Initialization method: {args.init}\n")
            if init_status['success']:
                f.write(f"Initialized with: {init_status['matched_name']}\n")
            else:
                f.write(f"Initialization failed: {init_status['error']} (used PCA fallback)\n")
            f.write("\n")
            for i, d in zip(similar_ids, distances):
                # Now construct the line using the integer i
                line = f"Name: {all_vtk_files[i]}, Index: {i}, Distance: {d:.4f}"
                print(line)
                f.write(line + "\n")

        # --- Inspect novel latent using clustering analysis ---

        # PCA Plot
        # Data loading
        latents = latent_codes.cpu().numpy()
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(latents)
        novel_coord = pca.transform(latent_novel.cpu().numpy())[0]
        similar_coords = coords_2d[similar_ids]

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], color='gray', alpha=0.3, label='Training Meshes')
        # Plot most similar (1st one) in pink
        plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', s=80, label='Most Similar')
        # Plot next 4 similar in blue
        if len(similar_coords) > 1:
            plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', s=60, label='Other Top-5 Similar')
        # Plot novel mesh in red
        plt.scatter(*novel_coord, color='red', s=80, label='Novel Mesh')
        # Annotate each of the top-5 similar meshes
        for mesh_idx, (x, y) in zip(similar_ids, similar_coords):
            plt.text(x, y, all_vtk_files[mesh_idx].split('.')[0], fontsize=6, color='black')
        plt.title("Latent Space Visualization (PCA)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outfpath + "/latent_space_pca_pca_regularized_95pct_cos.png", dpi=300)
        plt.close()

        # t-SNE Plot
        # Data loading
        latent_novel_np = latent_novel.detach().cpu().numpy()
        latents_with_novel = np.vstack([latents, latent_novel_np])
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        coords_with_novel = tsne.fit_transform(latents_with_novel)
        train_coords = coords_with_novel[:-1]
        novel_coord = coords_with_novel[-1]
        similar_coords = train_coords[similar_ids]

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(train_coords[:, 0], train_coords[:, 1], color='grey', alpha=0.1, label='Training Meshes')
        # Plot most similar (1st one) in pink
        plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', alpha=0.5, label='Most Similar')
        # Plot next 4 similar in blue
        if len(similar_coords) > 1:
            plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', alpha=0.5, label='Other Top-5 Similar')
        # Plot novel mesh in red
        plt.scatter(*novel_coord, color='red', alpha=0.5, label='Novel Mesh')
        # Annotate each of the top-5 similar meshes
        for mesh_idx, (x, y) in zip(similar_ids, similar_coords):
            plt.text(x, y, all_vtk_files[mesh_idx].split('.')[0], fontsize=6, color='black')
        plt.title("Latent Space Visualization (t-SNE)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outfpath + "/latent_space_tsne_pca_regularized_95pct_cos.png", dpi=300)
        plt.close()

        # --- Reconstruct optimized latent into mesh to confirm it looks normal ---

        output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + f"_decoded_novel_pca_regularized_95pct_cos_{suffix}.vtk"
        if args.save_vtk:
            # Reconstruction parameters
            recon_grid_origin = 1.0
            n_pts_per_axis = 256
            voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
            voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
            offset = np.array([0.0, 0.0, 0.0])
            scale = 1.0
            icp_transform = NumpyTransform(np.eye(4))
            objects = 1

            # Reconstruct the novel mesh
            mesh_out = create_mesh(
                decoder=model,
                latent_vector=latent_novel,
                n_pts_per_axis=n_pts_per_axis,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
                path_original_mesh=None,
                offset=offset,
                scale=scale,
                icp_transform=icp_transform,
                objects=objects,
                verbose=True,
                device=device,
                )

            # Ensure it's PyVista PolyData
            if isinstance(mesh_out, list):
                mesh_out = mesh_out[0]
            if not isinstance(mesh_out, pv.PolyData):
                mesh_pv = mesh_out.extract_geometry()
            else:
                mesh_pv = mesh_out

            mesh_pv.save(output_path)
            print(f"Novel mesh saved to: {output_path}")
        else:
            print(f"Skipping mesh reconstruction and VTK save (--save_vtk false)")

        # Save results to summary log
        # Get ground truth species and position
        labels, _ = parse_labels_from_filepaths([os.path.basename(vert_fname)])
        gt_species, gt_position = labels[0]

        # Check top-1 match
        labels, _ = parse_labels_from_filepaths([all_vtk_files[similar_ids[0]]])
        top1_species_pred, top1_position_pred = labels[0]
        top1_species_match = "yes" if gt_species and gt_species == top1_species_pred else "no"
        top1_region_match = "yes" if gt_position and top1_position_pred and gt_position[0] == top1_position_pred[0] else "no"
        gt_normpos = vtk_to_normpos.get(os.path.basename(vert_fname))
        top1_normpos = vtk_to_normpos.get(all_vtk_files[similar_ids[0]])
        if top1_region_match == "yes" and gt_normpos is not None and top1_normpos is not None:
            top1_position_error = abs(gt_normpos - top1_normpos)
        else:
            top1_position_error = "NA_region_mismatch"

        # Check top-5 matches
        labels, _ = parse_labels_from_filepaths([all_vtk_files[i] for i in similar_ids])
        top5_species_pred  = [s for s, _ in labels]
        top5_position_pred = [v for _, v in labels]
        top5_species_match = "yes" if (gt_species is not None and gt_species in top5_species_pred) else "no"
        top5_region_match = "yes" if (gt_position is not None and any(pred and pred[0].lower() == gt_position[0].lower() for pred in top5_position_pred)) else "no"
        position_errors = []
        if top5_region_match == "yes" and gt_normpos is not None:
            for idx, pred in zip(similar_ids, top5_position_pred):
                if pred and pred[0].lower() == gt_position[0].lower():
                    pred_normpos = vtk_to_normpos.get(all_vtk_files[idx])
                    if pred_normpos is not None:
                        position_errors.append(abs(gt_normpos - pred_normpos))
        top5_position_error = min(position_errors) if position_errors else "NA_region_mismatch"

        # Prepare summary log with top-5
        top_k_summary = {
            "mesh": os.path.basename(vert_fname),
            "output_mesh": output_path,
            "init_method": args.init,
            "init_success": init_status['success'],
            "init_matched_name": init_status['matched_name'],
            "parsed_family": parsed_info['family'],
            "parsed_genus": parsed_info['genus'],
            "parsed_species": parsed_info['species'],
            "top1_species_match": top1_species_match,
            "top5_species_match": top5_species_match,
            "top1_region_match": top1_region_match,
            "top1_position_error": top1_position_error,
            "top5_region_match": top5_region_match,
            "top5_position_error": top5_position_error,
        }
        # Add top-5 similar mesh names and distances
        for rank, (i, dist) in enumerate(zip(similar_ids, distances), 1):
            top_k_summary[f"similar_{rank}_name"] = all_vtk_files[i]
            top_k_summary[f"similar_{rank}_distance"] = dist
        summary_log.append(top_k_summary)

    # Write final summary statistics to log files
    with open(success_log_path, 'a') as f:
        f.write("-" * 80 + "\n")
        f.write(f"Total successful initializations: {success_count}/{len(mesh_list)}\n")

    with open(failure_log_path, 'a') as f:
        f.write("-" * 80 + "\n")
        f.write(f"Total failed initializations: {failure_count}/{len(mesh_list)}\n")

    # Export results to summary log
    df = pd.DataFrame(summary_log)
    df.to_csv(final_csv_name, index=False)
    print(f"\n{'='*60}")
    print(f"Summary saved to {final_csv_name}")
    print(f"Success log: {success_log_path}")
    print(f"Failure log: {failure_log_path}")
    print(f"Loss log: {loss_log_path}")
    print(f"Initialization: {success_count} success, {failure_count} failures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
