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

# Define PC index and model checkpoint to use for analysis of novel mdeshes
TRAIN_DIR = "run_v30" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# --- Load model config ---
config_path = 'model_params_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
device = config.get("device", "cuda:0")
# Randomly select test paths
#mesh_list = random.sample(config['test_paths'], 100)
#mesh_list = random.sample(config['val_paths'], 100) # TO DO: Choose val or test paths
mesh_list = config['val_paths']
# Define train paths for identifying closest mesh match
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# monkey patch
def safe_load_mesh_scalars(self):
    try:
        # Try PyVista mesh first
        if hasattr(self, 'mesh'):
            mesh = self.mesh
        elif hasattr(self, '_mesh'):
            mesh = self._mesh
        else:
            raise AttributeError("No mesh attribute found in Mesh object.")

        point_scalars = mesh.point_data
        cell_scalars = mesh.cell_data

        if point_scalars and len(point_scalars.keys()) > 0:
            self.mesh_scalar_names = list(point_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        elif cell_scalars and len(cell_scalars.keys()) > 0:
            self.mesh_scalar_names = list(cell_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        else:
            self.mesh_scalar_names = []
            self.scalar_name = None
            print("No scalar data found in mesh. Proceeding without scalars.")
    except Exception as e:
        print(f"Failed to load mesh scalars: {e}")
        self.mesh_scalar_names = []
        self.scalar_name = None
# Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars

def fixed_point_coords(self):
    if self.n_points < 1:
        raise AttributeError(f"No points found in mesh '{self}'")
    return self.points
meshes.Mesh.point_coords = property(fixed_point_coords)

def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device='cuda'):

    n_pts_total = samples.shape[0]

    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects, device=device) # KW patch from device mismatch
    
    if batch_size > n_pts_total:
        print('WARNING: batch_size is greater than the number of samples, setting batch_size to the number of samples')
        batch_size = n_pts_total

    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)
        sdf_values[current_idx : current_idx + current_batch_size, :] = decode_sdf(
            decoder, latent_vector, sampled_pts
        ) # removed .detach().cpu() bc of device mismatch

        current_idx += current_batch_size
        print(f"Processed {current_idx} / {n_pts_total} points")
    # sdf_values.squeeze(1)
    return sdf_values

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], dim=1)
    # Make sure inputs are going to the same device as everyone else
    inputs = inputs.to(next(decoder.parameters()).device)  
    
    return decoder(inputs)

#### end monkey patch

# Initialize latent near PCA offset mean
def pca_initialize_latent(mean_latent, latent_codes, top_k=10):
    # Convert to numpy
    latent_np = latent_codes.detach().cpu().numpy()
    mean_np = mean_latent.detach().cpu().numpy().squeeze()

    pca = PCA(n_components=latent_np.shape[1])
    pca.fit(latent_np)

    # Sample along top-K PCs
    top_components = pca.components_[:top_k]  # (K, D)
    top_eigenvalues = pca.explained_variance_[:top_k]
    scale = 0.01  # tune this
    coeffs = np.random.randn(top_k) * np.sqrt(top_eigenvalues) * scale

    pca_offset = np.dot(coeffs, top_components)  # D

    init_latent = mean_np + pca_offset
    return torch.tensor(init_latent, dtype=torch.float32, device=latent_codes.device).unsqueeze(0)

# Get top k PCA's based on defined explained variance threshold
def get_top_k_pcs(latent_codes, threshold=0.90):
    latent_np = latent_codes.cpu().numpy()
    pca = PCA()
    pca.fit(latent_np)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cum_var, threshold)
    print(f"Selected top {k+1} PCs to explain {threshold*100:.1f}% of variance")
    return pca, k + 1

# Convert ply file to vtk
def convert_ply_to_vtk(input_file, output_file=None):
    if not input_file.lower().endswith('.ply'):
        raise ValueError("Input file must have a .ply extension.")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".vtk"
    mesh = pv.read(input_file)
    mesh.save(output_file)
    print(f"Converted {input_file} → {output_file}")
    return output_file

# Optimie latent vector for inference (since DeepSDF has no encoder, this is how you run novel data through for inference)
def optimize_latent(decoder, points, sdf_vals, latent_size, iters=1000, lr=1e-3):
    init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg) # initialize near mean using PCAs for regularization
    latent = init_latent_torch.clone().detach().requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)
    sdf_vals = sdf_vals.to(device)
    decoder = decoder.to(device)
    points = points.to(device)
    for i in range(iters):
        optimizer.zero_grad()
        pred_sdf = get_sdfs(decoder, points, latent)
        loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals)
        loss.backward()
        optimizer.step()
        if i % 200 == 0 or i == iters - 1:
            print(f"[{i}/{iters}] Loss: {loss.item():.6f}")
    return latent.detach().to(device)

# Find the top 5 most similar meshes from training data to novel/input mesh - uses L2 (euclidian) distance in latent space
def find_similar(latent_novel, latent_codes, top_k=5, n_std=2):
    dists = torch.norm(latent_codes - latent_novel, dim=1)
    mean_dist = dists.mean()
    std_dist = dists.std()
    threshold = mean_dist + n_std * std_dist
    within = dists <= threshold
    sorted_idx = torch.argsort(dists[within])[:top_k]
    similar_ids = torch.nonzero(within).squeeze()[sorted_idx]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), dists[similar_ids].tolist()

# Find the top 5 most similar meshes from training data to novel/input mesh - uses cosine distance in latent space
def find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2):
    # Compute cosine similarity between each latent code and the novel latent code
    cosine_similarities = F.cosine_similarity(latent_codes, latent_novel, dim=1)
    # Cosine distance is 1 - cosine similarity
    cosine_distances = 1 - cosine_similarities
    # Compute mean and std of cosine distances
    mean_dist = cosine_distances.mean()
    std_dist = cosine_distances.std()
    # Apply threshold (mean + n_std * std)
    threshold = mean_dist + n_std * std_dist
    within = cosine_distances <= threshold
    # Sort distances within the threshold and get top_k
    within_indices = torch.nonzero(within, as_tuple=False).squeeze()
    if within_indices.numel() == 0:
        print("No similar items within the threshold.")
        return [], []
    # If only one index remains, ensure it's a 1D tensor
    if within_indices.ndim == 0:
        within_indices = within_indices.unsqueeze(0)
    sorted_indices = torch.argsort(cosine_distances[within_indices])[:top_k]
    similar_ids = within_indices[sorted_indices]  # 1D: shape [top_k]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), cosine_distances[similar_ids].tolist()

# Get species name using regex from filenames
def extract_species_prefix(filename):
    """Extracts species prefix using regex (e.g., Scincidae_Tribolonotus_novaeguineae)"""
    match = re.match(r"([A-Za-z]+_[A-Za-z]+_[a-z]+)", filename.lower())
    if match:
        return match.group(1)
    else:
        return None

# Utility class for ICP transform
class NumpyTransform:
    def __init__(self, matrix):
        self.matrix = matrix
    def GetMatrix(self):
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, self.matrix[i, j])
        return vtk_mat

# Load model and latents
print("Loading model and latents")
latent_ckpt = torch.load(LC_PATH, map_location=device)
latent_codes = latent_ckpt['latent_codes']['weight'].detach().to(device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

triplane_args = {
    'latent_dim': config['latent_size'],
    'n_objects': config['objects_per_decoder'],
    'conv_hidden_dims': config['conv_hidden_dims'],
    'conv_deep_image_size': config['conv_deep_image_size'],
    'conv_norm': config['conv_norm'], 
    'conv_norm_type': config['conv_norm_type'],
    'conv_start_with_mlp': config['conv_start_with_mlp'],
    'sdf_latent_size': config['sdf_latent_size'],
    'sdf_hidden_dims': config['sdf_hidden_dims'],
    'sdf_weight_norm': config['weight_norm'],
    'sdf_final_activation': config['final_activation'],
    'sdf_activation': config['activation'],
    'sdf_dropout_prob': config['dropout_prob'],
    'sum_sdf_features': config['sum_conv_output_features'],
    'conv_pred_sdf': config['conv_pred_sdf'],
}
model = TriplanarDecoder(**triplane_args)
model_ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(model_ckpt['model'])
model.to(device)
model.eval()

# Loop through meshes
summary_log = []
for i, vert_fname in enumerate(mesh_list):    
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    print(f"\033[32m\n=== Mesh {i} / {len(mesh_list)} ===\033[0m")
    # Make a new dir to save predictions
    outfpath = 'novel_meshes/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0]
    print("Making a new directory to save model predictions and outputs at: ", outfpath)
    os.makedirs(outfpath, exist_ok=True)

    # Convert plys to vtks
    if '.ply' in vert_fname:
        ply_fname = vert_fname
        vert_fname = convert_ply_to_vtk(ply_fname)

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
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]

    # Optimie latents
    print("Optimizing latents")
    latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'])
    print("Translated novel mesh into latent space!")

    # --- 5. Compare to Existing Latents ---
    similar_ids, distances = find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2)

    # Write most similar meshes to txt file
    sim_mesh_fpath = outfpath + '/' + 'similar_meshes_pca_regularized_95pct_cos.txt'
    with open(sim_mesh_fpath, "w") as f:
        print(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}\n")
        f.write(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}:\n")
        for i, d in zip(similar_ids, distances):
            # Now construct the line using the integer i
            line = f"Name: {all_vtk_files[i]}, Index: {i}, Distance: {d:.4f}"
            print(line)
            f.write(line + "\n")

    #---6. Inspect novel latent using clustering analysis

    # --- PCA Plot with Highlights ---
    latents = latent_codes.cpu().numpy()
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(latents)

    novel_coord = pca.transform(latent_novel.cpu().numpy())[0]
    similar_coords = coords_2d[similar_ids]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], color='gray', alpha=0.3, label='Training Meshes')

    # Plot most similar (1st one) in pink
    plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', s=80, label='Most Similar')
    # Plot next 4 similar in blue
    if len(similar_coords) > 1:
        plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', s=60, label='Other Top-5 Similar')
    # Plot novel mesh in red
    plt.scatter(*novel_coord, color='red', s=80, label='Novel Mesh')
    # Aannotate each of the top-5 similar meshes
    for idx, (x, y) in zip(similar_ids, similar_coords):
        plt.text(x, y, all_vtk_files[idx].split('.')[0], fontsize=6, color='black')

    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfpath + "/latent_space_pca_pca_regularized_95pct_cos.png", dpi=300)
    plt.close()


    # t-SNE Plot
    # --- t-SNE Plot with Highlights ---
    latent_novel_np = latent_novel.detach().cpu().numpy()
    latents_with_novel = np.vstack([latents, latent_novel_np])

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    coords_with_novel = tsne.fit_transform(latents_with_novel)

    train_coords = coords_with_novel[:-1]
    novel_coord = coords_with_novel[-1]
    similar_coords = train_coords[similar_ids]

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
    for idx, (x, y) in zip(similar_ids, similar_coords):
        plt.text(x, y, all_vtk_files[idx].split('.')[0], fontsize=6, color='black')

    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfpath + "/latent_space_tsne_pca_regularized_95pct_cos.png", dpi=300)
    plt.close()

    #---7. Reconstruct optimized latent into mesh to confirm it looks normal
    
    # --- Reconstruction parameters ---
    recon_grid_origin = 1.0
    n_pts_per_axis = 256
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # --- Reconstruct the novel mesh ---
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

    # --- Ensure it's PyVista PolyData ---
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]

    if not isinstance(mesh_out, pv.PolyData):
        mesh_pv = mesh_out.extract_geometry()
    else:
        mesh_pv = mesh_out

    # --- Save mesh ---
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_decoded_novel_pca_regularized_95pct_cos.vtk"
    mesh_pv.save(output_path)
    print(f"Novel mesh saved to: {output_path}")

    # Save results to summary log
    # Get species prefix
    mesh_species = extract_species_prefix(os.path.basename(vert_fname))

    # Check top-1 match
    similar_1_species = extract_species_prefix(all_vtk_files[similar_ids[0]])
    species_match = "yes" if mesh_species and mesh_species == similar_1_species else "no"

    # Check top-5 matches
    top5_match = any(extract_species_prefix(all_vtk_files[i]) == mesh_species
                    for i in similar_ids)
    top5_species_match = "yes" if top5_match else "no"


    # Prepare summary log with top-5
    top_k_summary = {
    "mesh": os.path.basename(vert_fname),
    "output_mesh": output_path,
    "species_match": species_match,
    "top5_species_match": top5_species_match,
    }

    # Add top-5 similar mesh names and distances
    for rank, (i, dist) in enumerate(zip(similar_ids, distances), 1):
        top_k_summary[f"similar_{rank}_name"] = all_vtk_files[i]
        top_k_summary[f"similar_{rank}_distance"] = dist

    summary_log.append(top_k_summary)

# Export results to summary log
df = pd.DataFrame(summary_log)
df.to_csv("summary_matches_95pct_cos.csv", index=False)
print("Summary saved to summary_matches_95pct_cos.csv")