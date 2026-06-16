# Make a grid of 2d images showing vertebrae sampled across isomap

import os
import json
import torch
import numpy as np
import open3d as o3d
import pyvista as pv
import vtk 
import gc
import cv2
from sklearn.manifold import Isomap
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents 
from NSM.traverse_latents import sample_latent_grid, solve_tsp_nearest_neighbor, interpolate_latent_loop

# Define parameters
TRAIN_DIR = "run_v41" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '1000' # TO DO: Choose model checkpoint to load
LC_PATH = f'latent_codes/{CKPT}.pth'
MODEL_PATH = f'model/{CKPT}.pth'
NUM_GRIDS_X = 8 # TO DO: Adjust based on desired grid format
NUM_GRIDS_Y = 8 # TO DO: Adjust based on desired grid format
full_grid = True # TO DO: Sample full grid or only space within distribution of latent codes
IMG_WIDTH_PER_MESH = 512
IMG_HEIGHT_PER_MESH = 512
MESH_RESOLUTION_N = 128
OUTPUT_FILENAME_BASE = f"isomap_grid_{NUM_GRIDS_X}x{NUM_GRIDS_Y}_{IMG_WIDTH_PER_MESH}p_{MESH_RESOLUTION_N}p.png" # TO DO: Update filename
if full_grid == True: 
    OUTPUT_FILENAME = OUTPUT_FILENAME_BASE + '_fullgrid.png'
else:  
    OUTPUT_FILENAME = OUTPUT_FILENAME_BASE + '.png'
EYE_OFFSET_FACTOR = 1.2
TINT_BG_FOR_SPARSITY = True
BASE_BG_COL = [0.38, 1, 0.98] # dark blue
MAX_TINT_BG_COL = [0.03, 0.11, 0.1] # aquamarine

# Define functions
def apply_sparsity_tints(latent_2d, tree):
    # Get the distances to the 5 nearest neighbors
    distances, _ = tree.query(latent_2d, k=6)  # k=6 because the closest point is itself, so we need the next 5 neighbors
    nearest_distances = distances[:, 1:]  # Ignore the first column (distance to itself)
    # Calculate the average distance to the 5 nearest neighbors
    avg_distances = nearest_distances.mean(axis=1)
    # Normalize the distances to [0, 1] where 0 = closest neighbors, 1 = furthest neighbors
    normalized_distances = (np.max(avg_distances) - avg_distances) / (np.max(avg_distances) - np.min(avg_distances))
    # Interpolate between the base and max tint color based on normalized distance
    base_bg_col_np = np.array(BASE_BG_COL)
    max_tint_bg_col_np = np.array(MAX_TINT_BG_COL)
    # Calculate the tints for each point
    tints = base_bg_col_np * (1 - normalized_distances[:, None]) + (max_tint_bg_col_np * normalized_distances[:, None])
    # Add the alpha channel (set to 1.0 for full opacity)
    tints_with_alpha = np.hstack([tints, np.ones((tints.shape[0], 1))])  # Append a column of 1s (alpha channel)
    print(f"\033[92mMax color for full background tint (dense) with alpha channel: {tints_with_alpha.max(axis=0)}\033[0m")
    print(f"\033[92mMin color for base background tint (sparse) with alpha channel: {tints_with_alpha.min(axis=0)}\033[0m")
    print(f"\033[92mMax color for full background tint (dense): {tints.max(axis=0)}\033[0m")
    print(f"\033[92mMin color for base background tint (sparse): {tints.min(axis=0)}\033[0m")
    return tints_with_alpha

def generate_mesh_from_latent_code(latent_code, model, dev, i):
    o3d_mesh_to_render = None
    latent_tensor = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(dev)
    icp = NumpyTransform(np.eye(4))
    origin_val = 1.0
    mesh_data_from_decoder = create_mesh(decoder=model, latent_vector=latent_tensor, n_pts_per_axis=MESH_RESOLUTION_N,
                    voxel_origin=(-origin_val,) * 3, voxel_size=(origin_val * 2) / (MESH_RESOLUTION_N - 1), path_original_mesh=None, offset=np.zeros(3), 
                    scale=1.0,icp_transform=icp, objects=1, verbose=False, device=dev)
    m_inter = mesh_data_from_decoder[0] if isinstance(mesh_data_from_decoder, list) else mesh_data_from_decoder
    if isinstance(m_inter, pv.PolyData):
        if m_inter.n_points > 0 :
            pv_m = m_inter.extract_geometry().triangulate()
            if pv_m.n_faces_strict > 0:
                pv_m = pv_m.compute_normals(cell_normals=False, point_normals=True, inplace=False, auto_orient_normals=True)
                o3d_mesh_to_render = pv_to_o3d(pv_m)
            else:
                print(f"\033[93mWarning: PyVista mesh for ({i}) has 0 faces after triangulation.\033[0m")
        else:
                print(f"\033[93mWarning: PyVista mesh for ({i}) has 0 points.\033[0m")
    elif isinstance(m_inter, o3d.geometry.TriangleMesh):
        if m_inter.has_vertices() and m_inter.has_triangles():
            o3d_mesh_to_render = m_inter
            o3d_mesh_to_render.compute_vertex_normals()
            o3d_mesh_to_render.paint_uniform_color([0.7,0.7,0.7])
        else:
            print(f"\033[93mWarning: Open3D mesh for ({latent_code}) has 0 triangles or vertices.\033[0m")
    else:
        print(f"\033[93mWarning: Mesh for ({latent_code}) is None or unexpected type: {type(m_inter)}\033[0m")
    return o3d_mesh_to_render

def render_mesh(ren, mesh, material, bg_color):
    rendered_img_np = np.full((IMG_HEIGHT_PER_MESH, IMG_WIDTH_PER_MESH, 3), 
                                      (np.array(bg_color[:3])*255).astype(np.uint8), # Use actual BG for placeholder
                                      dtype=np.uint8)
    ren.scene.clear_geometry()
    ren.scene.add_geometry("shape_mesh", mesh, material)
    # Set background color dynamically
    ren.scene.set_background(list(bg_color))
    # Set up camera
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    max_dim_extent = np.max(bounds.get_extent())
    if max_dim_extent < 1e-3: 
        max_dim_extent = 1.0
    eye = center + np.array([1.0, 1.0, 0.8]) * max_dim_extent * EYE_OFFSET_FACTOR
    up_vector = np.array([0.0, 0.0, 1.0])   
    ren.scene.camera.look_at(center, eye, up_vector)
    img_o3d = ren.render_to_image()
    return np.asarray(img_o3d)

def plot_latent_path(latent_2d, indices):
    neighbor_indices = indices[1]  # Indices of nearest neighbors for each sampled point
    path_points = []
    for i, neighbors in enumerate(neighbor_indices):
        for neighbor in neighbors:
            path_points.append(latent_2d[neighbor])
    path_points = np.array(path_points)
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], color='gray', alpha=0.3, label="All Latents")
    # Plot the path
    plt.plot(path_points[:, 0], path_points[:, 1], color='red', linewidth=2, label="Latent Path")
    # Highlight the start and end points
    plt.scatter(path_points[0, 0], path_points[0, 1], color='green', s=100, edgecolors='black', linewidths=1.5, label='Start')
    plt.scatter(path_points[-1, 0], path_points[-1, 1], color='blue', s=100, edgecolors='black', linewidths=1.5, label='End')
    plt.title("Latent Space Path")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend()
    figpath_base = "latent_space_path_overlay_kdtree" # TO DO: Change latent path plot filename
    if full_grid == True:
        figpath = figpath_base + "_fullgrid.png"
    else:
        figpath = figpath_base + ".png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"\033[92mSaved latent space path overlay to {figpath}\033[0m")

def resample_by_cumulative_distance(latents, n_frames):
    diffs = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    dists = np.concatenate([[0], np.cumsum(diffs)])
    dists /= dists[-1]  # Normalize to [0, 1]
    new_steps = np.linspace(0, 1, n_frames)
    new_latents = np.array([
        np.interp(new_steps, dists, latents[:, i]) for i in range(latents.shape[1])]).T
    return new_latents

# --- Main Logic ---

def main():
    # Load Configuration file
    cfg = load_config(config_path='model_params_config.json')
    
    # Load Model and Latent Codes
    device = cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu'
    mdl, ck2, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, cfg, device)
   
    # Latent Code Processing (Isomap)
    isomap = Isomap(n_neighbors=10, n_components=2)
    latent_2d = isomap.fit_transform(latent_codes)
    
    # k-d Tree and Sampling
    tree = cKDTree(latent_2d)
    sampled_points = sample_latent_grid(latent_2d, NUM_GRIDS_X, NUM_GRIDS_Y)
    indices = tree.query(sampled_points, k=3)
    weights = 1 / (indices[0] + 1e-5)  # Inverse distance weighting (avoid division by 0)
    weights /= weights.sum(axis=1)[:, None]  # Normalize the weights

    # Interpolate between the nearest latent codes
    latent_interp = []
    for i, sampled_point in enumerate(sampled_points):
        neighbors = indices[1][i]
        row = (latent_codes[neighbors] * weights[i][:, None]).sum(axis=0)
        latent_interp.append(row)
    latent_interp = np.array(latent_interp)
    
    # Explore full grid not just area within point bounds
    if full_grid == True:
        # Generate a pairwise distance matrix of latent codes
        dist_matrix = cdist(latent_interp, latent_interp, metric='cosine')

        # Use travelling salesman to determine nearest neighbor and reorder latent_interp for smooth transitions
        tsp_path = solve_tsp_nearest_neighbor(dist_matrix)
        tsp_path.append(tsp_path[0])  # To make it loop
        latent_interp_ordered = latent_interp[tsp_path]
        steps_per_segment = 20 # TO DO: adjust steps per segment
        dense_interp = interpolate_latent_loop(latent_interp_ordered, steps_per_segment=steps_per_segment)
        n_frames = NUM_GRIDS_X * NUM_GRIDS_Y
        smooth_latent_loop = resample_by_cumulative_distance(dense_interp, n_frames=n_frames)
        # Apply temporal smoothing filter over latent trajectory
        # window_length must be odd and <= length of array
        window_length = min(31, len(smooth_latent_loop) - 1 if len(smooth_latent_loop) % 2 == 0 else len(smooth_latent_loop))
        smooth_latent_loop = savgol_filter(smooth_latent_loop, window_length=window_length, polyorder=3, axis=0)
        latent_interp = smooth_latent_loop    
    print("\n\n\nLength: ", len(latent_interp))

    # Background Tint for Sparsity
    if TINT_BG_FOR_SPARSITY:
        tints = apply_sparsity_tints(sampled_points, tree)
    
    # Render Grid of Latent Codes
    ren = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH_PER_MESH, IMG_HEIGHT_PER_MESH)
    ren.scene.set_background(list(np.array(BASE_BG_COL)) + [1.0])
    ren.scene.scene.set_sun_light([0.707, 0.707, 0.0], [1,1,1], 75000)
    ren.scene.scene.enable_sun_light(True)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.base_roughness = 0.4
    material.base_metallic = 0.1
    
    generated_mesh_count = 0
    grid_images = []
    for i, latent_code in enumerate(latent_interp):
        generated_mesh_count += 1
        print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{len(latent_interp)}\033[0m")
        mesh = generate_mesh_from_latent_code(latent_code, mdl, device, i)
        
        # Determine the background color based on sparsity (i.e., count)
        bg_color = tints[i] if TINT_BG_FOR_SPARSITY else BASE_BG_COL
        print(f"\n\n\033[92mUsing background color: {bg_color} for mesh {str(i+1)}\n\033[0m")
        
        # Render the mesh with dynamic background
        img = render_mesh(ren, mesh, material, bg_color=bg_color)
        grid_images.append(img)
    
    # Save Final Grid Image
    final_grid_image = np.vstack([np.hstack(grid_images[i:i + NUM_GRIDS_Y]) for i in range(0, len(grid_images), NUM_GRIDS_Y)])
    cv2.imwrite(OUTPUT_FILENAME, cv2.cvtColor(final_grid_image, cv2.COLOR_RGB2BGR))
    print(f"\033[92mGrid image saved to {OUTPUT_FILENAME}\033[0m")

    # Optional: Plot Latent Space Path
    plot_latent_path(latent_2d, indices)

    # Cleanup
    del mdl, lc, ck2, latent_codes, ren
    gc.collect()

if __name__ == '__main__':
    main()
