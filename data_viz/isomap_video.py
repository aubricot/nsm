import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.manifold import Isomap
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import re
from scipy.signal import savgol_filter
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents, average_across_regions, overlay_text_on_frame, render_cameras, generate_and_render_mesh
from NSM.traverse_latents import sample_latent_grid, solve_tsp_nearest_neighbor, interpolate_latent_loop, project_to_isomap, resample_by_cumulative_distance, plot_latent_paths

# Define model parameters to use for video generation
TRAIN_DIR = "run_v47" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '2000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
USE_AVERAGES = True # TO DO: Use region averages or individual vertebrae?
vertebral_regions = ['C', 'T', 'L'] # TO DO: Define vertebral regions to inspect

# Load config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

# Define vertebral regions
latent_codes_subs = []
all_vtk_files_subs = []
# To find all vertebrae for each vertebral region across all specimens
for vert_region in vertebral_regions:
    r_p = r'[_-]' + vert_region + r'([1-9]|[1-3][0-9]|200)(?!\d)' # Match "_C1" to "_C200" or "-C1" to "-C200"
    pattern = re.compile(r_p, re.IGNORECASE)
    # Subset indices from all paths
    matches = [(i, int(pattern.search(fname).group(1))) for i, fname in enumerate(all_vtk_files) if pattern.search(fname)]
    indices = [i for i, _ in matches]

    # Filter latent codes and corresponding mesh paths
    vert_region_codes = latent_codes[indices]
    vert_region_files = [all_vtk_files[i] for i in indices]
    print(f"\n\nFound {len(vert_region_files)} latent codes for region: {vert_region}")
    print(f"Sample files: {vert_region_files[:5]}\n\n")

    # To average across vertebrae regions per specimen
    # Regex to extract specimen ID from filename by removing "_C1" or "-C1", etc.
    if USE_AVERAGES == True:
        r_p = r'^(.*?)(?:[-_]\d+[-_]' + vert_region + r'\d+)(?:.*)$'
        vert_region_files, vert_region_codes = average_across_regions(r_p, vert_region, vert_region_files, vert_region_codes)
    
    # Add to dictionary
    latent_codes_subs.extend(vert_region_codes)
    all_vtk_files_subs.extend(vert_region_files)

latent_codes_tensor = torch.stack([torch.tensor(latent) for latent in latent_codes_subs])
latent_codes_subs = latent_codes_tensor
print(f"Running analysis for : {len(all_vtk_files_subs)} averaged latent codes for vertebral regions {vertebral_regions}.")

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 256
voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset = np.array([0.0, 0.0, 0.0])
scale = 1.0
icp_transform = NumpyTransform(np.eye(4))
objects = 1

# Isomap: project latent codes to 2D manifold
latents_np = latent_codes_subs.numpy() if torch.is_tensor(latent_codes_subs) else latent_codes_subs
isomap = Isomap(n_neighbors=20, n_components=2)
isomap_2d = isomap.fit_transform(latents_np)  # Shape: (N, 2)

projections = isomap_2d[:, 0]
min_proj = isomap_2d[:, 0].min()
max_proj = isomap_2d[:, 0].max()

# Path params
n_rotations = 3
n_frames = 120 * n_rotations # TO DO: Adjust the number of frames to resample by

# 5. k-d Tree and Sampling
tree = cKDTree(isomap_2d)
NUM_GRIDS_X = 8
NUM_GRIDS_Y = 8
sampled_points = sample_latent_grid(isomap_2d, NUM_GRIDS_X, NUM_GRIDS_Y)
indices = tree.query(sampled_points, k=3)
weights = 1 / (indices[0] + 1e-5)  # Inverse distance weighting (avoid division by 0)
weights /= weights.sum(axis=1)[:, None]  # Normalize the weights

# Step 6: Interpolate between the nearest latent codes
latent_interp = []
for i, sampled_point in enumerate(sampled_points):
        neighbors = indices[1][i]
        row = (latents_np[neighbors] * weights[i][:, None]).sum(axis=0)
        latent_interp.append(row)
latent_interp = np.array(latent_interp)

# Generate a pairwise distance matrix of latent codes
dist_matrix = cdist(latent_interp, latent_interp, metric='cosine')

# Use travelling salesman to determine nearest neighbor and reorder latent_interp for smooth transitions
tsp_path = solve_tsp_nearest_neighbor(dist_matrix)
latent_interp_ordered = latent_interp[tsp_path]
steps_per_segment = 100 # TO DO: adjust steps per segment
dense_interp = interpolate_latent_loop(latent_interp_ordered, steps_per_segment=steps_per_segment)
smooth_latent_loop = resample_by_cumulative_distance(dense_interp, n_frames=n_frames)
# Apply temporal smoothing filter over latent trajectory
# window_length must be odd and <= length of array
window_length = min(31, len(smooth_latent_loop) - 1 if len(smooth_latent_loop) % 2 == 0 else len(smooth_latent_loop))
smooth_latent_loop = savgol_filter(smooth_latent_loop, window_length=window_length, polyorder=3, axis=0)
print("\n\n\nLength: ", len(smooth_latent_loop))

# Project interpolated path points back into isomap for plotting
loop_2d, _ = project_to_isomap(latent_interp, latents_np, isomap_2d)
tsp_2d, _ = project_to_isomap(latent_interp_ordered, latents_np, isomap_2d)
smooth_loop_2d, loop_idx = project_to_isomap(smooth_latent_loop, latents_np, isomap_2d)

# Get closest specimen labels for smooth latent loop
closest_specimens = [all_vtk_files_subs[i] for i in loop_idx]

# Plot path in latent space
plot_latent_paths(isomap_2d, loop_2d, tsp_2d, smooth_loop_2d, sampled_points, vertebral_regions, USE_AVERAGES)

# Setup Offscreen Renderers (4)
width, height = 640, 480
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height) for _ in range(4)]
for r in renderers:
    r.scene.set_background([0.0, 0.0, 0.0, 1.0])
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1.0, 1.0, 1.0, 1.0]

# Video Writer (2x2 grid → 1280x960)
video_path = "isomap" + "_4way_splitscreen_" + "C-T-L"
if USE_AVERAGES == True:
    video_path = video_path + "_avg" + ".mp4"
else:
    video_path = video_path + ".mp4"
fps = 15
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

generated_mesh_count = 0
loop_sequence = np.concatenate([smooth_latent_loop, smooth_latent_loop[::-1][1:]], axis=0)
loop_sequence_names = np.concatenate([closest_specimens, closest_specimens[::-1][1:]])
loop_sequence = smooth_latent_loop
loop_sequence_names = closest_specimens
for i, latent_code in enumerate(loop_sequence):
    try:
        # Generate and render mesh
        mesh_o3d = generate_and_render_mesh(latent_code, len(loop_sequence), i, device, model, n_pts_per_axis, 
                                            voxel_origin, voxel_size, offset, scale, icp_transform, objects, 
                                            generated_mesh_count, loop_sequence_names)
        # Render views of model for video
        combined = render_cameras(renderers, mesh_o3d, i, material, len(loop_sequence), n_rotations)
        # Overlay specimen name info onto each frame
        combined = overlay_text_on_frame(combined, i, loop_sequence_names)
        # Write video
        out_video.write(combined)
        print(f"Captured frame {i + 1}/{len(loop_sequence)}")
    except Exception as e:
        print(f"Error at frame {i}: {e}")
    finally:
        for var in ['mesh_out', 'mesh_pv', 'mesh_o3d', 'new_latent']:
            if var in locals():
                del locals()[var]
        gc.collect()
out_video.release()
print("Video saved as", video_path)
