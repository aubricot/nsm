# Make a 4-way split video exploring shape along PC axis

import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents, render_cameras, generate_and_render_mesh
from NSM.traverse_latents import generate_latent_path_plot
from pathlib import Path

# Define PC index and model checkpoint to use for video generation
cwd = Path.cwd()
base_wd = cwd.parent 
TRAIN_DIR = base_wd / "run_v57" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
PC_idx = 2   # TO DO: Choose PC index for PC of interest (ex: For PC1, choose 0)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Load config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 256
voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset = np.array([0.0, 0.0, 0.0])
scale = 1.0
icp_transform = NumpyTransform(np.eye(4))
objects = 1

# PCA on latent codes
latents_np = latent_codes.numpy()
latent_mean = np.mean(latents_np, axis=0)
centered = latents_np - latent_mean
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
pc1 = Vt[PC_idx]
projections = centered.dot(pc1)
dists = np.linalg.norm(latents_np - latent_mean, axis=1)
center_idx = np.argmin(dists)
center_sample = latents_np[center_idx]
center_proj = np.dot(center_sample - latent_mean, pc1)
max_proj = np.max(projections)
min_proj = np.min(projections)
high_delta = max_proj - center_proj
low_delta = min_proj - center_proj

# Alpha path
n_rotations = 3
amplify = 1.5
n_seg = 15 * n_rotations
alpha_vals = np.concatenate([
    np.linspace(+low_delta, 0, n_seg),
    np.linspace(0, +high_delta, n_seg),
    np.linspace(+high_delta, 0, n_seg),
    np.linspace(0, +low_delta, n_seg)
])
total_frames = len(alpha_vals)

# Setup Offscreen Renderers (4)
width, height = 640, 480
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height) for _ in range(4)]
for r in renderers:
    r.scene.set_background([0.0, 0.0, 0.0, 1.0])
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1.0, 1.0, 1.0, 1.0]

# Video Writer (2x2 grid → 1280x960)
video_path = f"pc{PC_idx+1}_4way_{n_pts_per_axis}p_{amplify}xpc.mp4"
fps = 15
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

# Loop through PC sweeps
generated_mesh_count = 0
for i, alpha in enumerate(alpha_vals):
    try:
        new_latent_np = center_sample + amplify * alpha * pc1
        new_latent = torch.tensor(new_latent_np, dtype=torch.float32).unsqueeze(0).to(device)
        # Compute PC1 projection for the latent space plot
        proj_val = np.dot(new_latent_np - latent_mean, pc1)
        # Generate and render mesh
        mesh_o3d = generate_and_render_mesh(new_latent_np, total_frames, i, device, model, n_pts_per_axis, 
                                            voxel_origin, voxel_size, offset, scale, icp_transform, objects, generated_mesh_count)
        generated_mesh_count += 1
        # Render views of model for video
        combined = render_cameras(renderers, mesh_o3d, i, material, total_frames, n_rotations)
        # Generate latent path plot image
        latent_path_img = generate_latent_path_plot(projections, proj_val, min_proj, max_proj, PC_idx, width=200, height=80)
        latent_path_img_bgr = cv2.cvtColor(latent_path_img, cv2.COLOR_RGB2BGR)

        # Overlay latent_path_img in the middle of the 2x2 grid
        # Calculate overlay position (centered horizontally & vertically)
        center_x = combined.shape[1] // 2
        center_y = combined.shape[0] // 2
        h, w, _ = latent_path_img_bgr.shape
        x_start = center_x - w // 2
        y_start = center_y - h // 2

        # Add transparency by blending (optional, here full opaque)
        combined[y_start:y_start+h, x_start:x_start+w] = latent_path_img_bgr
        
        # Write video
        out_video.write(combined)

        print(f"Captured frame {i + 1}/{total_frames}")

    except Exception as e:
        print(f"Error at frame {i}: {e}")
    finally:
        for var in ['mesh_out', 'mesh_pv', 'mesh_o3d', 'new_latent']:
            if var in locals():
                del locals()[var]
        gc.collect()

out_video.release()
print("Video saved as", video_path)