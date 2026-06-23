# Make a 4-way split video exploring shape along a CCA latent axis.
#
# Mirrors the CCA-loading logic of pc_mesh_grid_cca.py: instead of PCA on the
# latents, it walks along a canonical direction from {run_folder}/cca_analysis/
# (produced by merge_latents_to_landmarks_cca.py). Pick the axis with --cca.

import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc, argparse
import pandas as pd
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image
from NSM.helper_funcs import (NumpyTransform, pv_to_o3d, load_config, load_model_and_latents,
                              render_cameras, generate_and_render_mesh)

# --- Configuration --- 
CCA_SUBDIR = 'cca_analysis'   # Subfolder (inside the run folder) holding the CCA analysis outputs.
CKPT = '3000'            # checkpoint to analyze
N_ROTATIONS = 1
AMPLIFY = 3            # exaggerate the sweep beyond the observed score range
N_PTS_PER_AXIS = 128
WIDTH, HEIGHT = 640, 480
FPS = 15
# --- End Configuration ---


def generate_cca_path_plot(projections, proj_val, min_proj, max_proj, cca_k, r_val=None,
                           width=200, height=80):
    """1D position plot for the current canonical score along a CCA axis.

    Local CCA-labeled variant of NSM.traverse_latents.generate_latent_path_plot.
    """
    projections = projections.flatten()
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=200)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(min_proj, max_proj)
    ax.plot([min_proj, max_proj], [0, 0], color='paleturquoise', alpha=0.2, linewidth=1)
    ax.scatter(proj_val, 0, color='deeppink', s=10)
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(['0'])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = f"Latent Path (CCA{cca_k})" if r_val is None else f"Latent Path (CCA{cca_k}, r={r_val:.3f})"
    ax.set_title(title, fontsize=10, color='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)[..., :3]
    return img_np


def main():
    parser = argparse.ArgumentParser(
        description="4-way split video walking along a CCA latent axis.")
    parser.add_argument("--run_folder", required=True,
                        help='Run name (e.g. "run_v57n", resolved as ./run_v57n) '
                             'or a full/relative path to the run folder.')
    parser.add_argument("--cca", type=int, default=1,
                        help="Which CCA axis to walk, 1-based (e.g. --cca 1 uses CCA1).")
    args = parser.parse_args()

    # Resolve --run_folder: a plain name is treated as ./{name}; a path is used as-is.
    TRAIN_DIR = args.run_folder
    if os.path.basename(os.path.normpath(TRAIN_DIR)) == TRAIN_DIR.strip('/\\'):
        TRAIN_DIR = os.path.join('.', TRAIN_DIR)
    TRAIN_DIR = os.path.normpath(TRAIN_DIR)
    if not os.path.isdir(TRAIN_DIR):
        print(f"\033[31mError: run folder not found: {TRAIN_DIR}\033[0m")
        return
    os.chdir(TRAIN_DIR)

    LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
    MODEL_PATH = 'model' + '/' + CKPT + '.pth'

    # Load config + model + latents
    config = load_config(config_path='model_params_config.json')
    device = config.get("device", "cuda:0")
    model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

    # --- Load CCA analysis tables ---
    lat_w_path = os.path.join(CCA_SUBDIR, 'cca_lat_weights.csv')
    corr_path = os.path.join(CCA_SUBDIR, 'cca_canonical_correlations.csv')
    for p in (lat_w_path, corr_path):
        if not os.path.isfile(p):
            print(f"\033[31mError: missing {p}.\n"
                  f"Run merge_latents_to_landmarks_cca.py for this run first.\033[0m")
            return
    lat_w = pd.read_csv(lat_w_path, index_col=0)   # rows latent_0..latent_{n_lat-1}, cols CCA1..CCAK
    corr_df = pd.read_csv(corr_path)

    norm = f"CCA{args.cca}"
    if norm not in lat_w.columns:
        print(f"\033[31m{norm} is not available. Choose --cca in "
              f"1..{lat_w.shape[1]} (cols: {list(lat_w.columns)[:5]}...).\033[0m")
        return
    k = args.cca
    w = lat_w[norm].values.astype(np.float64)       # (n_lat,)
    wn2 = float(w @ w)
    if wn2 < 1e-12:
        print(f"\033[31m{norm} weight vector is ~zero; cannot walk.\033[0m")
        return
    r_k = float(corr_df.loc[corr_df['rank'] == k, 'r'].iloc[0]) if (corr_df['rank'] == k).any() else None

    # Mesh creation params
    recon_grid_origin = 1.0
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (N_PTS_PER_AXIS - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # --- Canonical direction in full latent space (walk param == canonical score) ---
    latents_np = latent_codes.numpy()
    latent_dim = latents_np.shape[1]
    n_lat = lat_w.shape[0]

    latent_mean = latents_np.mean(axis=0)
    mean_block = latent_mean[:n_lat]     

    direction_full = np.zeros(latent_dim, dtype=np.float64)
    direction_full[:n_lat] = w / wn2

    # Observed canonical scores for every training shape (same indexing as latents_np)
    projections = (latents_np[:, :n_lat] - mean_block).dot(w)
    s_mean = float(projections.mean())
    max_proj = float(projections.max())
    min_proj = float(projections.min())

    # Oscillating sweep: low -> mean -> high -> mean -> low
    n_seg = 15 * N_ROTATIONS
    alpha_vals = np.concatenate([
        np.linspace(min_proj, s_mean,   n_seg, endpoint=False),
        np.linspace(s_mean,   max_proj, n_seg, endpoint=False),
        np.linspace(max_proj, s_mean,   n_seg, endpoint=False),
        np.linspace(s_mean,   min_proj, n_seg),
    ]) 
    total_frames = len(alpha_vals)

    # Renderers (4) + material
    renderers = [o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT) for _ in range(4)]
    for r in renderers:
        r.scene.set_background([0.0, 0.0, 0.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]


    # Video writer (2x2 grid -> 1280x960) into the cca_analysis subfolder
    os.makedirs(CCA_SUBDIR, exist_ok=True)
    video_path = os.path.join(CCA_SUBDIR, f"cca{k}_4way_{N_PTS_PER_AXIS}p_{AMPLIFY}x.mp4")
    out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH * 2, HEIGHT * 2))
    
    alpha_vals_plot = alpha_vals / AMPLIFY
    generated_mesh_count = 0
    mesh_o3d = None
    try:
        for i, (alpha, proj_val) in enumerate(zip(alpha_vals, alpha_vals_plot)):
            try:
                new_latent_np = latent_mean + AMPLIFY * alpha * direction_full
                mesh_o3d = generate_and_render_mesh(
                    new_latent_np, total_frames, i, device, model, N_PTS_PER_AXIS,
                    voxel_origin, voxel_size, offset, scale, icp_transform,
                    objects, generated_mesh_count)
                generated_mesh_count += 1

                combined = render_cameras(renderers, mesh_o3d, i, material, total_frames, N_ROTATIONS)

                path_img = generate_cca_path_plot(
                    projections, proj_val, min_proj, max_proj, k, r_val=r_k, width=200, height=80)
                path_bgr = cv2.cvtColor(path_img, cv2.COLOR_RGB2BGR)
                h, ww = path_bgr.shape[:2]
                cy, cx = combined.shape[0] // 2, combined.shape[1] // 2
                combined[cy - h//2 : cy + h//2, cx - ww//2 : cx + ww//2] = path_bgr

                out_video.write(combined)
                print(f"Captured frame {i + 1}/{total_frames}")

            except Exception as e:
                print(f"Error at frame {i}: {e}")
            finally:
                mesh_o3d = None
                gc.collect()
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
    finally:
        out_video.release()
        print("Video saved as", video_path)


if __name__ == '__main__':
    main()
