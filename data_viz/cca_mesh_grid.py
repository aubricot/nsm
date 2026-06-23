import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import pyvista as pv
import vtk # Required by NumpyTransform if used by create_mesh implicitly
import gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents

# --- Configuration for the grid image ---
CCA_SUBDIR = 'cca_analysis' # Subfolder (inside the run folder) holding the CCA analysis outputs.

CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
N_STEPS_WALK = 8       # default number of steps along a CCA axis
N_SD = 4        # default number of standard deviations to walk across
IMG_WIDTH_PER_MESH = 512
IMG_HEIGHT_PER_MESH = 512
MESH_RESOLUTION_N = 256 # TO DO: was 256
EYE_OFFSET_FACTOR = 0.6

# Configuration for background tinting in sparse regions
TINT_BACKGROUND_FOR_SPARSITY = True
BASE_BACKGROUND_COLOR_RGB_LIST = [0.72, 0.89, 0.78]  # Pale Seafoam
MAX_TINT_BACKGROUND_COLOR_RGB_LIST = [0.03, 0.11, 0.08] # Dark Seafoam for sparse areas

# --- End Configuration ---


# ──────────────────────────────────────────────────────────────────────────────
# Shared rendering helpers (used by both the PC grid and the CCA walk)
# ──────────────────────────────────────────────────────────────────────────────
def setup_renderer():
    """Build an offscreen Open3D renderer + a vertebra-like lit material."""
    ren = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH_PER_MESH, IMG_HEIGHT_PER_MESH)
    ren.scene.set_background(BASE_BACKGROUND_COLOR_RGB_LIST + [1.0])
    ren.scene.scene.set_sun_light([0.707, 0.707, 0.0], [1, 1, 1], 75000)
    ren.scene.scene.enable_sun_light(True)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.base_roughness = 0.4
    material.base_metallic = 0.1
    return ren, material


def render_latent_to_image(ren, material, mdl, latent_vector_np, dev,
                           voxel_origin, voxel_size, offset, scale, icp, objs,
                           bg_color, label=None):
    """Decode a single latent vector to a mesh and render it to an RGB image.

    Returns an (H, W, 3) uint8 array. On any failure a blank cell (background
    colored) is returned so the grid/strip still stitches cleanly.
    """
    ren.scene.set_background(bg_color)
    latent_tensor = torch.tensor(latent_vector_np, dtype=torch.float32).unsqueeze(0).to(dev)

    rendered_img_np = np.full((IMG_HEIGHT_PER_MESH, IMG_WIDTH_PER_MESH, 3),
                              (np.array(bg_color[:3]) * 255).astype(np.uint8),
                              dtype=np.uint8)
    o3d_mesh_to_render = None
    try:
        mesh_data_from_decoder = create_mesh(
            decoder=mdl, latent_vector=latent_tensor, n_pts_per_axis=MESH_RESOLUTION_N,
            voxel_origin=voxel_origin, voxel_size=voxel_size,
            path_original_mesh=None, offset=offset, scale=scale,
            icp_transform=icp, objects=objs, verbose=False, device=dev
        )
        m_inter = mesh_data_from_decoder[0] if isinstance(mesh_data_from_decoder, list) else mesh_data_from_decoder

        if isinstance(m_inter, pv.PolyData):
            if m_inter.n_points > 0:
                pv_m = m_inter.extract_geometry().triangulate()
                if pv_m.n_faces_strict > 0:
                    pv_m = pv_m.compute_normals(cell_normals=False, point_normals=True,
                                                inplace=False, auto_orient_normals=True)
                    o3d_mesh_to_render = pv_to_o3d(pv_m)
                else:
                    print(f"\033[93mWarning: PyVista mesh has 0 faces after triangulation.\033[0m")
            else:
                print(f"\033[93mWarning: PyVista mesh has 0 points.\033[0m")
        elif isinstance(m_inter, o3d.geometry.TriangleMesh):
            if m_inter.has_vertices() and m_inter.has_triangles():
                o3d_mesh_to_render = m_inter
                o3d_mesh_to_render.compute_vertex_normals()
                o3d_mesh_to_render.paint_uniform_color([0.7, 0.7, 0.7])
            else:
                print(f"\033[93mWarning: Open3D mesh has 0 triangles or vertices.\033[0m")
        else:
            print(f"\033[93mWarning: Mesh is None or unexpected type: {type(m_inter)}\033[0m")

        if o3d_mesh_to_render and o3d_mesh_to_render.has_triangles():
            ren.scene.clear_geometry()
            ren.scene.add_geometry("shape_mesh", o3d_mesh_to_render, material)

            bounds = o3d_mesh_to_render.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            max_dim_extent = np.max(bounds.get_extent())
            if max_dim_extent < 1e-3:
                max_dim_extent = 1.0

            base = np.array([1.0, 1.0, 0.8])
            magnitude = np.linalg.norm(base)
            base_norm = base / magnitude

            az = np.radians(30)
            el = np.radians(-20)

            cos_az, sin_az = np.cos(az), np.sin(az)
            rot_z = np.array([[cos_az, -sin_az, 0],
                            [sin_az,  cos_az, 0],
                            [0,       0,      1]])

            rotated = rot_z @ base_norm
            lateral = np.cross(rotated, np.array([0.0, 0.0, 1.0]))
            lateral = lateral / np.linalg.norm(lateral)

            cos_el, sin_el = np.cos(el), np.sin(el)
            rot_lat = (cos_el * np.eye(3)
                    + sin_el * np.array([[0, -lateral[2], lateral[1]],
                                            [lateral[2], 0, -lateral[0]],
                                            [-lateral[1], lateral[0], 0]])
                    + (1 - cos_el) * np.outer(lateral, lateral))

            view_dir = rot_lat @ rotated
            eye = center + view_dir * magnitude * max_dim_extent * EYE_OFFSET_FACTOR

            up_vector = np.array([0.0, 0.0, 1.0])
            ren.scene.camera.look_at(center, eye, up_vector)
            img_o3d = ren.render_to_image()
            rendered_img_np = np.asarray(img_o3d).copy()
    except Exception as e:
        print(f"\033[31mError during mesh generation or rendering: {e}\033[0m")
        import traceback
        traceback.print_exc()

    if label is not None:
        cv2.putText(rendered_img_np, label, (10, IMG_HEIGHT_PER_MESH - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    # cleanup
    del latent_tensor
    if 'mesh_data_from_decoder' in locals():
        del mesh_data_from_decoder
    if 'm_inter' in locals():
        del m_inter
    if 'pv_m' in locals():
        del pv_m
    if o3d_mesh_to_render is not None:
        del o3d_mesh_to_render
    gc.collect()
    if dev.startswith('cuda'):
        torch.cuda.empty_cache()
    return rendered_img_np


def tint_for_distance(min_dist_to_observed, max_relevant_distance_for_tint,
                      base_bg_color_np, max_tint_bg_color_np):
    """Interpolate background color by how far a walk point is from observed data."""
    if not TINT_BACKGROUND_FOR_SPARSITY:
        return list(base_bg_color_np) + [1.0]
    if max_relevant_distance_for_tint < 1e-9:
        normalized_dist = 0.0 if min_dist_to_observed < 1e-9 else 1.0
    elif max_relevant_distance_for_tint == float('inf'):
        normalized_dist = 0.0
    else:
        normalized_dist = min(1.0, max(0.0, min_dist_to_observed / max_relevant_distance_for_tint))
    interp = base_bg_color_np * (1 - normalized_dist) + max_tint_bg_color_np * normalized_dist
    return list(interp) + [1.0]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: CCA axis walk (interactive)
# ──────────────────────────────────────────────────────────────────────────────
def run_cca_walk(mdl, L, dev, cca, n_steps, n_sd,
                 voxel_origin, voxel_size, offset, scale, icp, objs):
    """Render a strip of meshes walking along the chosen CCA axis (1-based) in
    latent space. The axis is selected via the --cca argument (e.g. --cca 1).
    """
    cca_dir = CCA_SUBDIR

    # --- Load required CCA files ---
    lat_w_path   = os.path.join(cca_dir, 'cca_lat_weights.csv')
    corr_path    = os.path.join(cca_dir, 'cca_canonical_correlations.csv')
    scores_path  = os.path.join(cca_dir, 'cca_variate_scores.csv')

    missing = [p for p in (lat_w_path, corr_path, scores_path) if not os.path.isfile(p)]
    if missing:
        print(f"\033[31mError: missing file(s):\n" +
              "\n".join(f"  {p}" for p in missing) +
              f"\nRun merge_latents_to_landmarks_cca.py for this run first.\033[0m")
        return

    lat_w     = pd.read_csv(lat_w_path,  index_col=0)  # (n_lat, K)  latent_0..latent_{n-1} × CCA1..CCAK
    corr_df   = pd.read_csv(corr_path)                 # rank, r, r2, p_value, significant
    scores_df = pd.read_csv(scores_path)               # ..., B_1, B_2, ...

    # --- Validate requested axis ---
    norm = f"CCA{cca}"
    if norm not in lat_w.columns:
        print(f"\033[31m{norm} is not available. "
              f"Choose --cca in 1..{lat_w.shape[1]} "
              f"(cols: {list(lat_w.columns)[:5]}...).\033[0m")
        return

    score_col = f"B_{cca}"
    if score_col not in scores_df.columns:
        print(f"\033[31mMissing {score_col} in variate scores; cannot set range.\033[0m")
        return

    # --- Build direction vector in full latent space ---
    n_lat      = lat_w.shape[0]
    latent_dim = L.shape[1]
    mean_full  = L.mean(0)                              # (latent_dim,)

    w   = lat_w[norm].values.astype(np.float64)         # (n_lat,)
    wn2 = float(w @ w)
    if wn2 < 1e-12:
        print(f"\033[31m{norm} weight vector is ~zero; cannot walk.\033[0m")
        return

    # Projecting a canonical score s back into latent space:
    #   latent = mean + s * (w / ||w||²)
    direction_full = np.zeros(latent_dim, dtype=np.float64)
    direction_full[:n_lat] = w / wn2

    # --- Score range from observed canonical variates ---
    observed    = scores_df[score_col].dropna().values.astype(np.float64)
    s_mean = float(observed.mean())
    s_sd   = float(observed.std())
    s_min  = s_mean - n_sd * s_sd
    s_max  = s_mean + n_sd * s_sd
    score_range  = np.linspace(s_min, s_max, n_steps)

    r_k = float(corr_df.loc[corr_df['rank'] == cca, 'r'].iloc[0])
    print(f"\033[92m[cca] {norm}: r={r_k:.4f}, "
          f"score range [{s_min:.3f}, {s_max:.3f}] over {n_steps} steps\033[0m")
    print(f"Max displacement norm : {np.linalg.norm(s_max * direction_full):.4f}")
    print(f"Mean latent norm      : {np.linalg.norm(mean_full):.4f}")

    # --- Background tint: one grid step worth of distance = fully tinted ---
    base_bg     = np.array(BASE_BACKGROUND_COLOR_RGB_LIST)
    max_tint_bg = np.array(MAX_TINT_BACKGROUND_COLOR_RGB_LIST)
    sorted_obs = np.sort(observed)
    nn_dists = np.diff(sorted_obs)
    max_rel_dist = float(np.median(nn_dists)) * 3.0  # typical gap between observations

    # --- Render each step ---
    ren, material = setup_renderer()
    row_images = []

    try:
        for i, s_val in enumerate(score_range):
            print(f"\033[92m  mesh {i+1}/{n_steps}  ({norm} score={s_val:+.3f})\033[0m")

            latent_vec = mean_full + s_val * direction_full
            min_dist   = float(np.min(np.abs(observed - s_val)))
            bg_color   = tint_for_distance(min_dist, max_rel_dist, base_bg, max_tint_bg)

            img = render_latent_to_image(
                ren, material, mdl, latent_vec, dev,
                voxel_origin, voxel_size, offset, scale, icp, objs,
                bg_color, label=f"{norm}={s_val:+.2f} SD"
            )
            row_images.append(img)
    finally:
        del ren
        gc.collect()
        if dev.startswith('cuda'):
            torch.cuda.empty_cache()

    # --- Save strip ---
    if row_images:
        strip    = np.hstack(row_images)
        out_name = os.path.join(cca_dir, f"cca_walk_{norm}_{n_steps}steps.png")
        cv2.imwrite(out_name, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"\033[92m[cca] saved walk image -> {out_name}\033[0m")
    else:
        print("\033[31mNo images were generated.\033[0m")

def main():
    parser = argparse.ArgumentParser(
        description="Render meshes along a CCA latent axis (default) or a 2D PCA grid.")
    parser.add_argument('--run_folder', required=True,
                        help='Run name (e.g. "run_v57n", resolved as ./run_v57n) '
                             'or a full/relative path to the run folder.')
    parser.add_argument('--cca', type=int, default=1,
                        help='Which CCA axis to walk, 1-based. e.g. --cca 1 uses CCA1.')
    parser.add_argument('--n_steps', type=int, default=N_STEPS_WALK,
                        help='Number of steps along the CCA axis.')
    parser.add_argument('--n_sd', type=int, default=N_SD,
                        help='Number of standard deviations to traverse along the CCA axis.')
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
    LC_PATH = 'latent_codes/' + CKPT + '.pth'
    MODEL_PATH = 'model/' + CKPT + '.pth'

    config = load_config(config_path='model_params_config.json')
    dev = config.get("device", "cuda:0")

    mdl, ck2, L = load_model_and_latents(MODEL_PATH, LC_PATH, config, dev)

    # Sampling grid / transform for SDF -> mesh
    origin_val = 1.0
    voxel_origin = (-origin_val,) * 3
    voxel_size = (origin_val * 2) / (MESH_RESOLUTION_N - 1)
    offset = np.zeros(3)
    scale = 1.0
    icp = NumpyTransform(np.eye(4))
    objs = 1

    run_cca_walk(mdl, L, dev, args.cca, args.n_steps, args.n_sd,
                     voxel_origin, voxel_size, offset, scale, icp, objs)


    del mdl, ck2, L
    gc.collect()
    if dev.startswith('cuda'):
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == '__main__':
    main()
