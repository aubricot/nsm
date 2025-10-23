import os
import json
import torch
import numpy as np
import cv2
import open3d as o3d
import pyvista as pv
import vtk # Required by NumpyTransform if used by create_mesh implicitly
import gc

# Assuming NSM.mesh and NSM.models are in PYTHONPATH or accessible
try:
    from NSM.mesh import create_mesh
    from NSM.models import TriplanarDecoder
except ImportError as e:
    print(f"\033[31mCould not import NSM modules. Make sure NSM is in your PYTHONPATH: {e}\033[0m")
    print("\033[31mPlease ensure the NSM package (containing mesh.py and models.py) is accessible.\033[0m")
    exit()

# --- Configuration for the grid image ---
TRAIN_DIR = "run_v30" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
NUM_STEPS_PC0 = 8
NUM_STEPS_PC1 = 8
IMG_WIDTH_PER_MESH = 512
IMG_HEIGHT_PER_MESH = 512
PC_INDEX_0 = 0 # TO DO: change to inspect other PCs
PC_INDEX_1 = PC_INDEX_0 + 1
MESH_RESOLUTION_N = 256 # TO DO: was 256
OUTPUT_FILENAME = f"pc{PC_INDEX_0+1}_vs_pc{PC_INDEX_1+1}_grid_{NUM_STEPS_PC0}x{NUM_STEPS_PC1}_{IMG_WIDTH_PER_MESH}p_{MESH_RESOLUTION_N}p.png" # TO DO: Update filename
EYE_OFFSET_FACTOR = 1.2

# Configuration for background tinting in sparse regions
TINT_BACKGROUND_FOR_SPARSITY = True
BASE_BACKGROUND_COLOR_RGB_LIST = [0.72, 0.89, 0.78]  # Pale Seafoam
MAX_TINT_BACKGROUND_COLOR_RGB_LIST = [0.03, 0.11, 0.08] # Dark Seafoam for sparse areas
# --- End Configuration ---

# For transforming and aligning meshes with ICP downstream
class NumpyTransform:
    def __init__(self, m):
        self.m = m

    def GetMatrix(self):
        M = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                M.SetElement(i, j, self.m[i, j])
        return M

# Convert pyvista object to open3d for rendering views in grid
def pv_to_o3d(pv_m):
    if pv_m is None or pv_m.points is None:
        return None
    pts = np.asarray(pv_m.points)
    if pv_m.faces is None or pv_m.n_faces_strict == 0:
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(pts)
        return m
    F = np.asarray(pv_m.faces)
    try:
        tris = F.reshape(-1, 4)[:, 1:4]
    except ValueError:
        temp_pv_mesh = pv.PolyData(pts, faces=F)
        if not temp_pv_mesh.is_all_triangles:
             temp_pv_mesh.triangulate(inplace=True)
        F_tri = np.asarray(temp_pv_mesh.faces)
        tris = F_tri.reshape(-1,4)[:,1:4]

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(pts)
    m.triangles = o3d.utility.Vector3iVector(tris)
    m.compute_vertex_normals()
    m.paint_uniform_color([0.7, 0.7, 0.7])
    return m

# Plot the grid
def main():
    # Load config
    config_path = 'model_params_config.json'
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        print(f"\033[92mLoaded config from {config_path}\033[0m")
    except FileNotFoundError:
        print(f"\033[31mError: model_params_config.json not found in {config_path}. Exiting.\033[0m")

    dev = cfg.get('device', 'cuda:0')
    if not torch.cuda.is_available() and dev.startswith('cuda'):
        print("CUDA device requested but not available. Switching to CPU.")
        dev = 'cpu'
    # Load trained model and latent code from last checkpoint
    try:
        lc = torch.load(LC_PATH, map_location=dev) 
        ck2 = torch.load(MODEL_PATH, map_location=dev)
    except FileNotFoundError as e:
        print(f"\033[31mError loading model/latent code files: {e}\033[0m")
        print(f"\033[31mMake sure {LC_PATH} and {MODEL_PATH} exist in the current directory.\033[0m")
        return

    # Get the latent codes
    L = lc['latent_codes']['weight'].detach().cpu().numpy()

    required_keys = ['latent_size','objects_per_decoder','conv_hidden_dims','conv_deep_image_size','conv_norm','conv_norm_type','conv_start_with_mlp','sdf_latent_size','sdf_hidden_dims','weight_norm','final_activation','activation','dropout_prob','sum_conv_output_features','conv_pred_sdf']
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        print(f"\033[31mError: Config file is missing required keys for model arguments: {missing_keys}\033[0m")
        return

    # Prepare trained model for inference using define params from config file
    args = {k: cfg[k] for k in required_keys}
    triplane_args = {
        'latent_dim': args['latent_size'], 'n_objects': args['objects_per_decoder'],
        'conv_hidden_dims': args['conv_hidden_dims'], 'conv_deep_image_size': args['conv_deep_image_size'],
        'conv_norm': args['conv_norm'], 'conv_norm_type': args['conv_norm_type'],
        'conv_start_with_mlp': args['conv_start_with_mlp'],
        'sdf_latent_size': args['sdf_latent_size'], 'sdf_hidden_dims': args['sdf_hidden_dims'],
        'sdf_weight_norm': args['weight_norm'], 'sdf_final_activation': args['final_activation'],
        'sdf_activation': args['activation'], 'sdf_dropout_prob': args['dropout_prob'],
        'sum_sdf_features': args['sum_conv_output_features'],
        'conv_pred_sdf': args['conv_pred_sdf']
    }
    mdl = TriplanarDecoder(**triplane_args)
    mdl.load_state_dict(ck2['model'])
    mdl.to(dev)
    mdl.eval()

    # Set centroids
    origin_val = 1.0 # TO DO: Adjust to fine tune sampled regions of shape for SDF calc (was 1.0)
    voxel_origin = (-origin_val,) * 3
    voxel_size = (origin_val * 2) / (MESH_RESOLUTION_N - 1)
    offset = np.zeros(3)
    scale = 1.0
    icp = NumpyTransform(np.eye(4)) # Iterative Closest Point to align and transform meshes
    objs = 1

    # Run PCA of embeddings
    mu = L.mean(0) # Get the mean shape vectors
    C = L - mu # Center the mean shape vectors
    U, S_val, Vt = np.linalg.svd(C, full_matrices=False) # Do singular value decomposition (SVD) to get principal components
    scores = C.dot(Vt.T) # project data into principal component space to get PC scores

    if scores.shape[1] <= max(PC_INDEX_0, PC_INDEX_1):
        print(f"\033[31mError: Not enough principal components. Requested indices {PC_INDEX_0}, {PC_INDEX_1}"
              f"but data only has {scores.shape[1]} components after PCA.\033[0m")
        return

    # Find min and max scores
    min_score_pc0 = scores[:, PC_INDEX_0].min()
    max_score_pc0 = scores[:, PC_INDEX_0].max()
    min_score_pc1 = scores[:, PC_INDEX_1].min()
    max_score_pc1 = scores[:, PC_INDEX_1].max()

    # Get range of PC scores to color grid by sparsity
    pc0_score_range = np.linspace(min_score_pc0, max_score_pc0, NUM_STEPS_PC0)
    pc1_score_range = np.linspace(min_score_pc1, max_score_pc1, NUM_STEPS_PC1)
    projected_observed_scores = scores[:, [PC_INDEX_0, PC_INDEX_1]]
    max_relevant_distance_for_tint = 0.0
    base_bg_color_np = np.array(BASE_BACKGROUND_COLOR_RGB_LIST)
    max_tint_bg_color_np = np.array(MAX_TINT_BACKGROUND_COLOR_RGB_LIST)

    if TINT_BACKGROUND_FOR_SPARSITY:
        step_pc0 = (max_score_pc0 - min_score_pc0) / (NUM_STEPS_PC0 - 1) if NUM_STEPS_PC0 > 1 else float('inf')
        step_pc1 = (max_score_pc1 - min_score_pc1) / (NUM_STEPS_PC1 - 1) if NUM_STEPS_PC1 > 1 else float('inf')
        
        min_step = float('inf')
        if NUM_STEPS_PC0 > 1 and step_pc0 > 1e-9: # Ensure step is meaningful
            min_step = min(min_step, step_pc0)
        if NUM_STEPS_PC1 > 1 and step_pc1 > 1e-9: # Ensure step is meaningful
            min_step = min(min_step, step_pc1)
        
        if min_step == float('inf') or min_step < 1e-9:
            # Grid is effectively a single point or line, or PC range is zero.
            # Default to a very large distance so effectively no tinting unless points are extremely far.
            max_relevant_distance_for_tint = float('inf')
            print("\033[93mWarning: Could not determine a meaningful grid step for tint normalization. Tinting may be minimal.\033[0m")
        else:
            max_relevant_distance_for_tint = min_step # Max tint if distance is one min_step
        print(f"\033[92mMax relevant distance for full background tint: {max_relevant_distance_for_tint:.3f}\033[0m")

    # Render scene
    ren = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH_PER_MESH, IMG_HEIGHT_PER_MESH)
    # Initial background, will be overridden per cell if tinting
    ren.scene.set_background(list(base_bg_color_np) + [1.0])
    ren.scene.scene.set_sun_light([0.707, 0.707, 0.0], [1,1,1], 75000)
    ren.scene.scene.enable_sun_light(True)
    
    # Material (vertebrae) properties
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.base_roughness = 0.4
    material.base_metallic = 0.1

    all_rows_images = []
    total_meshes_to_generate = NUM_STEPS_PC0 * NUM_STEPS_PC1
    generated_mesh_count = 0

    print(f"Generating {NUM_STEPS_PC0}x{NUM_STEPS_PC1} grid for PC{PC_INDEX_0 + 1} vs PC{PC_INDEX_1 + 1}...")
    # Generate grid for PC's
    for i, s1_val in enumerate(reversed(pc1_score_range)):
        current_row_image_list = []
        for j, s0_val in enumerate(pc0_score_range):
            generated_mesh_count += 1
            print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{total_meshes_to_generate}\033[0m"
                  f"(PC{PC_INDEX_0 + 1}={s0_val:.2f}, PC{PC_INDEX_1 + 1}={s1_val:.2f})...", end="")

            # Set background color
            current_bg_to_set = list(base_bg_color_np) + [1.0] # Default background
            if TINT_BACKGROUND_FOR_SPARSITY:
                current_grid_node_coords = np.array([s0_val, s1_val])
                dist_sq_to_observed = np.sum((projected_observed_scores - current_grid_node_coords)**2, axis=1)
                min_dist_to_observed = np.sqrt(np.min(dist_sq_to_observed))
                
                normalized_dist = 0.0
                if max_relevant_distance_for_tint < 1e-9: # Avoid division by zero if range is ~0
                    normalized_dist = 0.0 if min_dist_to_observed < 1e-9 else 1.0
                elif max_relevant_distance_for_tint == float('inf'):
                     normalized_dist = 0.0 # No tint if distance for normalization is infinite
                else:
                    normalized_dist = min(1.0, max(0.0, min_dist_to_observed / max_relevant_distance_for_tint))
                
                # Interpolate background color
                interpolated_bg_color = base_bg_color_np * (1 - normalized_dist) + max_tint_bg_color_np * normalized_dist
                current_bg_to_set = list(interpolated_bg_color) + [1.0]
                print(f"(dist_norm: {normalized_dist:.2f})", end="")
            print() # Newline for next mesh or message
            ren.scene.set_background(current_bg_to_set) # Set background for this specific cell
            target_coeffs = np.zeros(scores.shape[1])
            target_coeffs[PC_INDEX_0] = s0_val
            target_coeffs[PC_INDEX_1] = s1_val
            
            # Use trained model to reconstruct shape from embedded PC values 
            reconstructed_latent = mu + target_coeffs.dot(Vt)
            latent_tensor = torch.tensor(reconstructed_latent, dtype=torch.float32).unsqueeze(0).to(dev)

            # Build mesh
            o3d_mesh_to_render = None
            rendered_img_np = np.full((IMG_HEIGHT_PER_MESH, IMG_WIDTH_PER_MESH, 3), 
                                      (np.array(current_bg_to_set[:3])*255).astype(np.uint8), # Use actual BG for placeholder
                                      dtype=np.uint8)
            try:
                mesh_data_from_decoder = create_mesh(
                    decoder=mdl, latent_vector=latent_tensor, n_pts_per_axis=MESH_RESOLUTION_N,
                    voxel_origin=voxel_origin, voxel_size=voxel_size,
                    path_original_mesh=None, offset=offset, scale=scale,
                    icp_transform=icp, objects=objs, verbose=False, device=dev
                )
                
                m_inter = mesh_data_from_decoder[0] if isinstance(mesh_data_from_decoder, list) else mesh_data_from_decoder

                if isinstance(m_inter, pv.PolyData):
                    if m_inter.n_points > 0 :
                         pv_m = m_inter.extract_geometry().triangulate()
                         if pv_m.n_faces_strict > 0:
                            pv_m = pv_m.compute_normals(cell_normals=False, point_normals=True, inplace=False, auto_orient_normals=True)
                            o3d_mesh_to_render = pv_to_o3d(pv_m)
                         else:
                            print(f"\033[93mWarning: PyVista mesh for (PC0={s0_val:.2f}, PC1={s1_val:.2f}) has 0 faces after triangulation.\033[0m")
                    else:
                        print(f"\033[93mWarning: PyVista mesh for (PC0={s0_val:.2f}, PC1={s1_val:.2f}) has 0 points.\033[0m")
                elif isinstance(m_inter, o3d.geometry.TriangleMesh):
                    if m_inter.has_vertices() and m_inter.has_triangles():
                        o3d_mesh_to_render = m_inter
                        o3d_mesh_to_render.compute_vertex_normals()
                        o3d_mesh_to_render.paint_uniform_color([0.7,0.7,0.7])
                    else:
                        print(f"\033[93mWarning: Open3D mesh for (PC0={s0_val:.2f}, PC1={s1_val:.2f}) has 0 triangles or vertices.\033[0m")
                else:
                    print(f"\033[93mWarning: Mesh for (PC0={s0_val:.2f}, PC1={s1_val:.2f}) is None or unexpected type: {type(m_inter)}\033[0m")

                if o3d_mesh_to_render and o3d_mesh_to_render.has_triangles():
                    ren.scene.clear_geometry()
                    ren.scene.add_geometry("shape_mesh", o3d_mesh_to_render, material)
                    
                    bounds = o3d_mesh_to_render.get_axis_aligned_bounding_box()
                    center = bounds.get_center()
                    max_dim_extent = np.max(bounds.get_extent())
                    if max_dim_extent < 1e-3: max_dim_extent = 1.0

                    eye = center + np.array([1.0, 1.0, 0.8]) * max_dim_extent * EYE_OFFSET_FACTOR
                    up_vector = np.array([0.0, 0.0, 1.0])
                    
                    ren.scene.camera.look_at(center, eye, up_vector)
                    img_o3d = ren.render_to_image()
                    rendered_img_np = np.asarray(img_o3d)

                else:
                    if o3d_mesh_to_render and not o3d_mesh_to_render.has_triangles():
                         print(f"\033[93mSkipping render for mesh (PC0={s0_val:.2f}, PC1={s1_val:.2f}) as it has no triangles.\033[0m")
            except Exception as e:
                print(f"\033[31mError during mesh generation or rendering for (PC0={s0_val:.2f}, PC1={s1_val:.2f}): {e}\033[0m")
                import traceback
                traceback.print_exc()
            
            current_row_image_list.append(rendered_img_np)

            del latent_tensor, reconstructed_latent, target_coeffs
            if 'mesh_data_from_decoder' in locals(): del mesh_data_from_decoder
            if 'm_inter' in locals(): del m_inter
            if 'pv_m' in locals(): del pv_m
            if o3d_mesh_to_render is not None: del o3d_mesh_to_render
            gc.collect()
            if dev.startswith('cuda'):
                torch.cuda.empty_cache()
        
        if current_row_image_list:
            h_stitched_row = np.hstack(current_row_image_list)
            all_rows_images.append(h_stitched_row)

    # Build and save the grid
    if all_rows_images:
        final_grid_image = np.vstack(all_rows_images)
        cv2.imwrite(OUTPUT_FILENAME, cv2.cvtColor(final_grid_image, cv2.COLOR_RGB2BGR))
        print(f"\033[92mGrid image saved to {OUTPUT_FILENAME}\033[0m")
    else:
        print("\033[31mNo images were generated to create the grid.\033[0m")
    
    del mdl, lc, ck2, L, scores, U, S_val, Vt, C, mu, ren
    gc.collect()
    if dev.startswith('cuda'):
        torch.cuda.empty_cache()
    print("Done.")

if __name__ == '__main__':
    main()