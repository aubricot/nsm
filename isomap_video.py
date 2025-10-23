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

# Define PC index and model checkpoint to use for video generation
TRAIN_DIR = "run_v30" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
USE_AVERAGES = True # TO DO: Use region averages or individual vertebrae?

class NumpyTransform:
    def __init__(self, matrix):
        self.matrix = matrix
    def GetMatrix(self):
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, self.matrix[i, j])
        return vtk_mat

def pv_to_o3d(mesh_pv):
    pts = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces)
    tris = faces.reshape(-1,4)[:,1:4]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tris)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def sample_latent_grid(latent_2d, num_x, num_y):
    x_min, y_min = latent_2d.min(axis=0)
    x_max, y_max = latent_2d.max(axis=0)
    x_vals = np.linspace(x_min, x_max, num_x)
    y_vals = np.linspace(y_min, y_max, num_y)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_samples = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_samples

def solve_tsp_nearest_neighbor(dist_matrix):
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    path = [0]  # Start at point 0
    visited[0] = True
    for _ in range(1, N):
        last = path[-1]
        # Mask visited nodes
        dists = dist_matrix[last]
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        path.append(next_idx)
        visited[next_idx] = True
    return path

def interpolate_latent_loop(latents, steps_per_segment=10):
    loop_latents = []
    for i in range(len(latents) - 1):
        start = latents[i]
        end = latents[i + 1]
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            interp = (1 - t) * start + t * end
            loop_latents.append(interp)
    return np.array(loop_latents)

def resample_by_cumulative_distance(latents, n_frames):
    diffs = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    dists = np.concatenate([[0], np.cumsum(diffs)])
    dists /= dists[-1]  # Normalize to [0, 1]
    new_steps = np.linspace(0, 1, n_frames)
    new_latents = np.array([
        np.interp(new_steps, dists, latents[:, i]) for i in range(latents.shape[1])
    ]).T
    return new_latents

def project_to_isomap(latents_query, latents_all, isomap_2d):
    tree = cKDTree(latents_all)
    _, indices = tree.query(latents_query, k=1)
    return isomap_2d[indices], indices

def plot_latent_paths(isomap_2d, sampled_points, vert_region, use_averages=False, save_dir="."):
    # Plot settings
    line_width = 2
    alpha = 0.7
    start_marker_size = 50
    end_marker_size = 50
    path_dict = { # TO DO: Adjust plotting colors
    "loop": {
        "data": loop_2d,
        "color": "violet",
        "title": "Latent Interpolation Path"
    },
    "tsp": {
        "data": tsp_2d,
        "color": "olivedrab",
        "title": "TSP-Ordered Path"
    },
    "smooth": {
        "data": smooth_loop_2d,
        "color": "deepskyblue",
        "title": "Smoothed TSP Path"
    }
    }
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, (key, path_info) in zip(axs, path_dict.items()):
        path = path_info["data"]
        color = path_info["color"]
        title = path_info["title"]
        ax.set_title(title)
        ax.scatter(isomap_2d[:, 0], isomap_2d[:, 1], c='lightgray', marker='o', s=10, label='All codes')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], marker='x', s=20, color='dimgrey', label='Sample Grid')
        # Path line
        ax.plot(path[:, 0], path[:, 1], '-', lw=line_width, color=color, alpha=alpha, label=f'Path ({key})')
        # Start and end points
        ax.scatter(*path[0], color=color, edgecolor='black', marker='o', s=start_marker_size, alpha=alpha, label='Start')
        ax.scatter(*path[-1], color=color, edgecolor='black', marker='X', s=end_marker_size, alpha=alpha, label='End')
        ax.set_aspect('equal', adjustable='box')
    # Legend on the last subplot
    axs[-1].legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        borderaxespad=0.0,
        fontsize='small'
    )
    # Title and save
    plt.suptitle("Latent Interpolation Paths in Isomap 2D", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    # Save figure
    figpath = os.path.join(save_dir, f"latent_space_path_overlay_isomap_video_C-T-L") # TO DO: define file path
    if use_averages == True:
        figpath = figpath + '_avg' + '.png'
    else:
        figpath = figpath + '.png'
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\033[92mSaved latent space path overlay to {figpath}\033[0m")

# Load config
config_path = 'model_params_config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\033[92mLoaded config from {config_path}\033[0m")
except FileNotFoundError:
    raise FileNotFoundError(f"Error: model_params_config.json not found at {config_path}")

device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load latent codes
latent_ckpt = torch.load(LC_PATH, map_location=device)
latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()

# Define vertebral regions
vertebral_regions = ['C', 'T', 'L']
latent_codes_subs = []
all_vtk_files_subs = []
# Match "_C1" to "_C40" or "-C1" to "-C40"
for vert_region in vertebral_regions:
    r_p = r'[_-]' + vert_region + r'([1-9]|[1-3][0-9]|40)(?!\d)'
    pattern = re.compile(r_p, re.IGNORECASE)
    # Subset indices from all paths
    matches = [
                (i, int(pattern.search(fname).group(1)))
                for i, fname in enumerate(all_vtk_files)
                if pattern.search(fname)
                ]
    indices = [i for i, _ in matches]
    #cervical_nums = [num for _, num in matches]

    # Filter latent codes and corresponding mesh paths
    vert_region_codes = latent_codes[indices]
    vert_region_files = [all_vtk_files[i] for i in indices]
    print(f"\n\nFound {len(vert_region_files)} latent codes for region: {vert_region}")
    print(f"Sample files: {vert_region_files[:5]}\n\n")

    # To average across vertebrae regions
    # Regex to extract specimen ID from filename by removing "_C1" or "-C1", etc.
    if USE_AVERAGES == True:
        r_p_specimen = r'^(.*?)(?:[-_]\d+[-_]' + vert_region + r'\d+)(?:.*)$'
        specimen_pattern = re.compile(r_p_specimen, re.IGNORECASE)
        specimen_latents = {}
        specimen_files = {}
        for fname, latent in zip(vert_region_files, vert_region_codes):
            match = specimen_pattern.match(fname)
            if match:
                specimen_id = match.group(1)
                if specimen_id not in specimen_latents:
                    specimen_latents[specimen_id] = []
                    specimen_files[specimen_id] = []
                specimen_latents[specimen_id].append(latent.numpy())
                specimen_files[specimen_id].append(fname)
            else:
                print(f"\033[93mWarning: could not extract specimen ID from {fname}\033[0m")

        # Average the latent codes per specimen
        avg_latent_codes = []
        avg_specimen_ids = []
        for specimen_id, latents in specimen_latents.items():
            avg_latent = np.mean(latents, axis=0)
            avg_latent_codes.append(avg_latent)
            avg_specimen_ids.append(specimen_id + '_' + vert_region)
        # Convert to NumPy array
        avg_latent_codes = np.array(avg_latent_codes)
        print(f"\nAveraged latent codes for {len(avg_specimen_ids)} specimens.\nSample specimen IDs: {avg_specimen_ids[:5]}")
        vert_region_codes = avg_latent_codes
        vert_region_files = avg_specimen_ids
    # Add to dictionary
    latent_codes_subs.extend(vert_region_codes)
    all_vtk_files_subs.extend(vert_region_files)

latent_codes_tensor = torch.stack([torch.tensor(latent) for latent in latent_codes_subs])
latent_codes_subs = latent_codes_tensor
print(f"Running analysis for : {len(all_vtk_files_subs)} averaged latent codes for vertebral regions {vertebral_regions}.")

# Set up model
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

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 128
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
plot_latent_paths(isomap_2d, sampled_points, vert_region, USE_AVERAGES)

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
fps = 10
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

generated_mesh_count = 0
loop_sequence = np.concatenate([smooth_latent_loop, smooth_latent_loop[::-1][1:]], axis=0)
loop_sequence_names = np.concatenate([closest_specimens, closest_specimens[::-1][1:]])
loop_sequence = smooth_latent_loop
loop_sequence_names = closest_specimens
for i, latent_code in enumerate(loop_sequence):
    try:
        generated_mesh_count += 1
        print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{len(loop_sequence)}\033[0m")
        print(f"Frame {i}: Closest to {loop_sequence_names[i]}")
        new_latent = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(device)

        mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device
        )
        mesh_out = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
        mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_geometry()
        mesh_pv = mesh_pv.compute_normals(cell_normals=False, point_normals=True, inplace=False)
        mesh_o3d = pv_to_o3d(mesh_pv)

        for r in renderers:
            r.scene.clear_geometry()
            r.scene.add_geometry("mesh", mesh_o3d, material)

        # Camera setup
        pts = np.asarray(mesh_o3d.vertices)
        center = pts.mean(axis=0)
        r = np.linalg.norm(pts - center, axis=1).max()
        distance = 2.5 * r
        elevation = np.deg2rad(30)

        # Define 4 camera positions
        angle_deg = (i /  (len(loop_sequence) - 1)) * 360 * n_rotations
        angle_rad = np.deg2rad(angle_deg)
        cam_positions = [
            center + np.array([  # Top Left: rotating
                distance * np.cos(angle_rad) * np.cos(elevation),
                distance * np.sin(angle_rad) * np.cos(elevation),
                distance * np.sin(elevation)
            ]),
            center + np.array([0, -distance, 0]),  # Top Right: side
            center + np.array([distance, 0, 0]),  # Bottom Left: back (90° CCW from side)
            center + np.array([0, 0, distance])    # Bottom Right: top-down (90° CCW from side)
        ]
        ups = [
            [0, 0, 1],  # rotating
            [0, 0, 1],  # front
            [0, 0, 1],  # side
            [0, 1, 0],  # top-down
        ]

        for idx, (rdr, pos, up) in enumerate(zip(renderers, cam_positions, ups)):
            rdr.setup_camera(60, center, pos, up)

        # Render images
        imgs = [np.asarray(r.render_to_image()) for r in renderers]
        imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

        # Compose 4 views into 2x2 grid (width=640, height=480)
        top = np.hstack([imgs_bgr[0], imgs_bgr[1]])
        bottom = np.hstack([imgs_bgr[2], imgs_bgr[3]])
        combined = np.vstack([top, bottom])

        # Overlay specimen name info onto each frame
        specimen_name = loop_sequence_names[i]
        parts = specimen_name.split("_")
        family = parts[0] if len(parts) > 0 else specimen_name
        genus = parts[1]  if len(parts) > 1 else ""
        region = parts[-1] if len(parts) > 2 else ""
        if 'C' in region:
            reg_full = 'Cervical'
        elif 'T' in region:
            reg_full = 'Thoracic'
        elif 'L' in region:
            reg_full = 'Lumbar'
        else:
            reg_full = ''
        text = f"Closest Specimen: \n{family}\n{genus}\n{reg_full}"
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        # Position: center of the frame
        center_x = combined.shape[1] // 2
        center_y = combined.shape[0] // 2
        text_x = center_x - 120
        text_y = center_y
        # Put the text
        for j, line in enumerate(text.split("\n")):
                y = text_y + j * (text_size[1] + 10)
                cv2.putText(combined, line, (text_x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

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
